"""Question 5: train Qwen2.5-0.5B with LoRA ranks and full fine-tuning."""

import csv
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B"
LORA_RANKS = (1, 4, 16)
TRAIN_EXAMPLES = 1000
MAX_LENGTH = 512
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
WARMUP_STEPS = 4
LORA_LR = 2e-4
FULL_FT_LR = 2e-5
SEED = 42
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "ultrachat_train_1000_seed42.jsonl"


def encode_conversation(tokenizer, messages):
    """Apply loss only to assistant turns; truncate to the configured length."""
    input_ids, labels = [], []
    for end in range(1, len(messages) + 1):
        current = tokenizer.apply_chat_template(
            messages[:end], tokenize=True, add_generation_prompt=False
        )
        if hasattr(current, "keys"):
            current = current["input_ids"]
        if current[:len(input_ids)] != input_ids:
            raise ValueError("Chat template changed an earlier message prefix")
        new_ids = current[len(input_ids):]
        input_ids = current
        if messages[end - 1]["role"] == "assistant":
            labels.extend(new_ids)
        else:
            labels.extend([-100] * len(new_ids))
    input_ids, labels = input_ids[-MAX_LENGTH:], labels[-MAX_LENGTH:]
    if not input_ids or all(token == -100 for token in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def main():
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit(
            "Install SFT dependencies in the training environment: "
            "datasets peft accelerate"
        ) from error

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing fixed 1,000-example training subset: {TRAIN_PATH}. "
            "Prepare it from HuggingFaceH4/ultrachat_200k train_sft first."
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run requires a CUDA GPU with BF16 support")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with TRAIN_PATH.open(encoding="utf-8") as handle:
        messages = [json.loads(line)["messages"] for line in handle]
    if len(messages) != TRAIN_EXAMPLES:
        raise RuntimeError(f"Expected {TRAIN_EXAMPLES} examples; found {len(messages)}")
    rows = [encode_conversation(tokenizer, conversation) for conversation in messages]
    if any(row is None for row in rows):
        raise RuntimeError("A conversation has no assistant target tokens")
    train_dataset = Dataset.from_list(rows)
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)

    output_dir = ROOT / "results" / ("q5_sft_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "training_metrics.csv"
    summary_path = output_dir / "summary.csv"
    weights_dir = output_dir / "weights"
    weights_dir.mkdir()
    metric_fields = ("run", "step", "epoch", "loss")
    summary_fields = (
        "run", "trainable_parameters", "learning_rate", "training_seconds",
        "peak_allocated_gib", "peak_reserved_gib", "final_train_loss", "weights_path",
    )

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file, \
            summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        metrics_writer.writeheader()
        summary_writer.writeheader()

        runs = [(f"lora_r{rank}", rank) for rank in LORA_RANKS] + [("full_ft", None)]
        for run_name, rank in runs:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            if rank is not None:
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=rank,
                        lora_alpha=2 * rank,
                        lora_dropout=0.05,
                        target_modules="all-linear",
                        task_type=TaskType.CAUSAL_LM,
                    ),
                )
                model.enable_input_require_grads()
            model.config.use_cache = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            learning_rate = LORA_LR if rank is not None else FULL_FT_LR
            args = TrainingArguments(
                output_dir=str(output_dir / run_name),
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUMULATION,
                learning_rate=learning_rate,
                warmup_steps=WARMUP_STEPS,
                weight_decay=0.0,
                bf16=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                eval_strategy="no",
                logging_strategy="steps",
                logging_steps=10,
                logging_first_step=True,
                save_strategy="no",
                report_to="none",
                seed=SEED,
                data_seed=SEED,
                dataloader_num_workers=0,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                data_collator=collator,
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            trainer.train()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            history = trainer.state.log_history
            for record in history:
                if "loss" in record:
                    metrics_writer.writerow({
                        "run": run_name,
                        "step": record.get("step", ""),
                        "epoch": record.get("epoch", ""),
                        "loss": record["loss"],
                    })
            metrics_file.flush()
            final_loss = next(
                (record["loss"] for record in reversed(history) if "loss" in record), ""
            )
            model.config.use_cache = True
            checkpoint = weights_dir / run_name
            trainer.model.save_pretrained(checkpoint)
            summary_writer.writerow({
                "run": run_name,
                "trainable_parameters": trainable,
                "learning_rate": learning_rate,
                "training_seconds": round(elapsed, 2),
                "peak_allocated_gib": round(peak_allocated, 3),
                "peak_reserved_gib": round(peak_reserved, 3),
                "final_train_loss": final_loss,
                "weights_path": str(checkpoint),
            })
            summary_file.flush()
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Training results: {output_dir}")
    print("Note: this script records training loss only; held-out evaluation is separate.")


if __name__ == "__main__":
    main()
