"""Question 5: held-out assistant-token loss for the five saved checkpoints.

Run q5_prepare_data.py first. Uses the same chat template, assistant-only labels,
and final-512-token truncation as q5_sft_lora.py. This is a final-checkpoint
comparison; the original run saved no intermediate checkpoints, so it cannot
retroactively produce a validation-loss curve during training.
"""

import csv
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from q5_sft_lora import MAX_LENGTH, MODEL_ID, encode_conversation

ROOT = Path(__file__).resolve().parent
TEST_PATH = ROOT / "data" / "ultrachat_test_100_seed42.jsonl"
RUN_DIR = ROOT / "results" / "sft_lora_20260924_204124"
SUMMARY_PATH = RUN_DIR / "validation_summary.csv"
DETAIL_PATH = RUN_DIR / "validation_per_conversation.csv"
EXPECTED_EXAMPLES = 100
VARIANTS = (
    ("base", None),
    ("lora_r1", RUN_DIR / "weights" / "lora_r1"),
    ("lora_r4", RUN_DIR / "weights" / "lora_r4"),
    ("lora_r16", RUN_DIR / "weights" / "lora_r16"),
    ("full_ft", RUN_DIR / "weights" / "full_ft"),
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; wait until the shared GPU is available")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Prepare the held-out subset first: {TEST_PATH}")
    for name, checkpoint in VARIANTS:
        if checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(f"Missing {name} checkpoint: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    with TEST_PATH.open(encoding="utf-8") as handle:
        conversations = [json.loads(line)["messages"] for line in handle]
    if len(conversations) != EXPECTED_EXAMPLES:
        raise ValueError(f"Expected {EXPECTED_EXAMPLES} test conversations; got {len(conversations)}")
    encoded = [encode_conversation(tokenizer, messages) for messages in conversations]
    if any(row is None for row in encoded):
        raise ValueError("A test conversation has no assistant tokens after truncation")

    summaries, details = [], []
    for name, checkpoint in VARIANTS:
        source = checkpoint if name == "full_ft" else MODEL_ID
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda").eval()
        if name.startswith("lora_"):
            model = PeftModel.from_pretrained(model, checkpoint).eval()
        loss_sum, token_count = 0.0, 0
        with torch.inference_mode():
            for index, row in enumerate(encoded):
                inputs = torch.tensor([row["input_ids"]], device="cuda")
                labels = torch.tensor([row["labels"]], device="cuda")
                # CausalLM shifts labels left by one internally. The first label
                # has no prediction, and user/prompt tokens have label -100.
                count = int((labels[:, 1:] != -100).sum().item())
                if count == 0:
                    raise ValueError(f"No predictable assistant tokens in example {index}")
                mean_loss = model(input_ids=inputs, labels=labels).loss.item()
                weighted_loss = mean_loss * count
                loss_sum += weighted_loss
                token_count += count
                details.append({
                    "run": name, "example_index": index, "target_tokens": count,
                    "mean_loss_nats": mean_loss,
                })
        summaries.append({
            "run": name, "conversations": len(encoded),
            "target_tokens": token_count, "mean_loss_nats": loss_sum / token_count,
        })
        print(f"{name}: {loss_sum / token_count:.5f} nats/token on {token_count} target tokens", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for path, fields, rows in (
        (SUMMARY_PATH, ("run", "conversations", "target_tokens", "mean_loss_nats"), summaries),
        (DETAIL_PATH, ("run", "example_index", "target_tokens", "mean_loss_nats"), details),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {SUMMARY_PATH} and {DETAIL_PATH}")


if __name__ == "__main__":
    main()
