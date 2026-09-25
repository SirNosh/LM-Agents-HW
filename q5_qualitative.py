"""Question 5: compare base and trained models on new behavioral prompts.

The prompts are inspired by patterns in the saved 1,000 training conversations,
not copied from them. This is a small qualitative comparison, not a validation
loss or a random test_sft evaluation.
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "results" / "sft_lora_20260924_204124"
OUTPUT_PATH = RUN_DIR / "qualitative_comparison.txt"
MAX_NEW_TOKENS = 220

# Behaviors seen in the fixed training subset: detailed steps, adapting a prior
# answer to a follow-up, and grounding an answer in user-provided material.
PROMPTS = (
    (
        "Actionable, structured plan",
        [
            {"role": "user", "content": (
                "I volunteer at a neighborhood tool library. Give me a practical "
                "step-by-step plan for recording damaged tools, repairing them, "
                "and notifying borrowers."
            )},
        ],
    ),
    (
        "Conversational follow-up and revision",
        [
            {"role": "user", "content": (
                "Help me organize a weekend clothing-swap event in our community room."
            )},
            {"role": "assistant", "content": (
                "Set a time and venue, ask neighbors to bring clean wearable items, "
                "arrange labeled tables by size, and plan where leftovers will go."
            )},
            {"role": "user", "content": (
                "That helps. Could you adapt the plan for volunteers who have only "
                "two hours to set up and must make the event accessible to wheelchair users?"
            )},
        ],
    ),
    (
        "Answer grounded in a supplied passage",
        [
            {"role": "user", "content": (
                "Answer only from this notice: The repair cafe opens Saturday at "
                "10 a.m. Walk-ins are welcome until noon; after noon, visitors "
                "need a booking. Electrical items cannot be accepted this month. "
                "Can I bring a broken lamp at 11 a.m., and is a booking needed?"
            )},
        ],
    ),
)


def generate(model, tokenizer, messages):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to load these BF16 checkpoints")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    report = [
        "Question 5: qualitative comparison on new prompts (not test_sft scores)",
        "Same prompts and greedy decoding; max_new_tokens=220.",
        "Inspect instruction following, structure, grounding, and follow-up adaptation.",
        "",
    ]
    variants = (
        ("base", None),
        ("lora_r1", RUN_DIR / "weights" / "lora_r1"),
        ("lora_r4", RUN_DIR / "weights" / "lora_r4"),
        ("lora_r16", RUN_DIR / "weights" / "lora_r16"),
        ("full_ft", RUN_DIR / "weights" / "full_ft"),
    )
    outputs = {title: {} for title, _ in PROMPTS}
    for name, checkpoint in variants:
        if checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        source = checkpoint if name == "full_ft" else MODEL_ID
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda").eval()
        if name.startswith("lora_"):
            model = PeftModel.from_pretrained(model, checkpoint).eval()
        for title, messages in PROMPTS:
            outputs[title][name] = generate(model, tokenizer, messages)
        del model
        torch.cuda.empty_cache()
        print(f"Generated {name}", flush=True)

    for title, messages in PROMPTS:
        report += ["=" * 72, title, "Messages:", json.dumps(messages, ensure_ascii=False, indent=2), ""]
        for name, _ in variants:
            report += [f"[{name}]", outputs[title][name], ""]
    OUTPUT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
