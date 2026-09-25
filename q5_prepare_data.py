"""Question 5: save a fixed 1,000-conversation UltraChat train_sft subset."""

import json
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
SEED = 42
NUM_EXAMPLES = 1_000
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "ultrachat_train_1000_seed42.jsonl"


def main():
    if OUTPUT_PATH.exists():
        print(f"Keeping existing fixed subset: {OUTPUT_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".jsonl.tmp")
    rows = load_dataset(DATASET_ID, split="train_sft", streaming=True)
    rows = rows.shuffle(seed=SEED, buffer_size=10_000)

    count = 0
    with temporary_path.open("w", encoding="utf-8") as output:
        for row in rows:
            messages = row["messages"]
            if not any(message["role"] == "assistant" for message in messages):
                continue
            output.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
            if count == NUM_EXAMPLES:
                break
    if count != NUM_EXAMPLES:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {NUM_EXAMPLES} examples, got {count}")
    temporary_path.replace(OUTPUT_PATH)
    print(f"Saved {count} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
