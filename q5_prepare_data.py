"""Question 5: save fixed UltraChat training and held-out subsets."""

import json
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
SEED = 42
NUM_EXAMPLES = 1_000
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "ultrachat_train_1000_seed42.jsonl"
TEST_PATH = DATA_DIR / "ultrachat_test_100_seed42.jsonl"


def save_subset(split, count, output_path, buffer_size):
    if output_path.exists():
        print(f"Keeping existing fixed subset: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    rows = load_dataset(DATASET_ID, split=split, streaming=True)
    rows = rows.shuffle(seed=SEED, buffer_size=buffer_size)

    saved = 0
    with temporary_path.open("w", encoding="utf-8") as output:
        for row in rows:
            messages = row["messages"]
            if not any(message["role"] == "assistant" for message in messages):
                continue
            output.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            saved += 1
            if saved == count:
                break
    if saved != count:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {count} examples from {split}, got {saved}")
    temporary_path.replace(output_path)
    print(f"Saved {saved} examples from {split} to {output_path}")


def main():
    save_subset("train_sft", NUM_EXAMPLES, TRAIN_PATH, 10_000)
    save_subset("test_sft", 100, TEST_PATH, 100)


if __name__ == "__main__":
    main()
