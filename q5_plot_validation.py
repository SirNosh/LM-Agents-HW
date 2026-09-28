"""Question 5: plot final-checkpoint held-out loss (not a training-time curve)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "results" / "sft_lora_20260924_204124"
SUMMARY_PATH = RUN_DIR / "validation_summary.csv"
PLOT_PATH = RUN_DIR / "validation_loss.png"
ORDER = ("base", "lora_r1", "lora_r4", "lora_r16", "full_ft")
LABELS = ("Base", "LoRA r=1", "LoRA r=4", "LoRA r=16", "Full FT")


def main():
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = {row["run"]: row for row in csv.DictReader(handle)}
    values = [float(rows[run]["mean_loss_nats"]) for run in ORDER]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(LABELS, values, color=["#607d8b", "#87b4c5", "#6c9cb8", "#497997", "#354d73"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("Held-out assistant-token loss (nats/token)")
    ax.set_title("Final checkpoints on 100 fixed test_sft conversations")
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    print(f"Saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
