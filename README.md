# LM Agents Homework — coding artifacts

The filenames use the **sequential subsection numbers** in `homework/hw1.tex`: serving is Question 3, LoRA accounting is Question 4, and supervised fine-tuning is Question 5. The finished written answer is `homework/hw1.tex` and `homework/hw1.pdf` (Dev Vyas). The original template references are retained; no new external citations were added.

## Question 3 — dense vs. MoE serving

- Code: `q3_benchmark.py` (vLLM, one GPU, the three assigned checkpoints, online FP8 per-tensor weights and FP8 KV cache).
- Raw measurements: `results/q1_fp8_20260924_215129/raw_measurements.csv` (54 trials: 12 prefill and 6 decode per checkpoint).
- Visualizations: `results/q1_fp8_20260924_215129/prefill_throughput.png` and `decode_throughput.png`.

The result folder retains its original run ID (`q1_fp8_...`); the numbered submission script was simplified *after* this run. It preserves the tested model IDs, workload text, prompt token lengths, quantization, sampling, concurrency, and repeats. The completed experiment used vLLM 0.30.0 under WSL2 on one RTX 4070 SUPER. Plots show mean throughput with sample-standard-deviation bars. Online FP8 and Q4 use different kernels, so prior Q4 data is not mixed into this submission.

## Question 4 — LoRA accounting

- Code: `q4_lora_accounting.py` (the seven projection dimensions, rank 16, 24 layers, 16-bit parameter/gradient memory and two FP32 AdamW moments).
- Text results: `lora_accounting_results.txt`.

No training or model download is required for this calculation.

## Question 5 — supervised fine-tuning and qualitative comparison

- Data preparation: `q5_prepare_data.py` downloads a deterministic 1,000-conversation sample from `HuggingFaceH4/ultrachat_200k` `train_sft` (seed 42, shuffle buffer 10,000). The exact sampled data used in the completed training run is `data/ultrachat_train_1000_seed42.jsonl`.
- Training code: `q5_sft_lora.py` trains LoRA ranks 1, 4, 16 and full fine-tuning using that fixed sample.
- Raw results: `results/sft_lora_20260924_204124/summary.csv` (parameter counts, last logged loss, training time, peak GPU memory) and `training_metrics.csv` (logged training losses).
- Visualization: `results/sft_lora_20260924_204124/training_loss.png`.
- Qualitative code: `q5_qualitative.py` generates base/LoRA/full-FT responses to three new behavioral prompts; its recorded output is `results/sft_lora_20260924_204124/qualitative_comparison.txt`. `q5_subset_behavior_notes.txt` explains why these behaviors were chosen from the *saved 1,000 conversations* and honestly interprets the generations.

No random `test_sft` evaluation or validation-loss measurement was run. Thus the assignment's validation-loss requirement remains unmet; the qualitative comparison is **not** a quantitative substitute. Training checkpoint binaries are omitted from Git because the full model is about 988 MB and the three adapters total about 46 MB; `q5_qualitative.py` requires the locally saved checkpoints at `results/sft_lora_20260924_204124/weights/` to rerun. The recorded generations and measurements are included. Training and benchmark scripts were simplified after their completed runs; their published results are from the original runs, not new runs of the cleaned scripts.

## Running locally

Use a CUDA-capable environment with `torch`, `transformers`, `datasets`, `peft`, `accelerate`, and (for Q3) `vllm` and `matplotlib`. The completed runs used the local WSL environment and cached model checkpoints. `q5_prepare_data.py` preserves the included training subset if it already exists. Running Q3 or Q5 training again creates a new timestamped result directory; it does not overwrite these measurements.
