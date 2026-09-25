"""Question 3: single-GPU OLMo dense/MoE serving benchmark with FP8 weights."""

from __future__ import annotations

import csv
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Use the CUDA toolkit shipped with the vLLM environment under WSL.
venv_bin = Path(sys.executable).parent
cuda_home = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "nvidia"
    / "cu13"
)
os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")
if (cuda_home / "bin" / "nvcc").is_file():
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["PATH"] = str(cuda_home / "bin") + os.pathsep + os.environ["PATH"]

MODELS = (
    ("olmo2_1b", "allenai/OLMo-2-0425-1B-Instruct"),
    ("olmoe_1b_7b", "allenai/OLMoE-1B-7B-0924-Instruct"),
    ("olmo2_7b", "allenai/OLMo-2-1124-7B"),
)
CONTEXT_LENGTHS = (512, 1024, 2048, 3000)
CONCURRENCIES = (1, 2, 4)
PREFILL_REPEATS = 3
DECODE_REPEATS = 2
OUTPUT_TOKENS = 3000
MAX_MODEL_LEN = 4096
# Fixed deterministic technical text used for the measured prefill workloads.
TEXT_BLOCK = """
The variational posterior couples a nonstationary latent manifold to a sparse
expert-routing prior, while the optimizer's preconditioned gradient estimates
remain sensitive to activation anisotropy and the spectral condition number of
the Fisher information matrix. In a multiscale dynamical system, cross-attention
propagates a learned representation through a residual stream whose covariance
structure is constrained by layer normalization and an implicit low-rank
factorization. The resulting inference trajectory reflects both posterior
calibration and the inductive bias introduced by tokenization, positional
encoding, and autoregressive conditioning. A mixture-of-experts controller
assigns each token to a top-k subset using router logits, balancing conditional
computation against expert utilization, dispatch overhead, and load-balancing
regularization. At the systems level, paged key-value storage, memory bandwidth,
cache-line locality, and kernel fusion affect latency independently of nominal
parameter count. Quantization perturbs the weight distribution according to
group-wise scale estimation, zero-point selection, activation outliers, and
rounding error; these perturbations can interact with normalization statistics
and attention-score dynamic range. Robust measurement therefore separates
prefill complexity from iterative decoding, controls the effective batch size,
and records prompt tokens, generated tokens, wall-clock duration, and scheduling
conditions. The experimental protocol fixes the sampling policy, disables
shared-prefix reuse, and uses deterministic inputs so that differences in
throughput can be attributed to model architecture and serving implementation
rather than uncontrolled prompt variation. Uncertainty in the measured mean
arises from clock-state transitions, host scheduling, queueing, compilation,
and allocator behavior, so warm-up requests are excluded from the reported
observations. Statistical interpretation must distinguish active parameters
from resident parameters, arithmetic intensity from memory traffic, and
per-request latency from aggregate tokens per second. These distinctions are
especially important when comparing dense matrix multiplication with sparse
conditional execution over a heterogeneous expert bank.
""".strip()
PREFILL_INSTRUCTION = (
    "\n\nInstruction: Ignore the technical passage above and respond with only "
    "the word yes."
)
DECODE_PROMPT = "Generate a 3k token/2000 word essay on Global Warming"
CSV_FIELDS = (
    "model_key", "model_id", "weight_quantization", "kv_cache_dtype",
    "workload", "repeat", "concurrency", "input_tokens", "generated_tokens",
    "elapsed_seconds", "tokens_per_second",
)


def chat_prompt(tokenizer, body_ids):
    """Wrap body token IDs in the model chat template without retokenizing them."""
    if not tokenizer.chat_template:
        prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        return prefix + body_ids
    marker = "VLLM_BODY_MARKER_7193"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
    )
    index = rendered.find(marker)
    if index < 0:
        raise RuntimeError("Chat template did not preserve the content marker")
    before = tokenizer.encode(rendered[:index], add_special_tokens=False)
    after = tokenizer.encode(rendered[index + len(marker):], add_special_tokens=False)
    return before + body_ids + after


def make_prefill_prompt(tokenizer, target_length):
    block_ids = tokenizer.encode(TEXT_BLOCK, add_special_tokens=False)
    context_ids = (block_ids * math.ceil(target_length / len(block_ids)))[:target_length]
    instruction_ids = tokenizer.encode(PREFILL_INSTRUCTION, add_special_tokens=False)
    return chat_prompt(tokenizer, context_ids + instruction_ids)


def record(writer, model_key, model_id, workload, repeat, concurrency,
           input_tokens, generated_tokens, elapsed):
    token_count = input_tokens if workload == "prefill" else generated_tokens
    writer.writerow({
        "model_key": model_key,
        "model_id": model_id,
        "weight_quantization": "online FP8 per-tensor",
        "kv_cache_dtype": "fp8",
        "workload": workload,
        "repeat": repeat,
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed, 6),
        "tokens_per_second": round(token_count / elapsed, 4),
    })


def run_model(model_key, model_id, csv_path):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        quantization="fp8_per_tensor",
        kv_cache_dtype="fp8",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max(CONCURRENCIES),
        max_num_batched_tokens=MAX_MODEL_LEN,
        gpu_memory_utilization=0.86,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=42,
    )
    prefill_params = SamplingParams(
        temperature=0, max_tokens=1, min_tokens=1, ignore_eos=True
    )
    decode_params = SamplingParams(
        temperature=0, max_tokens=OUTPUT_TOKENS, min_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
    )
    warmup_params = SamplingParams(
        temperature=0, max_tokens=16, min_tokens=16, ignore_eos=True
    )

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        for context_length in CONTEXT_LENGTHS:
            prompt_ids = make_prefill_prompt(tokenizer, context_length)
            prompt = {"prompt_token_ids": prompt_ids}
            llm.generate([prompt], prefill_params, use_tqdm=False)
            for repeat in range(1, PREFILL_REPEATS + 1):
                start = time.perf_counter()
                output = llm.generate([prompt], prefill_params, use_tqdm=False)[0]
                elapsed = time.perf_counter() - start
                record(writer, model_key, model_id, "prefill", repeat, 1,
                       len(prompt_ids), len(output.outputs[0].token_ids), elapsed)
                handle.flush()

        decode_ids = chat_prompt(
            tokenizer, tokenizer.encode(DECODE_PROMPT, add_special_tokens=False)
        )
        prompt = {"prompt_token_ids": decode_ids}
        for concurrency in CONCURRENCIES:
            batch = [prompt] * concurrency
            llm.generate(batch, warmup_params, use_tqdm=False)
            for repeat in range(1, DECODE_REPEATS + 1):
                start = time.perf_counter()
                outputs = llm.generate(batch, decode_params, use_tqdm=False)
                elapsed = time.perf_counter() - start
                generated = sum(len(item.outputs[0].token_ids) for item in outputs)
                if generated != concurrency * OUTPUT_TOKENS:
                    raise RuntimeError("Decode batch did not produce the requested tokens")
                record(writer, model_key, model_id, "decode", repeat, concurrency,
                       len(decode_ids), generated, elapsed)
                handle.flush()


def make_plots(csv_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = {
        "olmo2_1b": ("OLMo-2 1B dense", "o"),
        "olmoe_1b_7b": ("OLMoE 1B active / 7B total", "s"),
        "olmo2_7b": ("OLMo-2 7B dense", "^"),
    }
    plots = (
        ("prefill", "input_tokens", "Input context length (tokens)",
         "Prompt tokens per second", "FP8 OLMo prefill throughput", "prefill_throughput.png"),
        ("decode", "concurrency", "Parallel generations",
         "Generated tokens per second", "FP8 OLMo decode throughput (3000 tokens/request)",
         "decode_throughput.png"),
    )
    for workload, x_field, xlabel, ylabel, title, filename in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, (label, marker) in labels.items():
            model_rows = [row for row in rows
                          if row["model_key"] == key and row["workload"] == workload]
            xs = sorted({int(row[x_field]) for row in model_rows})
            means, errors = [], []
            for x_value in xs:
                rates = [float(row["tokens_per_second"]) for row in model_rows
                         if int(row[x_field]) == x_value]
                means.append(statistics.mean(rates))
                errors.append(statistics.stdev(rates) if len(rates) > 1 else 0)
            if xs:
                ax.errorbar(xs, means, yerr=errors, marker=marker,
                            capsize=3, label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        model_key, model_id = next(model for model in MODELS if model[0] == sys.argv[2])
        run_model(model_key, model_id, Path(sys.argv[3]))
        return

    output_dir = Path(__file__).resolve().parent / "results" / (
        "q1_fp8_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "raw_measurements.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
    script = str(Path(__file__).resolve())
    for model_key, _ in MODELS:
        subprocess.run(
            [sys.executable, script, "--worker", model_key, str(csv_path)],
            check=True,
        )
    make_plots(csv_path, output_dir)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
