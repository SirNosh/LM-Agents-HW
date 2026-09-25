"""Question 4: LoRA parameter and optimizer-memory accounting."""

RANK = 16
LAYERS = 24
# (input width, output width), from the homework statement.
PROJECTIONS = (
    (896, 896),   # q_proj
    (896, 128),   # k_proj
    (896, 128),   # v_proj
    (896, 896),   # o_proj
    (896, 4864),  # gate_proj
    (896, 4864),  # up_proj
    (4864, 896),  # down_proj
)
FULL_FT_PARAMETERS = 490_000_000
MIB = 1024**2
GIB = 1024**3


def report_memory(label, parameters):
    # Two-byte parameter + two-byte gradient. AdamW adds two four-byte moments.
    parameters_and_gradients = parameters * 4
    with_adamw = parameters * 12
    print(f"{label}: {parameters:,} trainable parameters")
    print(
        f"  Parameters + gradients: {parameters_and_gradients:,} bytes "
        f"({parameters_and_gradients / MIB:.2f} MiB; "
        f"{parameters_and_gradients / GIB:.4f} GiB)"
    )
    print(
        f"  Including two FP32 AdamW moments: {with_adamw:,} bytes "
        f"({with_adamw / MIB:.2f} MiB; {with_adamw / GIB:.4f} GiB)"
    )


def main():
    per_layer = sum(RANK * (input_width + output_width)
                    for input_width, output_width in PROJECTIONS)
    lora_parameters = LAYERS * per_layer

    print(f"LoRA parameters per layer: {per_layer:,}")
    print(f"LoRA parameters across {LAYERS} layers: {lora_parameters:,}\n")
    report_memory("LoRA rank 16", lora_parameters)
    print()
    report_memory("Full fine-tuning (0.49B parameters)", FULL_FT_PARAMETERS)


if __name__ == "__main__":
    main()
