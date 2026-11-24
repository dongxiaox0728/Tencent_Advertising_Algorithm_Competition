# PCVRHyFormer with automatic mixed precision

**Comparison baseline** · [the TAAC baseline](../TAAC_baseline/) | [Project version guide](../../readme.md).

## Design and motivation

Adds CUDA autocast for the training forward pass, loss, and evaluation. BF16 is the default because its exponent range helps limit overflow and underflow. FP16 uses a gradient scaler that unscales both optimizers before clipping, steps each optimizer, then updates once. BF16 and FP32 use ordinary backward and optimizer steps. Unsupported BF16 hardware falls back to FP32 with a warning. Evaluation casts logits to FP32 for AUC and log loss.

The experiment isolates training cost from model design. The original engineering targets were 50–100% higher throughput and 25–35% lower peak memory. The validation plan compares loss, AUC, and NaN counts on a small matched subset, then measures full-epoch throughput and peak memory and checks checkpoint loading.

## Running this snapshot

Use `bash run.sh` for BF16, `bash run.sh --amp_dtype fp16` for FP16, or `bash run.sh --no_amp` for FP32. Match model arguments and schema when comparing modes. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

Historical `Baseline_AMP` recorded **0.8137 online AUC**. Matched throughput and peak-memory measurements are unavailable for this snapshot. [Experiment history](../baseline_film_decay_staleness_cosine_film_on_seq/#cumulative-experiment-chain-how-we-got-here).
