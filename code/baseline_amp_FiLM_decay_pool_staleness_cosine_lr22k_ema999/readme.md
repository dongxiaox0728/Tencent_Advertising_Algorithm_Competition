# Cosine learning-rate schedule and EMA

**Comparison baseline** · [the user-staleness version](../basel_amp_FiLM_decay_pool_user_staleness/) | [Project version guide](../../readme.md).

## Design and motivation

This version combines request-time FiLM, recency-weighted pooling, global staleness, and BF16 AMP with two training experiments. The optimizer still uses AdamW for dense parameters and Adagrad for embedding tables.

- **Warmup and cosine decay.** Ramp the learning rate for 500 steps, then decay it toward 10% of the peak. The launcher sets a 22,000-step schedule. Warmup limits abrupt early updates and the cosine tail reduces step size near the end of training.
- **Exponential moving average.** Update shadow weights after optimizer steps using `shadow = 0.999 * shadow + 0.001 * current`. Evaluation and checkpointing can use those smoothed weights. The intent was to test whether averaging recent steps improved generalization.
- **Recency alternatives.** Separate launchers add per-domain staleness or event-count embeddings. The former distinguishes the last event in each domain, while the latter summarizes activity volume independently of event age.

## Running this snapshot

Run `bash run.sh` for the 22k schedule and EMA 0.999. Use `run_per_domain_staleness.sh` and `run_activity_count.sh` for the alternatives. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

| Historical milestone | Online AUC |
| --- | ---: |
| Standalone calendar token and per-channel gate | 0.8146 |
| NS FiLM | 0.81485 |
| Recency-weighted pooling | 0.8149 |
| Global staleness | 0.8194 |
| Add warmup and cosine schedule, before EMA | 0.8212 |
| 22k cosine schedule with EMA 0.999 | 0.8198 |

The iteration notes also record negative AUC changes for four Fourier harmonics (about −0.016), random event masking at 0.15 (−0.001), per-domain staleness (−0.010), and per-domain counts (−0.021). The notes give approximate deltas, with limited details on matched controls. The later [sequence-FiLM control](../baseline_film_decay_staleness_cosine_film_on_seq/) uses a 30k schedule and disables EMA.
