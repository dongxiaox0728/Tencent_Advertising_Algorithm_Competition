# Sequence-FiLM model with torch.compile

**Comparison baseline** · [the sequence-FiLM control](../baseline_film_decay_staleness_cosine_film_on_seq/) | [Project version guide](../../readme.md).

## Design and motivation

This branch aims to reduce training overhead on the 0.823064 control architecture. It enables TF32 for supported CUDA operations, compiles `forward` and `predict` at the method level, and accumulates loss on device between log updates. Method-level wrapping preserves access to the model's optimizer-parameter helpers and checkpoint state. Fewer scalar reads reduce synchronization. Compilation adds warmup, and TF32 can slightly change numerical precision.

## Running this snapshot

Run `bash run.sh` to enable BF16 AMP and `torch.compile` in default mode. Benchmark warmup separately from steady-state throughput and compare AUC with the same schema, checkpoint policy, and RankMixer configuration. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

The parent model recorded **0.823064 online AUC**. The compile launcher estimates a **1.3–1.8×** steady-state speedup. A final measured speedup and independent online AUC are unavailable for this branch.
