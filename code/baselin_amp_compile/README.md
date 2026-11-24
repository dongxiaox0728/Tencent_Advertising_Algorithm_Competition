# AMP baseline with optional torch.compile

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

Adds an optional compile path to the AMP training workflow to improve steady-state throughput. Compare warmup time, steady-state throughput, peak memory, and validation metrics at the same precision and batch size.

## Running this snapshot

Run `bash run.sh --use_compile` to enable compilation. Plain `bash run.sh` uses BF16 with compilation disabled. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
