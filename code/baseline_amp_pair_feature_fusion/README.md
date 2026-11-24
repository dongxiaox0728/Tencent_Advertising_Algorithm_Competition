# Aligned integer and dense feature fusion

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

The baseline pools ID embeddings and projects numeric values separately. This branch preserves the association between each ID and its aligned value. The statistics path uses a value gate with target-aware attention. The quantile path combines ID, value-bin, and position embeddings. The additional tokens expose pair-level signal to the backbone.

## Running this snapshot

Run `bash run.sh` for `--use_aligned_dense` at width 96. Match width as well as training settings when isolating the feature change. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
