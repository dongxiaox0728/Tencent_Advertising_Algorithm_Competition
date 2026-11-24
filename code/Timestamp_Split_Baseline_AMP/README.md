# Parquet row-group timestamp diagnostics

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

Logs minimum and maximum `label_time` per Parquet row group before the standard partition and checks if group maxima are nondecreasing. The goal was to test whether file order approximates chronology. Overlapping groups can pass this diagnostic, and the actual split still follows file and row-group order.

## Running this snapshot

Run `bash run.sh` and look for the group ranges and the order check in the log. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
