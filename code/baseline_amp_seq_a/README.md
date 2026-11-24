# Sparse event extraction for seq_a

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

Compacts the selected domain after the ordinary loader fills its history buffer. It uses a positive ID in the first feature slot as the event marker, retains the last positions up to the sparse cap, and moves all other fields and age buckets in lockstep. This lets the sparse cap count active events within the portion loaded by the ordinary history limit.

## Running this snapshot

Run `bash run.sh` for `--sparse_domains seq_a:64`. Use the exact schema domain names when overriding the flag. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
