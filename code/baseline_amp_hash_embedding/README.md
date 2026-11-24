# Hash embeddings for high-cardinality histories

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

Sequence fields above the vocabulary threshold share a 500,000-row embedding table with modulo lookup. This recovers some signal from high-cardinality IDs within a fixed memory budget. Different features and IDs can collide. IDs divisible by 500,000 map to the reserved zero row.

## Running this snapshot

Run `bash run.sh`. The launcher sets a one-million vocabulary threshold. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
