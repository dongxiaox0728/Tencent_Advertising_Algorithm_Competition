# Adjacent event interval embeddings

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

Adds a learned embedding of the bucketed, nonnegative difference between each event timestamp and the preceding timestamp in stored order. The first event and padded positions use bucket zero. This offers a signal for the spacing between events in addition to their age at request time.

## Running this snapshot

Run `bash run.sh`. The interval embedding follows the time-bucket setting. Verify stored timestamp order when interpreting intervals. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
