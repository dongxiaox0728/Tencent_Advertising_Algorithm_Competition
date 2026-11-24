# Discrete request-time features

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

Encodes hour of day, a day-of-week index, and hour of week from the request timestamp using separate lookup tables and projects them into a calendar NS token. This tests whether time of request carries conversion signal after accounting for the age of historical events. The time tables are optimized with AdamW.

## Running this snapshot

Run `bash run.sh` to enable the calendar token. The launcher uses four user-int tokens so the typical total stays at `T=16`. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

The historical basic-time run `baseline_time_3` recorded **0.8142 online AUC**. The local snapshot has no independently identified evaluation task. [Experiment history](../baseline_film_decay_staleness_cosine_film_on_seq/#cumulative-experiment-chain-how-we-got-here).
