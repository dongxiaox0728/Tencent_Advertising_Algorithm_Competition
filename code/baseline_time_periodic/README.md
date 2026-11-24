# Cyclical request-time encoding

**Comparison baseline** · [the discrete time-feature experiment](../baseline_amp_time_feature/) | [Project version guide](../../readme.md).

## Design and motivation

Replaces three calendar lookups with sine/cosine pairs for periods of 24 hours, 7 days, and 168 hours. A small projection creates one NS token and a learned scalar sigmoid gate controls its contribution. This makes neighboring timestamps close in the representation even around midnight or the week boundary.

## Running this snapshot

Run `bash run.sh` to enable the cyclical time token. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
