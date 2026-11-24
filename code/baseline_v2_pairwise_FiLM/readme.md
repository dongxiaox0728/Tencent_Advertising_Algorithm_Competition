# Pairwise FiLM for aligned user features

**Comparison baseline** · [the 22k cosine and EMA experiment](../baseline_amp_FiLM_decay_pool_staleness_cosine_lr22k_ema999/) | [Project version guide](../../readme.md).

## Design and motivation

The base time branch conditions the user and ad representations on the request time. This version also conditions an individual user ID embedding on its paired numeric value before pooling. Training detects multivalue integer and dense fields with the same fid and length from the schema, covering 62–66 and 89–91 in the documented layout.

For each valid ID, a signed-log transform of the value goes through a zero-initialized affine projection to produce `gamma` and `beta`. The pooler computes `embedding * (1 + gamma) + beta` and masked-mean-pools the resulting vectors. Per-field modulation allows the value to influence each embedding channel. The separate global dense token is still part of the input, and this pair path adds no tokens.

## Running this snapshot

Run `bash run.sh` for the pairwise model with time FiLM, recency pooling, global staleness, a 22k schedule, and EMA 0.999. See the [implementation change log](CHANGES.md). See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

**0.811552 online AUC** for eval task **103121**. This run combined pairwise FiLM and the EMA recipe, so it does not isolate the effect of pairwise modulation against the sequence-FiLM control. [Paired-stat run notes](../baseline_pair_stat_weighted_pool/).
