# Time FiLM, recency-weighted pooling, and global user staleness

**Comparison baseline** · [the cyclical request-time version](../baseline_time_periodic/) | [Project version guide](../../readme.md).

## Design and motivation

Three changes give calendar time and behavior recency distinct roles before attention.

1. **Time FiLM.** The request-time embedding produces channel-wise scale and shift. On NS tokens, this is `x * (1 + gamma) + beta`. The FiLM projection starts at zero so the modulation starts as an identity.
2. **Time-decay pooling.** The domain summary for query generation weights valid events by `exp(-bucket_id / tau)`. Each domain learns its own positive decay scale. The bucket is an age index, so the decay is in bucket units.
3. **Global staleness.** The smallest nonzero event-age bucket across all domains represents the user's most recent recorded activity. Its embedding is added to the request-time representation. This can distinguish users with recent activity from users whose histories have become stale.

## Running this snapshot

Run `bash run.sh` for BF16 AMP, one sinusoidal harmonic, NS FiLM, decay pooling, and user staleness. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

**0.8194 online AUC** in the historical global-staleness run. Earlier notes give roughly **0.8149–0.8150** for the decay-pooling predecessor. [Experiment history](../baseline_film_decay_staleness_cosine_film_on_seq/#cumulative-experiment-chain-how-we-got-here).
