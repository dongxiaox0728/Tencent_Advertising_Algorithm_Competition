# Query-token allocation by behavior domain

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

The baseline allocates an equal number of queries to each sequence. This experiment gives some domains more queries to test whether they benefit from a larger retrieval budget. `train.py` resolves `--per_seq_queries` by exact domain name and falls back to `--num_queries` for other domains. The query generator and HyFormer blocks use the resulting list for generation, decoding, and token splitting.

Keep the total query count fixed at eight for the four-domain comparisons. With eight NS tokens and `d_model=64`, this gives `T=16` and satisfies full RankMixer. Equal token count controls the mixer size. Cross-attention cost can still vary with the length of the domain receiving the extra queries. The original plan proposed rotating a 4–2–1–1 allocation across the four domains. The logged runs used a 5–1–1–1 allocation and its permutations.

## Running this snapshot

Run `bash run.sh` for two queries per domain. Use a schema-matched override such as `bash run.sh --per_seq_queries seq_a:5,seq_b:1,seq_c:1,seq_d:1` for the recorded A-heavy allocation. Some historical examples use `domain_a` through `domain_d`. Check the resolved allocation printed at startup. The `--eval_only` branch skips training and evaluates once, but this snapshot does not load a checkpoint on that path. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

| Experiment | A / B / C / D | Best validation ROC-AUC | Change vs. uniform |
| --- | --- | ---: | ---: |
| E0 | 2 / 2 / 2 / 2 | 0.8624656796455383 | — |
| E1 | 5 / 1 / 1 / 1 | 0.8618979454040527 | −0.00056773 |
| E2 | 1 / 5 / 1 / 1 | 0.8621273636817932 | −0.00033832 |
| E3 | 1 / 1 / 5 / 1 | Unavailable | — |
| E4 | 1 / 1 / 1 / 5 | Unavailable | — |

The uniform allocation led the three recorded validation runs. A complete four-domain comparison and online evaluation would be needed to establish a preferred allocation.
