# UniRec V5 · handcrafted behavior and target-match tokens

**Comparison baseline** · [the UniRec interaction experiments](../UniRec_V3/) | [Project version guide](../../readme.md).

## Design and motivation

V5 gives activity, history coverage, and ad match separate non-sequence tokens. Their independent projections let the model use these signals directly alongside sparse ID embeddings and the original dense profile.

| Token | Inputs | Motivation |
| --- | --- | --- |
| `feat_time` (6 values) | Counts in the past 1, 7, and 30 days, time since last content activity, 7-day share of 30-day activity, and activity span. | Describe engagement frequency, freshness, and concentration. |
| `feat_seq_stat` (5 values) | Valid action, content, and item history lengths, unique item count, and item recency. | Describe how much history is available and how varied it is. |
| `feat_pair` (3 values) | Historical category match fraction, age of the most recent matching event, and recent matching count. | Relate historical interests to the candidate ad. |

The loader retains request-relative event timestamps long enough to compute these fields. The match path looks for item-side category fid 13 and the corresponding history field, and the distinct-item path uses fid 29. The match feature uses a normalized recency value of 1.0 when the category has no match and that path is available. `trainer.py` forwards the three tensors to the model. Each projects through its own linear layer and SiLU. The three added tokens bring the typical NS count from eight to eleven.

## Running this snapshot

Run `bash run.sh` for three UniREC blocks, sparse history settings, and `pos_weight=7.0`. Check the exact history names and field availability in the schema when comparing runs. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
