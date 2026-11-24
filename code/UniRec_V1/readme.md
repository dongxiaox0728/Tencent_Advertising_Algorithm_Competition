# UniRec V1 · bidirectional feature and sequence interaction

**Comparison baseline** · the [TAAC reference baseline](../TAAC_baseline/) | [Project version guide](../../readme.md).

## Design and motivation

- **Sparse behavior history.** The `domain_a` loader extracts positions with any nonzero side-information ID and retains the corresponding timestamps. The original notes describe a median raw history of about 578 positions with about 23 active events. This motivated spending the sequence budget on active events.
- **Class imbalance.** V1 adds BCE positive-class weighting (training default `7.0`) and increases the focal-loss positive alpha to `0.875`. The original notes estimated a positive rate of about 12.4%. Set `--pos_weight 1.0` for unweighted BCE.
- **UniRECBlock.** Each domain first encodes its sequence. The block updates NS features through token-wise FFNs, then cross-attends in both directions: NS reads the sequence, and the sequence reads NS. A learned vector gate built from summaries and their element-wise product blends the two residual updates. Domain queries read the enriched sequence. This gives user and ad features and the behavior history a direct route to update each other. The alternative block omits RankMixer, so its dimension need only satisfy the applicable attention constraints.

## Running this snapshot

`bash run.sh` sets the history limits and positive weight and selects HyFormer through the model default. Run `bash run.sh --block_type unirec` for the bidirectional block. Check whether the actual schema uses `domain_a` or `seq_a`; V1's sparse extraction checks the former explicitly. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
