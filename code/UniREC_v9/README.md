# UniREC V9 · typed dense feature alignment

**Comparison baseline** · [UniRec V6](../UniRec_V6/) and the [V8 token-dropout changes](V8_CHANGES.md) | [Project version guide](../../readme.md).

## Design and motivation

The loader and model use the same schema-driven approach to match integer and dense features that share a fid and length. V9 handles each type of value separately.

| Feature IDs | Paired input to user-int tokenizer | Dedicated user-dense token |
| --- | --- | --- |
| 61, 87 | None | L2 normalization and a linear projection. |
| 62–66 | Add an MLP residual from the ID embedding and signed-log value clipped to `[-20, 20]`, then masked-mean-pool. | Nonnegative log-scaled values and a linear projection. |
| 89–91 | Clip the value to `[-1,1]`, map it to one of 32 bins, and add value-bin and position embeddings to the ID embedding before pooling. | Clip the values and use a small Conv1d encoder when dimensions match the expected 3 × 10 layout. |

The three projections preserve a global view of the dense statistics while the aligned tokenizer preserves each ID's value. Sequence events still combine field embeddings with elapsed-time buckets. During training, user-token dropout randomly zeros entire user-side NS vectors and sequence-token dropout zeros whole event vectors. This reduces dependence on individual groups and events.

The launcher uses five user-int tokens, three user-dense tokens, two item-int tokens, and two queries for each of four domains. This makes `T=18` with the standard schema, so `d_model=72` satisfies full RankMixer's divisibility constraint. The forward path tokenizes the static fields, builds and masks histories, generates queries, runs the HyFormer stack, and classifies from the final queries. Inference rebuilds the alignment and dense-group configuration from the saved schema.

## Running this snapshot

Run `bash run.sh` for BF16 AMP, width 72, a learning rate of `8e-5`, ordinary dropout of `0.015`, and user and sequence token dropout of `0.05` each. See [V9 changes](V9_CHANGES.md). See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
