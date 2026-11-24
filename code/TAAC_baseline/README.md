# TAAC reference baseline · PCVRHyFormer

**Comparison baseline** · the original competition model. [Project version guide](../../readme.md).

## Design and motivation

The baseline estimates post-click conversion probability using a candidate ad, user features, and four behavior histories. The [English data-to-tokens walkthrough](DATA_TO_TOKENS_README.md) gives tensor shapes and worked examples.

1. **Parquet pipeline.** `FeatureSchema` maps feature IDs to positions in flat user and item tensors. The iterable loader assigns Parquet row groups to workers, streams Arrow batches, and shuffles training examples within a bounded buffer. The target is `label_type == 2`. Missing and invalid IDs map to zero. The loader can clip or reject out-of-vocabulary IDs and tracks out-of-bounds counts per feature. It pads or truncates lists and aligns each timestamp to its sequence event.
2. **Static features.** `GroupNSTokenizer` projects each semantic group from `ns_groups.json`. `RankMixerNSTokenizer` embeds individual fields, masked-mean-pools lists, then concatenates and splits the embeddings into projected user and item tokens. Dense user features get a separate projection. These are the non-sequence (NS) tokens.
3. **Behavior histories.** Each domain projects the concatenated embeddings of an event's fields and adds a learned embedding of its age bucket. The bucket comes from request time minus event time. Zero represents a missing timestamp. Sequence lengths define attention masks, and the vocabulary skip threshold limits embedding memory.
4. **Domain queries.** `MultiSeqQueryGenerator` normalizes the concatenated NS tokens and a masked mean of the corresponding history. Independent FFNs generate the domain's query tokens.
5. **HyFormer stack.** Each block updates the histories, lets each query group attend to its matching history, then concatenates the decoded queries with NS tokens. Full RankMixer rewires hidden-channel groups across tokens, applies a token-wise FFN with residual and normalization, and splits the stream back into domain queries and NS. It requires `d_model % T == 0`. The alternate modes are `ffn_only` and `none`.
6. **Classifier.** The final queries are flattened and projected before the binary classifier produces a logit. Apply sigmoid for a conversion probability. `predict()` also returns a pooled representation.

### Encoders and training

The default pre-normalized Transformer uses self-attention and an FFN. The SwiGLU encoder provides a token-wise gated projection. The Longer encoder can use a bounded set of queries to compress a long history and carries forward an updated padding mask. Optional RoPE rotates attention Q and K.

The trainer uses Adagrad for embeddings and AdamW for the other parameters, clips gradients with `foreach=False`, and supports reinitialization of selected high-cardinality embeddings. The default objective is BCE with logits, with a focal-loss option. Validation logs ROC-AUC and log loss. Early stopping tracks AUC, and checkpoints include the schema and training config, plus feature groups where applicable. The split follows sorted file and row-group order. `utils.py` implements seeding, logging, early stopping, and focal loss.

## Running this snapshot

Run `bash run.sh`. The launcher selects five user-int tokens, two item-int tokens, and two queries per domain. With one user-dense token and four domains, `T=16` and `d_model=64`. Defaults include two blocks and four attention heads. The launcher also contains a group-tokenizer example. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No online AUC is recorded for this exact reference snapshot. The earliest scored run in the experiment history is [UniRec V2](../UniRec_V2/).
