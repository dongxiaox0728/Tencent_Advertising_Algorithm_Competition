# UniRec V6 · residual value-conditioned pooling

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

When an integer feature and a dense feature share a fid and length, their positions represent paired IDs and numeric values. The baseline independently mean-pools the ID embeddings and projects the whole dense vector. V6 preserves the position-wise association before pooling.

At training startup, the code constructs a mapping from each aligned integer feature index to the dense offset and length in the schema. For each nonpadding ID, the paired value receives a signed `log1p` transform and is clipped to `[-20, 20]`. An MLP reads the ID embedding and transformed value, and adds a residual to the ID embedding before masked mean pooling. Both group and RankMixer tokenizers support the change. Unpaired IDs follow the ordinary pooling path. The dense vector also remains available as its own global NS token, so the ablation can measure the value of explicit pairing.

## Running this snapshot

Run `bash run.sh` for BF16 AMP. Use the same schema, split, and hyperparameters as the AMP baseline to isolate aligned tokenization. See [V6 implementation notes](V6_CHANGES.md). See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
