# Positive-class weighting for binary cross-entropy

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

Adds `pos_weight` to scale the positive examples in BCE with logits. The experiment asks if giving rare conversions more influence on training improves ranking. Use ROC-AUC and log loss to check the ranking and probability effects.

## Running this snapshot

Run `bash run.sh --pos_weight 7.0` for an illustrative weighted run. The default is `pos_weight=1.0`. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
