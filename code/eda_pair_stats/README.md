# Exploratory analysis of aligned integer and dense statistics

**Comparison baseline** · [the BF16 AMP baseline](../baseline_train_amp/) | [Project version guide](../../readme.md).

## Design and motivation

This diagnostic uses the AMP family’s Parquet layout to detect user int/dense fields with the same fid and a matching length greater than one. It scans nonzero paired statistics and computes quantiles, conversion rates by value quantile, and a univariate AUC using the associated row labels. Several values from one row can contribute to these summaries. The results help choose between monotonic weighting and bucket embeddings and are exploratory evidence for a model experiment.

The [weighted-pooling experiment](../baseline_pair_stat_weighted_pool/) uses statistics to reweight IDs. The [bucket-embedding experiment](../baseline_pair_stat_bucket_emb/) uses quantile boundaries to model effects that can vary across value ranges.

## Running this snapshot

Run `bash run.sh` on the competition Parquet data. The launcher scans up to 400 batches of 512 rows and writes `pair_stat_eda.json` to the checkpoint directory. `eda_local.py` supports local sample analysis. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

The deliverable is the per-field summary JSON, rather than a model score. The repository does not bundle the full-data analysis output.
