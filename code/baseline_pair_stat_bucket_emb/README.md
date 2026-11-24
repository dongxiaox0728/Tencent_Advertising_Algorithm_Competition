# baseline_pair_stat_bucket_emb

**Plan C for the pairwise int-dense feature line.**

For paired (`user_int_feats_k`, `user_dense_feats_k`) arrays on the user side,
**bucketizes the dense stat** (`sign(x) · log1p(|x|)` → K=32 quantile buckets)
**and sum-fuses the bucket embedding into the id embedding** before masked-mean pool.

The bucket embedding table is **zero-init**, so training step 0 is identical to
the `film_on_seq` baseline; capacity is only added as gradients warrant.

## Status

Ready to upload. Not yet evaluated. Expected to test the hypothesis that the U-shape
CTR vs stat-decile observed in EDA (see "Hypothesis" below) can be expressed by
bucket embeddings.

## Hypothesis

EDA on `demo_1000.csv` (see [`../eda_pair_stats/`](../eda_pair_stats/)) revealed two
distinct types of paired fids and a strong qualitative signal:

| Fid group | Type | CTR-vs-stat-decile shape |
|-----------|------|-------------------------|
| **62-66** | Count-like, heavy long-tail, positive | **U-shape**: high CTR at both extremes (cold-start + heavy users), low in middle (mid-engagement) |
| 89-91 | Signed similarity score in [-1, 1] | Flat (no signal in demo) |

A monotonic reweighting scheme (e.g. softmax pool weight, see Plan B at
[`../baseline_pair_stat_weighted_pool/`](../baseline_pair_stat_weighted_pool/) — online
AUC 0.8218, marginal negative) cannot express U-shape. **Bucket embeddings can express
any shape** because each bucket gets an independent learned vector.

## Design

- **Base config**: identical to the winning `film_on_seq` recipe
  (FiLM-on-NS + FiLM-on-seq + decay_pool + global staleness + cosine LR, EMA off).
- **Scope**: only fid 62-66 (5 fids). Fid 89-91 stay with plain mean pool because
  demo showed no usable signal there and bucketing signed scores has unclear semantics.
- **`StatBucketEmbedder`** (in `model.py`):
  ```python
  log_d = sign(dense) * log1p(|dense|)
  bucket = torch.bucketize(log_d, boundaries) + 1   # [1..K]
  bucket = bucket * mask.long()                      # padding → index 0
  return self.emb(bucket)                            # (B, L, E), zero-init
  ```
- **Fusion** (in `RankMixerNSTokenizer.forward`):
  ```python
  emb_all = id_emb_all + stat_emb        # element-wise sum
  fid_emb = (emb_all * mask).sum(L) / count   # original mean pool
  ```
  Sum fusion keeps `chunk_dim` invariant; no downstream changes needed.
- **K = 32** quantile buckets per fid, **independent embedding table per fid**
  (since fid 62-66 have different stat semantics: dwell time / click count / probability).
- **New parameters**: 5 fids × 33 buckets × 64 dim ≈ **10.5k**.

## Boundary management

`pair_boundaries.json` stores per-fid K-1 = 31 quantile cut points on
`sign(x) · log1p(|x|)`. Initial values are derived from `demo_1000.csv`.

### Online refresh (defends against demo↔platform distribution drift)

`--pair_bucket_refresh` (default **ON**) scans 100 batches (~51k samples) from
`train_loader` at startup, recomputes the K=32 quantile boundaries from the real
platform distribution, and **overrides** the JSON values for any fid that had
non-zero stats in the scan window. Refreshed boundaries are saved to
`$TRAIN_LOG_PATH/pair_boundaries_refreshed.json` for reproducibility.

Disable with `--no_pair_bucket_refresh` if you want to fall back to the demo-derived
JSON values exactly (useful for ablation).

### Why both demo JSON and online refresh

- **JSON is required**: declares *which* fids participate and the bucket count K
- **Refresh is optional**: replaces *the boundary values* with in-distribution ones
- The fid list and K cannot be inferred at runtime; the boundaries can.

## Files changed vs `film_on_seq` baseline

| File | Lines | What |
|------|-------|------|
| `model.py` | ~120 | New `StatBucketEmbedder`; `RankMixerNSTokenizer` accepts `paired_bucket_info`; signature parity on `GroupNSTokenizer`; `PCVRHyFormer.__init__` and `_build_representations` pass `user_dense_feats` |
| `train.py` | ~110 | New CLI flags `--pair_bucket_boundaries_json`, `--pair_bucket_refresh`, `--pair_bucket_refresh_batches`; boundary JSON loader; `compute_boundaries_online()` helper |
| `run.sh` | +1 | Adds `--pair_bucket_boundaries_json pair_boundaries.json` (refresh defaults to ON) |
| `pair_boundaries.json` | NEW | K=32 quantile cuts for fid 62-66, from demo EDA |
| `dataset.py`, `trainer.py` | 0 | Unchanged |

## Run

```bash
./run.sh
```

Expected startup log lines (sanity check):
```
pair_bucket: refreshing boundaries online from 100 train batches (K=32, fids=[62, 63, 64, 65, 66])
online-boundaries: fid=62 N=158,234 log1p range=[1.79, 16.50] cut[0]=6.95 cut[-1]=14.20
online-boundaries: fid=63 N=...
...
pair_bucket: refreshed boundaries saved to .../pair_boundaries_refreshed.json
pair_bucket: ready with 5 fids (fids=[62, 63, 64, 65, 66], num_buckets=32, refresh=on)
RankMixerNSTokenizer: ... pair_bucket_fids=5
```

If `pair_bucket_fids=5` appears, the hook is wired correctly.

## Why this might still fail

1. **Bucket embeddings need samples per bucket to learn good vectors.** With ~50k
   non-zero values per fid per batch and 32 buckets, ~1500 obs/bucket/batch — fine.
2. **Sum fusion may underfit.** If id_emb and stat_emb cancel out (e.g. opposite
   gradient directions), the residual signal is weaker than concat-and-project.
   If this proves a problem, switch to concat in a future variant.
3. **U-shape signal in demo may not hold at platform scale.** Online refresh shifts
   boundaries but doesn't change the underlying signal shape. If the platform
   distribution is more uniform, the U-shape advantage of bucket emb shrinks.

## Comparison plan

No final online AUC is recorded for the bucket-embedding snapshot. The comparison plan uses these controls:

| Experiment | Online AUC | Key difference |
|-----------|------------|----------------|
| baseline `film_on_seq` | 0.823064 | no stat use |
| `baseline_pair_stat_weighted_pool` (Plan B) | 0.821803 | softmax pool weight |
| **this** (Plan C) | Unavailable | bucket emb sum fusion |

| Result | Interpretation | Next step |
|--------|----------------|-----------|
| ≥ 0.8240 | U-shape signal real, bucket effective | Stack with DIN target-aware attention on these same fids |
| 0.8225 ~ 0.8240 | Marginal positive | Try K=64, or check refreshed bucket histogram for imbalance |
| ≈ 0.823 ± 0.001 | Bucket emb doesn't help — either U-shape was demo artifact, or sum-fusion underfits | Try concat-and-project fusion before abandoning |
| < 0.821 | Bucket emb hurts | The pair int-dense line is exhausted; pivot to seq-side or multi-task |

## Diagnostic for post-hoc analysis

Inspect the learned bucket embeddings to see if the U-shape was actually captured:

```python
ckpt = torch.load('best.pt')
for fid_idx in range(5):
    emb = ckpt['model'][f'user_ns_tokenizer.pair_bucket_modules.{fid_idx}.emb.weight']
    # emb shape: (33, 64). Compare L2 norms of buckets to see which ones learned
    norms = emb.norm(dim=-1)
    print(f'fid_idx={fid_idx} bucket norms: {norms.tolist()}')
```

If norms at low and high buckets (1-2 and 30-31) are much larger than mid buckets
(15-17), the model learned the U-shape signal. If norms are roughly uniform, no
useful signal was extracted.

## Related

- Baseline: [`../baseline_film_decay_staleness_cosine_film_on_seq/`](../baseline_film_decay_staleness_cosine_film_on_seq/)
- Plan B (predecessor): [`../baseline_pair_stat_weighted_pool/`](../baseline_pair_stat_weighted_pool/)
- EDA basis: [`../eda_pair_stats/`](../eda_pair_stats/) — local analysis on `demo_1000.csv`
- Likely successor (Plan A overlay): DIN target-aware attention on the same paired fids,
  using `(id_emb + stat_emb)` as attention keys/values and item NS summary as query.
