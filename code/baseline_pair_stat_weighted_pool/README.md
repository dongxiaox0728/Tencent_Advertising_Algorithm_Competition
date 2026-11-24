# baseline_pair_stat_weighted_pool

**Plan B for the pairwise int-dense feature line.**

Replaces the masked-mean pool of paired (`user_int_feats_k`, `user_dense_feats_k`) arrays
with a **softmax-weighted pool** whose logits are `log1p(|stat|) · sign(stat) / τ`.
One learnable scalar `τ` per fid (init `log τ = 3.0` → softmax ≈ uniform → step-0
equivalent to mean pool).

## Result

| Metric | Value |
|--------|-------|
| Online AUC (eval task 103972) | **0.821803** |
| Best baseline (`film_on_seq`) | 0.823064 |
| Δ vs baseline | **−0.001261** |
| Train valid AUC peak | 0.865488 @ step 10854 |
| Baseline valid AUC peak | 0.865654 @ step 10854 |

**Verdict: weakly negative / inconclusive.** The training trajectory almost overlaps
the baseline (per-step Δ within ±0.0008), suggesting either the `log_tau` parameters
never moved far from initialization (softmax stayed near uniform) or the stat signal
genuinely lacks utility when expressed as a monotonic pool weight.

This experiment serves as a **sanity check** for the larger question "can the stat
component of paired fids help at all?" Answer: not via monotonic reweighting.

## Hypothesis

The dense stat in each (id, stat) pair represents per-entity importance
(dwell time / interaction count / probability). The current mean pool treats every
non-zero entry equally; replacing with softmax weights conditioned on the stat
should amplify the entities the user actually engaged with.

## Design

- **Base config**: identical to the winning `film_on_seq` recipe — same FiLM-on-NS,
  FiLM-on-seq, decay_pool, global staleness, cosine LR, EMA off, 30k decay steps.
- **Pair detection**: at startup, find fids that appear in both `user_int_schema`
  and `user_dense_schema` with matching array length > 1 (yields 8 fids: 62-66, 89-91).
- **`StatWeightedPooler`** (in `model.py`):
  ```python
  s = sign(dense) * log1p(|dense|)
  logits = s / exp(log_tau).clamp(min=1e-3)
  logits = logits.masked_fill(~mask, -inf)  # all-padding rows → fall back to zero
  weights = softmax(logits, dim=L)
  pooled = (id_emb * weights).sum(L)
  ```
- **Init**: `log_tau = 3.0` per fid → τ ≈ 20 → softmax close to uniform at step 0.
- **New params**: 8 scalars total. The `RankMixerNSTokenizer` registers them in
  `pair_pool_modules` and routes paired fids through `StatWeightedPooler` instead
  of mean pool when `user_dense_feats` is provided.

## Files changed vs `film_on_seq` baseline

| File | Lines | What |
|------|-------|------|
| `model.py` | ~100 | New `StatWeightedPooler`; `RankMixerNSTokenizer` accepts `paired_fid_info`; signature parity on `GroupNSTokenizer`; `PCVRHyFormer.__init__` and `_build_representations` pass `user_dense_feats` |
| `train.py` | ~30 | New CLI flags `--use_pair_stat_pool`, `--pair_pool_init_log_tau`; pair fid detection logic |
| `run.sh` | +2 | Adds the two new flags |
| `dataset.py`, `trainer.py` | 0 | Unchanged |

## Run

```bash
./run.sh
```

Expected startup log lines (sanity check):
```
Pair-stat-pool: 8 paired user fids detected (fids=[62, 63, 64, 65, 66, 89, 90, 91])
RankMixerNSTokenizer: ... pair_pool_fids=8
```

## Diagnostic for post-hoc analysis

If you want to verify whether the `log_tau` scalars actually moved during training,
load the checkpoint and inspect:

```python
ckpt = torch.load('best.pt')
for k, v in ckpt['model'].items():
    if 'pair_pool_modules' in k and 'log_tau' in k:
        print(f'{k}: {v.item():.4f}')  # initial value was 3.0
```

τ values still near 3.0 indicate softmax stayed near uniform → mean-pool equivalent →
explains why the result essentially matches baseline ± noise.

## Why the result motivated Plan C

The EDA on `demo_1000.csv` (see `../eda_pair_stats/`) revealed that fid 64-66 have
**U-shape CTR vs stat-decile** (high CTR at both stat extremes, low in the middle).
Monotonic reweighting (this experiment) cannot express U-shape; bucket embeddings
(Plan C, see `../baseline_pair_stat_bucket_emb/`) can.

## Related experiments

- Baseline: [`../baseline_film_decay_staleness_cosine_film_on_seq/`](../baseline_film_decay_staleness_cosine_film_on_seq/)
  — control group at 0.8231
- Failed predecessor: `baseline_amp_ema999_LRWarmupDecay_pairWiseFiLM_TimeFiLMDecay_staleness`
  (eval 103121, 0.811552) — used FiLM on id_emb instead of pool weight, stacked
  with EMA(0.999); large negative
- Successor: [`../baseline_pair_stat_bucket_emb/`](../baseline_pair_stat_bucket_emb/)
  — replaces softmax-weight with bucket-embedding sum fusion, expected to express U-shape
- EDA basis: [`../eda_pair_stats/`](../eda_pair_stats/) — local analysis on `demo_1000.csv`
