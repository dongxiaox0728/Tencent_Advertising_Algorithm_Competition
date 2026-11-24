# baseline_film_decay_staleness_cosine_film_on_seq

**Best recorded online ROC-AUC · 0.823064 (eval task 94016, 2026-05-14).**

This is the PCVRHyFormer control for the downstream sequence-FiLM experiments. It combines the AMP backbone with request-time modulation, recency-weighted pooling, and global user staleness. [Project version guide](../../readme.md).

## Result

| Metric | Value |
|--------|-------|
| **Online AUC** | **0.823064** |
| Training job internal ID | 99345 |
| Training task ID | `angel_training_ams_2026_..._20260513132328_a938b931` |
| Eval task ID | 94016 (`baseline_film_decay_staleness_cosine_film_on_seq_eval_1778733393351`) |
| Published model ID | 52907 |
| Train valid AUC peak | 0.865654 @ step 10854 |
| Training date | 2026-05-13 |

## What's in the name

```
baseline_film_decay_staleness_cosine_film_on_seq
        │     │       │         │      └──────┴── ★ this run's new addition
        │     │       │         └─────────────── cosine LR schedule (warmup 500, decay 30k)
        │     │       └──────────────────────── global user staleness signal
        │     └────────────────────────────── time-decay weighted pool over seq
        └────────────────────────────────── FiLM-on-NS time modulation

★ = the experimental delta of this run.
```

The new contribution of this run: **broadcast the FiLM `(γ, β)` time-conditioning to
every seq event token `(B, L, D)`**, not just NS tokens. Zero-init `film_proj`
preserves step-0 behavior; the model only grows per-event time modulation as
gradients warrant.

**Result vs immediate predecessor** (`baseline_amp_FiLM_decay_pool_staleness_cosine_lr`,
AUC 0.8212): **+0.0019** — the largest single-step gain after switching to the FiLM
+ staleness recipe.

## Feature design (cumulative recipe)

### NS token construction

- **RankMixer tokenizer**: concat all int embeddings → split into 4 user-NS tokens
  + 2 item-NS tokens, each projected to `d_model=64` with LayerNorm.
- `user_ns_tokens=4, item_ns_tokens=2, num_queries=2`
- `emb_skip_threshold=1,000,000` — fids with vocab > 1M skipped (hash collision concern).

### Time conditioning

- **TimeFeatureEmbedder** consumes request `hour_of_day`, `day_of_week`,
  `hour_of_week` → 1 harmonic (`sin`, `cos` pair per period) → MLP → time embedding token.
- **FiLM-on-NS** (`--time_use_film`): time embedding produces `(γ, β)` of dim `d_model`,
  applied as `x * (1+γ) + β` to the user and item NS tokens before the calendar token is appended. `film_proj` zero-init.
- **★ FiLM-on-seq** (`--time_film_on_seq`): same `(γ, β)` also broadcast to
  every seq event token `(B, L, D)`. **The breakthrough of this run.**
- **Global user staleness** (`--use_user_staleness`): `min(bucket_id > 0)` across all
  seq domains becomes a per-sample scalar; embedded via zero-init `nn.Embedding`,
  added as residual to the time token. Captures "how stale is this user overall".

### Sequence pooling

- **Time-decay pool** (`--time_decay_pool`): in `MultiSeqQueryGenerator`, replace
  mean pool over seq tokens with `weight = exp(-bucket_id / τ_i)`. A separate learnable `τ_i` sets the decay scale for each domain, in age-bucket units.

### Other

- **AMP bf16** (`--use_amp --amp_dtype bf16`): no GradScaler needed; ~30% speedup.
- **EMA off** (`--ema_decay 0`): EMA was tried in `_lr22k_ema999` (AUC 0.8198) and
  found neutral-to-negative.
- **No seq event masking** (`--seq_event_mask_prob 0`): regularization not needed.
- **No cross-domain attention**: `--cross_domain_attn` off.

## Training recipe

| Hyperparameter | Value | Reasoning |
|----------------|-------|-----------|
| Optimizer | Dense AdamW + sparse Adagrad | Dual-optimizer for embedding tables |
| `lr` (dense) | `1e-4` | Default for AdamW |
| `sparse_lr` | `0.05` | High LR works for Adagrad on sparse embs |
| `lr_warmup_steps` | 500 | Stable early steps |
| `lr_decay_total_steps` | 30000 | Cosine over ~22k actual steps + plateau tail |
| `lr_min_ratio` | 0.1 | Min LR = 10% of peak, prevents over-decay |
| `batch_size` | 512 | Memory budget |
| `d_model`, `emb_dim` | 64, 64 | Baseline sizes |
| `num_hyformer_blocks` | 2 | Depth-2 HyFormer |
| `num_heads` | 4 | Standard multi-head |
| `dropout_rate` | 0.01 | Minimal; sparse Adagrad handles emb regularization |
| `loss_type` | bce | Unweighted; AUC is the eval metric |

## Cumulative experiment chain (how we got here)

Each row is one experiment that beat its predecessor. The diff column shows what
was added on top.

| Date | Experiment | Online AUC | Δ | New addition |
|------|-----------|-----------|---|--------------|
| 04-25 | `baseline_run_001` | — | — | original UniRec |
| 04-28 | `UniRec_V2` | 0.8130 | — | first stable baseline |
| 05-02 | `Baseline_AMP` | 0.8137 | +0.0007 | AMP bf16 |
| 05-09 | `baseline_time_3` | 0.8142 | +0.0005 | basic time features |
| 05-09 | `baseline_amp_time_feature_per_channel_gate` | 0.8146 | +0.0004 | per-channel time gate |
| 05-10 | `baseline_amp_FiLM_time_feature` | 0.8149 | +0.0003 | FiLM-on-NS replaces gate |
| 05-11 | `baseline_amp_FiLM_decay_pool` | 0.8150 | +0.0001 | time-decay pool |
| 05-11 | `baseline_amp_FiLM_decay_pool_user_staleness` | 0.8194 | +0.0044 | global staleness |
| 05-12 | `baseline_amp_FiLM_decay_pool_staleness_cosine_lr` | 0.8212 | +0.0018 | cosine LR |
| **05-13** | **`baseline_film_decay_staleness_cosine_film_on_seq` (this)** | **0.8231** | **+0.0019** | **FiLM-on-seq** |

A subsequent attempt to extend FiLM further to Q tokens (`film_on_seq_q`) gave
**0.8213** — slightly negative — suggesting per-event Q conditioning interferes
with the existing time signal pathway. **Stop at FiLM-on-seq.**

## Files

| File | Source |
|------|--------|
| `model.py`, `dataset.py`, `train.py`, `trainer.py`, `run.sh` | Downloaded from platform (training job 99345) |
| `utils.py`, `infer.py`, `ns_groups.json` | Copied from `../baseline_v2_pairwise_FiLM/` (identical across runs) |

## Run

```bash
./run.sh
```

Use the competition data and matching schema described in the [run setup](../../readme.md#running-a-snapshot). Under the four-domain schema with a user-dense token, expect four user-int, two item-int, one user-dense, and one time NS token, plus eight queries. Expected log lines:
```
RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=736, num_ns_tokens=4, pad=0
RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
PCVRHyFormer model created: num_ns=8, T=16, d_model=64, rank_mixer_mode=full
```

## Use as the control for experiments

This folder is the **standard control group** for downstream experiments. Common
pattern:

```bash
# Copy this folder, change one thing, A/B test
cp -r baseline_film_decay_staleness_cosine_film_on_seq baseline_<your_change>
cd baseline_<your_change>
# Modify model.py / train.py / run.sh for the new feature
./run.sh
```

Downstream experiments based on this baseline:

| Experiment | Goal | Status |
|-----------|------|--------|
| [`../model823_compile/`](../model823_compile/) | Speed only: `torch.compile` + TF32 + loss-sync deferral | No measured result recorded |
| [`../baseline_pair_stat_weighted_pool/`](../baseline_pair_stat_weighted_pool/) | Plan B: softmax pool weight from dense stat | 0.821803 online ROC-AUC |
| [`../baseline_pair_stat_bucket_emb/`](../baseline_pair_stat_bucket_emb/) | Plan C: bucket-emb sum fusion for paired fids | No final AUC recorded |

## Failure modes observed in extensions

Patterns that **didn't beat this baseline** (useful to know what to avoid):

| Experiment | AUC | Interpretation |
|-----------|-----|------------------|
| `film_on_seq_q` (FiLM also on Q tokens) | 0.8213 | Additional conditioning may be redundant with time information from NS |
| `pairWiseFiLM_TimeFiLMDecay_staleness` | 0.8116 | FiLM(stat) on pair id_emb; stacked with EMA(0.999) |
| `pair_stat_weighted_pool` | 0.8218 | Monotonic weighting cannot express the U-shaped conversion pattern observed in the sample |
| `scale_up_increase_block` (4 blocks) | 0.8150 | Increasing to four blocks did not improve this run |
| `lr22k_ema999` (cosine + EMA) | 0.8198 | EMA neutral to slightly negative on this task |

The recorded comparisons motivate further tests of feature representation:
1. Better use of dense stats in pair fids (Plan C in progress)
2. Target-aware attention (DIN-style) — see `../baseline_pair_stat_bucket_emb/README.md`
3. Auxiliary task heads (multi-task)
