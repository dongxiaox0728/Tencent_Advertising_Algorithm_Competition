# UniREC_v9 Changes

`UniREC_v9` starts from `UniREC_v8` and changes how aligned user dense
features are modeled.

## Alignment Split

- Fids `62-66` keep the V6 residual value-conditioned alignment:
  `pair_emb = id_emb + MLP([id_emb, sign(x) * log1p(abs(x))])`.
- Fids `89-91` are treated as quantile-like aligned signals:
  values are clamped to `[-1, 1]`, bucketed into quantile bins, embedded, and
  combined with the paired ID embedding plus a rank-position embedding.
- Fids `61` and `87` are not aligned with user int features.

## User Dense NS Tokens

The old single user-dense token is replaced by three tokens:

- embedding token: dense fids `61, 87`, with L2 normalization then
  `Linear + LayerNorm`.
- stat token: dense fids `62-66`, with `log1p + clamp` then
  `Linear + LayerNorm`.
- quantile token: dense fids `89-91`, through `QuantileTrendEncoder`.

## Default Launcher

The active `run.sh` uses `d_model=72`. With RankMixer full mode,
`num_queries=2`, four sequence domains, five user NS tokens, three user-dense
tokens, and two item NS tokens gives `T=18`, and `72 % 18 == 0`.
It also uses `lr=8e-5` and `dropout_rate=0.015` as the default safer profile
for the wider model.
