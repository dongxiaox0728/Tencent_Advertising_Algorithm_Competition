# UniRec_V6 Changes

V6 is copied from `baseline_train_amp` and keeps the AMP backbone unchanged.

The added modeling change is user-side aligned int/dense value-conditioned
pooling:

- `train.py` discovers overlapping `user_int` / `user_dense` fids dynamically
  from the real training `schema.json`.
- If the same fid exists on both sides and lengths match, `model.py` treats
  each `(user_int_id, user_dense_value)` position as an aligned pair.
- For aligned fids, the tokenizer computes:
  `pair_emb = id_emb + MLP([id_emb, normalized_dense_value])`, then masked-mean
  pools the pair embeddings.
- Dense values are normalized as `sign(x) * log1p(abs(x))` and clamped before
  the pair MLP, so both large positive values and signed values are supported.
- The original `user_dense_proj` token is kept unchanged for this first
  ablation, so no baseline dense signal is removed.
- If no aligned fids are found, or lengths mismatch, behavior falls back to the
  baseline mean pooling for those features.
