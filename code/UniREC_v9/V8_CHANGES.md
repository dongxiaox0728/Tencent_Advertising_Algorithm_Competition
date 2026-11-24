# UniREC_v8 Changes

`UniREC_v8` is copied from `UniRec_V6` and integrates the dropout features from
`baseline_amp_dropout`.

## Integrated Features

- V6 aligned user int/dense value-conditioned tokenization is preserved.
- Training-only whole-token dropout is added for user-side NS tokens via
  `--user_token_dropout_rate`.
- Training-only whole-token dropout is added for sequence timestep tokens via
  `--seq_token_dropout_rate`.
- Dropout rates are included in `train_config.json` through `model_args`, so
  inference can reconstruct checkpoints that include these constructor fields.
- `run.sh` enables the same default dropout rates as `baseline_amp_dropout`.

## Notes

- Dropout is only active while `model.training` is true.
- `predict()` keeps the deterministic V6 inference path.
- AMP, trainer, dataset, utils, and schema handling remain inherited from V6.
