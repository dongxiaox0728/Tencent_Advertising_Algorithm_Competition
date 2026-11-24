# Local competition code restoration audit

Source: `TAAC_code.zip`, compared against the `main` commit `6c7b3ab4be13bc0b12e40933f390527d98899a9c` on 2026-10-26.

## Comparison method

Compared all 122 source files using Git blob SHA-1 (path-independent exact byte equality), then compared unmatched source against same-named repository files by line-level diff. Similarity is a review aid, not proof of equivalent behavior.

| Uploaded files | Count |
| --- | ---: |
| Exact match at same path | 36 |
| Exact match at another repository path | 50 |
| Content not present in the repository | 34 |
| Existing path with different content | 2 |

## Restored experiments

Each new directory remains self-contained. Shared `dataset.py`, `utils.py`, configuration and README files are retained when needed to preserve the experiment layout. Git reuses identical blob objects; these are not new unique file contents. Older snapshots remain separate from later variants already in the repository. Inherited baseline READMEs are historical context and may not describe the specific variant.

| New directory under `code/` | Distinction |
| --- | --- |
| `Timestamp_Split_Baseline_AMP` | Adds row-group label_time min/max diagnostics. Despite the folder name, the added code checks ordering; it does not implement a new timestamp-based split. |
| `basel_amp_FiLM_decay_pool_user_staleness` | Preserves the earlier global-staleness + time FiLM + decay-pooling variant, distinct from the newer variants already present. |
| `baseline_amp_bce_loss_weight` | Adds configurable BCE positive-class weighting (--pos_weight). The default 1.0 retains ordinary BCE. |
| `baseline_amp_hash_embedding` | Uses hash buckets for high-cardinality sequence features instead of dropping them. |
| `baseline_amp_pair_feature_fusion` | Adds aligned integer-ID/dense-value tokenization, target-aware pooling, and a different sparse/dense parameter partition. |
| `baseline_amp_seq_a` | Preserves the seq_a preprocessing and matching training/inference changes. |
| `baseline_amp_time_feature` | Preserves an earlier time-feature embedding variant and its matching inference/trainer. |
| `baseline_amp_time_interval_feature` | Adds adjacent-event interval buckets and embeddings, wired through training and inference. |
| `baseline_amp_with_sincos_time_feature` | Preserves the combined periodic-time experiment, including FiLM on Q tokens, seq-token FiLM and cross-domain attention launchers. |

## Existing paths and filename corrections

- Add `code/UniRec_V5/infer.py`. Its model-input construction includes the existing V5 model's `feat_time`, `feat_seq_stat` and `feat_pair` fields.
- Keep the existing `code/UniRec_V5/readme.md`. The uploaded README exactly duplicates `code/Uni_Rec_V2/readme.md` and would replace a more specific V5 description.
- Keep `code/baseline_train_amp/run.sh`. Preserve the uploaded launch configuration as `run_local_snapshot.sh`; it omits the repository launcher's explicit `--batch_size 512`, so it uses the training parser default unless overridden.
- In `basel_amp_FiLM_decay_pool_user_staleness`, restore `train (1).py` as `train.py` and `trainer (1).py` as `trainer.py`, matching the launcher and sibling import. File contents are unchanged.
- Skip the unchanged TAAC baseline and UniRec V1/V3 snapshots, and unchanged files in existing directories. No existing files are overwritten or deleted.

## Validation and scope

- All 79 uploaded Python files parse successfully.
- All 53 restored Python files parse successfully; restored sibling imports resolve statically (V5 checked against its existing repository dependencies).
- All 15 restored shell scripts pass `bash -n`.
- Literal launcher options match their corresponding `train.py` argparse definitions in the new experiment directories.
- The source-only archive contains no competition dataset or model checkpoints. Full imports, forward passes, training, checkpoint compatibility and AUC reproduction have not been run. These require the appropriate dependencies, data and hardware. Static checks do not establish numerical correctness.
- This change restores historical code, not a general model-correctness audit. For example, the timestamp diagnostic does not establish non-overlapping time ranges or fix the split strategy.

## File decisions

`identical`: same path and bytes. `duplicate_elsewhere`: same bytes at a different path. `new_content`: no byte-identical repository blob. `modified`: existing path, different bytes.

| Uploaded relative path | Comparison | Destination / decision |
| --- | --- | --- |
| `TAAC_baseline/README.md` | identical | Skip; already present |
| `TAAC_baseline/dataset.py` | identical | Skip; already present |
| `TAAC_baseline/model.py` | identical | Skip; already present |
| `TAAC_baseline/ns_groups.json` | identical | Skip; already present |
| `TAAC_baseline/run.sh` | identical | Skip; already present |
| `TAAC_baseline/train.py` | identical | Skip; already present |
| `TAAC_baseline/trainer.py` | identical | Skip; already present |
| `TAAC_baseline/utils.py` | identical | Skip; already present |
| `Timestamp_Split_Baseline_AMP/README.md` | duplicate_elsewhere | `code/Timestamp_Split_Baseline_AMP/README.md` |
| `Timestamp_Split_Baseline_AMP/dataset.py` | new_content | `code/Timestamp_Split_Baseline_AMP/dataset.py` |
| `Timestamp_Split_Baseline_AMP/model.py` | duplicate_elsewhere | `code/Timestamp_Split_Baseline_AMP/model.py` |
| `Timestamp_Split_Baseline_AMP/ns_groups.json` | duplicate_elsewhere | `code/Timestamp_Split_Baseline_AMP/ns_groups.json` |
| `Timestamp_Split_Baseline_AMP/run.sh` | duplicate_elsewhere | `code/Timestamp_Split_Baseline_AMP/run.sh` |
| `Timestamp_Split_Baseline_AMP/train.py` | duplicate_elsewhere | `code/Timestamp_Split_Baseline_AMP/train.py` |
| `Timestamp_Split_Baseline_AMP/trainer.py` | duplicate_elsewhere | `code/Timestamp_Split_Baseline_AMP/trainer.py` |
| `Timestamp_Split_Baseline_AMP/utils.py` | duplicate_elsewhere | `code/Timestamp_Split_Baseline_AMP/utils.py` |
| `UniRec_V1/dataset.py` | identical | Skip; already present |
| `UniRec_V1/model.py` | identical | Skip; already present |
| `UniRec_V1/ns_groups.json` | identical | Skip; already present |
| `UniRec_V1/readme.md` | identical | Skip; already present |
| `UniRec_V1/run.sh` | identical | Skip; already present |
| `UniRec_V1/train.py` | identical | Skip; already present |
| `UniRec_V1/trainer.py` | identical | Skip; already present |
| `UniRec_V1/utils.py` | identical | Skip; already present |
| `UniRec_V3/dataset.py` | identical | Skip; already present |
| `UniRec_V3/infer.py` | identical | Skip; already present |
| `UniRec_V3/model.py` | identical | Skip; already present |
| `UniRec_V3/readme.md` | identical | Skip; already present |
| `UniRec_V3/train.py` | identical | Skip; already present |
| `UniRec_V3/trainer.py` | identical | Skip; already present |
| `UniRec_V3/utils.py` | identical | Skip; already present |
| `UniRec_V5/dataset.py` | identical | Skip; already present |
| `UniRec_V5/infer.py` | new_content | `code/UniRec_V5/infer.py` |
| `UniRec_V5/model.py` | identical | Skip; already present |
| `UniRec_V5/ns_groups.json` | identical | Skip; already present |
| `UniRec_V5/readme.md` | modified | Skip; keep repository version |
| `UniRec_V5/train.py` | identical | Skip; already present |
| `UniRec_V5/trainer.py` | identical | Skip; already present |
| `UniRec_V5/utils.py` | identical | Skip; already present |
| `basel_amp_FiLM_decay_pool_user_staleness/dataset.py` | duplicate_elsewhere | `code/basel_amp_FiLM_decay_pool_user_staleness/dataset.py` |
| `basel_amp_FiLM_decay_pool_user_staleness/infer.py` | new_content | `code/basel_amp_FiLM_decay_pool_user_staleness/infer.py` |
| `basel_amp_FiLM_decay_pool_user_staleness/model.py` | new_content | `code/basel_amp_FiLM_decay_pool_user_staleness/model.py` |
| `basel_amp_FiLM_decay_pool_user_staleness/ns_groups.json` | duplicate_elsewhere | `code/basel_amp_FiLM_decay_pool_user_staleness/ns_groups.json` |
| `basel_amp_FiLM_decay_pool_user_staleness/run.sh` | new_content | `code/basel_amp_FiLM_decay_pool_user_staleness/run.sh` |
| `basel_amp_FiLM_decay_pool_user_staleness/train (1).py` | new_content | `code/basel_amp_FiLM_decay_pool_user_staleness/train.py` |
| `basel_amp_FiLM_decay_pool_user_staleness/trainer (1).py` | new_content | `code/basel_amp_FiLM_decay_pool_user_staleness/trainer.py` |
| `basel_amp_FiLM_decay_pool_user_staleness/utils.py` | duplicate_elsewhere | `code/basel_amp_FiLM_decay_pool_user_staleness/utils.py` |
| `baseline_amp_bce_loss_weight/README.md` | duplicate_elsewhere | `code/baseline_amp_bce_loss_weight/README.md` |
| `baseline_amp_bce_loss_weight/dataset.py` | duplicate_elsewhere | `code/baseline_amp_bce_loss_weight/dataset.py` |
| `baseline_amp_bce_loss_weight/model.py` | duplicate_elsewhere | `code/baseline_amp_bce_loss_weight/model.py` |
| `baseline_amp_bce_loss_weight/ns_groups.json` | duplicate_elsewhere | `code/baseline_amp_bce_loss_weight/ns_groups.json` |
| `baseline_amp_bce_loss_weight/run.sh` | duplicate_elsewhere | `code/baseline_amp_bce_loss_weight/run.sh` |
| `baseline_amp_bce_loss_weight/train.py` | new_content | `code/baseline_amp_bce_loss_weight/train.py` |
| `baseline_amp_bce_loss_weight/trainer.py` | new_content | `code/baseline_amp_bce_loss_weight/trainer.py` |
| `baseline_amp_bce_loss_weight/utils.py` | duplicate_elsewhere | `code/baseline_amp_bce_loss_weight/utils.py` |
| `baseline_amp_hash_embedding/README.md` | duplicate_elsewhere | `code/baseline_amp_hash_embedding/README.md` |
| `baseline_amp_hash_embedding/dataset.py` | duplicate_elsewhere | `code/baseline_amp_hash_embedding/dataset.py` |
| `baseline_amp_hash_embedding/infer.py` | new_content | `code/baseline_amp_hash_embedding/infer.py` |
| `baseline_amp_hash_embedding/model.py` | new_content | `code/baseline_amp_hash_embedding/model.py` |
| `baseline_amp_hash_embedding/ns_groups.json` | duplicate_elsewhere | `code/baseline_amp_hash_embedding/ns_groups.json` |
| `baseline_amp_hash_embedding/run.sh` | duplicate_elsewhere | `code/baseline_amp_hash_embedding/run.sh` |
| `baseline_amp_hash_embedding/train.py` | duplicate_elsewhere | `code/baseline_amp_hash_embedding/train.py` |
| `baseline_amp_hash_embedding/trainer.py` | duplicate_elsewhere | `code/baseline_amp_hash_embedding/trainer.py` |
| `baseline_amp_hash_embedding/utils.py` | duplicate_elsewhere | `code/baseline_amp_hash_embedding/utils.py` |
| `baseline_amp_pair_feature_fusion/README.md` | duplicate_elsewhere | `code/baseline_amp_pair_feature_fusion/README.md` |
| `baseline_amp_pair_feature_fusion/dataset.py` | duplicate_elsewhere | `code/baseline_amp_pair_feature_fusion/dataset.py` |
| `baseline_amp_pair_feature_fusion/infer.py` | new_content | `code/baseline_amp_pair_feature_fusion/infer.py` |
| `baseline_amp_pair_feature_fusion/model.py` | new_content | `code/baseline_amp_pair_feature_fusion/model.py` |
| `baseline_amp_pair_feature_fusion/ns_groups.json` | duplicate_elsewhere | `code/baseline_amp_pair_feature_fusion/ns_groups.json` |
| `baseline_amp_pair_feature_fusion/run.sh` | new_content | `code/baseline_amp_pair_feature_fusion/run.sh` |
| `baseline_amp_pair_feature_fusion/train.py` | new_content | `code/baseline_amp_pair_feature_fusion/train.py` |
| `baseline_amp_pair_feature_fusion/trainer.py` | duplicate_elsewhere | `code/baseline_amp_pair_feature_fusion/trainer.py` |
| `baseline_amp_pair_feature_fusion/utils.py` | duplicate_elsewhere | `code/baseline_amp_pair_feature_fusion/utils.py` |
| `baseline_amp_seq_a/README.md` | duplicate_elsewhere | `code/baseline_amp_seq_a/README.md` |
| `baseline_amp_seq_a/dataset.py` | new_content | `code/baseline_amp_seq_a/dataset.py` |
| `baseline_amp_seq_a/infer.py` | new_content | `code/baseline_amp_seq_a/infer.py` |
| `baseline_amp_seq_a/model.py` | duplicate_elsewhere | `code/baseline_amp_seq_a/model.py` |
| `baseline_amp_seq_a/ns_groups.json` | duplicate_elsewhere | `code/baseline_amp_seq_a/ns_groups.json` |
| `baseline_amp_seq_a/run.sh` | new_content | `code/baseline_amp_seq_a/run.sh` |
| `baseline_amp_seq_a/train.py` | new_content | `code/baseline_amp_seq_a/train.py` |
| `baseline_amp_seq_a/trainer.py` | duplicate_elsewhere | `code/baseline_amp_seq_a/trainer.py` |
| `baseline_amp_seq_a/utils.py` | duplicate_elsewhere | `code/baseline_amp_seq_a/utils.py` |
| `baseline_amp_time_feature/README.md` | duplicate_elsewhere | `code/baseline_amp_time_feature/README.md` |
| `baseline_amp_time_feature/dataset.py` | duplicate_elsewhere | `code/baseline_amp_time_feature/dataset.py` |
| `baseline_amp_time_feature/infer.py` | new_content | `code/baseline_amp_time_feature/infer.py` |
| `baseline_amp_time_feature/model.py` | new_content | `code/baseline_amp_time_feature/model.py` |
| `baseline_amp_time_feature/ns_groups.json` | duplicate_elsewhere | `code/baseline_amp_time_feature/ns_groups.json` |
| `baseline_amp_time_feature/run.sh` | duplicate_elsewhere | `code/baseline_amp_time_feature/run.sh` |
| `baseline_amp_time_feature/train.py` | new_content | `code/baseline_amp_time_feature/train.py` |
| `baseline_amp_time_feature/trainer.py` | new_content | `code/baseline_amp_time_feature/trainer.py` |
| `baseline_amp_time_feature/utils.py` | duplicate_elsewhere | `code/baseline_amp_time_feature/utils.py` |
| `baseline_amp_time_interval_feature/README.md` | duplicate_elsewhere | `code/baseline_amp_time_interval_feature/README.md` |
| `baseline_amp_time_interval_feature/dataset.py` | new_content | `code/baseline_amp_time_interval_feature/dataset.py` |
| `baseline_amp_time_interval_feature/infer.py` | new_content | `code/baseline_amp_time_interval_feature/infer.py` |
| `baseline_amp_time_interval_feature/model.py` | new_content | `code/baseline_amp_time_interval_feature/model.py` |
| `baseline_amp_time_interval_feature/ns_groups.json` | duplicate_elsewhere | `code/baseline_amp_time_interval_feature/ns_groups.json` |
| `baseline_amp_time_interval_feature/run.sh` | duplicate_elsewhere | `code/baseline_amp_time_interval_feature/run.sh` |
| `baseline_amp_time_interval_feature/train.py` | duplicate_elsewhere | `code/baseline_amp_time_interval_feature/train.py` |
| `baseline_amp_time_interval_feature/trainer.py` | new_content | `code/baseline_amp_time_interval_feature/trainer.py` |
| `baseline_amp_time_interval_feature/utils.py` | duplicate_elsewhere | `code/baseline_amp_time_interval_feature/utils.py` |
| `baseline_amp_with_sincos_time_feature/ITERATION_NOTES.md` | duplicate_elsewhere | `code/baseline_amp_with_sincos_time_feature/ITERATION_NOTES.md` |
| `baseline_amp_with_sincos_time_feature/dataset.py` | duplicate_elsewhere | `code/baseline_amp_with_sincos_time_feature/dataset.py` |
| `baseline_amp_with_sincos_time_feature/infer.py` | new_content | `code/baseline_amp_with_sincos_time_feature/infer.py` |
| `baseline_amp_with_sincos_time_feature/model.py` | new_content | `code/baseline_amp_with_sincos_time_feature/model.py` |
| `baseline_amp_with_sincos_time_feature/ns_groups.json` | duplicate_elsewhere | `code/baseline_amp_with_sincos_time_feature/ns_groups.json` |
| `baseline_amp_with_sincos_time_feature/run.sh` | new_content | `code/baseline_amp_with_sincos_time_feature/run.sh` |
| `baseline_amp_with_sincos_time_feature/run_activity_count.sh` | duplicate_elsewhere | `code/baseline_amp_with_sincos_time_feature/run_activity_count.sh` |
| `baseline_amp_with_sincos_time_feature/run_cross_domain.sh` | new_content | `code/baseline_amp_with_sincos_time_feature/run_cross_domain.sh` |
| `baseline_amp_with_sincos_time_feature/run_film_on_q.sh` | new_content | `code/baseline_amp_with_sincos_time_feature/run_film_on_q.sh` |
| `baseline_amp_with_sincos_time_feature/run_film_on_seq.sh` | new_content | `code/baseline_amp_with_sincos_time_feature/run_film_on_seq.sh` |
| `baseline_amp_with_sincos_time_feature/run_per_domain_staleness.sh` | duplicate_elsewhere | `code/baseline_amp_with_sincos_time_feature/run_per_domain_staleness.sh` |
| `baseline_amp_with_sincos_time_feature/train.py` | new_content | `code/baseline_amp_with_sincos_time_feature/train.py` |
| `baseline_amp_with_sincos_time_feature/trainer.py` | duplicate_elsewhere | `code/baseline_amp_with_sincos_time_feature/trainer.py` |
| `baseline_amp_with_sincos_time_feature/utils.py` | duplicate_elsewhere | `code/baseline_amp_with_sincos_time_feature/utils.py` |
| `baseline_train_amp/README.md` | identical | Skip; already present |
| `baseline_train_amp/dataset.py` | identical | Skip; already present |
| `baseline_train_amp/model.py` | identical | Skip; already present |
| `baseline_train_amp/ns_groups.json` | identical | Skip; already present |
| `baseline_train_amp/run.sh` | modified | `code/baseline_train_amp/run_local_snapshot.sh` |
| `baseline_train_amp/train.py` | identical | Skip; already present |
| `baseline_train_amp/trainer.py` | identical | Skip; already present |
| `baseline_train_amp/utils.py` | identical | Skip; already present |
