#!/bin/bash
# Experiment: stat-bucket-embedding fusion for paired (int, dense) user fids.
# On top of the current best model (film_on_seq, test AUC 0.8231), adds a
# zero-init StatBucketEmbedder for fid 62-66: bucketize sign(x)*log1p(|x|)
# into K=32 quantile buckets (boundaries from demo EDA, see pair_boundaries.json),
# look up an embedding (E=64), SUM into the id_emb before masked-mean pool.
#
# Why bucket (not Plan B softmax-weight): demo EDA found U-shape CTR vs stat
# for fid 64-66 (high CTR at both stat extremes, low in middle). Linear /
# monotonic weighting can't express this; bucket embeddings can.
# Why only fid 62-66: 89-91 are signed similarity scores in [-1, 1]; demo
# shows no signal there; staying with mean pool reduces risk surface.
#
# Zero-init StatBucketEmbedder → step 0 identical to baseline; model only
# adds capacity as gradient warrants. EMA stays off (same as winning recipe).
#
# Drift defense: --pair_bucket_refresh is ON by default. Before training,
# scans 100 train batches (~51k samples) and re-estimates the K=32 quantile
# boundaries on the real platform distribution, overriding the demo-derived
# JSON values. Refreshed boundaries are saved to
# $TRAIN_LOG_PATH/pair_boundaries_refreshed.json. Pass --no_pair_bucket_refresh
# to disable and use the JSON as-is.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_amp \
    --amp_dtype bf16 \
    --use_time_features \
    --time_use_film \
    --time_num_harmonics 1 \
    --use_user_staleness \
    --time_decay_pool \
    --seq_event_mask_prob 0 \
    --time_film_on_seq \
    --pair_bucket_boundaries_json pair_boundaries.json \
    --lr_warmup_steps 500 \
    --lr_decay_total_steps 30000 \
    --lr_min_ratio 0.1 \
    --ema_decay 0 \
    --batch_size 512 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
