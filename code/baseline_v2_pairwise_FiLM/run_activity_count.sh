#!/bin/bash
# Experiment B: per-domain activity count.
# On top of the validated FiLM + decay_pool + global staleness baseline
# (test AUC 0.8194), adds a NEW signal axis: how many valid events per
# sequence domain. Per-sample valid counts are bucketed via log-spaced bins
# [1,2,5,10,20,50,100,200,500] → 10 buckets; S separate nn.Embeddings (one
# per domain) embed each, summed as a residual to TimeFeatureEmbedder.token.
# Zero-init so initial behavior is unchanged. Orthogonal to staleness:
# staleness = "recency of last action", activity_count = "intensity of action".
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
    --staleness_mode global \
    --use_activity_count \
    --time_decay_pool \
    --seq_event_mask_prob 0 \
    --batch_size 512 \
    "$@"
