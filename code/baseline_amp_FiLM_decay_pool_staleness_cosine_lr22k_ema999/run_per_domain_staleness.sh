#!/bin/bash
# Experiment A: per-domain staleness.
# Replaces the validated global staleness (one nn.Embedding over min bucket
# across ALL seq domains, +0.0045 AUC) with S separate tables, each over its
# own domain's min bucket. Captures "fresh on click, stale on convert" type
# structures that the global min flattens. Zero-init per-domain tables so
# initial behavior matches the global-staleness baseline (each table
# contributes 0 until trained).
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
    --staleness_mode per_domain \
    --time_decay_pool \
    --seq_event_mask_prob 0 \
    --batch_size 512 \
    "$@"
