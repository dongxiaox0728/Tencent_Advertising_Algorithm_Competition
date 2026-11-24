#!/bin/bash
# Experiment: extend time FiLM modulation to Q tokens.
# On top of the validated winning baseline (FiLM-on-NS + FiLM-on-seq + decay_pool
# + global staleness + cosine LR, test AUC 0.8231), broadcasts the same (γ, β)
# computed in TimeFeatureEmbedder to every freshly-generated Q token (B, Nq, D).
# Zero-init film_proj keeps step-0 identity; the model only grows per-Q time
# conditioning as the gradient warrants. Same logical extension that worked for
# FiLM-on-seq: let Q tokens "know what hour it is" before they enter the
# HyFormer block stack.
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
    --time_film_on_seq \
    --time_film_on_q \
    --time_num_harmonics 1 \
    --use_user_staleness \
    --time_decay_pool \
    --seq_event_mask_prob 0 \
    --lr_warmup_steps 500 \
    --lr_decay_total_steps 30000 \
    --lr_min_ratio 0.1 \
    --ema_decay 0 \
    --batch_size 512 \
    "$@"
