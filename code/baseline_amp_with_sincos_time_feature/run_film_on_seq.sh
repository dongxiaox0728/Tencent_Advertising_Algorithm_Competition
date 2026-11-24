#!/bin/bash
# Experiment: extend time FiLM modulation to seq tokens.
# On top of the validated baseline (FiLM-on-NS + decay_pool + global staleness
# + cosine LR, test AUC 0.8212), broadcasts the same (γ, β) computed in
# TimeFeatureEmbedder to every seq event token (B, L, D). Zero-init film_proj
# guarantees identical behavior at training step 0; the model only grows the
# per-event time modulation as the gradient warrants. EMA stays off.
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
    --lr_warmup_steps 500 \
    --lr_decay_total_steps 30000 \
    --lr_min_ratio 0.1 \
    --ema_decay 0 \
    --batch_size 512 \
    "$@"
