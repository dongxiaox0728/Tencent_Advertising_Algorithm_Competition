#!/bin/bash
# Experiment: stat-weighted pool for aligned (int, dense) pair fids.
# On top of the current best model (film_on_seq, test AUC 0.8231), replaces
# masked-mean pool with a softmax-weighted pool for the 8 paired user fids
# (62-66, 89-91, each is an aligned ID/stat array). Softmax logits are
# log1p(|stat|)·sign(stat) / τ with a per-fid learnable τ (init log_tau=3.0
# → τ≈20, softmax ≈ uniform at step 0, equivalent to mean pool).
#
# Hypothesis: the dense stat *is* a per-entity importance signal; routing it
# through pool weights (instead of FiLM modulation on the id embedding, which
# failed in pairwise_FiLM, online 0.8116) lets the model amplify the entities
# the user actually engaged with. All other settings identical to the winning
# film_on_seq run — only difference is `--use_pair_stat_pool`.
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
    --use_pair_stat_pool \
    --pair_pool_init_log_tau 3.0 \
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
