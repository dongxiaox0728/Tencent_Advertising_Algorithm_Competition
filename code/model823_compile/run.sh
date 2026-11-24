#!/bin/bash
# Experiment: speed-up only — torch.compile on the best model (film_on_seq,
# test AUC 0.8231). Numerics-preserving changes:
#   1. TF32 enabled (matmul + cuDNN) — < 1e-4 AUC delta in practice.
#   2. torch.compile wraps model.forward and model.predict at the bound-method
#      level (get_sparse_params / state_dict / EarlyStopping stay intact).
#   3. trainer defers loss .item() to every LOG_EVERY=50 steps; running sum
#      stays on-device as a tensor → epoch-avg loss still exact.
# Expected: ~1.3-1.8× train throughput on Ampere/Hopper, AUC unchanged.
# First epoch is slower due to inductor warmup; steady-state from epoch 2.
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
    --use_compile \
    --compile_mode default \
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
