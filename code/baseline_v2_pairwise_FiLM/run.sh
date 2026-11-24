#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v2 pairwise FiLM: Pair-level FiLM on aligned int-dense NS features ----
# Pairwise fids (62-66, 89-91) are auto-detected from schema.json at startup.
# For each pairwise fid, dense_stat → (γ,β) → FiLM-modulates id_emb before pool.
# Zero-initialized: training starts equivalent to plain mean pool (no capacity lost).
# All other settings identical to baseline_amp_FiLM_decay_pool_staleness_cosine_lr22k_ema999.
#
# Mixed precision: BF16 autocast is on by default (no GradScaler).
#   - Add `--no_amp`         to fall back to FP32.
#   - Add `--amp_dtype fp16` for older GPUs without BF16 (enables GradScaler).
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
    --lr_warmup_steps 500 \
    --lr_decay_total_steps 22000 \
    --lr_min_ratio 0.1 \
    --ema_decay 0.999 \
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

