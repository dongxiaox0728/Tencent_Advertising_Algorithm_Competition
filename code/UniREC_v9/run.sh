#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: V9 RankMixer + split user-dense tokens ----
# Mixed precision: BF16 autocast is on by default (no GradScaler).
#   - Add `--no_amp`         to fall back to FP32.
#   - Add `--amp_dtype fp16` for older GPUs without BF16 (enables GradScaler).
# First-run OOM tip: baseline FP32 already uses >90% GPU memory. If BF16
# triggers a transient OOM at startup, temporarily lower --batch_size (e.g.
# 256 -> 192) for the first epoch, then raise it back once stable.
python3 -u "${SCRIPT_DIR}/train.py" \
    --d_model 72 \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --lr 8e-5 \
    --dropout_rate 0.015 \
    --user_token_dropout_rate 0.05 \
    --seq_token_dropout_rate 0.05 \
    --batch_size 512 \
    --num_workers 8 \
    --use_amp \
    --amp_dtype bf16 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With three user-dense tokens, check d_model % T before switching configs.
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --d_model 72 \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
