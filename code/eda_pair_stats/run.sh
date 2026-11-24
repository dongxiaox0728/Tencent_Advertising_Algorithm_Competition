#!/bin/bash
# EDA: scan ~200k samples, dump per-fid stat distributions and recommended K.
# Output: $TRAIN_CKPT_PATH/pair_stat_eda.json + eda.log under $TRAIN_LOG_PATH.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/eda.py" \
    --batch_size 512 \
    --num_workers 4 \
    --num_batches 400 \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    "$@"