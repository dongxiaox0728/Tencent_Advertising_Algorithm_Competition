"""EDA for aligned (user_int_feats_k, user_dense_feats_k) pair fids.

Scans a sample of the training data and computes per-fid statistics that
inform how to bucketize the dense stat for downstream feature design:

  - log1p(stat) quantile boundaries for K ∈ {16, 32, 64}
  - log1p(stat) histogram (50 bins, for distribution shape)
  - CTR vs stat-quantile (10 bins) → monotonicity / signal strength
  - Per-fid AUC of "stat-only" predictor → upper bound on stat's standalone power
  - Recommendation of K based on heuristics

Output: outputs/pair_stat_eda.json under TRAIN_CKPT_PATH (or current dir).
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import get_pcvr_data, FeatureSchema  # noqa: F401
from utils import create_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str,
                   default=os.environ.get('TRAIN_DATA_PATH', '/data_ams/academic_training_data'))
    p.add_argument('--schema_path', type=str, default=None)
    p.add_argument('--ckpt_dir', type=str,
                   default=os.environ.get('TRAIN_CKPT_PATH', './eda_out'))
    p.add_argument('--log_dir', type=str,
                   default=os.environ.get('TRAIN_LOG_PATH', './eda_out'))
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--num_batches', type=int, default=400,
                   help='Number of batches to sample. 400 * 512 ≈ 200k samples '
                        '→ each paired fid sees ~1M-10M non-zero stat values.')
    p.add_argument('--seq_max_lens', type=str,
                   default='seq_a:256,seq_b:256,seq_c:512,seq_d:512')
    p.add_argument('--buffer_batches', type=int, default=20)
    p.add_argument('--train_ratio', type=float, default=1.0)
    p.add_argument('--valid_ratio', type=float, default=0.0)
    return p.parse_args()


def _parse_seq_maxlens(s: str) -> Dict[str, int]:
    out = {}
    for kv in s.split(','):
        if not kv.strip():
            continue
        k, v = kv.split(':')
        out[k.strip()] = int(v)
    return out


def detect_paired_fids(dataset) -> List[Dict]:
    """Find fids that have aligned (int, dense) arrays of the same length > 1."""
    int_fid_map = {fid: (idx, offset, length)
                   for idx, (fid, offset, length)
                   in enumerate(dataset.user_int_schema.entries)}
    dense_fid_map = {fid: (offset, length)
                     for fid, offset, length
                     in dataset.user_dense_schema.entries}
    paired = []
    for fid, (int_idx, int_offset, int_length) in int_fid_map.items():
        if fid in dense_fid_map:
            dense_offset, dense_length = dense_fid_map[fid]
            if dense_length == int_length and int_length > 1:
                paired.append({
                    'fid': fid,
                    'int_offset': int_offset,
                    'dense_offset': dense_offset,
                    'length': int_length,
                })
    return paired


def stat_only_auc(stats: np.ndarray, labels: np.ndarray) -> float:
    """AUC of using log1p(stat) directly as predictor. Manual Mann-Whitney U
    formulation: avoids sklearn dep and is fast on numpy."""
    if len(stats) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    # Rank-based AUC = (sum_ranks_positive - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    score = np.sign(stats) * np.log1p(np.abs(stats))
    order = np.argsort(score, kind='stable')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    sum_pos_ranks = float(ranks[labels == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def count_direction_changes(seq: List[float]) -> int:
    """Count how many times monotonicity flips. 0 = strictly monotonic."""
    changes = 0
    last_sign = 0
    for i in range(1, len(seq)):
        diff = seq[i] - seq[i - 1]
        if abs(diff) < 1e-12:
            continue
        sign = 1 if diff > 0 else -1
        if last_sign != 0 and sign != last_sign:
            changes += 1
        last_sign = sign
    return changes


def recommend_k(num_unique_rounded: int, direction_changes: int, auc: float) -> int:
    """Heuristic to pick K from EDA signals.

      - Highly non-monotonic + decent AUC → K=64 (need capacity to express shape)
      - Mildly non-monotonic OR strong AUC → K=32 (default)
      - Strictly monotonic + weak AUC → K=16 (don't waste params)
      - Almost no unique values → K = min(K, 2 * unique)
    """
    if direction_changes >= 3 and auc >= 0.52:
        k = 64
    elif direction_changes <= 1 and auc < 0.51:
        k = 16
    else:
        k = 32
    # Cap by effective resolution
    k = min(k, max(8, 2 * num_unique_rounded))
    return int(k)


def main():
    args = parse_args()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    create_logger(os.path.join(args.log_dir, 'eda.log'))
    logging.info(f"EDA args: {vars(args)}")

    schema_path = args.schema_path or os.path.join(args.data_dir, 'schema.json')
    seq_max_lens = _parse_seq_maxlens(args.seq_max_lens)

    dataset, _ = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        buffer_batches=args.buffer_batches,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seq_max_lens=seq_max_lens,
    )

    paired = detect_paired_fids(dataset)
    logging.info(f"Detected {len(paired)} paired fids: {[p['fid'] for p in paired]}")

    # Collect raw stat values + labels per fid (across N batches).
    collected: Dict[int, Dict[str, List]] = {
        p['fid']: {'stats': [], 'labels': []} for p in paired
    }

    loader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers,
                        collate_fn=lambda x: x)
    n_seen = 0
    for batch in loader:
        labels = batch['label'].numpy()                  # (B,)
        user_int = batch['user_int_feats'].numpy()       # (B, total_int_dim)
        user_dense = batch['user_dense_feats'].numpy()   # (B, total_dense_dim)

        for p in paired:
            fid = p['fid']
            L = p['length']
            int_slice = user_int[:, p['int_offset']: p['int_offset'] + L]   # (B, L)
            dense_slice = user_dense[:, p['dense_offset']: p['dense_offset'] + L]  # (B, L)
            mask = int_slice != 0
            if not mask.any():
                continue
            # Broadcast labels to per-element entries
            sample_idx, _ = np.where(mask)
            stat_vals = dense_slice[mask]
            label_vals = labels[sample_idx]
            collected[fid]['stats'].append(stat_vals)
            collected[fid]['labels'].append(label_vals)

        n_seen += 1
        if n_seen >= args.num_batches:
            break
        if n_seen % 50 == 0:
            logging.info(f"Processed {n_seen} batches")

    # Compute per-fid statistics.
    result = {
        'num_batches_scanned': n_seen,
        'batch_size': args.batch_size,
        'fids': {},
    }

    for p in paired:
        fid = p['fid']
        c = collected[fid]
        if not c['stats']:
            logging.warning(f"fid={fid}: no non-zero stats collected, skipping")
            continue
        stats = np.concatenate(c['stats']).astype(np.float64)
        labels = np.concatenate(c['labels']).astype(np.int64)

        log_stats = np.sign(stats) * np.log1p(np.abs(stats))

        # Effective resolution: unique values after rounding log1p to 2 decimals
        unique_rounded = len(np.unique(np.round(log_stats, 2)))

        # log1p quantile boundaries for K ∈ {16, 32, 64}
        boundaries = {}
        for K in [16, 32, 64]:
            qs = np.linspace(0, 1, K + 1)[1:-1]  # interior boundaries (K-1 of them)
            boundaries[K] = np.quantile(log_stats, qs).tolist()

        # CTR per quantile (10 equal-frequency buckets) for monotonicity check
        cuts = np.quantile(log_stats, np.linspace(0, 1, 11))
        bins = np.clip(np.searchsorted(cuts[1:-1], log_stats, side='right'), 0, 9)
        ctr_per_bin = []
        n_per_bin = []
        for b in range(10):
            mask_b = bins == b
            if mask_b.sum() == 0:
                ctr_per_bin.append(0.0)
                n_per_bin.append(0)
            else:
                ctr_per_bin.append(float(labels[mask_b].mean()))
                n_per_bin.append(int(mask_b.sum()))
        dir_changes = count_direction_changes(ctr_per_bin)
        auc = stat_only_auc(stats, labels)

        rec_k = recommend_k(unique_rounded, dir_changes, auc)

        result['fids'][fid] = {
            'array_length': p['length'],
            'num_nonzero_values': int(len(stats)),
            'unique_rounded_log2dp': int(unique_rounded),
            'raw_stat_quantiles': {
                'min': float(stats.min()),
                'p01': float(np.quantile(stats, 0.01)),
                'p25': float(np.quantile(stats, 0.25)),
                'p50': float(np.quantile(stats, 0.50)),
                'p75': float(np.quantile(stats, 0.75)),
                'p99': float(np.quantile(stats, 0.99)),
                'max': float(stats.max()),
                'mean': float(stats.mean()),
                'std': float(stats.std()),
            },
            'log1p_stat_quantiles': {
                'min': float(log_stats.min()),
                'p25': float(np.quantile(log_stats, 0.25)),
                'p50': float(np.quantile(log_stats, 0.50)),
                'p75': float(np.quantile(log_stats, 0.75)),
                'max': float(log_stats.max()),
            },
            'ctr_per_decile': ctr_per_bin,
            'n_per_decile': n_per_bin,
            'direction_changes': int(dir_changes),
            'stat_only_auc': float(auc),
            'global_ctr': float(labels.mean()),
            'boundaries_log1p': boundaries,
            'recommended_K': int(rec_k),
        }

        logging.info(
            f"fid={fid}: N={len(stats):>10,}  L={p['length']:>3}  "
            f"unique≈{unique_rounded:>5}  "
            f"AUC={auc:.4f}  "
            f"dir_chg={dir_changes}  "
            f"global_CTR={labels.mean():.4f}  "
            f"→ recommend K={rec_k}"
        )
        logging.info(f"  CTR/decile = {[f'{c:.4f}' for c in ctr_per_bin]}")

    # Global recommendation = mode of per-fid Ks
    if result['fids']:
        ks = [v['recommended_K'] for v in result['fids'].values()]
        from collections import Counter
        global_k = Counter(ks).most_common(1)[0][0]
        result['global_recommended_K'] = int(global_k)
        logging.info(f"Global recommended K = {global_k} (per-fid Ks: {ks})")

    out_path = os.path.join(args.ckpt_dir, 'pair_stat_eda.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f"Wrote {out_path}")
    print(f"EDA result saved to {out_path}")


if __name__ == '__main__':
    main()