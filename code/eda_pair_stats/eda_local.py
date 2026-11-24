"""Local EDA on demo_1000.csv: per-fid (int, dense) pair stat distributions.

Reads the CSV directly (avoids platform dataset loader) and computes:
  - log1p(stat) quantile boundaries for K ∈ {16, 32, 64}
  - log1p(stat) histogram (50 bins)
  - CTR vs stat-decile → monotonicity / signal strength
  - stat-only AUC → upper bound on stat's standalone power
  - Recommendation of K based on heuristics

Output: pair_stat_eda_local.json + concise console summary.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PAIRED_FIDS = [62, 63, 64, 65, 66, 89, 90, 91]

# Array element regex — handles ints and floats (incl. scientific notation).
_NUM_RE = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')


def parse_array(cell: str) -> List[float]:
    """Parse a numpy-style array string like '[928 556 538 739  94]' or empty."""
    if cell is None or cell == '' or cell == 'nan':
        return []
    return [float(x) for x in _NUM_RE.findall(cell)]


def read_demo(csv_path: str, max_rows: int = None) -> Tuple[Dict[int, List[Tuple[List[float], List[float], int]]], int]:
    """Return: {fid: [(int_arr, dense_arr, label), ...]} and total_rows seen.

    label here is the binary conversion target: (label_type == 2).
    """
    # Increase CSV field size limit (embedding columns can be very long).
    csv.field_size_limit(sys.maxsize)

    int_cols = {fid: f'user_int_feats_{fid}' for fid in PAIRED_FIDS}
    dense_cols = {fid: f'user_dense_feats_{fid}' for fid in PAIRED_FIDS}

    collected: Dict[int, List[Tuple[List[float], List[float], int]]] = {fid: [] for fid in PAIRED_FIDS}

    rows_seen = 0
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        # The embedding columns (user_dense_feats_61, _87) contain multi-line
        # text inside quotes; csv module handles that via quoting.
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = 1 if int(row.get('label_type', 0) or 0) == 2 else 0
            except Exception:
                label = 0
            for fid in PAIRED_FIDS:
                int_arr = parse_array(row.get(int_cols[fid], ''))
                dense_arr = parse_array(row.get(dense_cols[fid], ''))
                if not int_arr or not dense_arr:
                    continue
                # Align: take only positions where int_id != 0
                n = min(len(int_arr), len(dense_arr))
                int_arr = int_arr[:n]
                dense_arr = dense_arr[:n]
                collected[fid].append((int_arr, dense_arr, label))
            rows_seen += 1
            if max_rows and rows_seen >= max_rows:
                break
            if rows_seen % 50000 == 0:
                print(f'  read {rows_seen} rows...', file=sys.stderr)
    return collected, rows_seen


def stat_only_auc(stats: np.ndarray, labels: np.ndarray) -> float:
    """AUC of log1p(stat) directly as predictor (Mann-Whitney U)."""
    if len(stats) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    score = np.sign(stats) * np.log1p(np.abs(stats))
    order = np.argsort(score, kind='stable')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    sum_pos_ranks = float(ranks[labels == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def count_direction_changes(seq: List[float]) -> int:
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
    if direction_changes >= 3 and auc >= 0.52:
        k = 64
    elif direction_changes <= 1 and auc < 0.51:
        k = 16
    else:
        k = 32
    return int(min(k, max(8, 2 * num_unique_rounded)))


def analyze_fid(fid: int, entries: List[Tuple[List[float], List[float], int]]) -> Dict:
    """Compute all metrics for one fid."""
    if not entries:
        return {'error': 'no data'}

    # Two views:
    #  (A) per-element: each (id, stat) pair → one (stat_value, label) sample
    #      Use for: per-element stat distribution, stat-only AUC at element level,
    #      CTR-vs-stat-decile (treating each non-zero element as one observation).
    #  (B) per-sample aggregated: mean/max stat per sample → label
    #      Use for: stat-only AUC at sample level (closer to model use case).
    elem_stats = []
    elem_labels = []
    sample_mean_stat = []
    sample_max_stat = []
    sample_labels = []
    array_lengths = []
    for int_arr, dense_arr, label in entries:
        arr = np.array(dense_arr, dtype=np.float64)
        ids = np.array(int_arr, dtype=np.float64)
        # Filter aligned non-zero positions (mask = id != 0)
        mask = ids != 0
        if not mask.any():
            continue
        vals = arr[mask]
        elem_stats.append(vals)
        elem_labels.append(np.full(len(vals), label, dtype=np.int64))
        sample_mean_stat.append(float(vals.mean()))
        sample_max_stat.append(float(vals.max()))
        sample_labels.append(label)
        array_lengths.append(int(mask.sum()))

    if not elem_stats:
        return {'error': 'no nonzero entries'}

    elem_stats = np.concatenate(elem_stats)
    elem_labels = np.concatenate(elem_labels)
    sample_mean_stat = np.array(sample_mean_stat)
    sample_max_stat = np.array(sample_max_stat)
    sample_labels = np.array(sample_labels, dtype=np.int64)
    array_lengths = np.array(array_lengths)

    log_stats = np.sign(elem_stats) * np.log1p(np.abs(elem_stats))
    unique_rounded = int(len(np.unique(np.round(log_stats, 2))))

    # Per-K quantile boundaries.
    boundaries = {}
    for K in [16, 32, 64]:
        qs = np.linspace(0, 1, K + 1)[1:-1]
        boundaries[K] = np.quantile(log_stats, qs).tolist()

    # CTR per decile (element-level).
    cuts = np.quantile(log_stats, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(cuts[1:-1], log_stats, side='right'), 0, 9)
    ctr_per_decile = []
    n_per_decile = []
    for b in range(10):
        m = bins == b
        ctr_per_decile.append(float(elem_labels[m].mean()) if m.any() else 0.0)
        n_per_decile.append(int(m.sum()))
    dir_changes = count_direction_changes(ctr_per_decile)

    elem_auc = stat_only_auc(elem_stats, elem_labels)
    sample_mean_auc = stat_only_auc(sample_mean_stat, sample_labels)
    sample_max_auc = stat_only_auc(sample_max_stat, sample_labels)

    rec_k = recommend_k(unique_rounded, dir_changes, elem_auc)

    return {
        'num_samples': int(len(sample_labels)),
        'num_nonzero_elements': int(len(elem_stats)),
        'array_length_p50': float(np.median(array_lengths)),
        'array_length_p95': float(np.quantile(array_lengths, 0.95)),
        'array_length_max': int(array_lengths.max()),
        'unique_rounded_log2dp': unique_rounded,
        'raw_stat': {
            'min': float(elem_stats.min()),
            'p01': float(np.quantile(elem_stats, 0.01)),
            'p25': float(np.quantile(elem_stats, 0.25)),
            'p50': float(np.quantile(elem_stats, 0.50)),
            'p75': float(np.quantile(elem_stats, 0.75)),
            'p99': float(np.quantile(elem_stats, 0.99)),
            'max': float(elem_stats.max()),
            'mean': float(elem_stats.mean()),
            'std': float(elem_stats.std()),
            'frac_zero': float((elem_stats == 0).mean()),
            'frac_negative': float((elem_stats < 0).mean()),
        },
        'log1p_stat': {
            'min': float(log_stats.min()),
            'p25': float(np.quantile(log_stats, 0.25)),
            'p50': float(np.quantile(log_stats, 0.50)),
            'p75': float(np.quantile(log_stats, 0.75)),
            'max': float(log_stats.max()),
        },
        'ctr_per_decile': ctr_per_decile,
        'n_per_decile': n_per_decile,
        'direction_changes': int(dir_changes),
        'global_ctr': float(elem_labels.mean()),
        'stat_only_auc_elem': float(elem_auc),
        'stat_only_auc_sample_mean': float(sample_mean_auc),
        'stat_only_auc_sample_max': float(sample_max_auc),
        'boundaries_log1p': boundaries,
        'recommended_K': int(rec_k),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='demo_1000.csv')
    p.add_argument('--max_rows', type=int, default=None,
                   help='Cap on rows scanned (default: all)')
    p.add_argument('--out', type=str, default='pair_stat_eda_local.json')
    args = p.parse_args()

    print(f"Reading {args.csv} ...", file=sys.stderr)
    collected, total_rows = read_demo(args.csv, args.max_rows)
    print(f"Done: {total_rows} rows read", file=sys.stderr)

    result = {'csv': args.csv, 'rows_scanned': total_rows, 'fids': {}}
    for fid in PAIRED_FIDS:
        result['fids'][str(fid)] = analyze_fid(fid, collected[fid])

    # Global K recommendation
    ks = [v.get('recommended_K') for v in result['fids'].values() if 'recommended_K' in v]
    if ks:
        result['global_recommended_K'] = Counter(ks).most_common(1)[0][0]

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    # Console summary
    print()
    print(f"{'fid':<5} {'N_elem':>10} {'L_p50':>6} {'unique':>7} {'auc_e':>7} {'auc_s_mean':>11} {'auc_s_max':>10} {'dir_chg':>8} {'rec_K':>6}")
    print('-' * 90)
    for fid in PAIRED_FIDS:
        s = result['fids'][str(fid)]
        if 'error' in s:
            print(f"{fid:<5} {s['error']}")
            continue
        print(f"{fid:<5} {s['num_nonzero_elements']:>10,} "
              f"{s['array_length_p50']:>6.0f} {s['unique_rounded_log2dp']:>7} "
              f"{s['stat_only_auc_elem']:>7.4f} {s['stat_only_auc_sample_mean']:>11.4f} "
              f"{s['stat_only_auc_sample_max']:>10.4f} {s['direction_changes']:>8} "
              f"{s['recommended_K']:>6}")
    print()
    if 'global_recommended_K' in result:
        print(f"Global recommended K = {result['global_recommended_K']}")
    print(f"Full details saved to {args.out}")


if __name__ == '__main__':
    main()