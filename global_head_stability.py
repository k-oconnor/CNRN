#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_global_summary(run_dir: Path, top_k_values: list[int]) -> None:
    summary = pd.read_csv(run_dir / 'head_leaf_summary.csv')
    delta = pd.read_csv(run_dir / 'head_leaf_delta.csv')

    q0 = summary.loc[summary['head'] == 'q0', ['literal', 'total_score']].rename(columns={'total_score': 'q0_score'})
    q1 = summary.loc[summary['head'] == 'q1', ['literal', 'total_score']].rename(columns={'total_score': 'q1_score'})
    merged = q0.merge(q1, on='literal', how='outer').fillna(0.0)

    weighted_corr = merged[['q0_score', 'q1_score']].corr(method='pearson').iloc[0, 1]
    spearman_corr = merged[['q0_score', 'q1_score']].corr(method='spearman').iloc[0, 1]

    overlap_rows = []
    q0_ranked = q0.sort_values('q0_score', ascending=False)
    q1_ranked = q1.sort_values('q1_score', ascending=False)
    for k in top_k_values:
        q0_top = set(q0_ranked.head(k)['literal'])
        q1_top = set(q1_ranked.head(k)['literal'])
        inter = q0_top & q1_top
        union = q0_top | q1_top
        overlap_rows.append({
            'top_k': k,
            'intersection_n': len(inter),
            'union_n': len(union),
            'jaccard': (len(inter) / len(union)) if union else 1.0,
        })
    overlap_df = pd.DataFrame(overlap_rows)

    dominant_counts = delta['dominant_head'].value_counts(dropna=False).rename_axis('dominant_head').reset_index(name='count')
    global_summary = pd.DataFrame([{
        'n_literals': int(len(merged)),
        'pearson_score_corr': float(weighted_corr) if pd.notna(weighted_corr) else None,
        'spearman_score_corr': float(spearman_corr) if pd.notna(spearman_corr) else None,
        'mean_abs_delta': float(delta['score_delta'].abs().mean()),
        'median_abs_delta': float(delta['score_delta'].abs().median()),
    }])

    top_q1 = delta.sort_values('score_delta', ascending=False).head(25)
    top_q0 = delta.sort_values('score_delta', ascending=True).head(25)

    global_summary.to_csv(run_dir / 'global_head_stability_summary.csv', index=False)
    overlap_df.to_csv(run_dir / 'global_head_topk_overlap.csv', index=False)
    dominant_counts.to_csv(run_dir / 'global_head_dominant_counts.csv', index=False)
    top_q1.to_csv(run_dir / 'global_head_top_q1_literals.csv', index=False)
    top_q0.to_csv(run_dir / 'global_head_top_q0_literals.csv', index=False)
    print(f'Wrote global head stability outputs to {run_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize global head stability from structural head-inspection outputs.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--top-k', type=int, nargs='+', default=[10, 25, 50])
    args = parser.parse_args()
    compute_global_summary(Path(args.run_dir), top_k_values=args.top_k)


if __name__ == '__main__':
    main()
