#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd


def _safe_name(text: str) -> str:
    return str(text).replace('\n', ' ').strip()


def _predicate_label(name: str, weight: float) -> str:
    name = _safe_name(name)
    return f'NOT({name})' if weight < 0 else name


def _collect_leaf_scores(block, channel: int, out_feature: int, cumulative_score: float, depth: int, max_depth: int, rows: list[dict]) -> None:
    weights = block.weights[channel, :, out_feature].detach().cpu().numpy()
    mask = block.mask[channel, out_feature, :].detach().cpu().numpy()
    operand = block.operands
    logic_type = getattr(block, 'logic_type', type(block).__name__)

    if hasattr(operand, 'operands') and not hasattr(operand, 'weights'):
        names = operand.operands
        for local_idx, (w, pred_idx) in enumerate(zip(weights, mask)):
            score = cumulative_score * abs(float(w))
            rows.append({
                'depth': depth,
                'logic_type': logic_type,
                'channel': channel,
                'out_feature': int(out_feature),
                'local_rank': local_idx,
                'child_index': int(pred_idx),
                'weight': float(w),
                'abs_weight': abs(float(w)),
                'score': float(score),
                'predicate': _safe_name(names[int(pred_idx)]),
                'literal': _predicate_label(names[int(pred_idx)], float(w)),
            })
        return

    if depth >= max_depth:
        return

    for local_idx, (w, child_out) in enumerate(zip(weights, mask)):
        next_score = cumulative_score * abs(float(w))
        _collect_leaf_scores(operand, channel, int(child_out), next_score, depth + 1, max_depth, rows)


def _render_tree(block, channel: int, out_feature: int, depth: int, max_depth: int, prefix: str = '') -> list[str]:
    lines: list[str] = []
    weights = block.weights[channel, :, out_feature].detach().cpu().numpy()
    mask = block.mask[channel, out_feature, :].detach().cpu().numpy()
    logic_type = getattr(block, 'logic_type', type(block).__name__)
    operand = block.operands

    header = f"{prefix}{logic_type}[channel={channel}, node={out_feature}]"
    lines.append(header)

    if hasattr(operand, 'operands') and not hasattr(operand, 'weights'):
        names = operand.operands
        for w, pred_idx in zip(weights, mask):
            literal = _predicate_label(names[int(pred_idx)], float(w))
            lines.append(f"{prefix}  w={float(w):+.4f} -> {literal}")
        return lines

    if depth >= max_depth:
        return lines

    for w, child_out in zip(weights, mask):
        lines.append(f"{prefix}  w={float(w):+.4f} -> child_node={int(child_out)}")
        lines.extend(_render_tree(operand, channel, int(child_out), depth + 1, max_depth, prefix + '    '))
    return lines


def inspect_run(run_dir: Path, max_depth: int, top_k: int) -> None:
    model = joblib.load(run_dir / 'cnrn_model.joblib')
    sm = model.model.symbolic_model
    head_map = {
        'q0': sm.output_layer_1,
        'q1': sm.output_layer_2,
    }

    all_rows: list[dict] = []
    tree_lines: list[str] = []

    for head_name, head in head_map.items():
        for channel in range(head.weights.shape[0]):
            rows: list[dict] = []
            _collect_leaf_scores(head, channel=channel, out_feature=0, cumulative_score=1.0, depth=0, max_depth=max_depth, rows=rows)
            for row in rows:
                row['head'] = head_name
            all_rows.extend(rows)

            tree_lines.append(f'[{head_name} channel={channel}]')
            tree_lines.extend(_render_tree(head, channel=channel, out_feature=0, depth=0, max_depth=max_depth))
            tree_lines.append('')

    leaf_df = pd.DataFrame(all_rows)
    if leaf_df.empty:
        raise RuntimeError('No leaf scores extracted from head structure.')

    summary = (
        leaf_df.groupby(['head', 'literal', 'predicate'], as_index=False)
        .agg(total_score=('score', 'sum'), mean_abs_weight=('abs_weight', 'mean'), count=('score', 'size'))
        .sort_values(['head', 'total_score'], ascending=[True, False])
    )

    top_summary = summary.groupby('head', group_keys=False).head(top_k).copy()

    q0 = set(summary.loc[summary['head'] == 'q0', 'literal'])
    q1 = set(summary.loc[summary['head'] == 'q1', 'literal'])
    compare_rows = []
    for literal in sorted(q0 | q1):
        compare_rows.append({
            'literal': literal,
            'in_q0': literal in q0,
            'in_q1': literal in q1,
            'group': 'shared' if literal in q0 and literal in q1 else ('q0_only' if literal in q0 else 'q1_only'),
        })
    compare_df = pd.DataFrame(compare_rows)
    delta_df = summary.pivot_table(index=['literal', 'predicate'], columns='head', values='total_score', fill_value=0.0).reset_index()
    if 'q0' not in delta_df.columns:
        delta_df['q0'] = 0.0
    if 'q1' not in delta_df.columns:
        delta_df['q1'] = 0.0
    delta_df['score_delta'] = delta_df['q1'] - delta_df['q0']
    delta_df['dominant_head'] = delta_df['score_delta'].apply(lambda x: 'q1' if x > 0 else ('q0' if x < 0 else 'tie'))
    delta_df = delta_df.sort_values('score_delta', ascending=False)

    (run_dir / 'head_structure_tree.txt').write_text('\n'.join(tree_lines), encoding='utf-8')
    leaf_df.to_csv(run_dir / 'head_leaf_scores.csv', index=False)
    summary.to_csv(run_dir / 'head_leaf_summary.csv', index=False)
    top_summary.to_csv(run_dir / 'head_leaf_top.csv', index=False)
    compare_df.to_csv(run_dir / 'head_leaf_overlap.csv', index=False)
    delta_df.to_csv(run_dir / 'head_leaf_delta.csv', index=False)
    print(f'Wrote head structure outputs to {run_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect CNRN causal heads directly from saved model structure.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=25)
    args = parser.parse_args()
    inspect_run(Path(args.run_dir), max_depth=args.max_depth, top_k=args.top_k)


if __name__ == '__main__':
    main()
