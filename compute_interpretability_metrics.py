#!/usr/bin/env python3
"""
Compute interpretability metrics (§6.3) for CNRN structural reads.

Definitions (for paper reporting):
- Sparsity: Number (or fraction) of literals with structural score above threshold.
  Here: count of literals with total_score >= score_threshold, and fraction of all literals.
- Stability: Consistency of top-k literals across seeds/replications. Jaccard similarity
  of top-k sets between pairs of runs; reported as mean pairwise Jaccard over run dirs.
- Signal alignment: Correlation of treatment-head vs control-head literal scores (shared
  outcome signal). Pearson and Spearman of (q0_score, q1_score) across literals.

Output: CSV with columns suitable for paper table (sparsity, stability, signal_alignment).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_head_summary(run_dir: Path) -> pd.DataFrame:
    """Load head_leaf_summary.csv; fallback to head_leaf_top.csv."""
    for name in ("head_leaf_summary.csv", "head_leaf_top.csv"):
        p = run_dir / name
        if p.exists():
            df = pd.read_csv(p)
            if "literal" not in df.columns and "predicate" in df.columns:
                df = df.rename(columns={"predicate": "literal"})
            return df
    raise FileNotFoundError(f"No head summary in {run_dir}")


def _load_head_delta(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "head_leaf_delta.csv"
    return pd.read_csv(p) if p.exists() else None


def sparsity(
    summary: pd.DataFrame,
    score_threshold: float = 0.0,
    use_fraction: bool = True,
) -> float:
    """
    Sparsity: fraction (or count) of literals with total_score > score_threshold.
    Uses max score per literal across heads if multiple rows per literal.
    """
    lit_scores = summary.groupby("literal", as_index=False)["total_score"].max()
    n_total = len(lit_scores)
    if n_total == 0:
        return 0.0
    n_active = (lit_scores["total_score"] >= score_threshold).sum()
    return (n_active / n_total) if use_fraction else float(n_active)


def topk_set(summary: pd.DataFrame, head: str, k: int) -> set[str]:
    """Top-k literals by total_score for a given head."""
    sub = summary.loc[summary["head"] == head]
    if sub.empty:
        return set()
    sub = sub.sort_values("total_score", ascending=False).head(k)
    return set(sub["literal"].astype(str).tolist())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union else 0.0


def stability_topk(
    run_dirs: list[Path],
    top_k: int = 10,
    head: str = "q1",
) -> float:
    """
    Mean pairwise Jaccard of top-k literal sets across run dirs.
    """
    sets_per_run: list[set[str]] = []
    for rd in run_dirs:
        summary = _load_head_summary(rd)
        sets_per_run.append(topk_set(summary, head, top_k))
    n = len(sets_per_run)
    if n <= 1:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard(sets_per_run[i], sets_per_run[j])
            count += 1
    return (total / count) if count else 1.0


def signal_alignment(summary: pd.DataFrame) -> dict[str, float]:
    """
    Correlation of q0 vs q1 scores across literals (shared outcome signal).
    Returns Pearson and Spearman; NaN if only one head present.
    """
    q0 = summary.loc[summary["head"] == "q0", ["literal", "total_score"]].rename(columns={"total_score": "q0_score"})
    q1 = summary.loc[summary["head"] == "q1", ["literal", "total_score"]].rename(columns={"total_score": "q1_score"})
    merged = q0.merge(q1, on="literal", how="outer").fillna(0.0)
    if len(merged) < 2:
        return {"pearson": np.nan, "spearman": np.nan}
    pearson = merged["q0_score"].corr(merged["q1_score"], method="pearson")
    spearman = merged["q0_score"].corr(merged["q1_score"], method="spearman")
    return {"pearson": float(pearson), "spearman": float(spearman)}


def compute_single_run(
    run_dir: Path,
    score_threshold: float = 0.0,
) -> dict:
    """Compute sparsity and signal alignment for one run dir."""
    summary = _load_head_summary(run_dir)
    sparsity_frac = sparsity(summary, score_threshold=score_threshold, use_fraction=True)
    sparsity_count = sparsity(summary, score_threshold=score_threshold, use_fraction=False)
    n_literals = len(summary.groupby("literal"))
    align = signal_alignment(summary)
    return {
        "run_dir": str(run_dir),
        "n_literals": n_literals,
        "sparsity_fraction": sparsity_frac,
        "sparsity_active_count": sparsity_count,
        "signal_alignment_pearson": align["pearson"],
        "signal_alignment_spearman": align["spearman"],
    }


def compute_multi_run(
    run_dirs: list[Path],
    top_k: int = 10,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    """Compute metrics for each run and add stability across runs."""
    rows = []
    for rd in run_dirs:
        row = compute_single_run(rd, score_threshold=score_threshold)
        rows.append(row)
    df = pd.DataFrame(rows)

    if len(run_dirs) > 1:
        stab_q0 = stability_topk(run_dirs, top_k=top_k, head="q0")
        stab_q1 = stability_topk(run_dirs, top_k=top_k, head="q1")
        df["stability_topk_q0"] = stab_q0
        df["stability_topk_q1"] = stab_q1
        df["stability_topk_mean"] = (stab_q0 + stab_q1) / 2.0
    else:
        df["stability_topk_q0"] = np.nan
        df["stability_topk_q1"] = np.nan
        df["stability_topk_mean"] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute interpretability metrics (sparsity, stability, signal alignment) for paper §6.3."
    )
    parser.add_argument("--run-dir", action="append", dest="run_dirs", default=[], help="Run directory (repeat for multiple seeds/replications).")
    parser.add_argument("--run-dir-list", type=str, default=None, help="Text file: one path per line (alternative to --run-dir).")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for stability Jaccard.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Score threshold for sparsity (literals with score >= this count as active).")
    parser.add_argument("--out-csv", type=str, default=None, help="Output CSV path; default: print to stdout.")
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for d in args.run_dirs:
        run_dirs.append(Path(d))
    if args.run_dir_list:
        with open(args.run_dir_list, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    run_dirs.append(Path(line))

    if not run_dirs:
        parser.error("Provide at least one --run-dir or --run-dir-list")

    df = compute_multi_run(
        run_dirs,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
    )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(df.to_string(index=False))

    # Summary row when multiple runs (for paper table)
    if len(run_dirs) > 1:
        summary_row = {
            "run_dir": "aggregate",
            "n_literals": df["n_literals"].mean(),
            "sparsity_fraction": df["sparsity_fraction"].mean(),
            "sparsity_active_count": df["sparsity_active_count"].mean(),
            "signal_alignment_pearson": df["signal_alignment_pearson"].mean(),
            "signal_alignment_spearman": df["signal_alignment_spearman"].mean(),
            "stability_topk_q0": df["stability_topk_q0"].iloc[0],
            "stability_topk_q1": df["stability_topk_q1"].iloc[0],
            "stability_topk_mean": df["stability_topk_mean"].iloc[0],
        }
        summary_df = pd.DataFrame([summary_row])
        if args.out_csv:
            summary_path = Path(args.out_csv)
            summary_path = summary_path.parent / (summary_path.stem + "_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Wrote summary {summary_path}")


if __name__ == "__main__":
    main()
