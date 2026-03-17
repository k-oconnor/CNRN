#!/usr/bin/env python3
"""
Produce a short seed-sensitivity summary for the main text (reviewer request).

Reads multi-seed results (e.g. from repeat_worst_ihdp_5x.py or a CSV of per-seed metrics)
and writes a one-row summary: min, max, mean, std for naive_ate_error, tmle_ate_error, pehe.
Use this to add a robustness caveat next to Table 1, e.g.:
"On replication X, plug-in ATE error ranged [min]–[max] across 5 seeds; TMLE ATE error ranged ..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("naive_ate_error", "tmle_ate_error", "pehe"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_from_run_dirs(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for rd in run_dirs:
        metrics_path = rd / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "run_dir": str(rd),
            "naive_ate_error": metrics.get("naive_ate_error"),
            "tmle_ate_error": metrics.get("tmle_ate_error"),
            "pehe": metrics.get("pehe"),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out = {"n_seeds": len(df)}
    for col in ("naive_ate_error", "tmle_ate_error", "pehe"):
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if vals.empty:
            continue
        out[f"{col}_min"] = float(vals.min())
        out[f"{col}_max"] = float(vals.max())
        out[f"{col}_mean"] = float(vals.mean())
        out[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize seed sensitivity for main-text robustness caveat."
    )
    parser.add_argument("--csv", type=str, default=None, help="CSV with columns naive_ate_error, tmle_ate_error, pehe (e.g. ihdp_instability_repeat5.csv).")
    parser.add_argument("--run-dirs", type=str, nargs="+", default=[], help="Run directories containing metrics.json.")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path; default: print to stdout.")
    args = parser.parse_args()

    if args.csv:
        df = load_from_csv(Path(args.csv))
    elif args.run_dirs:
        df = load_from_run_dirs([Path(d) for d in args.run_dirs])
    else:
        # Default: look for paper artifact
        default_csv = Path(__file__).resolve().parent / "paper_artifacts" / "ihdp_instability_repeat5.csv"
        if default_csv.exists():
            df = load_from_csv(default_csv)
        else:
            parser.error("Provide --csv or --run-dirs, or place ihdp_instability_repeat5.csv in paper_artifacts/")
            return

    if df.empty:
        print("No rows to summarize.")
        return

    summary = summarize(df)
    summary_df = pd.DataFrame([summary])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
    else:
        pd.set_option("display.max_columns", None)
        print(summary_df.to_string(index=False))

    # One-line caveat text
    na = summary.get("naive_ate_error_min"), summary.get("naive_ate_error_max")
    tm = summary.get("tmle_ate_error_min"), summary.get("tmle_ate_error_max")
    if na[0] is not None and na[1] is not None:
        print("\n# Main-text caveat (paste near Table 1):")
        print(f"  Plug-in ATE error ranged {na[0]:.2f}–{na[1]:.2f} across {summary['n_seeds']} seeds;")
    if tm[0] is not None and tm[1] is not None:
        print(f"  TMLE ATE error ranged {tm[0]:.2f}–{tm[1]:.2f}.")


if __name__ == "__main__":
    main()
