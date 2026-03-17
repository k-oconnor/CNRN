#!/usr/bin/env python3
"""
Smoke test for the CNRN reproduce package.

Runs the full CNRN pipeline on a single sample IHDP replication with
minimal training (5 epochs) and writes all outputs to smoke_artifacts/.
This lets a reviewer verify the environment is correctly set up without
waiting for a full training run.

Expected runtime: ~30-120 seconds on CPU depending on hardware.

Usage:
    python smoke_test.py
    python smoke_test.py --epochs 2   # faster, metrics will be noisy

After success, smoke_artifacts/ will contain:
    ihdp_run_*/
        metrics.json            -- ATE / PEHE metrics for this run
        head_leaf_summary.csv   -- structural scores for each literal per head
        head_leaf_delta.csv     -- score delta (q1 - q0) per literal
        global_head_stability_summary.csv  -- Pearson/Spearman head correlation
        global_head_topk_overlap.csv       -- Jaccard overlap at top-k
    smoke_summary.json          -- pass/fail and key numbers
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _check_sample_data() -> Path:
    sample = ROOT / "data" / "ihdp_sims_subset_500" / "ihdp_sim_001.csv"
    if not sample.exists():
        raise FileNotFoundError(
            f"Sample data not found: {sample}\n"
            "Make sure you cloned the repo including the data/ sample files."
        )
    return sample


def _run(cmd: list[str], step: str) -> None:
    print(f"\n[smoke_test] {step}")
    print("  $", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[smoke_test] FAILED: {step} (exit={result.returncode})")
        sys.exit(result.returncode)
    print(f"[smoke_test] OK: {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CNRN smoke test")
    parser.add_argument("--epochs", type=int, default=5,
                        help="CNRN training epochs (default 5; increase for more realistic metrics)")
    parser.add_argument("--out", default="smoke_artifacts",
                        help="Output directory (default: smoke_artifacts/)")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ihdp_out = out_dir / "ihdp_runs"
    ihdp_out.mkdir(parents=True, exist_ok=True)

    sample_file = _check_sample_data()
    print(f"\n[smoke_test] Using sample file: {sample_file.name}")
    print(f"[smoke_test] Output directory:  {out_dir}")
    print(f"[smoke_test] Training epochs:   {args.epochs}")

    # ── Step 1: Run CNRN on 1 IHDP file ──────────────────────────────────────
    _run(
        [
            sys.executable,
            str(ROOT / "interp_experiment_trace.py"),
            "--dataset", "ihdp",
            "--ihdp-file", str(sample_file),
            "--has-headers",
            "--cnrn-epochs", str(args.epochs),
            "--propensity-epochs", str(args.epochs),
            "--random-seed", "42",
            "--out-root", str(ihdp_out),
        ],
        "CNRN training on ihdp_sim_001.csv",
    )

    # ── Step 2: Find the run directory produced ───────────────────────────────
    run_dirs = sorted(ihdp_out.glob("*_tracebatch"))
    if not run_dirs:
        # fall back to any subdirectory with metrics.json
        run_dirs = [d for d in ihdp_out.iterdir() if (d / "metrics.json").exists()]
    if not run_dirs:
        print("[smoke_test] FAILED: no run directory found under", ihdp_out)
        sys.exit(1)
    run_dir = run_dirs[0]
    print(f"\n[smoke_test] Run directory: {run_dir.name}")

    # ── Step 3: Extract head leaf structure (generates head_leaf_summary.csv) ─
    _run(
        [
            sys.executable,
            str(ROOT / "inspect_causal_heads.py"),
            "--run-dir", str(run_dir),
        ],
        "Head structure extraction (inspect_causal_heads)",
    )

    # ── Step 4: Run head stability analysis ───────────────────────────────────
    _run(
        [
            sys.executable,
            str(ROOT / "global_head_stability.py"),
            "--run-dir", str(run_dir),
        ],
        "Head stability analysis",
    )

    # ── Step 5: Compute interpretability metrics (sparsity etc.) ──────────────
    if run_dir.exists():
        _run(
            [
                sys.executable,
                str(ROOT / "compute_interpretability_metrics.py"),
                "--run-dir", str(run_dir),
                "--out-csv", str(out_dir / "smoke_interp_metrics.csv"),
            ],
            "Interpretability metrics (sparsity / signal alignment)",
        )

    # ── Step 5: Load metrics.json and build summary ───────────────────────────
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    # Pull key numbers
    plugin_err  = metrics.get("naive_ate_error") or metrics.get("plugin_ate_error")
    tmle_err    = metrics.get("tmle_ate_error")
    pehe        = metrics.get("pehe")

    summary = {
        "status": "ok",
        "sample_file": sample_file.name,
        "run_dir": run_dir.name,
        "epochs": args.epochs,
        "plugin_ate_error": plugin_err,
        "tmle_ate_error": tmle_err,
        "pehe": pehe,
        "note": (
            "Metrics from 5-epoch smoke run; expect high error. "
            "Paper values come from 200-epoch runs over 100/500 replications."
        ),
    }

    summary_path = out_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("[smoke_test] ALL STEPS PASSED")
    print(f"  plug-in ATE error : {plugin_err}")
    print(f"  TMLE ATE error    : {tmle_err}")
    print(f"  PEHE              : {pehe}")
    print(f"\nFull outputs: {out_dir}")
    print(f"Summary:      {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
