from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CF_ROOT = ROOT / "baselines" / "causal_forest"
RUN_SCRIPT = CF_ROOT / "src" / "run_causal_forest.py"
IHDP_DATA_DIR = ROOT / "data" / "ihdp_sims_subset_500"
ACIC_DATA_DIR = ROOT / "data" / "acic_2018"


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[exit={proc.returncode}]\n")
        return proc.returncode


def ihdp_files_from_final_cnrn() -> list[Path]:
    interp_runs = ROOT / "interp_runs_ihdp_q05_w128_out4"
    files = []
    for p in sorted(interp_runs.iterdir()):
        if p.is_dir() and p.name.startswith("ihdp_sim_") and p.name.endswith("_tracebatch") and (p / "metrics.json").exists():
            files.append(IHDP_DATA_DIR / f"{p.name.replace('_tracebatch', '')}.csv")
    return files


def acic_files_from_matched_manifest() -> list[tuple[Path, Path]]:
    df = pd.read_csv(ROOT / "paper_artifacts" / "final" / "matched_dataset_ids.csv")
    rows = []
    for base in df[df["dataset"] == "ACIC 2018"]["base"].astype(str):
        factual = ACIC_DATA_DIR / f"{base}.csv"
        cf = ACIC_DATA_DIR / f"{base}_cf.csv"
        rows.append((factual, cf))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for CausalForest baseline.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-ihdp", action="store_true")
    parser.add_argument("--run-acic", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--n-estimators-y", type=int, default=200)
    parser.add_argument("--n-estimators-t", type=int, default=200)
    parser.add_argument("--min-samples-leaf-y", type=int, default=5)
    parser.add_argument("--min-samples-leaf-t", type=int, default=5)
    parser.add_argument("--propensity-trials", type=int, default=5)
    parser.add_argument("--propensity-epochs", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out-root", default=str(CF_ROOT / "runs"))
    parser.add_argument("--log-dir", default=str(CF_ROOT / "logs"))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    log_path = Path(args.log_dir) / "batch_causal_forest.log"

    if args.run_ihdp:
        ihdp_list = ihdp_files_from_final_cnrn()
        n_ihdp = len(ihdp_list)
        print(f"CausalForest IHDP: {n_ihdp} files, out_root={out_root}", flush=True)
        for i, ihdp_file in enumerate(ihdp_list, 1):
            run_name = ihdp_file.stem + "_cf"
            run_dir = out_root / run_name
            if (run_dir / "metrics.json").exists() and (run_dir / "predictions.csv").exists():
                print(f"  [{i}/{n_ihdp}] Skip {run_name}", flush=True)
                continue
            print(f"  [{i}/{n_ihdp}] Running {run_name} ...", flush=True)
            cmd = [
                args.python,
                str(RUN_SCRIPT),
                "--dataset", "ihdp",
                "--ihdp-file", str(ihdp_file),
                "--has-headers",
                "--run-name", run_name,
                "--out-root", str(out_root),
                "--n-estimators", str(args.n_estimators),
                "--min-samples-leaf", str(args.min_samples_leaf),
                "--max-depth", str(args.max_depth),
                "--n-estimators-y", str(args.n_estimators_y),
                "--n-estimators-t", str(args.n_estimators_t),
                "--min-samples-leaf-y", str(args.min_samples_leaf_y),
                "--min-samples-leaf-t", str(args.min_samples_leaf_t),
                "--propensity-trials", str(args.propensity_trials),
                "--propensity-epochs", str(args.propensity_epochs),
                "--n-jobs", str(args.n_jobs),
            ]
            exit_code = run_cmd(cmd, log_path)
            print(f"  [{i}/{n_ihdp}] {run_name} exit={exit_code}", flush=True)
            if exit_code != 0:
                break
        print("CausalForest IHDP batch done.", flush=True)

    if args.run_acic:
        for factual, cf in acic_files_from_matched_manifest():
            run_name = factual.stem + "_cf"
            run_dir = out_root / run_name
            if (run_dir / "metrics.json").exists() and (run_dir / "predictions.csv").exists():
                continue
            cmd = [
                args.python,
                str(RUN_SCRIPT),
                "--dataset", "acic",
                "--acic-file", str(factual),
                "--acic-cf", str(cf),
                "--acic-x", str(ACIC_DATA_DIR / "x.csv"),
                "--run-name", run_name,
                "--out-root", str(out_root),
                "--n-estimators", str(args.n_estimators),
                "--min-samples-leaf", str(args.min_samples_leaf),
                "--max-depth", str(args.max_depth),
                "--n-estimators-y", str(args.n_estimators_y),
                "--n-estimators-t", str(args.n_estimators_t),
                "--min-samples-leaf-y", str(args.min_samples_leaf_y),
                "--min-samples-leaf-t", str(args.min_samples_leaf_t),
                "--propensity-trials", str(args.propensity_trials),
                "--propensity-epochs", str(args.propensity_epochs),
                "--n-jobs", str(args.n_jobs),
            ]
            if run_cmd(cmd, log_path) != 0:
                break


if __name__ == "__main__":
    main()
