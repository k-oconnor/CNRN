from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARNET_ROOT = ROOT / "tarnet"
RUN_SCRIPT = TARNET_ROOT / "src" / "run_tarnet.py"
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


def ihdp_files_from_tracebatch(interp_root: Path | None = None) -> list[Path]:
    interp_runs = interp_root or ROOT / "interp_runs_ihdp_q05_w128_out4"
    if not interp_runs.exists():
        interp_runs = ROOT / "interp_runs"
    files = []
    for p in sorted(interp_runs.iterdir()):
        if p.is_dir() and p.name.startswith("ihdp_sim_") and p.name.endswith("_tracebatch") and (p / "metrics.json").exists():
            files.append(IHDP_DATA_DIR / f"{p.name.replace('_tracebatch', '')}.csv")
    return files


def acic_files_from_csv(file_list: Path) -> list[tuple[Path, Path]]:
    df = pd.read_csv(file_list)
    rows = []
    for factual in df["file"].astype(str).str.strip():
        cf = factual.replace(".csv", "_cf.csv")
        rows.append((ACIC_DATA_DIR / factual, ACIC_DATA_DIR / cf))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for PyTorch TARNet baseline.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-ihdp", action="store_true")
    parser.add_argument("--run-acic", action="store_true")
    parser.add_argument("--acic-file-list", default=str(Path(r"C:\Users\kevin\OneDrive\Desktop\dragonnet\results\acic_top50_files.csv")))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--propensity-trials", type=int, default=5)
    parser.add_argument("--propensity-epochs", type=int, default=30)
    parser.add_argument("--device", default=("cuda:0" if __import__("torch").cuda.is_available() else "cpu"))
    parser.add_argument("--out-root", default=str(TARNET_ROOT / "runs"))
    parser.add_argument("--log-dir", default=str(TARNET_ROOT / "logs"))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    log_path = Path(args.log_dir) / "batch_tarnet.log"

    if args.run_ihdp:
        ihdp_list = ihdp_files_from_tracebatch()
        n_ihdp = len(ihdp_list)
        print(f"TARNet IHDP: {n_ihdp} files, out_root={out_root}", flush=True)
        for i, ihdp_file in enumerate(ihdp_list, 1):
            run_name = ihdp_file.stem + "_tarnet"
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
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--propensity-trials", str(args.propensity_trials),
                "--propensity-epochs", str(args.propensity_epochs),
                "--device", str(args.device),
            ]
            exit_code = run_cmd(cmd, log_path)
            print(f"  [{i}/{n_ihdp}] {run_name} exit={exit_code}", flush=True)
            if exit_code != 0:
                break
        print("TARNet IHDP batch done.", flush=True)

    if args.run_acic:
        for factual, cf in acic_files_from_csv(Path(args.acic_file_list)):
            run_name = factual.stem + "_tarnet"
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
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--propensity-trials", str(args.propensity_trials),
                "--propensity-epochs", str(args.propensity_epochs),
                "--device", str(args.device),
            ]
            if run_cmd(cmd, log_path) != 0:
                break


if __name__ == "__main__":
    main()
