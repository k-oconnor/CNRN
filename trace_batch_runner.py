#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
IHDP_DIR = ROOT / 'data' / 'ihdp_sims_subset_500'
ACIC_TOP101 = ROOT / 'final_acic_2018_top101.csv'
ACIC_DATA_DIR = ROOT / 'data' / 'acic_2018'
TRACE_SCRIPT = ROOT / 'interp_experiment_trace.py'


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\n$ ' + ' '.join(cmd) + '\n')
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write(f'[exit={proc.returncode}]\n')
        return proc.returncode


def ihdp_files(count: int) -> list[Path]:
    return sorted(IHDP_DIR.glob('*.csv'))[:count]


def acic_files(count: int) -> list[tuple[str, str]]:
    df = pd.read_csv(ACIC_TOP101)
    files = []
    for factual in df['file'].astype(str).str.strip().head(count):
        cf = factual.replace('.csv', '_cf.csv')
        files.append((factual, cf))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch runner for trace-architecture CNRN reruns.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ihdp-count', type=int, default=25)
    parser.add_argument('--acic-count', type=int, default=25)
    parser.add_argument('--cnrn-epochs', type=int, default=200)
    parser.add_argument('--early-stopping-patience', type=int, default=200)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--disable-scheduler', action='store_true')
    parser.add_argument('--target-lower-quantile', type=float, default=0.5)
    parser.add_argument('--target-upper-quantile', type=float, default=0.95)
    parser.add_argument('--ihdp-layer-sizes', default='75,75,75')
    parser.add_argument('--ihdp-n-selected-features-input', type=int, default=25)
    parser.add_argument('--ihdp-n-selected-features-internal', type=int, default=20)
    parser.add_argument('--ihdp-n-selected-features-output', type=int, default=2)
    parser.add_argument('--ihdp-weight-init', type=float, default=0.5)
    parser.add_argument('--acic-layer-sizes', default='100,100,100')
    parser.add_argument('--acic-n-selected-features-input', type=int, default=25)
    parser.add_argument('--acic-n-selected-features-internal', type=int, default=20)
    parser.add_argument('--acic-n-selected-features-output', type=int, default=4)
    parser.add_argument('--acic-weight-init', type=float, default=0.7)
    parser.add_argument('--propensity-trials', type=int, default=5)
    parser.add_argument('--propensity-epochs', type=int, default=30)
    parser.add_argument('--out-root', default=str(ROOT / 'interp_runs'))
    parser.add_argument('--log-dir', default=str(ROOT / 'logs' / 'trace_batch'))
    parser.add_argument('--ihdp-only', action='store_true', help='Run only IHDP; skip ACIC.')
    args = parser.parse_args()

    out_root = Path(args.out_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / 'trace_batch.log'

    ihdp_list = ihdp_files(args.ihdp_count)
    n_ihdp = len(ihdp_list)
    print(f"IHDP: {n_ihdp} files, out_root={out_root}, log={run_log}", flush=True)

    for i, ihdp_file in enumerate(ihdp_list, 1):
        run_name = ihdp_file.stem + '_tracebatch'
        run_dir = out_root / run_name
        if run_dir.exists() and (run_dir / 'metrics.json').exists() and (run_dir / 'predictions.csv').exists():
            print(f"  [{i}/{n_ihdp}] Skip {run_name} (already done)", flush=True)
            continue
        print(f"  [{i}/{n_ihdp}] Running {run_name} ...", flush=True)
        cmd = [
            args.python,
            str(TRACE_SCRIPT),
            '--dataset', 'ihdp',
            '--ihdp-file', str(ihdp_file),
            '--has-headers',
            '--cnrn-epochs', str(args.cnrn_epochs),
            '--early-stopping-patience', str(args.early_stopping_patience),
            '--random-seed', str(args.random_seed),
            '--target-lower-quantile', str(args.target_lower_quantile),
            '--target-upper-quantile', str(args.target_upper_quantile),
            '--ihdp-layer-sizes', str(args.ihdp_layer_sizes),
            '--ihdp-n-selected-features-input', str(args.ihdp_n_selected_features_input),
            '--ihdp-n-selected-features-internal', str(args.ihdp_n_selected_features_internal),
            '--ihdp-n-selected-features-output', str(args.ihdp_n_selected_features_output),
            '--ihdp-weight-init', str(args.ihdp_weight_init),
            '--propensity-trials', str(args.propensity_trials),
            '--propensity-epochs', str(args.propensity_epochs),
            '--lambda-pehe', '0.0',
            '--run-name', run_name,
            '--out-root', str(out_root),
        ]
        if args.disable_scheduler:
            cmd.append('--disable-scheduler')
        exit_code = run_cmd(cmd, run_log)
        print(f"  [{i}/{n_ihdp}] {run_name} exit={exit_code}", flush=True)
        if exit_code != 0:
            break

    if args.ihdp_only:
        print("IHDP-only run finished (ACIC skipped).", flush=True)
        return

    acic_list = acic_files(args.acic_count)
    n_acic = len(acic_list)
    print(f"ACIC: {n_acic} files", flush=True)
    for i, (factual, cf) in enumerate(acic_list, 1):
        run_name = Path(factual).stem + '_tracebatch'
        run_dir = out_root / run_name
        if run_dir.exists() and (run_dir / 'metrics.json').exists() and (run_dir / 'predictions.csv').exists():
            print(f"  [{i}/{n_acic}] Skip {run_name} (already done)", flush=True)
            continue
        print(f"  [{i}/{n_acic}] Running {run_name} ...", flush=True)
        cmd = [
            args.python,
            str(TRACE_SCRIPT),
            '--dataset', 'acic',
            '--acic-file', str(ACIC_DATA_DIR / factual),
            '--acic-cf', str(ACIC_DATA_DIR / cf),
            '--acic-x', str(ACIC_DATA_DIR / 'x.csv'),
            '--max-covariates', '150',
            '--cnrn-epochs', str(args.cnrn_epochs),
            '--early-stopping-patience', str(args.early_stopping_patience),
            '--random-seed', str(args.random_seed),
            '--target-lower-quantile', str(args.target_lower_quantile),
            '--target-upper-quantile', str(args.target_upper_quantile),
            '--acic-layer-sizes', str(args.acic_layer_sizes),
            '--acic-n-selected-features-input', str(args.acic_n_selected_features_input),
            '--acic-n-selected-features-internal', str(args.acic_n_selected_features_internal),
            '--acic-n-selected-features-output', str(args.acic_n_selected_features_output),
            '--acic-weight-init', str(args.acic_weight_init),
            '--propensity-trials', str(args.propensity_trials),
            '--propensity-epochs', str(args.propensity_epochs),
            '--lambda-pehe', '0.0',
            '--run-name', run_name,
            '--out-root', str(out_root),
        ]
        if args.disable_scheduler:
            cmd.append('--disable-scheduler')
        exit_code = run_cmd(cmd, run_log)
        print(f"  [{i}/{n_acic}] {run_name} exit={exit_code}", flush=True)
        if exit_code != 0:
            break

    print("Batch finished.", flush=True)


if __name__ == '__main__':
    main()
