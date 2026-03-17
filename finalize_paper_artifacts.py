from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from acic_exclusion_config import ACIC_PRESPECIFIED_EXCLUDED_BASES
except ImportError:
    ACIC_PRESPECIFIED_EXCLUDED_BASES = frozenset({"ae021576c9b248b5942fbc7c1a0539df"})

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper_artifacts"
FINAL = PAPER / "final"
ARCHIVE = PAPER / "archive"

IHDP_CNRN_ROOT = ROOT / "interp_runs_ihdp_q05_w128_out4"
ACIC_CNRN_ROOT = ROOT / "interp_runs_acic_testsplit"
TARNET_ROOT = ROOT / "tarnet" / "runs"

# Use pre-specified exclusion (see acic_exclusion_config.py and REPRODUCIBILITY.md)
ACIC_DROP_BASES = set(ACIC_PRESPECIFIED_EXCLUDED_BASES)


def _read_metrics(root: Path, suffix: str, method: str) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in sorted(root.glob(f"*{suffix}")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        status = metrics.get("status", "ok")
        row = {
            "method": method,
            "run": run_dir.name,
            "base": run_dir.name[: -len(suffix)],
            "status": status,
            "split": metrics.get("evaluation_split", "unknown"),
            "source_file": metrics.get("source_file"),
            "naive_ate_error": metrics.get("naive_ate_error"),
            "tmle_ate_error": metrics.get("tmle_ate_error"),
            "pehe": metrics.get("pehe"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _sem(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    n = len(series)
    if n <= 1:
        return float("nan")
    return float(series.std(ddof=1) / math.sqrt(n))


def _fmt(mean: float, sem: float) -> str:
    return f"{mean:.4f} +/- {sem:.4f}"


def _summarize(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        row = {
            "Dataset": dataset,
            "Method": method,
            "Split": "test",
            "N": int(len(g)),
            "Plug-in ATE Error": _fmt(float(g["naive_ate_error"].mean()), _sem(g["naive_ate_error"])),
            "TMLE ATE Error": _fmt(float(g["tmle_ate_error"].mean()), _sem(g["tmle_ate_error"])),
            "PEHE": _fmt(float(g["pehe"].mean()), _sem(g["pehe"])),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _clean_ok(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["status"] == "ok"].copy()
    for col in ["naive_ate_error", "tmle_ate_error", "pehe"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    ihdp_cnrn = _clean_ok(_read_metrics(IHDP_CNRN_ROOT, "_tracebatch", "CNRN"))
    ihdp_cnrn = ihdp_cnrn[ihdp_cnrn["source_file"].notna()]

    ihdp_tarnet = _clean_ok(_read_metrics(TARNET_ROOT, "_tarnet", "TARNet"))
    ihdp_tarnet = ihdp_tarnet[ihdp_tarnet["source_file"].str.startswith("ihdp_sim_", na=False)].copy()

    ihdp_bases = sorted(set(ihdp_cnrn["base"]).intersection(ihdp_tarnet["base"]))
    ihdp_cnrn = ihdp_cnrn[ihdp_cnrn["base"].isin(ihdp_bases)].copy()
    ihdp_tarnet = ihdp_tarnet[ihdp_tarnet["base"].isin(ihdp_bases)].copy()

    acic_cnrn_raw = _clean_ok(_read_metrics(ACIC_CNRN_ROOT, "_tracebatch", "CNRN"))
    acic_cnrn_raw = acic_cnrn_raw[acic_cnrn_raw["source_file"].notna()].copy()
    acic_cnrn_raw = acic_cnrn_raw[pd.notna(acic_cnrn_raw["naive_ate_error"]) & pd.notna(acic_cnrn_raw["tmle_ate_error"]) & pd.notna(acic_cnrn_raw["pehe"])].copy()
    acic_cnrn_raw = acic_cnrn_raw[acic_cnrn_raw["naive_ate_error"].map(math.isfinite)].copy()

    acic_tarnet_raw = _clean_ok(_read_metrics(TARNET_ROOT, "_tarnet", "TARNet"))
    acic_tarnet_raw = acic_tarnet_raw[acic_tarnet_raw["source_file"].str.match(r"^[0-9a-f]{32}\.csv$", na=False)].copy()
    acic_tarnet_raw = acic_tarnet_raw[pd.notna(acic_tarnet_raw["naive_ate_error"]) & pd.notna(acic_tarnet_raw["tmle_ate_error"]) & pd.notna(acic_tarnet_raw["pehe"])].copy()
    acic_tarnet_raw = acic_tarnet_raw[acic_tarnet_raw["naive_ate_error"].map(math.isfinite)].copy()

    # Excluded (pre-specified): main table
    acic_cnrn = acic_cnrn_raw[~acic_cnrn_raw["base"].isin(ACIC_DROP_BASES)].copy()
    acic_tarnet = acic_tarnet_raw[~acic_tarnet_raw["base"].isin(ACIC_DROP_BASES)].copy()
    acic_bases = sorted(set(acic_cnrn["base"]).intersection(acic_tarnet["base"]))
    acic_cnrn = acic_cnrn[acic_cnrn["base"].isin(acic_bases)].copy()
    acic_tarnet = acic_tarnet[acic_tarnet["base"].isin(acic_bases)].copy()

    # With excluded: for reviewer comparability (report both N=44 and N=45)
    acic_bases_all = sorted(set(acic_cnrn_raw["base"]).intersection(acic_tarnet_raw["base"]))
    acic_cnrn_all = acic_cnrn_raw[acic_cnrn_raw["base"].isin(acic_bases_all)].copy()
    acic_tarnet_all = acic_tarnet_raw[acic_tarnet_raw["base"].isin(acic_bases_all)].copy()

    ihdp_long = pd.concat([ihdp_cnrn, ihdp_tarnet], ignore_index=True)
    acic_long = pd.concat([acic_cnrn, acic_tarnet], ignore_index=True)
    acic_long_all = pd.concat([acic_cnrn_all, acic_tarnet_all], ignore_index=True)

    ihdp_summary = _summarize(ihdp_long, "IHDP")
    acic_summary = _summarize(acic_long, "ACIC 2018")
    acic_summary_all = _summarize(acic_long_all, "ACIC 2018 (all)")
    combined = pd.concat([ihdp_summary, acic_summary], ignore_index=True)

    ihdp_long.to_csv(FINAL / "ihdp_results_long.csv", index=False)
    acic_long.to_csv(FINAL / "acic_results_long.csv", index=False)
    acic_long_all.to_csv(FINAL / "acic_results_long_with_excluded.csv", index=False)
    ihdp_summary.to_csv(FINAL / "table_ihdp_results.csv", index=False)
    acic_summary.to_csv(FINAL / "table_acic_results.csv", index=False)
    acic_summary_all.to_csv(FINAL / "table_acic_results_with_excluded.csv", index=False)
    combined.to_csv(FINAL / "benchmark_tables.csv", index=False)

    matched = pd.DataFrame({
        "dataset": ["IHDP"] * len(ihdp_bases) + ["ACIC 2018"] * len(acic_bases),
        "base": ihdp_bases + acic_bases,
    })
    matched.to_csv(FINAL / "matched_dataset_ids.csv", index=False)

    manifest = {
        "ihdp_cnrn_root": str(IHDP_CNRN_ROOT),
        "ihdp_tarnet_root": str(TARNET_ROOT),
        "acic_cnrn_root": str(ACIC_CNRN_ROOT),
        "acic_tarnet_root": str(TARNET_ROOT),
        "acic_prespecified_excluded_bases": sorted(ACIC_DROP_BASES),
        "ihdp_n": len(ihdp_bases),
        "acic_n": len(acic_bases),
        "acic_n_with_excluded": len(acic_bases_all),
        "notes": [
            "IHDP uses the completed q05 / 128,128,128 / out4 CNRN batch.",
            "ACIC uses the completed matched test-split comparison from interp_runs_acic_testsplit.",
            "ACIC pre-specified exclusion: one dataset base (extreme heavy-tailed outcomes); see acic_exclusion_config.py.",
            "table_acic_results.csv = excluding pre-specified; table_acic_results_with_excluded.csv = all matched.",
        ],
    }
    (FINAL / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
