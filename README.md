# CNRN Reproduction Package

Reproduction kit for the paper:

> **Causal Neural Reasoning Networks: Neuro-Symbolic Causal Inference**

This folder is self-contained: the `torchlogic/` logic-network library is vendored directly, so only the dependencies in `requirements.txt` need to be installed externally. No private repositories or custom wheels are required.

---

## What this package reproduces

| Paper item | Reproduced by |
|---|---|
| **Table 1** — IHDP results (N=100, N=500) | `finalize_paper_artifacts.py` |
| **Table 2** — ACIC 2018 results (N=44/43) | `finalize_paper_artifacts.py` |
| **Table 3** — Literal stability across 500 IHDP runs | `global_head_stability.py` |
| **Table 4** — Case-study interpretability structure | `inspect_causal_heads.py` |
| **Appendix fig.** — Top-10 ACIC head structure | `inspect_causal_heads.py` |
| **Appendix** — Seed sensitivity numbers | `run_ihdp_seed_sensitivity.py` |

Pre-computed results from the paper's full runs are already in `paper_artifacts/final/` — see that folder before re-running everything.

---

## Quick start (smoke test — ~60 seconds, no full data needed)

The sample data committed here (5 IHDP files, 3 ACIC pairs) is enough to verify the environment is correctly set up.

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .          # registers the vendored torchlogic package

# 2. Run the smoke test
python smoke_test.py
```

Expected output (last few lines):

```
[smoke_test] ALL STEPS PASSED
  plug-in ATE error : <some number>  (5-epoch run — metrics are noisy)
  TMLE ATE error    : <some number>
  PEHE              : <some number>

Full outputs: smoke_artifacts/
Summary:      smoke_artifacts/smoke_summary.json
```

The smoke test exercises the full pipeline in order:
1. Trains CNRN on `data/ihdp_sims_subset_500/ihdp_sim_001.csv` (5 epochs)
2. Extracts head structure → `head_leaf_summary.csv`, `head_leaf_delta.csv`
3. Computes head stability (Pearson/Spearman correlation, Jaccard overlap)
4. Computes interpretability metrics (sparsity, signal alignment)

---

## Full reproduction (requires full datasets)

### Step 0: Download the data

**IHDP 500 replications (~81 MB)**

Download the standard IHDP `.npz` files from the [CFRNet repository](https://github.com/clinicalml/cfrnet) and convert to per-replication CSVs, or obtain `data/ihdp_sims_subset_500/` directly from the authors. Place all `ihdp_sim_001.csv` … `ihdp_sim_500.csv` into `data/ihdp_sims_subset_500/`.

**ACIC 2018 (~3.5 GB)**

Download from the [ACIC 2018 data release](https://acic2018.mathematica-mpr.com/). Place all `<hash>.csv` and `<hash>_cf.csv` files into `data/acic_2018/`. The covariates file `x.csv` must also be present there. `final_acic_2018_top101.csv` (already committed) specifies which datasets are used.

### Step 1: CNRN — train on IHDP (500 replications)

```bash
python trace_batch_runner.py \
    --ihdp-count 500 \
    --cnrn-epochs 200 \
    --early-stopping-patience 200 \
    --target-lower-quantile 0.05 \
    --ihdp-n-selected-features-output 4 \
    --out-root interp_runs_ihdp_q05_w128_out4
```

Each run takes 1–4 minutes on CPU; use `--python path/to/faster/python` if you have a GPU env.

### Step 2: CNRN — train on ACIC

```bash
python trace_batch_runner.py \
    --acic-count 101 \
    --cnrn-epochs 200 \
    --early-stopping-patience 200 \
    --out-root interp_runs_acic_testsplit
```

### Step 3: TARNet baseline

```bash
python tarnet/src/batch_run_tarnet.py --run-ihdp
```

### Step 4: CausalForest baseline

```bash
python baselines/causal_forest/src/batch_run_causal_forest.py --run-ihdp
```

### Step 5: Generate head structure files (for interpretability)

Run `inspect_causal_heads.py` on each CNRN run directory. For the IHDP batch:

```bash
for d in interp_runs_ihdp_q05_w128_out4/*/; do
    python inspect_causal_heads.py --run-dir "$d"
done
```

On Windows PowerShell:

```powershell
Get-ChildItem interp_runs_ihdp_q05_w128_out4 -Directory | ForEach-Object {
    python inspect_causal_heads.py --run-dir $_.FullName
}
```

### Step 6: Aggregate results → paper tables

```bash
python finalize_paper_artifacts.py
```

Outputs land in `paper_artifacts/final/`:

| File | Paper item |
|---|---|
| `table_ihdp_results.csv` | Table 1 |
| `table_acic_results.csv` | Table 2 |
| `manifest.json` | Run counts, exclusion record |

### Step 7: Stability table (Table 3)

```bash
# Run on any single CNRN run directory that has head_leaf_summary.csv:
python global_head_stability.py --run-dir interp_runs_ihdp_q05_w128_out4/ihdp_sim_001_tracebatch/
```

The paper's Table 3 numbers were computed from `paper_artifacts/final/ihdp_topk10_stability_summary.csv`.

### Step 8: Seed sensitivity (Appendix)

```bash
python run_ihdp_seed_sensitivity.py --help
```

---

## Pre-computed results

`paper_artifacts/final/` contains the tables and artifacts from the full paper runs (500 IHDP replications × CNRN + TARNet; 44 ACIC datasets × CNRN + TARNet + CausalForest).

| File | Contents |
|---|---|
| `table_ihdp_results.csv` | Table 1 — IHDP results |
| `table_acic_results.csv` | Table 2 — ACIC results (exclusion applied) |
| `ihdp_topk10_stability_summary.csv` | Table 3 — literal stability |
| `benchmark_tables.csv` | Both tables combined |
| `manifest.json` | Run counts and exclusion notes |

The pre-specified ACIC exclusion (dataset `ae021576...`, extreme heavy-tailed outcomes) is documented in `acic_exclusion_config.py`.

---

## Repository layout

```
reproduce/
├── torchlogic/                   ← vendored NRN framework (no separate install)
├── data/
│   ├── ihdp_sims_subset_500/     ← 5 sample CSVs (full 500 downloaded separately)
│   ├── acic_2018/                ← 3 sample dataset pairs (full set downloaded separately)
│   └── acic_2018_dgp_samples.csv
├── tarnet/src/                   ← TARNet baseline
├── baselines/causal_forest/src/  ← CausalForest baseline
├── paper_artifacts/final/        ← pre-computed paper tables
├── trace_batch_runner.py         ← CNRN batch driver
├── interp_experiment_trace.py    ← single CNRN run
├── inspect_causal_heads.py       ← head structure extraction
├── global_head_stability.py      ← head stability metrics
├── compute_interpretability_metrics.py
├── run_ihdp_seed_sensitivity.py
├── TAR_ihdp_experiment.py        ← shared IHDP utilities
├── TAR_acic2018_experiment.py    ← shared ACIC utilities
├── acic_exclusion_config.py      ← pre-specified ACIC exclusion
├── finalize_paper_artifacts.py   ← aggregates results → paper tables
├── final_acic_2018_top101.csv
├── smoke_test.py                 ← end-to-end verification (no full data needed)
├── setup.py
└── requirements.txt
```

---

## Python version and dependencies

Python **3.9.x or 3.10.x** is required (`torchlogic` is incompatible with Python 3.11+).

| Package | Version tested |
|---|---|
| Python | 3.9 / 3.10 |
| torch | 2.0.x |
| numpy | 1.25.x |
| pandas | 2.0.x |
| scikit-learn | 1.2.x |
| pytorch_optimizer | 2.12.x |
| econml | 0.15.x |

---

## Citation

```bibtex
@article{cnrn2025,
  title={Causal Neural Reasoning Networks: Neuro-Symbolic Causal Inference},
  author={Kevin O'Connor},
  year={2025}
}
```
