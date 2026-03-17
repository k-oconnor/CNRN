from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TAR_ihdp_experiment import fit_prop_model as fit_prop_model_ihdp, load_ihdp_data, logit, _to_1d_probs
from TAR_acic2018_experiment import fit_prop_model as fit_prop_model_acic, load_acic_2018_data
from torchlogic.sklogic.causal.semi_parametric_estimation.ate import (
    psi_aiptw,
    psi_iptw,
    psi_naive,
    psi_tmle_cont_outcome,
)


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fit_propensity_ihdp(X_tr_full: pd.DataFrame, covariate_cols: list[str], n_trials: int, epochs: int):
    prop_model = fit_prop_model_ihdp(
        X_tr_full[covariate_cols], X_tr_full["treatment"], covariate_cols=covariate_cols, n_trials=n_trials, epochs=epochs
    )
    g_train_raw = _to_1d_probs(prop_model.predict_proba(X_tr_full[covariate_cols]))
    calibrator = LogisticRegression(n_jobs=1)
    calibrator.fit(logit(g_train_raw).reshape(-1, 1), X_tr_full["treatment"].values.astype(int))
    return prop_model, calibrator


def _fit_propensity_acic(X_tr_full: pd.DataFrame, covariate_cols: list[str], n_trials: int, epochs: int):
    prop_model, best_auc, best_params = fit_prop_model_acic(
        X_tr_full[covariate_cols], X_tr_full["treatment"], covariate_cols=covariate_cols, n_trials=n_trials, epochs=epochs
    )
    return prop_model, {"best_auc": float(best_auc), "best_params": best_params}


def _predict_propensity(prop_model, X_cov: pd.DataFrame, calibrator=None) -> np.ndarray:
    g = _to_1d_probs(prop_model.predict_proba(X_cov))
    if calibrator is not None:
        g = calibrator.predict_proba(logit(g).reshape(-1, 1))[:, 1]
        g = np.clip(g.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return g


def _build_forest(args: argparse.Namespace) -> CausalForestDML:
    model_y = RandomForestRegressor(
        n_estimators=args.n_estimators_y,
        min_samples_leaf=args.min_samples_leaf_y,
        random_state=args.random_seed,
        n_jobs=args.n_jobs,
    )
    model_t = RandomForestClassifier(
        n_estimators=args.n_estimators_t,
        min_samples_leaf=args.min_samples_leaf_t,
        random_state=args.random_seed,
        n_jobs=args.n_jobs,
    )
    return CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        discrete_treatment=True,
        random_state=args.random_seed,
        n_jobs=args.n_jobs,
    )


def _fit_and_predict(forest: CausalForestDML, X_train: np.ndarray, t_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray):
    forest.fit(Y=y_train, T=t_train, X=X_train)
    ite = forest.effect(X_eval)
    q0 = forest.const_marginal_effect(X_eval).reshape(-1) * 0.0
    # Recover potential outcomes by combining CATE with E[Y|X,T=0] from model_y nuisance.
    # EconML does not expose both heads directly in a stable public API, so use the final-stage CATE
    # with a separate control regression fitted on control units only for q0 calibration.
    return ite


def _fit_control_regression(X_train: np.ndarray, t_train: np.ndarray, y_train: np.ndarray, random_seed: int, n_jobs: int) -> RandomForestRegressor:
    reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=random_seed, n_jobs=n_jobs)
    reg.fit(X_train[t_train == 0], y_train[t_train == 0])
    return reg


def run_ihdp(args: argparse.Namespace, run_dir: Path) -> None:
    df, true_ate, true_att, _ = load_ihdp_data(str(args.ihdp_file), has_headers=args.has_headers)
    covariate_cols = [f"x{i}" for i in range(1, 26)]
    X_full = df[covariate_cols + ["treatment"]].copy()
    y_full = pd.DataFrame({"y_factual": df["y_factual"]})

    X_tr_full, X_ts_full, y_tr, y_ts = train_test_split(
        X_full, y_full, test_size=0.27, random_state=args.random_seed, stratify=X_full["treatment"]
    )

    prop_model, calibrator = _fit_propensity_ihdp(X_tr_full, covariate_cols, args.propensity_trials, args.propensity_epochs)
    g = _predict_propensity(prop_model, X_ts_full[covariate_cols], calibrator=calibrator)

    x_scaler = StandardScaler().fit(X_tr_full[covariate_cols])
    X_train = x_scaler.transform(X_tr_full[covariate_cols])
    X_eval = x_scaler.transform(X_ts_full[covariate_cols])
    t_train = X_tr_full["treatment"].values.astype(int)
    t_eval = X_ts_full["treatment"].values.astype(float)
    y_train = y_tr["y_factual"].values.astype(float)
    y_obs = y_ts["y_factual"].values.astype(float)

    forest = _build_forest(args)
    ite = _fit_and_predict(forest, X_train, t_train, y_train, X_eval)
    control_reg = _fit_control_regression(X_train, t_train, y_train, args.random_seed, args.n_jobs)
    q_t0 = control_reg.predict(X_eval)
    q_t1 = q_t0 + ite

    test_indices = y_ts.index
    true_ite_test = (df.loc[test_indices, "mu1"] - df.loc[test_indices, "mu0"]).values
    true_ate_test = float(true_ite_test.mean())

    tmle_ate, tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(
        q_t0, q_t1, g, t_eval, y_obs, truncate_level=0.001
    )
    aiptw_ate = psi_aiptw(q_t0, q_t1, g, t_eval, y_obs, truncate_level=0.001)
    iptw_ate = psi_iptw(q_t0, q_t1, g, t_eval, y_obs, truncate_level=0.001)
    naive_ate = psi_naive(q_t0, q_t1, g, t_eval, y_obs, truncate_level=0.0)
    simple_ate = float(y_obs[t_eval == 1].mean() - y_obs[t_eval == 0].mean())
    pehe = float(np.sqrt(np.mean(((q_t1 - q_t0) - true_ite_test) ** 2)))

    factual_preds = t_eval * q_t1 + (1 - t_eval) * q_t0
    counterfactual_preds = (1 - t_eval) * q_t1 + t_eval * q_t0
    counterfactual_true = df.loc[test_indices, "y_cfactual"].values

    metrics = {
        "dataset": "ihdp",
        "method": "CausalForest",
        "source_file": Path(args.ihdp_file).name,
        "evaluation_split": "test",
        "tmle_ate": float(tmle_ate),
        "tmle_std": float(tmle_std),
        "aiptw_ate": float(aiptw_ate),
        "iptw_ate": float(iptw_ate),
        "naive_ate": float(naive_ate),
        "simple_ate": float(simple_ate),
        "true_ate": float(true_ate),
        "true_att": float(true_att),
        "true_ate_test": true_ate_test,
        "tmle_ate_error": abs(float(tmle_ate) - true_ate_test),
        "simple_ate_error": abs(float(simple_ate) - true_ate_test),
        "naive_ate_error": abs(float(naive_ate) - true_ate_test),
        "aiptw_ate_error": abs(float(aiptw_ate) - true_ate_test),
        "iptw_ate_error": abs(float(iptw_ate) - true_ate_test),
        "pehe": pehe,
        "propensity_auc": float(roc_auc_score(t_eval, g)),
        "factual_mse": float(mean_squared_error(y_obs, factual_preds)),
        "counterfactual_mse": float(mean_squared_error(counterfactual_true, counterfactual_preds)),
        "epsilon_hat": float(eps_hat),
        "tmle_initial_loss": float(initial_loss),
        "tmle_final_loss": float(final_loss),
        "tmle_g_loss": float(g_loss),
    }

    pd.DataFrame({
        "row_index": test_indices,
        "treatment": t_eval,
        "y_obs": y_obs,
        "y_cfactual": counterfactual_true,
        "q_t0": q_t0,
        "q_t1": q_t1,
        "g": g,
        "pred_ite": q_t1 - q_t0,
        "true_ite": true_ite_test,
        "ite_error": (q_t1 - q_t0) - true_ite_test,
    }).to_csv(run_dir / "predictions.csv", index=False)
    _save_json(run_dir / "metrics.json", metrics)


def run_acic(args: argparse.Namespace, run_dir: Path) -> None:
    df, true_ate, true_att, covariate_cols = load_acic_2018_data(str(args.acic_x), str(args.acic_file), str(args.acic_cf))
    if args.max_covariates and len(covariate_cols) > args.max_covariates:
        covariate_cols = covariate_cols[: args.max_covariates]

    X_full = df[covariate_cols + ["treatment"]].copy()
    y_full = pd.DataFrame({"y_factual": df["y_factual"]})
    X_tr_full, X_eval_full, y_tr, y_eval = train_test_split(
        X_full, y_full, test_size=0.30, random_state=args.random_seed, stratify=X_full["treatment"]
    )

    prop_model, prop_meta = _fit_propensity_acic(X_tr_full, covariate_cols, args.propensity_trials, args.propensity_epochs)
    g = _predict_propensity(prop_model, X_eval_full[covariate_cols])

    tmle_truncate = 0.01
    dr_truncate = 0.05
    keep = (g >= tmle_truncate) & (g <= (1.0 - tmle_truncate))
    if keep.sum() < 10:
        metrics = {
            "dataset": "acic",
            "method": "CausalForest",
            "source_file": Path(args.acic_file).name,
            "counterfactual_file": Path(args.acic_cf).name,
            "evaluation_split": "test",
            "status": "skipped_low_retained",
            "n_total": len(df),
            "n_eval": len(X_eval_full),
            "n_retained": int(keep.sum()),
            "retained_frac": float(keep.mean()),
            "tmle_truncate": tmle_truncate,
            "dr_truncate": dr_truncate,
            "true_ate": float(true_ate),
            "true_att": float(true_att),
            "propensity_auc_full_eval": float(roc_auc_score(X_eval_full["treatment"].values.astype(float), np.clip(g, 1e-6, 1.0 - 1e-6))),
            "skip_reason": f"Too few retained points after truncation: {int(keep.sum())}",
            "propensity_auc_val": prop_meta["best_auc"],
        }
        pd.DataFrame(columns=["row_index", "treatment", "y_obs", "y_cfactual", "q_t0", "q_t1", "g", "pred_ite", "true_ite", "ite_error"]).to_csv(run_dir / "predictions.csv", index=False)
        _save_json(run_dir / "metrics.json", metrics)
        return

    x_scaler = StandardScaler().fit(X_tr_full[covariate_cols])
    X_train = x_scaler.transform(X_tr_full[covariate_cols])
    X_eval = x_scaler.transform(X_eval_full[covariate_cols])
    t_train = X_tr_full["treatment"].values.astype(int)
    t_eval = X_eval_full["treatment"].values.astype(float)[keep]
    y_train = y_tr["y_factual"].values.astype(float)
    y_obs = y_eval["y_factual"].values.astype(float)[keep]

    forest = _build_forest(args)
    ite_all = _fit_and_predict(forest, X_train, t_train, y_train, X_eval)
    control_reg = _fit_control_regression(X_train, t_train, y_train, args.random_seed, args.n_jobs)
    q0_all = control_reg.predict(X_eval)
    q1_all = q0_all + ite_all

    q_t0 = q0_all[keep]
    q_t1 = q1_all[keep]
    g = g[keep]
    true_ite_all = df.loc[X_eval_full.index, "true_ite"].values
    true_ite = true_ite_all[keep]
    true_ate_eval = float(true_ite.mean())

    tmle_ate, tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(
        q_t0, q_t1, g, t_eval, y_obs, truncate_level=tmle_truncate
    )
    aiptw_ate = psi_aiptw(q_t0, q_t1, g, t_eval, y_obs, truncate_level=dr_truncate)
    iptw_ate = psi_iptw(q_t0, q_t1, g, t_eval, y_obs, truncate_level=dr_truncate)
    naive_ate = psi_naive(q_t0, q_t1, g, t_eval, y_obs, truncate_level=dr_truncate)
    simple_ate = float(y_obs[t_eval == 1].mean() - y_obs[t_eval == 0].mean())
    estimated_ite = q_t1 - q_t0
    pehe = float(np.sqrt(np.mean((estimated_ite - true_ite) ** 2)))

    factual_preds = t_eval * q_t1 + (1 - t_eval) * q_t0
    counterfactual_true = df.loc[X_eval_full.index, "y_cfactual"].values[keep]
    counterfactual_preds = (1 - t_eval) * q_t1 + t_eval * q_t0

    metrics = {
        "dataset": "acic",
        "method": "CausalForest",
        "source_file": Path(args.acic_file).name,
        "counterfactual_file": Path(args.acic_cf).name,
        "evaluation_split": "test",
        "n_total": len(df),
        "n_eval": len(X_eval_full),
        "n_retained": int(keep.sum()),
        "retained_frac": float(keep.mean()),
        "tmle_ate": float(tmle_ate),
        "tmle_std": float(tmle_std),
        "aiptw_ate": float(aiptw_ate),
        "iptw_ate": float(iptw_ate),
        "naive_ate": float(naive_ate),
        "simple_ate": float(simple_ate),
        "true_ate": float(true_ate),
        "true_att": float(true_att),
        "true_ate_eval": true_ate_eval,
        "tmle_ate_error": abs(float(tmle_ate) - true_ate_eval),
        "simple_ate_error": abs(float(simple_ate) - true_ate_eval),
        "naive_ate_error": abs(float(naive_ate) - true_ate_eval),
        "aiptw_ate_error": abs(float(aiptw_ate) - true_ate_eval),
        "iptw_ate_error": abs(float(iptw_ate) - true_ate_eval),
        "pehe": pehe,
        "propensity_auc": float(roc_auc_score(t_eval, g)),
        "propensity_auc_val": prop_meta["best_auc"],
        "factual_mse": float(mean_squared_error(y_obs, factual_preds)),
        "counterfactual_mse": float(mean_squared_error(counterfactual_true, counterfactual_preds)),
        "epsilon_hat": float(eps_hat),
        "tmle_initial_loss": float(initial_loss),
        "tmle_final_loss": float(final_loss),
        "tmle_g_loss": float(g_loss),
    }

    pd.DataFrame({
        "row_index": X_eval_full.index[keep],
        "treatment": t_eval,
        "y_obs": y_obs,
        "y_cfactual": counterfactual_true,
        "q_t0": q_t0,
        "q_t1": q_t1,
        "g": g,
        "pred_ite": estimated_ite,
        "true_ite": true_ite,
        "ite_error": estimated_ite - true_ite,
    }).to_csv(run_dir / "predictions.csv", index=False)
    _save_json(run_dir / "metrics.json", metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ihdp", "acic"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-root", default=str(ROOT / "baselines" / "causal_forest" / "runs"))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--propensity-trials", type=int, default=5)
    parser.add_argument("--propensity-epochs", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--n-estimators-y", type=int, default=200)
    parser.add_argument("--n-estimators-t", type=int, default=200)
    parser.add_argument("--min-samples-leaf-y", type=int, default=5)
    parser.add_argument("--min-samples-leaf-t", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--ihdp-file")
    parser.add_argument("--has-headers", action="store_true")

    parser.add_argument("--acic-file")
    parser.add_argument("--acic-cf")
    parser.add_argument("--acic-x")
    parser.add_argument("--max-covariates", type=int, default=150)
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    run_dir = Path(args.out_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "run_config.json", vars(args).copy())

    if args.dataset == "ihdp":
        if not args.ihdp_file:
            raise ValueError("--ihdp-file is required for IHDP runs")
        run_ihdp(args, run_dir)
    else:
        if not all([args.acic_file, args.acic_cf, args.acic_x]):
            raise ValueError("--acic-file, --acic-cf, and --acic-x are required for ACIC runs")
        run_acic(args, run_dir)


if __name__ == "__main__":
    main()
