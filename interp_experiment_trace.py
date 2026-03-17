#!/usr/bin/env python3
from __future__ import annotations
from torchlogic.sklogic.causal.TARNRNTraceRegressor import TARNRNTraceRegressor

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from TAR_ihdp_experiment import (
    TARNetLoss,
    fit_prop_model as fit_prop_model_ihdp,
    load_ihdp_data,
    extract_model_predictions as extract_ihdp_predictions,
    logit,
    _to_1d_probs,
)
from TAR_acic2018_experiment import (
    fit_prop_model as fit_prop_model_acic,
    load_acic_2018_data,
    extract_model_predictions as extract_acic_predictions,
)
from torchlogic.sklogic.causal.semi_parametric_estimation.ate import (
    psi_aiptw,
    psi_iptw,
    psi_naive,
    psi_tmle_cont_outcome,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_IHDP_FILE = ROOT / "data" / "ihdp_sims_subset_500" / "ihdp_sim_047.csv"
DEFAULT_ACIC_FILE = "7f9b2da504a14860a55de4a0a19db383.csv"
DEFAULT_ACIC_CF = "7f9b2da504a14860a55de4a0a19db383_cf.csv"
DEFAULT_ACIC_X = ROOT / "data" / "acic_2018" / "x.csv"
DEFAULT_OUT = ROOT / "interp_runs"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _write_explanations(path: Path, model: TARNRNTraceRegressor, X: pd.DataFrame, label_col: str = "pred_ite") -> None:
    preds = model.predict(X, treatment=None)
    pred_ite = preds[:, 1] - preds[:, 0]
    work = X.copy()
    work[label_col] = pred_ite
    high_idx = int(np.argmax(pred_ite))
    low_idx = int(np.argmin(pred_ite))
    lines = []
    for name, idx in [("highest_pred_ite", high_idx), ("lowest_pred_ite", low_idx)]:
        lines.append(f"[{name}] row_index={idx} pred_ite={pred_ite[idx]:.6f}")
        try:
            lines.append(model.explain_sample(X.copy(), sample_index=idx, quantile=1.0))
        except Exception as exc:
            lines.append(f"Explanation failed: {exc}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_model_for_dump(model: TARNRNTraceRegressor) -> TARNRNTraceRegressor:
    return model




def _save_model_bundle(run_dir: Path, model: TARNRNTraceRegressor) -> None:
    torch.save(model.model.state_dict(), run_dir / "cnrn_model_state.pt")
    bundle = {
        "feature_names": list(model.feature_names) if getattr(model, "feature_names", None) is not None else None,
        "target_name": getattr(model, "target_name", None),
        "treatment_col": getattr(model, "treatment_col", None),
        "propensity_col": getattr(model, "propensity_col", None),
        "instance_params": model._get_instance_params(),
        "feature_scaler": getattr(model, "feature_scaler", None),
        "target_scaler": getattr(model, "target_scaler", None),
        "encoder": getattr(model, "encoder", None),
        "binarizer": getattr(model, "binarizer", None),
        "fbt": getattr(model, "fbt", None),
        "numeric_features": getattr(model, "numeric_features", None),
        "categorical_features": getattr(model, "categorical_features", None),
    }
    joblib.dump(bundle, run_dir / "cnrn_bundle.joblib")


def run_ihdp(args: argparse.Namespace, run_dir: Path) -> None:
    df, true_ate, true_att, true_ite = load_ihdp_data(str(args.ihdp_file), has_headers=args.has_headers)
    covariate_cols = [f"x{i}" for i in range(1, 26)]
    X_full = df[covariate_cols + ["treatment"]].copy()
    y_full = pd.DataFrame({"y_factual": df["y_factual"]})

    X_tr_full, X_ts_full, y_tr, y_ts = train_test_split(
        X_full,
        y_full,
        test_size=0.27,
        random_state=args.random_seed,
        stratify=X_full["treatment"],
    )

    prop_model = fit_prop_model_ihdp(
        X_tr_full[covariate_cols],
        X_tr_full["treatment"],
        covariate_cols=covariate_cols,
        n_trials=args.propensity_trials,
        epochs=args.propensity_epochs,
    )
    g_train_raw = _to_1d_probs(prop_model.predict_proba(X_tr_full[covariate_cols]))
    X_tr_full["g_hat"] = g_train_raw
    calibrator = LogisticRegression(n_jobs=1)
    calibrator.fit(logit(g_train_raw).reshape(-1, 1), X_tr_full["treatment"].values.astype(int))

    model = TARNRNTraceRegressor(
        feature_names=covariate_cols,
        target_name="y_factual",
        treatment_col="treatment",
        propensity_col="g_hat",
        normal_form="cnf",
        epochs=args.cnrn_epochs,
        learning_rate=0.05,
        weight_decay=0.0001,
        layer_sizes=_parse_int_list(args.ihdp_layer_sizes, [75, 75, 75]),
        head_layer_sizes=_parse_int_list(args.ihdp_head_layer_sizes, []),
        mlp_head_hidden_dim=args.ihdp_mlp_head_hidden_dim,
        n_selected_features_input=args.ihdp_n_selected_features_input,
        n_selected_features_internal=args.ihdp_n_selected_features_internal,
        n_selected_features_output=args.ihdp_n_selected_features_output,
        add_negations=False,
        weight_init=args.ihdp_weight_init,
        t_0=40,
        t_mult=4,
        batch_size=32,
        early_stopping_plateau_count=args.early_stopping_patience,
        loss_fn=TARNetLoss(
            lambda_pehe=args.lambda_pehe,
            balance_classes=True,
            eps=1e-6,
            epsilon_init=0.1,
            loss_on_original_scale=args.loss_on_original_scale,
        ),
        holdout_pct=0.1,
        device=args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        beta_l1=0.0,
        target_lower_quantile=args.target_lower_quantile,
        target_upper_quantile=args.target_upper_quantile,
        target_lower_expand_ratio=args.target_lower_expand_ratio,
        target_upper_expand_ratio=args.target_upper_expand_ratio,
        loss_on_original_scale=args.loss_on_original_scale,
        disable_scheduler=args.disable_scheduler,
    )
    model.fit(X_tr_full, y_tr)

    q_t0, q_t1, g = extract_ihdp_predictions(model, prop_model, X_ts_full, calibrator=calibrator)
    t = X_ts_full["treatment"].values.astype(float)
    y_obs = y_ts["y_factual"].values.astype(float)
    test_indices = y_ts.index
    true_ite_test = (df.loc[test_indices, "mu1"] - df.loc[test_indices, "mu0"]).values
    true_ate_test = float(true_ite_test.mean())

    tmle_ate, tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(
        q_t0, q_t1, g, t, y_obs, truncate_level=0.001
    )
    aiptw_ate = psi_aiptw(q_t0, q_t1, g, t, y_obs, truncate_level=0.001)
    iptw_ate = psi_iptw(q_t0, q_t1, g, t, y_obs, truncate_level=0.001)
    naive_ate = psi_naive(q_t0, q_t1, g, t, y_obs, truncate_level=0.0)
    simple_ate = float(y_obs[t == 1].mean() - y_obs[t == 0].mean())
    estimated_ite = q_t1 - q_t0
    pehe = float(np.sqrt(np.mean((estimated_ite - true_ite_test) ** 2)))

    t_tensor = torch.tensor(t).float().view(-1, 1)
    q_t0_tensor = torch.tensor(q_t0).float().view(-1, 1)
    q_t1_tensor = torch.tensor(q_t1).float().view(-1, 1)
    factual_preds = (t_tensor * q_t1_tensor + (1 - t_tensor) * q_t0_tensor).detach().cpu().numpy().flatten()
    counterfactual_preds = ((1 - t_tensor) * q_t1_tensor + t_tensor * q_t0_tensor).detach().cpu().numpy().flatten()
    counterfactual_true = df.loc[test_indices, "y_cfactual"].values

    metrics = {
        "dataset": "ihdp",
        "source_file": Path(args.ihdp_file).name,
        "evaluation_split": "test",
        "lambda_pehe": args.lambda_pehe,
        "best_val_loss": float(model.training_history_.get("best_val_loss")) if getattr(model, "training_history_", None) and model.training_history_.get("best_val_loss") is not None else None,
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
        "propensity_auc": float(roc_auc_score(t, g)),
        "factual_mse": float(mean_squared_error(y_obs, factual_preds)),
        "counterfactual_mse": float(mean_squared_error(counterfactual_true, counterfactual_preds)),
        "epsilon_hat": float(eps_hat),
        "tmle_initial_loss": float(initial_loss),
        "tmle_final_loss": float(final_loss),
        "tmle_g_loss": float(g_loss),
    }

    pred_df = pd.DataFrame({
        "row_index": test_indices,
        "treatment": t,
        "y_obs": y_obs,
        "y_cfactual": counterfactual_true,
        "q_t0": q_t0,
        "q_t1": q_t1,
        "g": g,
        "pred_ite": estimated_ite,
        "true_ite": true_ite_test,
        "ite_error": estimated_ite - true_ite_test,
    })

    model = _prepare_model_for_dump(model)
    model = _prepare_model_for_dump(model)
    _save_model_bundle(run_dir, model)
    joblib.dump(model, run_dir / "cnrn_model.joblib")
    joblib.dump(prop_model, run_dir / "propensity_model.joblib")
    joblib.dump(calibrator, run_dir / "propensity_calibrator.joblib")
    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    _save_json(run_dir / "metrics.json", metrics)
    _write_explanations(run_dir / "sample_explanations.txt", model, X_ts_full.drop(columns=["g_hat"], errors="ignore"))


def run_acic(args: argparse.Namespace, run_dir: Path) -> None:
    x_path = str(args.acic_x)
    df, true_ate, true_att, covariate_cols = load_acic_2018_data(x_path, str(args.acic_file), str(args.acic_cf))
    if args.max_covariates and len(covariate_cols) > args.max_covariates:
        covariate_cols = covariate_cols[: args.max_covariates]

    X_full = df[covariate_cols + ["treatment"]].copy()
    y_full = pd.DataFrame({"y_factual": df["y_factual"]})
    if args.use_all_data:
        X_tr_full, y_tr = X_full.copy(), y_full.copy()
        X_eval_full, y_eval = X_full.copy(), y_full.copy()
    else:
        X_tr_full, X_eval_full, y_tr, y_eval = train_test_split(
            X_full,
            y_full,
            test_size=0.30,
            random_state=args.random_seed,
            stratify=X_full["treatment"],
        )

    prop_model, prop_best_auc, prop_best_params = fit_prop_model_acic(
        X_tr_full[covariate_cols],
        X_tr_full["treatment"],
        covariate_cols=covariate_cols,
        n_trials=args.propensity_trials,
        epochs=args.propensity_epochs,
    )
    g_train = np.asarray(prop_model.predict_proba(X_tr_full[covariate_cols])).reshape(-1)
    X_tr_full["g_hat"] = g_train

    model = TARNRNTraceRegressor(
        feature_names=covariate_cols,
        target_name="y_factual",
        treatment_col="treatment",
        propensity_col="g_hat",
        normal_form="cnf",
        epochs=args.cnrn_epochs,
        learning_rate=0.05,
        weight_decay=0.0001,
        layer_sizes=_parse_int_list(args.acic_layer_sizes, [100, 100, 100]),
        head_layer_sizes=_parse_int_list(args.acic_head_layer_sizes, []),
        mlp_head_hidden_dim=args.acic_mlp_head_hidden_dim,
        n_selected_features_input=args.acic_n_selected_features_input,
        n_selected_features_internal=args.acic_n_selected_features_internal,
        n_selected_features_output=args.acic_n_selected_features_output,
        add_negations=False,
        weight_init=args.acic_weight_init,
        t_0=40,
        t_mult=4,
        batch_size=16,
        early_stopping_plateau_count=args.early_stopping_patience,
        loss_fn=TARNetLoss(
            lambda_pehe=args.lambda_pehe,
            balance_classes=True,
            eps=1e-6,
            epsilon_init=0.1,
            loss_on_original_scale=args.loss_on_original_scale,
        ),
        holdout_pct=0.1,
        device=args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        beta_l1=0.0,
        target_lower_quantile=args.target_lower_quantile,
        target_upper_quantile=args.target_upper_quantile,
        target_lower_expand_ratio=args.target_lower_expand_ratio,
        target_upper_expand_ratio=args.target_upper_expand_ratio,
        loss_on_original_scale=args.loss_on_original_scale,
        disable_scheduler=args.disable_scheduler,
    )
    model.fit(X_tr_full, y_tr)

    q_t0, q_t1, g = extract_acic_predictions(model, prop_model, X_eval_full)
    tmle_truncate = 0.01
    dr_truncate = 0.05
    keep = (g >= tmle_truncate) & (g <= (1.0 - tmle_truncate))
    if keep.sum() < 10:
        metrics = {
            "dataset": "acic",
            "source_file": Path(args.acic_file).name,
            "counterfactual_file": Path(args.acic_cf).name,
            "evaluation_split": "all" if args.use_all_data else "test",
            "lambda_pehe": args.lambda_pehe,
            "status": "skipped_low_retained",
            "retained_points": int(keep.sum()),
            "tmle_truncate": tmle_truncate,
            "dr_truncate": dr_truncate,
        }
        pd.DataFrame(columns=["q_t0", "q_t1", "g", "t", "y_obs", "true_ite", "estimated_ite"]).to_csv(
            run_dir / "predictions.csv", index=False
        )
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return

    q_t0 = q_t0[keep]
    q_t1 = q_t1[keep]
    g = g[keep]
    t = X_eval_full["treatment"].values.astype(float)[keep]
    y_obs = y_eval["y_factual"].values.astype(float)[keep]
    true_ite_all = df.loc[X_eval_full.index, "true_ite"].values
    true_ite = true_ite_all[keep]
    true_ate_all = float(true_ite_all.mean())
    true_ate_eval = float(true_ite.mean())
    ate_ref = true_ate_all if args.use_all_data else true_ate_eval

    tmle_ate, tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(
        q_t0, q_t1, g, t, y_obs, truncate_level=tmle_truncate
    )
    aiptw_ate = psi_aiptw(q_t0, q_t1, g, t, y_obs, truncate_level=dr_truncate)
    iptw_ate = psi_iptw(q_t0, q_t1, g, t, y_obs, truncate_level=dr_truncate)
    naive_ate = psi_naive(q_t0, q_t1, g, t, y_obs, truncate_level=dr_truncate)
    simple_ate = float(y_obs[t == 1].mean() - y_obs[t == 0].mean())
    estimated_ite = q_t1 - q_t0
    pehe = float(np.sqrt(np.mean((estimated_ite - true_ite) ** 2)))

    t_tensor = torch.tensor(t).float().view(-1, 1)
    q_t0_tensor = torch.tensor(q_t0).float().view(-1, 1)
    q_t1_tensor = torch.tensor(q_t1).float().view(-1, 1)
    factual_preds = (t_tensor * q_t1_tensor + (1 - t_tensor) * q_t0_tensor).detach().cpu().numpy().flatten()
    counterfactual_true = df.loc[X_eval_full.index, "y_cfactual"].values[keep]
    counterfactual_preds = ((1 - t_tensor) * q_t1_tensor + t_tensor * q_t0_tensor).detach().cpu().numpy().flatten()

    metrics = {
        "dataset": "acic",
        "source_file": Path(args.acic_file).name,
        "counterfactual_file": Path(args.acic_cf).name,
        "evaluation_split": "all" if args.use_all_data else "test",
        "lambda_pehe": args.lambda_pehe,
        "best_val_loss": float(model.training_history_.get("best_val_loss")) if getattr(model, "training_history_", None) and model.training_history_.get("best_val_loss") is not None else None,
        "tmle_ate": float(tmle_ate),
        "tmle_std": float(tmle_std),
        "aiptw_ate": float(aiptw_ate),
        "iptw_ate": float(iptw_ate),
        "naive_ate": float(naive_ate),
        "simple_ate": float(simple_ate),
        "true_ate_all": true_ate_all,
        "true_ate_eval": true_ate_eval,
        "true_ate_ref": ate_ref,
        "tmle_ate_error": abs(float(tmle_ate) - ate_ref),
        "simple_ate_error": abs(float(simple_ate) - ate_ref),
        "naive_ate_error": abs(float(naive_ate) - ate_ref),
        "aiptw_ate_error": abs(float(aiptw_ate) - ate_ref),
        "iptw_ate_error": abs(float(iptw_ate) - ate_ref),
        "pehe": pehe,
        "propensity_auc": float(roc_auc_score(t, g)),
        "propensity_auc_val": float(prop_best_auc),
        "factual_mse": float(mean_squared_error(y_obs, factual_preds)),
        "counterfactual_mse": float(mean_squared_error(counterfactual_true, counterfactual_preds)),
        "epsilon_hat": float(eps_hat),
        "tmle_initial_loss": float(initial_loss),
        "tmle_final_loss": float(final_loss),
        "tmle_g_loss": float(g_loss),
        "n_retained": int(keep.sum()),
    }

    pred_df = pd.DataFrame({
        "row_index": np.asarray(X_eval_full.index)[keep],
        "treatment": t,
        "y_obs": y_obs,
        "y_cfactual": counterfactual_true,
        "q_t0": q_t0,
        "q_t1": q_t1,
        "g": g,
        "pred_ite": estimated_ite,
        "true_ite": true_ite,
        "ite_error": estimated_ite - true_ite,
    })

    model = _prepare_model_for_dump(model)
    _save_model_bundle(run_dir, model)
    joblib.dump(model, run_dir / "cnrn_model.joblib")
    joblib.dump(prop_model, run_dir / "propensity_model.joblib")
    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    _save_json(run_dir / "metrics.json", metrics)
    _save_json(run_dir / "propensity_search.json", {"best_auc": float(prop_best_auc), "best_params": prop_best_params})
    _write_explanations(run_dir / "sample_explanations.txt", model, X_eval_full.iloc[np.asarray(keep)].drop(columns=["g_hat"], errors="ignore"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun one strong CNRN model and save checkpoints for interpretability work.")
    parser.add_argument("--dataset", choices=["ihdp", "acic"], required=True)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--propensity-trials", type=int, default=5)
    parser.add_argument("--propensity-epochs", type=int, default=30)
    parser.add_argument("--cnrn-epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--target-lower-quantile", type=float, default=0.5)
    parser.add_argument("--target-upper-quantile", type=float, default=0.95)
    parser.add_argument("--target-lower-expand-ratio", type=float, default=0.0)
    parser.add_argument("--target-upper-expand-ratio", type=float, default=0.0)
    parser.add_argument("--lambda-pehe", type=float, default=0.0)
    parser.add_argument("--loss-on-original-scale", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--disable-scheduler", action="store_true")
    parser.add_argument("--ihdp-layer-sizes", default="75,75,75")
    parser.add_argument("--ihdp-head-layer-sizes", default="")
    parser.add_argument("--ihdp-mlp-head-hidden-dim", type=int, default=0)
    parser.add_argument("--ihdp-n-selected-features-input", type=int, default=25)
    parser.add_argument("--ihdp-n-selected-features-internal", type=int, default=20)
    parser.add_argument("--ihdp-n-selected-features-output", type=int, default=2)
    parser.add_argument("--ihdp-weight-init", type=float, default=0.5)
    parser.add_argument("--acic-layer-sizes", default="100,100,100")
    parser.add_argument("--acic-head-layer-sizes", default="")
    parser.add_argument("--acic-mlp-head-hidden-dim", type=int, default=0)
    parser.add_argument("--acic-n-selected-features-input", type=int, default=25)
    parser.add_argument("--acic-n-selected-features-internal", type=int, default=20)
    parser.add_argument("--acic-n-selected-features-output", type=int, default=4)
    parser.add_argument("--acic-weight-init", type=float, default=0.7)

    parser.add_argument("--ihdp-file", default=str(DEFAULT_IHDP_FILE))
    parser.add_argument("--has-headers", action="store_true")

    parser.add_argument("--acic-file", default=str(ROOT / "data" / "acic_2018" / DEFAULT_ACIC_FILE))
    parser.add_argument("--acic-cf", default=str(ROOT / "data" / "acic_2018" / DEFAULT_ACIC_CF))
    parser.add_argument("--acic-x", default=str(DEFAULT_ACIC_X))
    parser.add_argument("--max-covariates", type=int, default=150)
    parser.add_argument("--use-all-data", action="store_true")
    args = parser.parse_args()
    if args.early_stopping_patience is None:
        args.early_stopping_patience = args.cnrn_epochs

    out_root = Path(args.out_root)
    run_name = args.run_name or f"{args.dataset}_{_now_tag()}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "dataset": args.dataset,
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "random_seed": args.random_seed,
        "propensity_trials": args.propensity_trials,
        "propensity_epochs": args.propensity_epochs,
        "cnrn_epochs": args.cnrn_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "target_lower_quantile": args.target_lower_quantile,
        "target_upper_quantile": args.target_upper_quantile,
        "target_lower_expand_ratio": args.target_lower_expand_ratio,
        "target_upper_expand_ratio": args.target_upper_expand_ratio,
        "lambda_pehe": args.lambda_pehe,
        "loss_on_original_scale": bool(args.loss_on_original_scale),
        "device": args.device,
        "disable_scheduler": bool(args.disable_scheduler),
        "ihdp_layer_sizes": args.ihdp_layer_sizes,
        "ihdp_head_layer_sizes": args.ihdp_head_layer_sizes,
        "ihdp_mlp_head_hidden_dim": args.ihdp_mlp_head_hidden_dim,
        "ihdp_n_selected_features_input": args.ihdp_n_selected_features_input,
        "ihdp_n_selected_features_internal": args.ihdp_n_selected_features_internal,
        "ihdp_n_selected_features_output": args.ihdp_n_selected_features_output,
        "ihdp_weight_init": args.ihdp_weight_init,
        "acic_layer_sizes": args.acic_layer_sizes,
        "acic_head_layer_sizes": args.acic_head_layer_sizes,
        "acic_mlp_head_hidden_dim": args.acic_mlp_head_hidden_dim,
        "acic_n_selected_features_input": args.acic_n_selected_features_input,
        "acic_n_selected_features_internal": args.acic_n_selected_features_internal,
        "acic_n_selected_features_output": args.acic_n_selected_features_output,
        "acic_weight_init": args.acic_weight_init,
    }
    if args.dataset == "ihdp":
        resolved["ihdp_file"] = str(Path(args.ihdp_file).resolve())
        resolved["has_headers"] = bool(args.has_headers)
    else:
        resolved["acic_file"] = str(Path(args.acic_file).resolve())
        resolved["acic_cf"] = str(Path(args.acic_cf).resolve())
        resolved["acic_x"] = str(Path(args.acic_x).resolve())
        resolved["max_covariates"] = int(args.max_covariates)
        resolved["use_all_data"] = bool(args.use_all_data)
    _save_json(run_dir / "run_config.json", resolved)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.dataset == "ihdp":
        run_ihdp(args, run_dir)
    else:
        run_acic(args, run_dir)

    print(f"Saved interpretability artifacts to: {run_dir}")


if __name__ == "__main__":
    main()




