import warnings
warnings.filterwarnings("ignore", message="Choices for a categorical distribution should be a tuple")
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended")
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"aix360\.algorithms\.rbm\.features",
    message=r".*A value is trying to be set on a copy of a DataFrame or Series.*",
)

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import argparse
import random
import time
from sklearn.metrics import roc_auc_score
from joblib import parallel_backend

from torchlogic.sklogic.causal.semi_parametric_estimation.ate import (
    psi_tmle_cont_outcome, psi_aiptw, psi_iptw, psi_naive
)
from torchlogic.sklogic.causal.TARNRNRegressor import TARNRNRegressor
from torchlogic.utils.trainers.tarnrn_trainer import TARNetLoss
def fit_prop_model(X, y, covariate_cols, n_trials=5, epochs=30):
    from RNRN_tuner import RNRNClassifierTuner

    tuner = RNRNClassifierTuner(feature_names=covariate_cols, n_trials=n_trials, epochs=epochs)
    y_df = pd.DataFrame(y)
    with parallel_backend("threading", n_jobs=1):
        best_model, best_auc, best_params, _ = tuner.tune(X, y_df, verbose=False)
    print(f"Best propensity AUC (val): {best_auc:.4f}")
    print(f"Best propensity params: {best_params}")
    return best_model, best_auc, best_params


def load_acic_2018_data(x_path, factual_path, counterfactual_path):
    X_df = pd.read_csv(x_path)
    factual_df = pd.read_csv(factual_path)
    cf_df = pd.read_csv(counterfactual_path)

    df = pd.merge(X_df, factual_df, on="sample_id")
    df = pd.merge(df, cf_df, on="sample_id")

    df = df.rename(columns={"z": "treatment", "y": "y_factual", "y0": "mu0", "y1": "mu1"})
    df["y_cfactual"] = (1 - df["treatment"]) * df["mu1"] + df["treatment"] * df["mu0"]
    df["true_ite"] = df["mu1"] - df["mu0"]

    df["treatment"] = pd.to_numeric(df["treatment"], errors="coerce").astype(int)
    if not set(df["treatment"].unique()).issubset({0, 1}):
        raise ValueError(f"Treatment values are not 0/1: {df['treatment'].unique()}")

    covariate_cols = [
        col for col in df.columns if col not in {
            "sample_ID", "sample_id", "z", "y", "y0", "y1",
            "treatment", "y_factual", "mu0", "mu1", "y_cfactual", "true_ite"
        }
    ]

    true_ate = df["true_ite"].mean()
    true_att = df.loc[df["treatment"] == 1, "true_ite"].mean()
    return df, true_ate, true_att, covariate_cols


def extract_model_predictions(model, prop_model, X_full: pd.DataFrame):
    covariate_cols = [c for c in X_full.columns if c not in ["treatment", "g_hat"]]
    X_cov = X_full[covariate_cols].copy()

    y_pair = model.predict(X_full, treatment=None)  # shape: (n, 2)
    q_t0 = y_pair[:, 0]
    q_t1 = y_pair[:, 1]

    g_hat = prop_model.predict_proba(X_cov)
    if isinstance(g_hat, pd.DataFrame):
        g_hat = g_hat.values
    g_hat = np.asarray(g_hat)

    if g_hat.ndim == 2:
        if g_hat.shape[1] == 2:
            g = g_hat[:, 1]
        elif g_hat.shape[1] == 1:
            g = g_hat[:, 0]
        else:
            raise ValueError(f"Unexpected predict_proba shape: {g_hat.shape}")
    else:
        g = g_hat.reshape(-1)

    g = np.clip(g.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return q_t0, q_t1, g

def run_single_analysis(
    x_path,
    factual_path,
    counterfactual_path,
    data_dir="data/acic_2018/",
    verbose=True,
    use_all_data=True,
    seed=None,
    max_covariates=150,
    max_propensity_auc=0.9,
    beta_l1=0.0,
    lambda_pehe=0.0,
    propensity_trials=5,
    propensity_epochs=30,
    device=None,
    cnrn_epochs=60,
):
    try:
        run_t0 = time.time()
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        factual_full = factual_path if os.path.isabs(factual_path) else os.path.join(data_dir, factual_path)
        cf_full = counterfactual_path if os.path.isabs(counterfactual_path) else os.path.join(data_dir, counterfactual_path)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(factual_full)}")
            print('='*60)

        df, true_ate, true_att, covariate_cols = load_acic_2018_data(x_path, factual_full, cf_full)
        n_covariates_total = len(covariate_cols)
        if max_covariates is not None and max_covariates > 0 and len(covariate_cols) > max_covariates:
            covariate_cols = covariate_cols[:max_covariates]
        n_covariates_used = len(covariate_cols)
        X_full = df[covariate_cols + ["treatment"]].copy()
        y_full = pd.DataFrame({"y_factual": df["y_factual"]})

        if use_all_data:
            X_tr_full, y_tr = X_full.copy(), y_full.copy()
            X_eval_full, y_eval = X_full.copy(), y_full.copy()
        else:
            X_tr_full, X_eval_full, y_tr, y_eval = train_test_split(
                X_full, y_full, test_size=0.30, random_state=42, stratify=X_full["treatment"]
            )

        if verbose:
            print(
                f"Treatment rates - Train: {X_tr_full['treatment'].mean():.3f}, "
                f"Eval: {X_eval_full['treatment'].mean():.3f}"
            )
            print(f"Using covariates: {len(covariate_cols)}")

        prop_t0 = time.time()
        prop_model, prop_best_auc, prop_best_params = fit_prop_model(
            X_tr_full[covariate_cols],
            X_tr_full["treatment"],
            covariate_cols=covariate_cols,
            n_trials=propensity_trials,
            epochs=propensity_epochs,
        )
        prop_fit_seconds = time.time() - prop_t0
        if max_propensity_auc is not None and prop_best_auc > float(max_propensity_auc):
            if verbose:
                print(
                    f"Skipping run: propensity val AUC {prop_best_auc:.4f} "
                    f"> threshold {float(max_propensity_auc):.4f}"
                )
            return {
                "status": "skipped_high_propensity_auc",
                "filename": os.path.basename(factual_full),
                "counterfactual_file": os.path.basename(cf_full),
                "seed": seed,
                "use_all_data": bool(use_all_data),
                "evaluation_split": "all" if use_all_data else "test",
                "max_covariates": max_covariates,
                "n_covariates_total": n_covariates_total,
                "n_covariates_used": n_covariates_used,
                "propensity_auc_val": float(prop_best_auc),
                "propensity_auc_threshold": float(max_propensity_auc),
                "propensity_fit_seconds": prop_fit_seconds,
                "runtime_seconds": time.time() - run_t0,
            }

        g_train = np.asarray(prop_model.predict_proba(X_tr_full[covariate_cols])).reshape(-1)
        X_tr_full["g_hat"] = g_train

        tmle_truncate = 0.01
        dr_truncate = 0.05
        tar_model_params = {
            "normal_form": "cnf",
            "epochs": cnrn_epochs,
            "learning_rate": 0.05,
            "weight_decay": 0.0001,
            "layer_sizes": [100, 100, 100],
            "n_selected_features_input": 25,
            "n_selected_features_internal": 20,
            "n_selected_features_output": 4,
            "add_negations": False,
            "weight_init": 0.7,
            "t_0": 40,
            "t_mult": 4,
            "batch_size": 16,
            "early_stopping_plateau_count": 80,
            "lambda_pehe": lambda_pehe,
            "balance_classes": True,
            "holdout_pct": 0.1,
            "device": device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
            "beta_l1": float(beta_l1),
        }
        model = TARNRNRegressor(
            feature_names=covariate_cols,
            target_name='y_factual',
            treatment_col ='treatment',
            propensity_col = 'g_hat',
            normal_form=tar_model_params["normal_form"],
            epochs=tar_model_params["epochs"],
            learning_rate=tar_model_params["learning_rate"],
            weight_decay=tar_model_params["weight_decay"],
            layer_sizes=tar_model_params["layer_sizes"],
            n_selected_features_input=tar_model_params["n_selected_features_input"],
            n_selected_features_internal=tar_model_params["n_selected_features_internal"],
            n_selected_features_output=tar_model_params["n_selected_features_output"],
            add_negations=tar_model_params["add_negations"],
            weight_init=tar_model_params["weight_init"],
            t_0=tar_model_params["t_0"],
            t_mult=tar_model_params["t_mult"],
            batch_size=tar_model_params["batch_size"],
            early_stopping_plateau_count=tar_model_params["early_stopping_plateau_count"],
            loss_fn=TARNetLoss(lambda_pehe=tar_model_params["lambda_pehe"], balance_classes=True, eps=1e-6, epsilon_init=0.1),
            holdout_pct=tar_model_params["holdout_pct"],
            device=tar_model_params["device"],
            beta_l1=tar_model_params["beta_l1"],
            )


        if verbose:
            print("Fitting model...")
        tar_fit_t0 = time.time()
        model.fit(X_tr_full, y_tr)
        tar_fit_seconds = time.time() - tar_fit_t0
        q_t0, q_t1, g = extract_model_predictions(model, prop_model, X_eval_full)
        g_min_pre = float(np.min(g))
        g_max_pre = float(np.max(g))
        g_mean_pre = float(np.mean(g))
        g_std_pre = float(np.std(g))

        t_all = X_eval_full["treatment"].values.astype(float)
        y_obs_all = y_eval["y_factual"].values.astype(float)
        true_ite_all = df.loc[X_eval_full.index, "true_ite"].values
        true_ate_all = true_ite_all.mean()
        true_att_all = df.loc[X_eval_full.index[X_eval_full["treatment"] == 1], "true_ite"].mean()

        # Dragonnet paper excludes points with extreme propensity values for estimation.
        keep = (g >= tmle_truncate) & (g <= (1.0 - tmle_truncate))
        if keep.sum() < 10:
            raise ValueError(f"Too few points after propensity truncation: {keep.sum()} / {len(keep)}")

        q_t0 = q_t0[keep]
        q_t1 = q_t1[keep]
        g = g[keep]
        t = t_all[keep]
        y_obs = y_obs_all[keep]
        true_ite = true_ite_all[keep]
        true_ate = true_ite.mean()
        true_att = true_ite[t == 1].mean() if (t == 1).any() else np.nan
        # Error reference:
        # - paper-style all-data protocol: compare against full-data true ATE
        # - split/eval protocol: compare against retained-eval true ATE
        ate_error_ref = true_ate_all if use_all_data else true_ate
        ate_error_ref_name = "true_ate_all" if use_all_data else "true_ate_eval"

        tmle_ate, tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(
            q_t0, q_t1, g, t, y_obs, truncate_level=tmle_truncate
        )
        aiptw_ate = psi_aiptw(q_t0, q_t1, g, t, y_obs, truncate_level=dr_truncate)
        iptw_ate = psi_iptw(q_t0, q_t1, g, t, y_obs, truncate_level=dr_truncate)
        naive_ate = psi_naive(q_t0, q_t1, g, t, y_obs, truncate_level=dr_truncate)
        simple_ate = y_obs[t == 1].mean() - y_obs[t == 0].mean()

        t_tensor = torch.tensor(t).float().view(-1, 1)
        q_t0_tensor = torch.tensor(q_t0).float().view(-1, 1)
        q_t1_tensor = torch.tensor(q_t1).float().view(-1, 1)
        factual_preds = (t_tensor * q_t1_tensor + (1 - t_tensor) * q_t0_tensor).detach().cpu().numpy().flatten()
        estimated_ite = q_t1 - q_t0
        pehe = np.sqrt(np.mean((estimated_ite - true_ite) ** 2))
        sqrt_pehe = pehe

        ss_res = np.sum((y_obs - factual_preds) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        factual_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        counterfactual_true = df.loc[X_eval_full.index, "y_cfactual"].values[keep]
        counterfactual_preds = ((1 - t_tensor) * q_t1_tensor + t_tensor * q_t0_tensor).detach().cpu().numpy().flatten()
        counterfactual_mse = mean_squared_error(counterfactual_true, counterfactual_preds)
        overall_mse = mean_squared_error(
            np.concatenate([y_obs, counterfactual_true]),
            np.concatenate([factual_preds, counterfactual_preds])
        )

        if verbose:
            print(f"Propensity AUC: {roc_auc_score(t, g):.3f}")
            print(f"True ATE (eval): {true_ate:.3f}")
            print(f"True ATE (all): {true_ate_all:.3f}")
            print(f"ATE error reference: {ate_error_ref_name}")
            print(f"TMLE ATE: {tmle_ate:.3f} (+/- {1.96 * tmle_std:.3f})")
            print(f"Retained after truncation: {keep.sum()} / {len(keep)}")

        return {
            "filename": os.path.basename(factual_full),
            "counterfactual_file": os.path.basename(cf_full),
            "status": "ok",
            "seed": seed,
            "use_all_data": bool(use_all_data),
            "evaluation_split": "all" if use_all_data else "test",
            "max_covariates": max_covariates,
            "n_covariates_total": n_covariates_total,
            "n_covariates_used": n_covariates_used,
            "n_total": len(df),
            "n_eval": len(X_eval_full),
            "n_retained": int(keep.sum()),
            "retained_frac": float(keep.mean()),
            "treatment_rate": df["treatment"].mean(),
            "y_obs_mean": float(np.mean(y_obs_all)),
            "y_obs_std": float(np.std(y_obs_all)),
            "true_ate_all": true_ate_all,
            "true_att_all": true_att_all,
            "true_ate_eval": true_ate,
            "true_att_eval": true_att,
            "ate_error_reference": ate_error_ref_name,
            "true_ate_ref": ate_error_ref,
            "tmle_truncate": tmle_truncate,
            "dr_truncate": dr_truncate,
            "tmle_ate": tmle_ate,
            "tmle_std": tmle_std,
            "aiptw_ate": aiptw_ate,
            "iptw_ate": iptw_ate,
            "naive_ate": naive_ate,
            "simple_ate": simple_ate,
            "tmle_ate_error": abs(tmle_ate - ate_error_ref),
            "aiptw_ate_error": abs(aiptw_ate - ate_error_ref),
            "iptw_ate_error": abs(iptw_ate - ate_error_ref),
            "naive_ate_error": abs(naive_ate - ate_error_ref),
            "simple_ate_error": abs(simple_ate - ate_error_ref),
            "factual_mse": mean_squared_error(y_obs, factual_preds),
            "factual_r2": factual_r2,
            "pehe": pehe,
            "sqrt_pehe": sqrt_pehe,
            "counterfactual_mse": counterfactual_mse,
            "overall_mse": overall_mse,
            "propensity_auc": roc_auc_score(t, g),
            "propensity_auc_val": float(prop_best_auc),
            "propensity_g_min_pre": g_min_pre,
            "propensity_g_max_pre": g_max_pre,
            "propensity_g_mean_pre": g_mean_pre,
            "propensity_g_std_pre": g_std_pre,
            "propensity_train_mean": float(np.mean(g_train)),
            "propensity_train_std": float(np.std(g_train)),
            "epsilon_hat": eps_hat,
            "tmle_initial_loss": initial_loss,
            "tmle_final_loss": final_loss,
            "tmle_g_loss": g_loss,
            "propensity_fit_seconds": prop_fit_seconds,
            "tar_fit_seconds": tar_fit_seconds,
            "runtime_seconds": time.time() - run_t0,
        }
    except Exception as e:
        print(f"Error processing {factual_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "filename": os.path.basename(factual_path),
            "counterfactual_file": os.path.basename(counterfactual_path),
            "seed": seed,
            "error": str(e),
            "runtime_seconds": time.time() - run_t0,
        }


def _load_factual_file_list(list_path: str):
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"File list not found: {list_path}")

    if list_path.lower().endswith(".txt"):
        with open(list_path, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        df_list = pd.read_csv(list_path)
        if "file" in df_list.columns:
            files = df_list["file"].astype(str).str.strip().tolist()
        else:
            files = df_list.iloc[:, 0].astype(str).str.strip().tolist()

    files = [f for f in files if f]
    if not files:
        raise ValueError(f"No factual files listed in: {list_path}")
    return files


def run_batch_analysis(
    data_dir="data/acic_2018",
    n_sample=25,
    n_runs=1,
    use_all_data=True,
    random_seed=42,
    file_list="selected_files_list.txt",
    max_covariates=150,
    max_propensity_auc=0.9,
    beta_l1=0.0,
    lambda_pehe=0.0,
    propensity_trials=5,
    propensity_epochs=30,
    device=None,
    cnrn_epochs=60,
    output_file="TAR_acic2018_batch_results.csv",
):
    print("ACIC 2018 TAR Batch Analysis")
    print("=" * 50)

    x_path = os.path.join(data_dir, "x.csv")
    list_path = file_list if os.path.isabs(file_list) else os.path.join(os.path.dirname(__file__), file_list)
    try:
        all_factuals = _load_factual_file_list(list_path)
    except Exception as e:
        print(f"Error loading file list: {e}")
        return None

    n_sample = min(n_sample, len(all_factuals))
    random.seed(random_seed)
    selected_factuals = random.sample(all_factuals, n_sample)
    print(f"Using file list: {list_path}")
    print(f"Random seed: {random_seed}")
    print(f"Selected {n_sample} files (random sample):")
    for i, f in enumerate(selected_factuals, 1):
        print(f"  {i:2d}. {os.path.basename(f)}")

    results = []
    skipped_runs = []
    failed_runs = []
    output_path = output_file if os.path.isabs(output_file) else os.path.join(os.path.dirname(__file__), output_file)
    existing_results = pd.DataFrame()
    completed_counts = {}
    if os.path.exists(output_path):
        try:
            existing_results = pd.read_csv(output_path)
            if not existing_results.empty and "filename" in existing_results.columns:
                completed_counts = existing_results.groupby("filename").size().to_dict()
                print(f"Resuming from existing output: {output_path} ({len(existing_results)} rows)")
        except Exception as e:
            print(f"Warning: could not load existing output '{output_path}': {e}")
            existing_results = pd.DataFrame()

    for i, factual_path in enumerate(selected_factuals, 1):
        cf_path = (factual_path[:-4] if factual_path.endswith(".csv") else factual_path) + "_cf.csv"
        factual_full = os.path.join(data_dir, factual_path)
        cf_full = os.path.join(data_dir, cf_path)

        print(f"\n[{i:2d}/{n_sample}] Starting: {os.path.basename(factual_path)}")
        if not os.path.exists(factual_full) or not os.path.exists(cf_full):
            print(f"Skipping missing pair: {factual_path}, {cf_path}")
            continue

        file_runs = []
        completed_for_file = int(completed_counts.get(os.path.basename(factual_path), 0))
        if completed_for_file >= n_runs:
            print(f"Already completed: {os.path.basename(factual_path)} | existing runs={completed_for_file}")
            continue

        for r in range(completed_for_file, n_runs):
            result = run_single_analysis(
                x_path=x_path,
                factual_path=factual_path,
                counterfactual_path=cf_path,
                data_dir=data_dir,
                verbose=False,
                use_all_data=use_all_data,
                seed=random_seed + r,
                max_covariates=max_covariates,
                max_propensity_auc=max_propensity_auc,
                beta_l1=beta_l1,
                lambda_pehe=lambda_pehe,
                propensity_trials=propensity_trials,
                propensity_epochs=propensity_epochs,
                device=device,
                cnrn_epochs=cnrn_epochs,
            )
            if result is not None and result.get("status") == "ok":
                result["run_id"] = r
                file_runs.append(result)
                results.append(result)
                row_df = pd.DataFrame([result])
                write_header = not os.path.exists(output_path)
                row_df.to_csv(output_path, mode='a', header=write_header, index=False)
                completed_counts[os.path.basename(factual_path)] = int(completed_counts.get(os.path.basename(factual_path), 0)) + 1
            elif result is not None and result.get("status") == "skipped_high_propensity_auc":
                result["run_id"] = r
                skipped_runs.append(result)
            elif result is not None and result.get("status") == "failed":
                result["run_id"] = r
                failed_runs.append(result)

        if file_runs:
            tmle_err_mean = np.mean([rr["tmle_ate_error"] for rr in file_runs])
            print(f"Completed: {os.path.basename(factual_path)} | runs={len(file_runs)} | mean TMLE error={tmle_err_mean:.3f}")
        elif any(
            sr.get("filename") == os.path.basename(factual_path)
            for sr in skipped_runs
        ):
            print(f"Skipped: {os.path.basename(factual_path)} | all runs filtered by propensity AUC > {max_propensity_auc}")
        else:
            print(f"Failed: {os.path.basename(factual_path)}")

    if skipped_runs:
        skipped_file = "TAR_acic2018_skipped_runs.csv"
        pd.DataFrame(skipped_runs).to_csv(skipped_file, index=False)
        print(f"\nSkipped runs logged to: {skipped_file} (count={len(skipped_runs)})")
    if failed_runs:
        failed_file = "TAR_acic2018_failed_runs.csv"
        pd.DataFrame(failed_runs).to_csv(failed_file, index=False)
        print(f"Failed runs logged to: {failed_file} (count={len(failed_runs)})")

    if os.path.exists(output_path):
        df_results = pd.read_csv(output_path)
    elif results:
        df_results = pd.DataFrame(results)
    else:
        print("No successful analyses to summarize.")
        return None

    out_file = output_path

    print(f"\n{'='*80}")
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Successfully processed runs: {len(results)} over {n_sample} datasets")
    print(f"Skipped runs (high propensity AUC): {len(skipped_runs)}")
    print(f"Failed runs: {len(failed_runs)}")

    numeric_cols = [c for c in df_results.columns if pd.api.types.is_numeric_dtype(df_results[c])]
    summary_rows = []
    for col in numeric_cols:
        values = df_results[col].dropna()
        n = len(values)
        if n == 0:
            continue
        std = values.std()
        sem = std / np.sqrt(n) if n > 1 else np.nan
        summary_rows.append({
            "metric": col,
            "n": n,
            "mean": values.mean(),
            "std": std,
            "se": sem,
            "min": values.min(),
            "max": values.max(),
        })

    summary_df = pd.DataFrame(summary_rows).set_index("metric")
    print(summary_df[["n", "mean", "std", "se", "min", "max"]])

    paper_metrics = {
        "TMLE ATE Error": "tmle_ate_error",
        "AIPTW ATE Error": "aiptw_ate_error",
        "PEHE": "pehe",
        "sqrt(PEHE)": "sqrt_pehe",
    }
    print("\nPaper-style summary (mean +/- SE):")
    for name, col in paper_metrics.items():
        if col in df_results.columns:
            values = df_results[col].dropna()
            n = len(values)
            if n > 0:
                std = values.std()
                se = std / np.sqrt(n) if n > 1 else np.nan
                print(f"  {name}: {values.mean():.3f} +/- {se:.3f} (N={n})")

    print(f"\nDetailed results saved to: {out_file}")
    return df_results


def main():
    parser = argparse.ArgumentParser(description="ACIC 2018 TARNet-style Causal Inference Analysis")
    parser.add_argument("--mode", choices=["single", "batch"], default="batch")
    parser.add_argument("--data_dir", type=str, default="data/acic_2018")
    parser.add_argument("--n_sample", type=int, default=25)
    parser.add_argument("--n_runs", type=int, default=1, help="Repeat each dataset this many times (paper used 25 for ACIC)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed used for file sampling and run seeds")
    parser.add_argument("--file_list", type=str, default="selected_files_list.txt", help="Path to factual file list (.txt or .csv)")
    parser.add_argument("--max_covariates", type=int, default=150, help="Cap number of covariates from x.csv (in file order)")
    parser.add_argument("--max_propensity_auc", type=float, default=0.9, help="Skip run if tuned propensity val AUC exceeds this threshold")
    parser.add_argument("--beta_l1", type=float, default=0.0, help="L1 regularization weight.")
    parser.add_argument("--lambda_pehe", type=float, default=0.0, help="PEHE regularization weight for CNRN.")
    parser.add_argument("--propensity_trials", type=int, default=5, help="Optuna trials for the propensity tuner.")
    parser.add_argument("--propensity_epochs", type=int, default=30, help="Training epochs per propensity tuning trial.")
    parser.add_argument("--device", type=str, default=None, help="Torch device for CNRN training, e.g. cuda:0 or cpu.")
    parser.add_argument("--cnrn_epochs", type=int, default=60, help="Training epochs for the CNRN model.")
    parser.add_argument("--output_file", type=str, default="TAR_acic2018_batch_results.csv", help="Batch results CSV path.")
    parser.add_argument("--use_all_data", action="store_true", help="Train and estimate on all rows (paper protocol for ACIC)")
    parser.add_argument("--factual_file", type=str, default=None)
    parser.add_argument("--counterfactual_file", type=str, default=None)
    args = parser.parse_args()

    x_path = os.path.join(args.data_dir, "x.csv")
    if args.mode == "single":
        if not args.factual_file:
            raise ValueError("--factual_file is required for single mode")
        cf_file = args.counterfactual_file
        if not cf_file:
            base = args.factual_file[:-4] if args.factual_file.endswith(".csv") else args.factual_file
            cf_file = f"{base}_cf.csv"
        run_single_analysis(
            x_path, args.factual_file, cf_file,
            data_dir=args.data_dir, verbose=True, use_all_data=args.use_all_data,
            max_covariates=args.max_covariates,
            max_propensity_auc=args.max_propensity_auc,
            beta_l1=args.beta_l1,
            lambda_pehe=args.lambda_pehe,
            propensity_trials=args.propensity_trials,
            propensity_epochs=args.propensity_epochs,
            device=args.device,
            cnrn_epochs=args.cnrn_epochs,
        )
    else:
        run_batch_analysis(
            data_dir=args.data_dir,
            n_sample=args.n_sample,
            n_runs=args.n_runs,
            use_all_data=args.use_all_data,
            random_seed=args.random_seed,
            file_list=args.file_list,
            max_covariates=args.max_covariates,
            max_propensity_auc=args.max_propensity_auc,
            beta_l1=args.beta_l1,
            lambda_pehe=args.lambda_pehe,
            propensity_trials=args.propensity_trials,
            propensity_epochs=args.propensity_epochs,
            device=args.device,
            cnrn_epochs=args.cnrn_epochs,
            output_file=args.output_file,
        )


if __name__ == "__main__":
    main()
