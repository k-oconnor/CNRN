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

import copy
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import glob
import random
import argparse
from datetime import datetime
from sklearn.metrics import roc_auc_score

from torchlogic.sklogic.causal.semi_parametric_estimation.ate import (
        psi_tmle_cont_outcome, psi_aiptw, psi_iptw, psi_naive)
from torchlogic.sklogic.causal.TARNRNRegressor import TARNRNRegressor
from torchlogic.utils.trainers.tarnrn_trainer import TARNRNTrainer,TARNetLoss
from sklearn.linear_model import LogisticRegression
from joblib import parallel_backend


def logit(x, eps=1e-6):
            x = np.clip(x, eps, 1-eps)
            return np.log(x) - np.log(1-x)


def _to_1d_probs(pred):
    if isinstance(pred, pd.DataFrame):
        vals = pred.values
    elif isinstance(pred, pd.Series):
        vals = pred.values
    else:
        vals = np.asarray(pred)

    vals = np.asarray(vals)
    if vals.ndim == 2:
        if vals.shape[1] == 2:
            vals = vals[:, 1]
        elif vals.shape[1] == 1:
            vals = vals[:, 0]
        else:
            vals = vals.reshape(-1)
    else:
        vals = vals.reshape(-1)
    return np.clip(vals.astype(np.float64), 1e-6, 1.0 - 1e-6)

def fit_prop_model(x, y, covariate_cols, n_trials=5, epochs=30):
    try:
        from RNRN_tuner import RNRNClassifierTuner
    except ModuleNotFoundError:
        model = LogisticRegression(max_iter=1000, n_jobs=1)
        model.fit(x[covariate_cols], np.asarray(y).astype(int))
        return model

    tuner = RNRNClassifierTuner(
        feature_names=covariate_cols,
        n_trials=n_trials,
        epochs=epochs,
        binarization=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        force_single_core=(os.cpu_count() == 1),
    )
    y = pd.DataFrame(y)
    # Force thread backend + single worker to avoid Windows loky permission failures.
    with parallel_backend("threading", n_jobs=1):
        best_model, best_auc, best_params, study = tuner.tune(x,y,verbose=False)
    return best_model

def estimate_propensity_scores(model, X):
    
    predictions_probs = model.predict_proba(X)
    print(f"Samples before clipping: {len(probs)}")
    probs = np.clip(predictions_probs, 0.0001, 0.99999)
    print(f"Samples after clipping: {len(probs)}")

    return probs


def load_header_columns(header_file="ihdp_headers.txt"):
    """Load column names from header file"""
    try:
        with open(header_file, 'r') as f:
            header_line = f.read().strip()
            columns = [col.strip() for col in header_line.split(',')]
        return columns
    except FileNotFoundError:
        print(f"Header file {header_file} not found. Using default IHDP columns.")
        # Default IHDP columns as fallback
        return ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1'] + [f'x{i}' for i in range(1, 26)]

def load_ihdp_data(filepath='data/ihdp_data.csv', has_headers=False, columns=None):
    """
    Load IHDP dataset with or without headers
    
    The IHDP dataset contains:
    - treatment: binary treatment indicator (0/1 or True/False)
    - y_factual: observed outcome
    - y_cfactual: counterfactual outcome (for evaluation)
    - mu0: true control outcome function
    - mu1: true treatment outcome function  
    - x1-x25: covariates (mix of continuous and binary features)
    """
    if has_headers:
        df = pd.read_csv(filepath)
    else:
        columns = load_header_columns('ihdp_headers.txt')
        df = pd.read_csv(filepath, names=columns)
    
    # Handle treatment column - convert to numeric if it's boolean
    print(f"Treatment column dtype: {df['treatment'].dtype}")
    print(f"Treatment unique values: {df['treatment'].unique()}")
    
    if df['treatment'].dtype == 'bool' or df['treatment'].dtype == 'object':
        # Convert boolean or string to numeric
        df['treatment'] = df['treatment'].astype(int)
        print(f"Converted treatment to numeric: {df['treatment'].unique()}")
    
    # Ensure treatment is 0/1
    if not set(df['treatment'].unique()).issubset({0, 1}):
        print(f"Warning: Treatment values are not 0/1: {df['treatment'].unique()}")
    
    # Calculate true ATE and ATT for evaluation
    true_ate = (df['mu1'] - df['mu0']).mean()
    treated_mask = df['treatment'] == 1
    true_att = (df['mu1'][treated_mask] - df['mu0'][treated_mask]).mean()
    
    # Calculate true ITE for each individual
    true_ite = df['mu1'] - df['mu0']
    
    print(f"IHDP Dataset loaded:")
    print(f"  - Sample size: {len(df)}")
    print(f"  - Treatment rate: {df['treatment'].mean():.3f}")
    print(f"  - True ATE: {true_ate:.3f}")
    print(f"  - True ATT: {true_att:.3f}")
    print(f"  - Outcome range: [{df['y_factual'].min():.2f}, {df['y_factual'].max():.2f}]")
    
    return df, true_ate, true_att, true_ite

def extract_model_predictions(model, prop_model, X_full: pd.DataFrame, calibrator=None, device='cpu'):
    """
    Given the full IHDP dataframe (including 'treatment'), return:
      - q_t0: E[Y | X, T=0]
      - q_t1: E[Y | X, T=1]
      - g:    estimated propensity P(T=1 | X)
    """
    eps = 1e-6

    # Covariates only for propensity model
    covariate_cols = [c for c in X_full.columns if c not in ['treatment', 'g_hat']]
    X_cov = X_full[covariate_cols].copy()

    # Outcome heads from TARNRNRegressor API (already inverse-transformed)
    # predict() accepts full frame and will drop treatment/propensity columns internally.
    y_pair = model.predict(X_full, treatment=None)  # shape: (n, 2)
    q_t0 = y_pair[:, 0]
    q_t1 = y_pair[:, 1]

    # Propensity from external model; calibrate if provided.
    g = _to_1d_probs(prop_model.predict_proba(X_cov))
    if calibrator is not None:
        g = calibrator.predict_proba(logit(g).reshape(-1, 1))[:, 1]
        g = np.clip(g.astype(np.float64), 1e-6, 1.0 - 1e-6)

    return q_t0, q_t1, g

def run_single_analysis(
        filepath,
        has_headers=True,
        columns=None,
        verbose=True,
        beta_l1=0.0,
        lambda_pehe=0.0,
        propensity_trials=5,
        propensity_epochs=30,
        device=None,
        cnrn_epochs=60,
):
    """Run analysis on a single dataset"""
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(filepath)}")
            print('='*60)
        
        # 1) Load raw data
        df, true_ate, true_att, true_ite = load_ihdp_data(filepath, has_headers=has_headers)
        
        # 2) Define covariate columns and keep treatment for later
        covariate_cols = [f'x{i}' for i in range(1, 26)]
        X_full = df[covariate_cols + ['treatment']].copy()
        y_full = pd.DataFrame({'y_factual': df['y_factual']})

        # 3) Split into train / val / test, stratifying on treatment
        X_tr_full, X_ts_full, y_tr, y_ts = train_test_split(
            X_full, y_full, test_size=0.27, random_state=42,stratify=X_full['treatment'],
        )
        
        #print(covariate_cols)
        print(f"Treatment rates - Train: {X_tr_full['treatment'].mean():.3f}, Test: {X_ts_full['treatment'].mean():.3f}")
        

        prop_model = fit_prop_model(
            X_tr_full[covariate_cols],
            X_tr_full['treatment'],
            covariate_cols=covariate_cols,
            n_trials=propensity_trials,
            epochs=propensity_epochs,
        )
        g_train_raw = _to_1d_probs(prop_model.predict_proba(X_tr_full[covariate_cols]))
        X_tr_full['g_hat'] = g_train_raw
        calibrator = LogisticRegression(n_jobs=1)
        calibrator.fit(logit(g_train_raw).reshape(-1, 1), X_tr_full['treatment'].values.astype(int))
        
        print(f"Covariates: {covariate_cols}")
        print(f"Training data shape: {X_tr_full.shape}")


        model = TARNRNRegressor(
            feature_names=covariate_cols,
            target_name='y_factual',
            treatment_col ='treatment',
            propensity_col = 'g_hat',
            normal_form='cnf',
            epochs=cnrn_epochs,
            learning_rate=0.05,
            weight_decay=0.0001,
            layer_sizes=[75,75,75],
            n_selected_features_input=25,
            n_selected_features_internal=20,
            n_selected_features_output=2, 
            add_negations=False,
            weight_init=.5,
            t_0=40,
            t_mult=4,
            batch_size=32,
            early_stopping_plateau_count=40,
            loss_fn=TARNetLoss(lambda_pehe=lambda_pehe, balance_classes=True, eps=1e-6, epsilon_init=0.1),
            holdout_pct=0.1,
            device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu'),
            beta_l1=float(beta_l1),
            )

        print("Fitting model...")
        model.fit(X_tr_full, y_tr)
        q_t0, q_t1, g = extract_model_predictions(model, prop_model, X_ts_full, calibrator=calibrator)
   

        print(f"q_t0 shape: {q_t0.shape}, q_t1 shape: {q_t1.shape}, g shape: {g.shape}")
        print(f"Propensity score range: [{g.min():.3f}, {g.max():.3f}]")

        
        # 7) Evaluate propensity AUC
        t = X_ts_full['treatment'].values.astype(float)
        y_obs = y_ts['y_factual'].values.astype(float)

        print("Propensity AUC:", roc_auc_score(t, g))
        print(f"Test set treatment prevalence: {t.mean():.3f}")
        print(f"Test set outcome range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
        
        # Get true values for test set
        test_indices = y_ts.index
        true_ite_test = (df.loc[test_indices, 'mu1'] - df.loc[test_indices, 'mu0']).values
        true_ate_test = true_ite_test.mean()
        true_att_test = (df.loc[test_indices[df.loc[test_indices, 'treatment'] == 1], 'mu1'] - 
                        df.loc[test_indices[df.loc[test_indices, 'treatment'] == 1], 'mu0']).mean()
        
        print(f"True ATE (test): {true_ate_test:.3f}")
        print(f"True ATT (test): {true_att_test:.3f}")
        
        
        # Compute causal estimates using semi-parametric methods
        print("\n=== ATE Estimates ===")
        # TMLE
        tmle_ate, tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(
            q_t0, q_t1, g, t, y_obs, truncate_level=0.001
        )
        print(f"TMLE ATE: {tmle_ate:.3f} (±{1.96*tmle_std:.3f})")
        print(f"  - True ATE: {true_ate:.3f}")
        print(f"  - ATE Error: {abs(tmle_ate - true_ate_test):.3f}")
        print(f"  - Epsilon: {eps_hat:.6f}")
        print(f"  - Initial loss: {initial_loss:.3f}")
        print(f"  - Final loss: {final_loss:.3f}")
        print(f"  - Propensity loss: {g_loss:.3f}")
        

        # AIPTW
        aiptw_ate = psi_aiptw(q_t0, q_t1, g, t, y_obs, truncate_level=0.001)
        print(f"AIPTW ATE: {aiptw_ate:.3f}")
        print(f"  - ATE Error: {abs(aiptw_ate - true_ate_test):.3f}")
        
        # IPTW
        iptw_ate = psi_iptw(q_t0, q_t1, g, t, y_obs, truncate_level=0.001
                            )
        print(f"IPTW ATE: {iptw_ate:.3f}")
        print(f"  - ATE Error: {abs(iptw_ate - true_ate_test):.3f}")
        
        # Naive (plug-in)
        naive_ate = psi_naive(q_t0, q_t1, g, t, y_obs, truncate_level=0)
        print(f"Naive ATE: {naive_ate:.3f}")
        print(f"  - ATE Error: {abs(naive_ate - true_ate_test):.3f}")
        
        # Simple difference in means
        simple_ate = y_obs[t == 1].mean() - y_obs[t == 0].mean()
        print(f"Simple ATE: {simple_ate:.3f}")
        print(f"  - ATE Error: {abs(simple_ate - true_ate_test):.3f}")
    
        
        # Model performance
        print("\n=== Model Performance ===")
        # Factual prediction accuracy
        t_tensor = torch.tensor(t).float().view(-1, 1)
        q_t0_tensor = torch.tensor(q_t0).float().view(-1, 1)
        q_t1_tensor = torch.tensor(q_t1).float().view(-1, 1)
        
        factual_preds = t_tensor * q_t1_tensor + (1 - t_tensor) * q_t0_tensor
        factual_preds = factual_preds.detach().cpu().numpy().flatten()
        
        mse = mean_squared_error(y_obs, factual_preds)
        print(f"Factual MSE: {mse:.3f}")
        
        # R-squared
        ss_res = np.sum((y_obs - factual_preds) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"Factual R²: {r2:.3f}")
        
        # Counterfactual prediction accuracy (only possible with IHDP)
        counterfactual_true = df.loc[test_indices, 'y_cfactual'].values
        counterfactual_preds = (1 - t_tensor) * q_t1_tensor + t_tensor * q_t0_tensor
        counterfactual_preds = counterfactual_preds.detach().cpu().numpy().flatten()
        
        cf_mse = mean_squared_error(counterfactual_true, counterfactual_preds)
        print(f"Counterfactual MSE: {cf_mse:.3f}")
        
        # Overall prediction accuracy (factual + counterfactual)
        all_true = np.concatenate([y_obs, counterfactual_true])
        all_preds = np.concatenate([factual_preds, counterfactual_preds])
        overall_mse = mean_squared_error(all_true, all_preds)
        print(f"Overall MSE: {overall_mse:.3f}")

        estimated_ite = q_t1 - q_t0
        pehe = np.sqrt(np.mean((estimated_ite - true_ite_test) ** 2))
        sqrt_pehe = pehe

        #from torchinfo import summary

        #summary(model.model, input_size=(64, 25))  # Batch size 16, 25 features
       # print(prop_model.explain_sample(X_full[covariate_cols], quantile=.5,sample_index=1))


        
        # Return results dictionary
        return {
            'filename': os.path.basename(filepath),
            'n_total': len(df),
            'n_test': len(X_ts_full),
            'treatment_rate': df['treatment'].mean(),
            'true_ate': true_ate,
            'true_att': true_att,
            'true_ate_test': true_ate_test,
            'evaluation_split': 'test',
            'tmle_ate': tmle_ate,
            'tmle_std': tmle_std,
            'aiptw_ate': aiptw_ate,
            'iptw_ate': iptw_ate,
            'naive_ate': naive_ate,
            'simple_ate': simple_ate,
            'tmle_ate_error': abs(tmle_ate - true_ate_test).mean(),
            'aiptw_ate_error': abs(aiptw_ate - true_ate_test).mean(),
            'iptw_ate_error': abs(iptw_ate - true_ate_test).mean(),
            'naive_ate_error': abs(naive_ate - true_ate_test).mean(),
            'simple_ate_error': abs(simple_ate - true_ate_test).mean(),
            'pehe': pehe,
            'sqrt_pehe': sqrt_pehe,
            'factual_mse': mse,
            'factual_r2': r2,
            'counterfactual_mse': cf_mse,
            'overall_mse': overall_mse
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_batch_analysis(
        data_dir="data/ihdp_sims",
        n_sample=25,
        has_headers=False,
        beta_l1=0.0,
        lambda_pehe=0.0,
        propensity_trials=5,
        propensity_epochs=30,
        device=None,
        cnrn_epochs=60,
        output_file="TAR_ihdp_batch_results.csv",
        overwrite_output=False,
        resume_output=True,
        file_list=None,
        random_seed=42,
):
    """Run batch analysis on multiple IHDP datasets"""
    print("IHDP Batch Analysis")
    print("="*50)
    
    # Load column headers if needed
    columns = None
    if not has_headers:
        columns = load_header_columns()
        print(f"Using columns from header file: {len(columns)} columns")
    
    # Find datasets
    patterns = [
        os.path.join(data_dir, "*.csv"),
        os.path.join(data_dir, "**/*.csv")
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    # De-duplicate when multiple glob patterns overlap.
    files = sorted(set(files))
    
    if not files:
        print(f"Error: No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(files)} dataset files")
    
    # Select datasets: explicit file list (exact cut) or random sample.
    if file_list:
        list_path = file_list
        if not os.path.isabs(list_path):
            list_path = os.path.join(os.getcwd(), list_path)
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"File list not found: {list_path}")

        if list_path.lower().endswith(".txt"):
            with open(list_path, "r", encoding="utf-8") as f:
                requested = [line.strip() for line in f if line.strip()]
        else:
            list_df = pd.read_csv(list_path)
            if "filename" in list_df.columns:
                requested = list_df["filename"].astype(str).str.strip().tolist()
            else:
                requested = list_df.iloc[:, 0].astype(str).str.strip().tolist()

        requested = [os.path.basename(x) for x in requested if x]
        file_map = {os.path.basename(fp): fp for fp in files}
        selected_files = []
        missing = []
        for name in requested:
            if name in file_map:
                selected_files.append(file_map[name])
            else:
                missing.append(name)
        if missing:
            print(f"Warning: {len(missing)} files from list not found under {data_dir}")
        if not selected_files:
            print("Error: No valid dataset files resolved from file_list")
            return
        n_sample = len(selected_files)
        print(f"Using explicit file list: {list_path}")
        print(f"Resolved {n_sample} datasets from file list")
    else:
        n_sample = min(n_sample, len(files))
        random.seed(random_seed)
        selected_files = random.sample(files, n_sample)
        print(f"Random seed: {random_seed}")
    
    print(f"Selected {n_sample} datasets for analysis:")
    for i, f in enumerate(selected_files, 1):
        print(f"  {i:2d}. {os.path.basename(f)}")

    output_path = output_file
    existing_df = None
    processed = set()
    if os.path.exists(output_path):
        if overwrite_output:
            os.remove(output_path)
            print(f"Overwriting existing output file: {output_path}")
        elif resume_output:
            try:
                existing_df = pd.read_csv(output_path)
                if "filename" in existing_df.columns:
                    processed = set(existing_df["filename"].astype(str).tolist())
                    print(
                        f"Resuming from existing output: {output_path} "
                        f"({len(processed)} completed rows found)"
                    )
            except Exception as e:
                print(f"Warning: could not read existing output for resume: {e}")
        else:
            base, ext = os.path.splitext(output_path)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{base}_{stamp}{ext or '.csv'}"
            print(f"Output exists, writing to new file instead: {output_path}")
    
    # Run analysis on each dataset
    results = []
    to_run = [fp for fp in selected_files if os.path.basename(fp) not in processed]
    if processed:
        print(f"Skipping {len(selected_files) - len(to_run)} already completed datasets.")
    for i, filepath in enumerate(to_run, 1):
        print(f"\n[{i:2d}/{len(to_run)}] Starting analysis...")
        result = run_single_analysis(
            filepath,
            has_headers,
            columns,
            verbose=False,
            beta_l1=beta_l1,
            lambda_pehe=lambda_pehe,
            propensity_trials=propensity_trials,
            propensity_epochs=propensity_epochs,
            device=device,
        )
        if result:
            results.append(result)
            print(f"[OK] Completed: {os.path.basename(filepath)}")
            print(f"  TMLE ATE: {result['tmle_ate']:.3f} (true: {result['true_ate_test']:.3f})")
            row_df = pd.DataFrame([result])
            write_header = not os.path.exists(output_path)
            row_df.to_csv(output_path, mode='a', header=write_header, index=False)
        else:
            print(f"[FAILED] {os.path.basename(filepath)}")
    
    # Compute summary statistics
    if os.path.exists(output_path):
        print(f"\n{'='*80}")
        print("BATCH ANALYSIS SUMMARY")
        print("="*80)
        df_results = pd.read_csv(output_path)
        if "filename" in df_results.columns:
            df_results = df_results.drop_duplicates(subset=["filename"], keep="last")
        print(f"Successfully processed: {len(df_results)}/{n_sample} datasets")
        
        # Key metrics to summarize
        metrics = {
            'True ATE': 'true_ate_test',
            'TMLE ATE': 'tmle_ate',
            'AIPTW ATE': 'aiptw_ate',
            'IPTW ATE': 'iptw_ate',
            'Naive ATE': 'naive_ate',
            'Simple ATE': 'simple_ate',
            'Simple ATE Error': 'simple_ate_error',
            'TMLE Error': 'tmle_ate_error',
            'AIPTW Error': 'aiptw_ate_error',
            'IPTW Error': 'iptw_ate_error',
            'PEHE': 'pehe',
            'sqrt(PEHE)': 'sqrt_pehe',
            'ITE Correlation': 'ite_correlation'
        }

        print(f"\n{'Metric':<20} {'N':<6} {'Mean':<10} {'Std':<10} {'SE':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
        print("-" * 100)

        for name, col in metrics.items():
            if col in df_results.columns:
                values = df_results[col].dropna()
                n = len(values)
                if n > 0:
                    mean = values.mean()
                    std = values.std()
                    se = std / np.sqrt(n) if n > 1 else np.nan
                    median = values.median()
                    min_v = values.min()
                    max_v = values.max()
                    print(f"{name:<20} {n:<6d} {mean:<10.3f} {std:<10.3f} {se:<10.3f} {median:<10.3f} {min_v:<10.3f} {max_v:<10.3f}")

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

        # Rewrite de-duplicated table in place to keep resume file clean.
        df_results.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
        
        print("\nBatch analysis completed!")
    else:
        print("No successful analyses to summarize.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='IHDP Causal Inference Analysis')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Run mode: single file or batch analysis')
    parser.add_argument('--file', type=str, default='data/ihdp_data.csv',
                       help='Path to single IHDP dataset file')
    parser.add_argument('--data_dir', type=str, default='data/ihdp',
                       help='Directory containing IHDP datasets for batch mode')
    parser.add_argument('--n_sample', type=int, default=25,
                       help='Number of datasets to sample for batch analysis. Standard IHDP protocol (Shalit et al. 2017, Shi et al. 2019) uses 1000 replications; use 1000 when available for comparability.')
    parser.add_argument('--has_headers', action='store_true',
                       help='Whether CSV files have header rows')
    parser.add_argument(
        '--beta_l1',
        type=float,
        default=1e-7,
        help='L1 energy/complexity penalty weight.'
    )
    parser.add_argument(
        '--lambda_pehe',
        type=float,
        default=0.0,
        help='PEHE regularization weight for CNRN.'
    )
    parser.add_argument(
        '--propensity_trials',
        type=int,
        default=5,
        help='Optuna trials for the propensity tuner.'
    )
    parser.add_argument(
        '--propensity_epochs',
        type=int,
        default=30,
        help='Training epochs per propensity tuning trial.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Torch device for CNRN training, e.g. cuda:0 or cpu.'
    )
    parser.add_argument(
        '--cnrn_epochs',
        type=int,
        default=60,
        help='Training epochs for the CNRN model.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='TAR_ihdp_batch_results.csv',
        help='Batch results CSV path.'
    )
    parser.add_argument(
        '--file_list',
        type=str,
        default=None,
        help='Optional dataset file list (.txt or .csv). If set, uses exact listed files.'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed used for dataset sampling when file_list is not provided.'
    )
    parser.add_argument(
        '--overwrite_output',
        action='store_true',
        help='Allow overwriting an existing output_file.'
    )
    parser.add_argument(
        '--no_resume_output',
        action='store_true',
        help='Do not resume from existing output_file; create a timestamped file instead.'
    )
    parser.add_argument(
        '--force_single_core',
        action='store_true',
        help='Force single-core execution to avoid Windows joblib/loky pipe permission errors.'
    )
    
    args = parser.parse_args()
    if args.force_single_core:
        os.cpu_count = (lambda: 1)
        print("Forcing single-core mode (os.cpu_count -> 1).")
    
    if args.mode == 'single':
        print("Running single file analysis...")
        columns = None if args.has_headers else load_header_columns()
        result = run_single_analysis(
            args.file,
            args.has_headers,
            columns,
            beta_l1=args.beta_l1,
            lambda_pehe=args.lambda_pehe,
            propensity_trials=args.propensity_trials,
            propensity_epochs=args.propensity_epochs,
            device=args.device,
            cnrn_epochs=args.cnrn_epochs,
        )
        if result:
            print("\nScript completed successfully!")
    else:
        print("Running batch analysis...")
        run_batch_analysis(
            args.data_dir,
            args.n_sample,
            args.has_headers,
            beta_l1=args.beta_l1,
            lambda_pehe=args.lambda_pehe,
            propensity_trials=args.propensity_trials,
            propensity_epochs=args.propensity_epochs,
            device=args.device,
            cnrn_epochs=args.cnrn_epochs,
            output_file=args.output_file,
            overwrite_output=args.overwrite_output,
            resume_output=not args.no_resume_output,
            file_list=args.file_list,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
