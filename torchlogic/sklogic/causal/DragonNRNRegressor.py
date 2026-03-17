import numpy as np
import pandas as pd
import torch
from torchlogic.sklogic.causal.dragon_module import DragonNet
from torchlogic.utils.trainers.dragonnrntrainer import DragonNRNTrainer
from torchlogic.sklogic.causal.dragon_loss import DragonLoss,AdaptiveDragonLoss
from torchlogic.sklogic.base.base_estimator import BaseSKLogicEstimator
from torchlogic.sklogic.datasets.causal_dataset import CausalDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torchlogic.utils.policy.mic_e import compute_mic_matrix
from torch import optim
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit


class DragonNRNRegressor(BaseSKLogicEstimator):
    def __init__(
        self,
        target_name=None,
        feature_names=None,
        treatment_col = 'treatment',
        layer_sizes=[8, 8],
        propensity_branch_layer=0,
        n_selected_features_input=4,
        n_selected_features_internal=4,
        n_selected_features_output=4,
        perform_prune_quantile=0.5,
        ucb_scale=1.5,
        add_negations=False,
        normal_form='dnf',
        weight_init=0.1,
        binarization=True,
        tree_num=10,
        tree_depth=5,
        tree_feature_selection=0.5,
        thresh_round=3,
        loss_fn=AdaptiveDragonLoss(),
        learning_rate=0.01,
        g_lr_scale=1.0,
        weight_decay=0.001,
        t_0=3,
        t_mult=4,
        epochs=200,
        batch_size=32,
        holdout_pct=0.2,
        early_stopping_plateau_count=20,
        evaluation_metric=mean_squared_error,
        pin_memory=False,
        persistent_workers=False,
        num_workers=0,
        g_pretrain_epochs=5
    ):
        super().__init__(
            binarization=binarization,
            tree_num=tree_num,
            tree_depth=tree_depth,
            tree_feature_selection=tree_feature_selection,
            thresh_round=thresh_round,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            t_0=t_0,
            t_mult=t_mult,
            epochs=epochs,
            batch_size=batch_size,
            holdout_pct=holdout_pct,
            early_stopping_plateau_count=early_stopping_plateau_count,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers
        )

        self.target_name = target_name
        self.feature_names = feature_names
        self.treatment_col = treatment_col
        self.layer_sizes = layer_sizes
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.propensity_branch_layer = propensity_branch_layer
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale
        self.add_negations = add_negations
        self.normal_form = normal_form
        self.weight_init = weight_init
        self.loss_fn = loss_fn
        self.evaluation_metric = evaluation_metric
        self.g_lr_scale = g_lr_scale
        self.g_pretrain_epochs = g_pretrain_epochs

        # Separate scalers for targets and features
        self.target_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()

    def _fit_transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.fit_transform(y), columns=[self.target_name], dtype=np.float32)

    def _transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.transform(y), columns=[self.target_name])

    def _inverse_transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.inverse_transform(y), columns=[self.target_name])

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        if not isinstance(y, pd.DataFrame):
            y = self._handle_non_dataframe_targets(y)
        y = y.copy()
        
        # Fit target scaler on original y values
        self.target_scaler.fit(y.values.reshape(-1, 1))
        
        # Scale y to [0,1]
        y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1)).flatten()
        y_arr = y_scaled.astype(np.float32)

        # Pull treatment before fitting transformation
        if self.treatment_col not in X.columns:
            raise ValueError(f"'{self.treatment_col}' must be present in the input data.")
        t_arr = X[self.treatment_col].values.astype(np.float32)
        X_cov = X.drop(columns=[self.treatment_col])

        X_cov = self._handle_empty_feature_names(X_cov)

        X_cov = self._fit_transform_encode_data(X_cov)
        if self.binarization:
            # IMPORTANT: Fit binarizer against treatment labels to expose propensity signal
            # rather than outcome-driven thresholds.
            y_bin_target = pd.DataFrame({'treatment': t_arr.astype(np.float32)})
            X_cov = self._fit_transform_binarize_features(X_cov, y_bin_target)

        # Fit the feature scaler on the processed features
        self.feature_scaler.fit(X_cov)

        feature_names = X_cov.columns  # pass these into DragonNet
        X_arr = X_cov.values
        
        # Build CausalDataset with (X_cov, t, y_scaled)
        dataset = CausalDataset(X_arr, t_arr, y_arr)
        train_dl, train_holdout_dl = self._generate_stratified_training_data_loaders(dataset, t_arr, 0.27)

        # Seed initial input feature selection via MIC (no minepy dependency)
        # IMPORTANT: For propensity, compute MIC against treatment labels, not outcome.
        try:
            t_df = pd.DataFrame({'treatment': t_arr.astype(np.float32)})
            mic_c_policy, _ = compute_mic_matrix(X_cov, t_df, alpha=.45, c=6)
            mic_policy_tensor = torch.tensor(mic_c_policy).float().view(-1)
        except Exception:
            mic_policy_tensor = None

        if X_cov.shape[1] < self.n_selected_features_input:
            print(f"Warning: The number of features ({X_cov.shape[1]}) is less than 'n_selected_features_input' ({self.n_selected_features_input}). Using number of features instead.")
            self.n_selected_features_input = X_cov.shape[1]

        self.model = DragonNet(
            input_size=X_cov.shape[1],
            layer_sizes=self.layer_sizes,
            feature_names=list(feature_names),
            n_selected_features_input=self.n_selected_features_input,
            n_selected_features_internal=self.n_selected_features_internal,
            n_selected_features_output=self.n_selected_features_output,
            perform_prune_quantile=self.perform_prune_quantile,
            ucb_scale=self.ucb_scale,
            normal_form=self.normal_form,
            add_negations=self.add_negations,
            weight_init=self.weight_init,
            propensity_branch_layer=self.propensity_branch_layer
        )

        # Initialize input mask using MIC policy if available
        if mic_policy_tensor is not None:
            try:
                self.model.symbolic_model.init_input_mask_from_policy(mic_policy_tensor, retain_ratio=0.7)
            except Exception:
                pass

        # Parameter groups: boost g head learning if desired
        try:
            g_params = list(self.model.symbolic_model.head_g.parameters())
            if hasattr(self.model.symbolic_model, 'prop_intermediate') and self.model.symbolic_model.prop_intermediate is not None:
                g_params += list(self.model.symbolic_model.prop_intermediate.parameters())
            g_param_ids = {id(p) for p in g_params}
            other_params = [p for p in self.model.parameters() if id(p) not in g_param_ids]
            param_groups = [
                {'params': other_params, 'lr': self.learning_rate, 'weight_decay': self.weight_decay},
                {'params': g_params, 'lr': self.learning_rate * float(self.g_lr_scale), 'weight_decay': self.weight_decay},
            ]
            optimizer = optim.AdamW(param_groups)
        except Exception:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.t_0, T_mult=self.t_mult)

        trainer = DragonNRNTrainer(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=self.loss_fn,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_plateau_count,
            target_scaler=self.target_scaler,
            lambda_cal=0.1,
            g_pretrain_epochs=self.g_pretrain_epochs,
            perform_prune_plateau_count=5,
            increase_prune_plateau_count=5,
            bandit_policy=mic_policy_tensor,
            mic_retain_ratio=0.7,
            mask_swap_frac=0.2
        )

        # Train model
        trainer.train(train_dl, train_holdout_dl)

    def predict(self, X: pd.DataFrame, treatment: int = None, return_propensity: bool = False):
        """
        Predict potential outcomes and optionally propensity scores.
        
        Args:
            X (pd.DataFrame): Input features
            treatment (int): If None, return both Y0 and Y1. If 0 or 1, return that outcome.
            return_propensity (bool): If True, return propensity scores as well.
        
        Returns:
            np.ndarray or tuple: 
                - If return_propensity=False: Predictions in original units
                - If return_propensity=True: (predictions, propensity_scores)
        """
        
        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()
        
        # Remove treatment column if present
        if self.treatment_col in X.columns:
            X = X.drop(columns=[self.treatment_col])
        
        # Transform features (encode + binarize)
        X = self._encode_prediction_data(X)
        
        # Forward pass - predictions in [0,1]
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
            y0_pred, y1_pred, g_pred = self.model(X_tensor)
        
        # Convert to numpy
        y0_pred = y0_pred.cpu().numpy()
        y1_pred = y1_pred.cpu().numpy()
        g_pred = g_pred.cpu().numpy().flatten() 
        
        # Inverse transform outcomes to original units
        y0_original = self.target_scaler.inverse_transform(y0_pred.reshape(-1, 1)).flatten()
        y1_original = self.target_scaler.inverse_transform(y1_pred.reshape(-1, 1)).flatten()

        # Return based on treatment argument
        if treatment == 0:
            outcomes = y0_original
        elif treatment == 1:
            outcomes = y1_original
        else:
            # Return both as columns: [Y0, Y1]
            outcomes = np.column_stack([y0_original, y1_original])
        
        if return_propensity:
            return outcomes, g_pred
        else:
            return outcomes


    def _get_instance_params(self):
        """
        Get parameters for this model type.

        Returns:
            dict: parameters and values
        """
        return {
            'layer_sizes': self.layer_sizes,
            'n_selected_features_input': self.n_selected_features_input,
            'n_selected_features_internal': self.n_selected_features_internal,
            'n_selected_features_output': self.n_selected_features_output,
            'perform_prune_quantile': self.perform_prune_quantile,
            'ucb_scale': self.ucb_scale,
            'add_negations': self.add_negations,
            'weight_init': self.weight_init,
            'loss_fn': self.loss_fn,
            'early_stopping_plateau_count': self.early_stopping_plateau_count,
            'evaluation_metric': self.evaluation_metric,
        }

    def explain_sample(
            self,
            X: pd.DataFrame,
            sample_index: int = 0,
            quantile: float = 1.0
    ) -> str:
        """
        Generate a sample explanation

        Args:
            X (pd.DataFrame): DataFrame of input features.
            sample_index (int): Index of sample to explain.
            quantile (float): Percent of model to explain

        Returns:
            str: explanation for selected sample
        """
        
        # create textual feature names if they are not given by user
        if self.feature_names is None:
            self.feature_names = [f"the {x} was" for x in X.columns]
            X.columns = self.feature_names
        else:
            X.columns = self.feature_names

        X = self._encode_prediction_data(X)

        self.min_max_features_dict = {
            col: {'min': self.numeric_features.iloc[:, i].min(), 'max': self.numeric_features.iloc[:, i].max()}
            for i, col in enumerate(self.numeric_features.columns)
        }

        device = next(self.model.parameters()).device
        X_tensor = torch.tensor(X.values[sample_index:sample_index+1], dtype=torch.float32).to(device)

        return self.model.explain_samples(
            X_tensor,
            quantile=quantile,
            target_names=[self.target_name],
            explain_type='both',
            sample_explanation_prefix="The sample has",
            print_type='logical',
            ignore_uninformative=True,
            rounding_precision=3,
            show_bounds=not self.binarization,
            simplify=True,
            exclusions=None,
            inverse_transform_target=self.target_scaler.inverse_transform
        )

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.feature_scaler, 'scale_'):
            raise RuntimeError("Feature scaler not fitted. Did you forget to call fit()?")

        return pd.DataFrame(
            self.feature_scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
