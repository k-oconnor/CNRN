import numpy as np
import pandas as pd
import torch
from torchlogic.sklogic.causal.tar_trace_module import TARTraceNet
from torchlogic.utils.trainers.tarnrn_trainer import TARNRNTrainer, TARNetLoss
from torchlogic.sklogic.base.base_estimator import BaseSKLogicEstimator
from torchlogic.sklogic.datasets.causal_dataset import CausalDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from torchlogic.utils.policy.mic_e import compute_mic_matrix
from torch import optim

class ClippedMinMaxScaler:
    def __init__(
        self,
        lower_quantile=0.5,
        upper_quantile=0.95,
        lower_expand_ratio=0.0,
        upper_expand_ratio=0.0,
        eps=1e-6,
    ):
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.lower_expand_ratio = float(lower_expand_ratio)
        self.upper_expand_ratio = float(upper_expand_ratio)
        self.eps = float(eps)
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, y):
        arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        base_min = float(np.quantile(arr, self.lower_quantile))
        base_max = float(np.quantile(arr, self.upper_quantile))
        base_span = max(base_max - base_min, self.eps)
        self.data_min_ = base_min - self.lower_expand_ratio * base_span
        self.data_max_ = base_max + self.upper_expand_ratio * base_span
        if self.data_max_ <= self.data_min_:
            self.data_max_ = self.data_min_ + self.eps
        return self

    def transform(self, y):
        arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        arr = np.clip(arr, self.data_min_, self.data_max_)
        return ((arr - self.data_min_) / (self.data_max_ - self.data_min_)).astype(np.float32)

    def inverse_transform(self, y):
        arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * (self.data_max_ - self.data_min_) + self.data_min_).astype(np.float32)


class TARNRNTraceRegressor(BaseSKLogicEstimator):
    @staticmethod
    def _expand_feature_names_for_encoded_columns(raw_cols, encoded_cols, provided_feature_names):
        """Map user-provided raw feature names onto encoded/binarized columns."""
        mapping = dict(zip(raw_cols, provided_feature_names))
        ordered_raw = sorted(raw_cols, key=len, reverse=True)
        expanded = []

        for col in encoded_cols:
            new_col = col
            matched = None

            # Prefer structural matches first
            for raw in ordered_raw:
                if col == raw or col.startswith(raw + " ") or col.startswith(raw + "_"):
                    matched = raw
                    break

            # Fallback to containment
            if matched is None:
                for raw in ordered_raw:
                    if raw in col:
                        matched = raw
                        break

            if matched is not None:
                new_col = col.replace(matched, mapping[matched], 1)

            expanded.append(new_col)

        return expanded

    def __init__(
        self,
        target_name=None,
        feature_names=None,
        treatment_col = 'treatment',
        propensity_col=None,
        layer_sizes=[8, 8],
        head_layer_sizes=None,
        mlp_head_hidden_dim=0,
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
        loss_fn=TARNetLoss(lambda_pehe=1.0, lambda_tar=0.1, epsilon_init=0.01),
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
        g_pretrain_epochs=5,
        beta_l1=0.0,
        target_lower_quantile=0.5,
        target_upper_quantile=0.95,
        target_lower_expand_ratio=0.0,
        target_upper_expand_ratio=0.0,
        loss_on_original_scale=False,
        disable_scheduler=False,
        device=None
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
        self.propensity_col = propensity_col
        self.layer_sizes = layer_sizes
        self.head_layer_sizes = head_layer_sizes or []
        self.mlp_head_hidden_dim = int(mlp_head_hidden_dim or 0)
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
        self.beta_l1 = float(beta_l1 or 0.0)
        self.target_lower_quantile = float(target_lower_quantile)
        self.target_upper_quantile = float(target_upper_quantile)
        self.target_lower_expand_ratio = float(target_lower_expand_ratio)
        self.target_upper_expand_ratio = float(target_upper_expand_ratio)
        self.loss_on_original_scale = bool(loss_on_original_scale)
        self.disable_scheduler = bool(disable_scheduler)
        self.device = device

        # The trace architecture uses bounded logical outcome heads, so target
        # scaling must also be bounded.
        self.target_scaler = ClippedMinMaxScaler(
            lower_quantile=self.target_lower_quantile,
            upper_quantile=self.target_upper_quantile,
            lower_expand_ratio=self.target_lower_expand_ratio,
            upper_expand_ratio=self.target_upper_expand_ratio,
        )
        self.feature_scaler = RobustScaler()

    def _fit_transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.fit_transform(y), columns=[self.target_name], dtype=np.float32)

    def _transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.transform(y), columns=[self.target_name])

    def _inverse_transform_encode_target(self, y: pd.DataFrame):
        return pd.DataFrame(self.target_scaler.inverse_transform(y), columns=[self.target_name])

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, propensity_scores=None) -> None:
        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        if not isinstance(y, pd.DataFrame):
            y = self._handle_non_dataframe_targets(y)
        y = y.copy()
        
        # Fit target scaler on original y values
        self.target_scaler.fit(y.values.reshape(-1, 1))
        
        # Scale y to [0,1]
        y_original = y.values.reshape(-1, 1).astype(np.float32)
        y_scaled = self.target_scaler.transform(y_original).flatten()
        y_arr = y_scaled.astype(np.float32)

        # Pull treatment before fitting transformation
        if self.treatment_col not in X.columns:
            raise ValueError(f"'{self.treatment_col}' must be present in the input data.")
        t_arr = X[self.treatment_col].values.astype(np.float32)
        drop_cols = [self.treatment_col]

        # Optionally pull propensity labels from a dedicated input column.
        # These are used only in the loss, never as representation features.
        g_from_col = None
        if self.propensity_col is not None:
            if self.propensity_col not in X.columns:
                raise ValueError(f"'{self.propensity_col}' must be present in the input data when propensity_col is set.")
            g_from_col = X[self.propensity_col].values.astype(np.float32)
            drop_cols.append(self.propensity_col)

        X_cov = X.drop(columns=drop_cols)

        X_cov = self._fit_transform_encode_data(X_cov)
        if self.binarization:
            # IMPORTANT: Fit binarizer against treatment labels to expose propensity signal
            # rather than outcome-driven thresholds.
            y_bin_target = pd.DataFrame({'treatment': t_arr.astype(np.float32)})
            X_cov = self._fit_transform_binarize_features(X_cov, y_bin_target)

        # Generate feature names for the symbolic model without mutating
        # the raw training columns expected by preprocessing pipelines.
        if self.feature_names is None:
            model_feature_names = [f"the {x} was" for x in X_cov.columns]
            self.feature_names = model_feature_names
        else:
            if len(self.feature_names) == X_cov.shape[1]:
                # User provided encoded-level names
                model_feature_names = self.feature_names
            elif len(self.feature_names) == len(X_cov.columns):
                # Defensive fallback; same as encoded size path
                model_feature_names = self.feature_names
            else:
                # User provided raw-level names (before encoding/binarization)
                raw_cols = list(X.drop(columns=drop_cols).columns)
                if len(self.feature_names) == len(raw_cols):
                    model_feature_names = self._expand_feature_names_for_encoded_columns(
                        raw_cols=raw_cols,
                        encoded_cols=list(X_cov.columns),
                        provided_feature_names=list(self.feature_names)
                    )
                else:
                    raise ValueError(
                        f"'feature_names' length ({len(self.feature_names)}) must match either raw covariate count "
                        f"({len(raw_cols)}) or encoded feature count ({X_cov.shape[1]})."
                    )

            # Store encoded-level names for downstream explanation/model predicates.
            self.feature_names = list(model_feature_names)

        # Fit the feature scaler on the processed features
        self.feature_scaler.fit(X_cov)

        feature_names = model_feature_names
        X_arr = X_cov.values
        
        # Build CausalDataset with (X_cov, t, y_scaled, g).
        # NOTE: The representation network never sees propensity scores;
        # g is used only inside the loss function by the trainer.
        if propensity_scores is None:
            if g_from_col is None:
                g_arr = None
            else:
                g_arr = np.asarray(g_from_col, dtype=np.float32).reshape(-1)
        else:
            if g_from_col is not None:
                raise ValueError("Pass propensity via either 'propensity_scores' or 'propensity_col', not both.")
            g_arr = np.asarray(propensity_scores, dtype=np.float32).reshape(-1)
            if g_arr.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    f"'propensity_scores' length ({g_arr.shape[0]}) must match number of rows ({X_arr.shape[0]})."
                )
        if g_arr is not None:
            g_arr = np.clip(g_arr, 1e-6, 1.0 - 1e-6)
            if g_arr.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    f"Propensity length ({g_arr.shape[0]}) must match number of rows ({X_arr.shape[0]})."
                )
        dataset = CausalDataset(X_arr, t_arr, y_arr, g=g_arr, y_original=y_original)
        train_dl, train_holdout_dl = self._generate_stratified_training_data_loaders(
            dataset, t_arr, self.holdout_pct
        )

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

        self.model = TARTraceNet(
            input_size=X_cov.shape[1],
            layer_sizes=self.layer_sizes,
            head_layer_sizes=self.head_layer_sizes,
            feature_names=list(feature_names),
            n_selected_features_input=self.n_selected_features_input,
            n_selected_features_internal=self.n_selected_features_internal,
            n_selected_features_output=self.n_selected_features_output,
            perform_prune_quantile=self.perform_prune_quantile,
            ucb_scale=self.ucb_scale,
            mlp_head_hidden_dim=self.mlp_head_hidden_dim,
            normal_form=self.normal_form,
            add_negations=self.add_negations,
            weight_init=self.weight_init
        )

        # Initialize input mask using MIC policy if available
        if mic_policy_tensor is not None:
            try:
                self.model.symbolic_model.init_input_mask_from_policy(mic_policy_tensor, retain_ratio=0.7)
            except Exception:
                pass

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = None if self.disable_scheduler else optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.t_0, T_mult=self.t_mult
        )
        if hasattr(self.loss_fn, "set_output_scaler"):
            self.loss_fn.set_output_scaler(self.target_scaler.data_min_, self.target_scaler.data_max_)

        trainer = TARNRNTrainer(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=self.loss_fn,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_plateau_count,
            beta_l1=self.beta_l1,
            device=self.device
        )

        # Train model
        history = trainer.train(train_dl, train_holdout_dl)
        self.training_history_ = history

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
        if self.propensity_col is not None and self.propensity_col in X.columns:
            X = X.drop(columns=[self.propensity_col])
        
        # Transform features (encode + binarize)
        X = self._encode_prediction_data(X)
        
        # Forward pass - predictions in [0,1]
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            X_values = np.asarray(X.values, dtype=np.float32)
            X_tensor = torch.tensor(X_values, dtype=torch.float32).to(device)
            y0_pred, y1_pred = self.model(X_tensor)
        
        # Convert to numpy
        y0_pred = y0_pred.cpu().numpy()
        y1_pred = y1_pred.cpu().numpy()
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
            raise NotImplementedError("TARNRNRegressor/TARNet does not provide propensity predictions.")
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
            'head_layer_sizes': self.head_layer_sizes,
            'mlp_head_hidden_dim': self.mlp_head_hidden_dim,
            'propensity_col': self.propensity_col,
            'n_selected_features_input': self.n_selected_features_input,
            'n_selected_features_internal': self.n_selected_features_internal,
            'n_selected_features_output': self.n_selected_features_output,
            'perform_prune_quantile': self.perform_prune_quantile,
            'ucb_scale': self.ucb_scale,
            'add_negations': self.add_negations,
            'weight_init': self.weight_init,
            'loss_fn': self.loss_fn,
            'beta_l1': self.beta_l1,
            'loss_on_original_scale': self.loss_on_original_scale,
            'device': self.device,
            'disable_scheduler': self.disable_scheduler,
            'early_stopping_plateau_count': self.early_stopping_plateau_count,
            'evaluation_metric': self.evaluation_metric,
        }

    def explain_sample(
            self,
            X: pd.DataFrame,
            sample_index: int = 0,
            quantile: float = 1.0,
            output_channel: int = 0
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
        elif len(self.feature_names) == X.shape[1]:
            X.columns = self.feature_names

        X = self._encode_prediction_data(X)

        self.min_max_features_dict = {
            col: {'min': self.numeric_features.iloc[:, i].min(), 'max': self.numeric_features.iloc[:, i].max()}
            for i, col in enumerate(self.numeric_features.columns)
        }

        device = next(self.model.parameters()).device
        X_tensor = torch.tensor(X.values[sample_index:sample_index+1], dtype=torch.float32).to(device)

        target_names = [
            f"{self.target_name}_control" if self.target_name else "control_outcome",
            f"{self.target_name}_treated" if self.target_name else "treated_outcome",
        ]

        return self.model.explain_samples(
            X_tensor,
            quantile=quantile,
            target_names=target_names,
            explain_type='both',
            sample_explanation_prefix="The sample has",
            print_type='logical',
            ignore_uninformative=True,
            rounding_precision=3,
            show_bounds=not self.binarization,
            simplify=True,
            exclusions=None,
            inverse_transform_target=self.target_scaler.inverse_transform,
            output_channel=output_channel
        )

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.feature_scaler, 'scale_'):
            raise RuntimeError("Feature scaler not fitted. Did you forget to call fit()?")

        return pd.DataFrame(
            self.feature_scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

