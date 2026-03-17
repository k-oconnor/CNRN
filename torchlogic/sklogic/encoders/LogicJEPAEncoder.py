import numpy as np
import pandas as pd

import torch
from torch import optim

from torchlogic.modules.logic_jepa import LogicJEPA
from torchlogic.utils.trainers.logicjepatrainer import LogicJEPATrainer, LogicJEPALoss

from ..base.base_estimator import BaseSKLogicEstimator
from ..datasets.simple_dataset import SimpleDataset


class LogicJEPAEncoder(BaseSKLogicEstimator):
    """
    Standalone NRN-based JEPA encoder.

    This estimator is intentionally separate from outcome/treatment estimators.
    """

    def __init__(
            self,
            feature_names: list = None,
            layer_sizes: list = [8, 8],
            embedding_dim: int = 16,
            predictor_hidden_dim: int = 64,
            predictor_depth: int = 2,
            n_selected_features_input: int = 4,
            n_selected_features_internal: int = 4,
            n_selected_features_output: int = 4,
            perform_prune_quantile: float = 0.5,
            ucb_scale: float = 1.5,
            add_negations: bool = False,
            normal_form: str = 'cnf',
            weight_init: float = 0.2,
            binarization: bool = False,
            tree_num: int = 10,
            tree_depth: int = 5,
            tree_feature_selection: float = 0.5,
            thresh_round: int = 3,
            learning_rate: float = 0.01,
            weight_decay: float = 0.001,
            t_0: int = 3,
            t_mult: int = 2,
            epochs: int = 200,
            batch_size: int = 32,
            holdout_pct: float = 0.2,
            early_stopping_plateau_count: int = 20,
            ema_momentum: float = 0.99,
            ema_momentum_end: float = None,
            view_mask_prob: float = 0.15,
            view_mask_prob_end: float = None,
            view_noise_std: float = 0.01,
            beta_l1: float = 0.0,
            random_state: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            num_workers: int = 0,
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
        self.feature_names = feature_names
        self.layer_sizes = layer_sizes
        self.embedding_dim = int(embedding_dim)
        self.predictor_hidden_dim = int(predictor_hidden_dim)
        self.predictor_depth = int(predictor_depth)
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale
        self.add_negations = add_negations
        self.normal_form = normal_form
        self.weight_init = weight_init
        self.ema_momentum = float(ema_momentum)
        self.ema_momentum_end = float(ema_momentum if ema_momentum_end is None else ema_momentum_end)
        self.view_mask_prob = float(view_mask_prob)
        self.view_mask_prob_end = float(view_mask_prob if view_mask_prob_end is None else view_mask_prob_end)
        self.view_noise_std = float(view_noise_std)
        self.beta_l1 = float(beta_l1)
        self.random_state = int(random_state)
        self.device = device
        self.training_history_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()
        X = self._handle_empty_feature_names(X)
        X = self._fit_transform_encode_data(X)

        if self.binarization:
            if y is None:
                raise ValueError("When binarization=True, provide y so the feature binarizer can be fit.")
            if not isinstance(y, pd.DataFrame):
                y = self._handle_non_dataframe_targets(y)
            X = self._fit_transform_binarize_features(X, y)

        feature_names = list(X.columns)
        dummy_target = np.zeros((X.shape[0], 1), dtype=np.float32)
        dataset = SimpleDataset(X.values, dummy_target)
        train_dl, holdout_dl = self._generate_training_data_loaders(dataset)

        if X.shape[1] < self.n_selected_features_input:
            self.n_selected_features_input = X.shape[1]

        self.model = LogicJEPA(
            input_size=X.shape[1],
            feature_names=feature_names,
            layer_sizes=self.layer_sizes,
            embedding_dim=self.embedding_dim,
            predictor_hidden_dim=self.predictor_hidden_dim,
            predictor_depth=self.predictor_depth,
            n_selected_features_input=self.n_selected_features_input,
            n_selected_features_internal=self.n_selected_features_internal,
            n_selected_features_output=self.n_selected_features_output,
            perform_prune_quantile=self.perform_prune_quantile,
            ucb_scale=self.ucb_scale,
            normal_form=self.normal_form,
            add_negations=self.add_negations,
            weight_init=self.weight_init
        )

        trainable_params = list(self.model.online_encoder.parameters()) + list(self.model.predictor.parameters())
        optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.t_0, T_mult=self.t_mult)

        trainer = LogicJEPATrainer(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=LogicJEPALoss(),
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_plateau_count,
            ema_momentum=self.ema_momentum,
            ema_momentum_end=self.ema_momentum_end,
            view_mask_prob=self.view_mask_prob,
            view_mask_prob_end=self.view_mask_prob_end,
            view_noise_std=self.view_noise_std,
            beta_l1=self.beta_l1,
            device=self.device
        )
        self.training_history_ = trainer.train(train_dl, holdout_dl)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Estimator is not fitted. Call fit first.")

        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()
        X = self._handle_empty_feature_names(X)
        X = self._encode_prediction_data(X)

        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(X.values, dtype=np.float32), dtype=torch.float32).to(device)
            z = self.model.encode(x_tensor, use_target=False).cpu().numpy()

        cols = self.get_feature_names_out()
        return pd.DataFrame(z, columns=cols, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        return self.fit(X, y=y).transform(X)

    def get_feature_names_out(self):
        if self.embedding_dim is None:
            raise RuntimeError("Encoder has no embedding_dim configured.")
        return [f"jepa_embedding_{i}" for i in range(self.embedding_dim)]

    def _get_instance_params(self):
        return {
            'layer_sizes': self.layer_sizes,
            'embedding_dim': self.embedding_dim,
            'predictor_hidden_dim': self.predictor_hidden_dim,
            'predictor_depth': self.predictor_depth,
            'n_selected_features_input': self.n_selected_features_input,
            'n_selected_features_internal': self.n_selected_features_internal,
            'n_selected_features_output': self.n_selected_features_output,
            'perform_prune_quantile': self.perform_prune_quantile,
            'ucb_scale': self.ucb_scale,
            'add_negations': self.add_negations,
            'normal_form': self.normal_form,
            'weight_init': self.weight_init,
            'ema_momentum': self.ema_momentum,
            'ema_momentum_end': self.ema_momentum_end,
            'view_mask_prob': self.view_mask_prob,
            'view_mask_prob_end': self.view_mask_prob_end,
            'view_noise_std': self.view_noise_std,
            'beta_l1': self.beta_l1,
            'random_state': self.random_state,
            'device': self.device,
        }
