import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional,Tuple
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Assuming these are in your project structure
from torchlogic.sklogic.base.base_estimator import BaseSKLogicEstimator
from torchlogic.sklogic.datasets.causal_dataset import CausalDataset
from torchlogic.utils.trainers.twostagednrntrainer import TwoPhaseDragonNRNTrainer
from torchlogic.nn import LukasiewiczChannelOrBlock, LukasiewiczChannelAndBlock, Predicates
from torchlogic.sklogic.causal.dragon_loss import DragonLoss, AdaptiveDragonLoss
from torchlogic.sklogic.causal.causal_tree_test import CausalTreeBanditPolicy

logging.basicConfig(level=logging.INFO)

class UnifiedDragonModule(nn.Module):
    """
    A single, unified DragonNet module.
    This combines the logic of the previous IntegratedDragonNRNModule and the core network.
    """
    def __init__(
        self,
        input_size: int,
        layer_sizes: List[int],
        feature_names: List[str],
        n_selected_features_input: int,
        n_selected_features_internal: int,
        n_selected_features_output: int,
        add_negations: bool = True,
        normal_form: str = 'dnf',
        weight_init: float = 0.1
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Representation Layers (Trunk)
        self.shared_trunk = self._build_trunk(
            input_size, layer_sizes, feature_names, n_selected_features_input,
            n_selected_features_internal, add_negations, normal_form, weight_init
        )
        
        # Head Layers
        final_features = layer_sizes[-1]
        last_layer_of_trunk = self.shared_trunk[-1]
        
        self.head_f0 = self._build_head(final_features, n_selected_features_output, last_layer_of_trunk, 'head_f0', weight_init)
        self.head_f1 = self._build_head(final_features, n_selected_features_output, last_layer_of_trunk, 'head_f1', weight_init)
        self.head_g = self._build_head(final_features, n_selected_features_output, last_layer_of_trunk, 'head_g', weight_init)

    def _build_trunk(self, input_size, layer_sizes, feature_names, n_input, n_internal, add_negations, normal_form, weight_init):
        layers = []
        current_in = input_size
        
        # Input Layer
        layers.append(LukasiewiczChannelAndBlock(
            channels=1, in_features=current_in, out_features=layer_sizes[0],
            n_selected_features=n_input, parent_weights_dimension='out_features',
            operands=Predicates(feature_names=feature_names), outputs_key='layer_0',
            add_negations=add_negations, weight_init=weight_init
        ))
        current_in = layer_sizes[0]

        # Internal Layers
        for i in range(1, len(layer_sizes)):
            Block = LukasiewiczChannelOrBlock if i % 2 != 0 else LukasiewiczChannelAndBlock
            layers.append(Block(
                channels=1, in_features=current_in, out_features=layer_sizes[i],
                n_selected_features=n_internal, parent_weights_dimension='out_features',
                operands=layers[-1], outputs_key=f'layer_{i}', weight_init=weight_init
            ))
            current_in = layer_sizes[i]
            
        return nn.Sequential(*layers)

    def _build_head(self, in_features, n_selected, operand, key, weight_init):
        return LukasiewiczChannelAndBlock(
            channels=1, in_features=in_features, out_features=1,
            n_selected_features=n_selected, parent_weights_dimension='out_features',
            operands=operand, outputs_key=key, weight_init=weight_init
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trunk_output = self.shared_trunk(x)
        q_t0 = self.head_f0(trunk_output).squeeze(-1).squeeze(-1)
        q_t1 = self.head_f1(trunk_output).squeeze(-1).squeeze(-1)
        g = self.head_g(trunk_output).squeeze(-1).squeeze(-1)
        return q_t0, q_t1, g

class UnifiedDragonRegressor(BaseSKLogicEstimator):
    """
    A single, refactored DragonNet-style regressor for causal inference.
    This class combines the functionality of DragonNRNRegressor and TwoPhaseDragonNRNRegressor.
    """
    def __init__(
        self,
        # Core Architecture
        feature_names: List[str],
        target_name: str,
        treatment_col: str = 'treatment',
        layer_sizes: List[int] = [100, 200, 50],
        n_selected_features_input: int = 83,
        n_selected_features_internal: int = 60,
        n_selected_features_output: int = 40,
        add_negations: bool = True,
        normal_form: str = 'dnf',
        weight_init: float = 0.2,
        
        # Training & Loss
        loss_fn: nn.Module = AdaptiveDragonLoss(alpha=1, lambda_reg=1, lambda_tar=1),
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        batch_size: int = 64,
        early_stopping_plateau_count: int = 60,
        
        # Two-Phase Propensity Pre-training
        pretrain_g: bool = True,
        g_pretrain_epochs: int = 100,
        g_pretrain_lr: float = 1e-2,
        g_pretrain_patience: int = 20,
        g_lr_scale: float = 1.0,
        freeze_g_epochs: int = 5,
        g_vector: Optional[List[float]] = None,
        
        # Causal Tree Bandit (Optional)
        enable_tree_bandit: bool = False,
        tree_max_depth: int = 3,
        confounding_weight: float = 0.1,
        treatment_effect_weight: float = 0.9,
        bandit_update_frequency: int = 10,
        
        # Other inherited params
        **kwargs
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs,
            batch_size=batch_size, early_stopping_plateau_count=early_stopping_plateau_count,
            **kwargs
        )
        
        # Store all parameters
        self.feature_names = feature_names
        self.target_name = target_name
        self.treatment_col = treatment_col
        self.layer_sizes = layer_sizes
        self.n_selected_features_input = n_selected_features_input
        self.n_selected_features_internal = n_selected_features_internal
        self.n_selected_features_output = n_selected_features_output
        self.add_negations = add_negations
        self.normal_form = normal_form
        self.weight_init = weight_init
        self.loss_fn = loss_fn
        
        self.pretrain_g = pretrain_g
        self.g_pretrain_epochs = g_pretrain_epochs
        self.g_pretrain_lr = g_pretrain_lr
        self.g_pretrain_patience = g_pretrain_patience
        self.g_lr_scale = g_lr_scale
        self.freeze_g_epochs = freeze_g_epochs
        
        self.enable_tree_bandit = enable_tree_bandit
        self.tree_max_depth = tree_max_depth
        self.confounding_weight = confounding_weight
        self.treatment_effect_weight = treatment_effect_weight
        self.bandit_update_frequency = bandit_update_frequency
        self.g_vector = g_vector
        
        self.feature_scaler = StandardScaler()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[UnifiedDragonModule] = None
        self.causal_tree_bandit: Optional[CausalTreeBanditPolicy] = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        X = X.copy()
        y = y.copy()
        
        # 1. Separate Treatment and Covariates
        if self.treatment_col not in X.columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' not found in X.")
        t_arr = X[self.treatment_col].values.astype(np.float32)
        X_cov = X.drop(columns=[self.treatment_col])
        y_arr = y[self.target_name].values.astype(np.float32)

        # 2. Scale Covariates
        X_scaled = self.feature_scaler.fit_transform(X_cov)
        
        # 3. Initialize Bandit (Optional)
        if self.enable_tree_bandit:
            self._initialize_bandit(X_cov)

        # 4. Create Dataset and Dataloaders
        dataset = CausalDataset(X_scaled, t_arr, y_arr,g=self.g_vector)
        train_dl, val_dl = self._generate_stratified_training_data_loaders(dataset, t_arr, self.holdout_pct)

        # 5. Build Model
        self.model = UnifiedDragonModule(
            input_size=X_scaled.shape[1],
            layer_sizes=self.layer_sizes,
            feature_names=self.feature_names,
            n_selected_features_input=self.n_selected_features_input,
            n_selected_features_internal=self.n_selected_features_internal,
            n_selected_features_output=self.n_selected_features_output,
            add_negations=self.add_negations,
            normal_form=self.normal_form,
            weight_init=self.weight_init
        )

        # 6. Setup Optimizer and Trainer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.early_stopping_plateau_count // 2)

        trainer = TwoPhaseDragonNRNTrainer(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=self.loss_fn,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_plateau_count,
            causal_tree_bandit=self.causal_tree_bandit,
            bandit_update_frequency=self.bandit_update_frequency,
            pretrain_g=self.pretrain_g,
            g_pretrain_epochs=self.g_pretrain_epochs,
            g_pretrain_lr=self.g_pretrain_lr,
            g_lr_scale=self.g_lr_scale,
            freeze_g_epochs=self.freeze_g_epochs,
            g_pretrain_patience=self.g_pretrain_patience
        )

        # 7. Train the model
        self.logger.info("Starting model training...")
        raw_data_for_bandit = (X, y) if self.enable_tree_bandit else None
        trainer.train(train_dl, val_dl, raw_data=raw_data_for_bandit)
        self.logger.info("Model training completed.")

    def _initialize_bandit(self, X_cov: pd.DataFrame):
        self.logger.info("Initializing Causal Tree Bandit...")
        self.causal_tree_bandit = CausalTreeBanditPolicy(
            input_size=X_cov.shape[1],
            feature_names=list(self.feature_names),
            treatment_col=self.treatment_col,
            tree_max_depth=self.tree_max_depth,
            confounding_weight=self.confounding_weight,
            treatment_effect_weight=self.treatment_effect_weight
        )

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scales features using the fitted scaler."""
        if not hasattr(self.feature_scaler, 'scale_'):
            raise RuntimeError("Scaler has not been fitted. Call fit() before transforming.")
        return self.feature_scaler.transform(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the Individual Treatment Effect (ITE)."""
        self.model.eval()
        X_scaled = self.transform_features(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            q0, q1, _ = self.model(X_tensor)
        ite = (q1 - q0).cpu().numpy()
        return ite

    def get_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Gets all head predictions: q0, q1, and g."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        self.model.eval()
        
        X_cov = X.drop(columns=[self.treatment_col], errors='ignore')
        X_scaled = self.transform_features(X_cov)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            q0, q1, g_logits = self.model(X_tensor)
            
        return {
            "q_t0": q0.cpu().numpy(),
            "q_t1": q1.cpu().numpy(),
            "g": torch.sigmoid(g_logits).cpu().numpy() # Return propensity score
        }