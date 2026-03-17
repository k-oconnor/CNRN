# integrated_dragon_nrn.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from torch.utils.data import DataLoader

# Add import for torch.nn.functional at the top
import torch.nn.functional as F
from torchlogic.sklogic.causal.DragonNRNRegressor import DragonNRNRegressor
from torchlogic.sklogic.causal.causal_tree_test import CausalTreeBanditPolicy 
from torchlogic.utils.trainers.twostagednrntrainer import TwoPhaseDragonNRNTrainer
        
from torchlogic.nn import LukasiewiczChannelOrBlock, LukasiewiczChannelAndBlock, Predicates


class IntegratedDragonNRNModule(nn.Module):
    """
    DragonNRN module that uses tree-bandit discovered features for initialization.
    """
    def __init__(
        self,
        input_size,
        output_size,
        layer_sizes,
        feature_names,
        n_selected_features_input,
        n_selected_features_internal,
        n_selected_features_output,
        perform_prune_quantile,
        ucb_scale,
        normal_form='dnf',
        add_negations=False,
        weight_init=0.2,
        causal_tree_bandit=None
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.normal_form = normal_form
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.causal_tree_bandit = causal_tree_bandit

        # Use bandit-guided feature selection for initialization
        if causal_tree_bandit is not None:
            self._initialize_with_bandit_features()
        
        # Build model layers as before but with potentially bandit-selected features
        self._build_model_layers(
            input_size, output_size, layer_sizes, feature_names,
            n_selected_features_input, n_selected_features_internal,
            n_selected_features_output, normal_form, add_negations, weight_init
        )
        
        # Store unused parameters
        self.perform_prune_quantile = perform_prune_quantile
        self.ucb_scale = ucb_scale

    def _initialize_with_bandit_features(self):
        """Initialize feature selection based on bandit policy."""
        if self.causal_tree_bandit is None:
            return
            
        # Get current feature policies for each head
        self.head_feature_priorities = {}
        for head_idx in range(3):  # y0, y1, g
            self.head_feature_priorities[head_idx] = (
                self.causal_tree_bandit.feature_policy[head_idx].numpy()
            )
            
        self.logger.info("Initialized with bandit-guided feature priorities")

    def _build_model_layers(self, input_size, output_size, layer_sizes, feature_names,
                           n_selected_features_input, n_selected_features_internal,
                           n_selected_features_output, normal_form, add_negations, weight_init):
        """Build model layers (same as original but with logging for features used)."""
        
        assert normal_form in ['dnf', 'cnf'], "'normal_form' must be 'dnf' or 'cnf'."
        assert len(layer_sizes) > 0, "layer_sizes must not be empty"

        # Input Layer
        if normal_form == 'dnf':
            input_layer = LukasiewiczChannelAndBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='layer_0',
                add_negations=add_negations,
                weight_init=weight_init
            )
        else:  # CNF
            input_layer = LukasiewiczChannelOrBlock(
                channels=output_size,
                in_features=input_size,
                out_features=layer_sizes[0],
                n_selected_features=n_selected_features_input,
                parent_weights_dimension='out_features',
                operands=Predicates(feature_names=feature_names),
                outputs_key='layer_0',
                add_negations=add_negations,
                weight_init=weight_init
            )

        model_layers = [input_layer]

        # Internal Layers
        for i in range(1, len(layer_sizes)):
            prev_layer = model_layers[-1]
            outputs_key = f'layer_{i}'

            if normal_form == 'dnf':
                use_and_block = (i % 2 == 0)
            else:  # CNF
                use_and_block = (i % 2 == 1)

            if use_and_block:
                block_cls = LukasiewiczChannelAndBlock
            else:
                block_cls = LukasiewiczChannelOrBlock

            internal_layer = block_cls(
                channels=output_size,
                in_features=layer_sizes[i - 1],
                out_features=layer_sizes[i],
                n_selected_features=n_selected_features_internal,
                parent_weights_dimension='out_features',
                operands=prev_layer,
                outputs_key=outputs_key,
                weight_init=weight_init
            )
            model_layers.append(internal_layer)

        # Shared Trunk
        self.shared_trunk = nn.Sequential(*model_layers)
        final_layer = model_layers[-1]
        final_features = layer_sizes[-1]

        # Heads - potentially use bandit-guided feature selection here
        self.head_f0 = LukasiewiczChannelAndBlock(
            channels=output_size,
            in_features=final_features,
            out_features=1,
            n_selected_features=n_selected_features_output,
            parent_weights_dimension='out_features',
            operands=final_layer,
            outputs_key='head_f0',
            weight_init=weight_init
        )

        self.head_f1 = LukasiewiczChannelAndBlock(
            channels=output_size,
            in_features=final_features,
            out_features=1,
            n_selected_features=n_selected_features_output,
            parent_weights_dimension='out_features',
            operands=final_layer,
            outputs_key='head_f1',
            weight_init=weight_init
        )

        self.head_g = LukasiewiczChannelAndBlock(
            channels=output_size,
            in_features=final_features,
            out_features=1,
            n_selected_features=n_selected_features_output,
            parent_weights_dimension='out_features',
            operands=final_layer,
            outputs_key='head_g',
            weight_init=weight_init
        )

    def forward(self, x: torch.Tensor):
        trunk_output = self.shared_trunk(x)

        # Each head output shape: [B, C, F]
        q_t0 = self.head_f0(trunk_output).mean(dim=(1, 2))
        q_t1 = self.head_f1(trunk_output).mean(dim=(1, 2))
        g = self.head_g(trunk_output).mean(dim=(1,2))
        
        return q_t0, q_t1, g

    def get_layer_info(self):
        """Debug method to inspect layer structure"""
        info = []
        info.append(f"Normal form: {self.normal_form}")
        info.append(f"Layer sizes: {self.layer_sizes}")
        
        for i, layer in enumerate(self.shared_trunk):
            layer_type = "AND" if isinstance(layer, LukasiewiczChannelAndBlock) else "OR"
            info.append(f"Layer {i}: {layer_type} - {layer.in_features} -> {layer.out_features}")
        
        if self.causal_tree_bandit:
            info.append("\nBandit Feature Priorities:")
            for head_idx, priorities in self.head_feature_priorities.items():
                top_features = np.argsort(priorities)[-5:][::-1]  # Top 5
                head_name = ['y0', 'y1', 'g'][head_idx]
                info.append(f"  {head_name}: features {top_features} (priorities: {priorities[top_features]})")
        
        return "\n".join(info)



class FullyIntegratedDragonNRNRegressor(DragonNRNRegressor):
    """
    Fully integrated DragonNRN regressor that connects all bandit features.
    """
    
    def __init__(
        self,
        enable_tree_bandit: bool = True,
        tree_max_depth: int = 3,
        confounding_weight: float = 0.3,
        treatment_effect_weight: float = 0.7,
        bandit_update_frequency: int = 10,
        pretrain_g=True,
        g_pretrain_epochs=50,
        g_pretrain_lr=1e-3,
        g_lr_scale=0.1,  # Scale down g head LR in phase 2
        freeze_g_epochs=10,  # Epochs to freeze g in phase 2
        g_pretrain_patience=10,
        **kwargs
    ):
        # Initialize bandit before calling super().__init__
        self.enable_tree_bandit = enable_tree_bandit
        self.tree_max_depth = tree_max_depth
        self.confounding_weight = confounding_weight
        self.treatment_effect_weight = treatment_effect_weight
        self.bandit_update_frequency = bandit_update_frequency
        self.causal_tree_bandit = None
        self.pretrain_g = pretrain_g
        self.g_pretrain_epochs = g_pretrain_epochs
        self.g_pretrain_lr = g_pretrain_lr
        self.g_lr_scale = g_lr_scale
        self.freeze_g_epochs = freeze_g_epochs
        self.g_pretrain_patience = g_pretrain_patience
        
        super().__init__(**kwargs)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _initialize_policy_from_mic(self, mic_values: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Initialize feature selection policy using MIC values.
        
        Args:
            mic_values: Tensor of shape [input_size] containing MIC values for each feature
                       Higher MIC values indicate stronger associations
        
        Returns:
            policy: Tensor of shape [3, input_size] with probabilities for each head
        """
        if mic_values is None:
            # Fallback to uniform initialization
            self.logger.warning("No MIC values provided, using uniform initialization")
            fallback_size = len(self.feature_names) if getattr(self, 'feature_names', None) is not None else None
            if fallback_size is None:
                # As a last resort, use 1 to avoid shape errors
                fallback_size = 1
            return torch.ones(3, fallback_size) / float(fallback_size)
        
        # Ensure proper shape and normalize MIC values
        input_size = int(mic_values.shape[0])
        
        # Normalize MIC values to create base probabilities
        mic_normalized = mic_values / (mic_values.sum() + 1e-8)
        
        # Apply softmax to create valid probability distributions
        mic_probs = F.softmax(mic_values, dim=0)
        
        # Initialize different strategies for each head based on MIC interpretation
        policy = torch.zeros(3, input_size)
        
        # Head 0 (y0 - control outcomes): Use MIC values directly
        # Features with high MIC are likely important for predicting outcomes
        policy[0] = mic_probs
        
        # Head 1 (y1 - treated outcomes): Use MIC values directly (same reasoning)
        policy[1] = mic_probs
        
        # Head 2 (g - propensity): Use inverted MIC emphasis
        # Features with moderate MIC might be better for propensity (less confounded)
        # Apply some transformation to differentiate from outcome heads
        mic_inverted = 1.0 / (mic_values + 1e-6)  # Invert while avoiding division by zero
        policy[2] = F.softmax(mic_inverted, dim=0)
        
        self.logger.info(f"Initialized bandit policy from MIC values. "
                        f"MIC range: [{mic_values.min():.4f}, {mic_values.max():.4f}]")
        
        return policy
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Enhanced fit with full bandit integration."""
        
        # Initialize bandit BEFORE any data processing
        if self.enable_tree_bandit:
            # Compute MIC values early, before feature processing
            X_cov = X.drop(columns=[self.treatment_col])
            mic_values = self._compute_mic_values(X_cov, y)
            self._initialize_bandit_with_mic(X_cov, mic_values)
        
        # Follow parent class fit process with modifications
        if not isinstance(X, pd.DataFrame):
            X = self._handle_non_dataframe_features(X)
        X = X.copy()

        if not isinstance(y, pd.DataFrame):
            y = self._handle_non_dataframe_targets(y)
        y = y.copy()
        y_arr = y.values.astype(np.float32).reshape(-1)

        # Pull treatment before fitting transformation
        if self.treatment_col not in X.columns:
            raise ValueError(f"'{self.treatment_col}' must be present in the input data.")
        t_arr = X[self.treatment_col].values.astype(np.float32)
        X_cov = X.drop(columns=[self.treatment_col])

        X_cov = self._handle_empty_feature_names(X_cov)
        X_cov = self._fit_transform_encode_data(X_cov)
        
        if self.binarization:
            X_cov = self._fit_transform_binarize_features(X_cov, y_arr > y.mean())

        # Fit the feature scaler on the processed features
        self.feature_scaler.fit(X_cov)

        feature_names = X_cov.columns
        X_arr = X_cov.values

        # Build CausalDataset
        from torchlogic.sklogic.datasets.causal_dataset import CausalDataset
        from torchlogic.utils.policy.mic_e import compute_mic_matrix
        from torch import optim
        dataset = CausalDataset(X_arr, t_arr, y_arr,g=self.g_vector)
        train_dl, train_holdout_dl = self._generate_stratified_training_data_loaders(dataset, t_arr, 0.27)

        # Use parent class MIC computation for model initialization
        mic_c_policy, _ = compute_mic_matrix(X_cov, y, alpha=.45, c=6)
        mic_c_policy = torch.tensor(mic_c_policy)

        if X_cov.shape[1] < self.n_selected_features_input:
            print(f"Warning: The number of features ({X_cov.shape[1]}) is less than 'n_selected_features_input' ({self.n_selected_features_input}). Using number of features instead.")
            self.n_selected_features_input = X_cov.shape[1]

        # Create integrated model instead of base DragonNet
        self.model = IntegratedDragonNRNModule(
            input_size=X_cov.shape[1],
            output_size=1,
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
            causal_tree_bandit=self.causal_tree_bandit  # Pass bandit to model
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.t_0, T_mult=self.t_mult)

        # Create integrated trainer
        self.trainer = TwoPhaseDragonNRNTrainer(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=self.loss_fn,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_plateau_count,
            causal_tree_bandit=self.causal_tree_bandit,
            bandit_update_frequency=self.bandit_update_frequency
        )

        # Train with bandit integration - pass raw data for bandit updates
        raw_data = (X, y)  # Original data before processing
        self.trainer.train(train_dl, train_holdout_dl, raw_data=raw_data)
        
        self.logger.info("Fully integrated DragonNRN training completed")
    
    def _initialize_bandit_with_mic(self, X_cov: pd.DataFrame, mic_values: torch.Tensor):
        """Initialize bandit with MIC values."""
        
        self.causal_tree_bandit = CausalTreeBanditPolicy(
            input_size=X_cov.shape[1],
            feature_names=list(X_cov.columns),
            treatment_col=self.treatment_col,
            #mic_values=mic_values,
            tree_max_depth=self.tree_max_depth,
            confounding_weight=self.confounding_weight,
            treatment_effect_weight=self.treatment_effect_weight
        )
        
        self.logger.info(f"Bandit initialized with MIC values for {X_cov.shape[1]} features")
    
    def _initialize_and_run_initial_bandit_analysis(self, X: pd.DataFrame, y: pd.DataFrame):
        """Run initial bandit analysis before model creation."""
        
        X_cov = X.drop(columns=[self.treatment_col])
        
        # Compute MIC values for bandit initialization
        mic_values = self._compute_mic_values(X_cov, y)
        
        self.causal_tree_bandit = CausalTreeBanditPolicy(
            input_size=X_cov.shape[1],
            feature_names=list(X_cov.columns),
            treatment_col=self.treatment_col,
            mic_values=mic_values,  # Pass MIC values for initialization
            tree_max_depth=self.tree_max_depth,
            confounding_weight=self.confounding_weight,
            treatment_effect_weight=self.treatment_effect_weight
        )
        
        # Run initial pattern discovery
        t_arr = X[self.treatment_col].values.astype(np.float32)
        y_arr = y.values.astype(np.float32).reshape(-1)
        
        # Use parent class encoding methods
        X_processed = self._prepare_features_for_bandit(X_cov, y_arr)
        
        patterns = self.causal_tree_bandit.discover_causal_patterns(
            X_processed.values, t_arr, y_arr
        )
        
        # Initial policy update
        initial_performance = {'factual_loss': 1.0, 'propensity_loss': 1.0, 'pehe_reg': 1.0}
        self.causal_tree_bandit.update_causal_policy(patterns, initial_performance)
        
        self.logger.info(f"Initial bandit analysis with MIC initialization: "
                        f"{len(patterns['confounding'])} confounders, "
                        f"{len(patterns['treatment_effect'])} heterogeneity patterns")
    
    def _prepare_features_for_bandit(self, X_cov: pd.DataFrame, y_arr: np.ndarray) -> pd.DataFrame:
        """
        Prepare features for bandit analysis using parent class methods.
        """
        try:
            # Handle feature names
            X_processed = self._handle_empty_feature_names(X_cov)
            
            # Apply encoding (without fitting - this is just for analysis)
            if hasattr(self, '_transform_encode_data'):
                X_processed = self._transform_encode_data(X_processed)
            
            # Apply binarization if enabled (without fitting)
            if self.binarization and hasattr(self, '_transform_binarize_features'):
                try:
                    X_processed = self._transform_binarize_features(X_processed, y_arr > np.mean(y_arr))
                except:
                    # If binarization fails, skip it for initial analysis
                    pass
            
            return X_processed
            
        except Exception as e:
            self.logger.warning(f"Feature preparation failed: {e}, using raw features")
            return X_cov
    
    
    def _compute_mic_values(self, X: pd.DataFrame, y: pd.DataFrame) -> torch.Tensor:
        """
        Compute Maximal Information Coefficient values for each feature.
        
        Args:
            X: Feature dataframe
            y: Target variable dataframe
            
        Returns:
            mic_values: Tensor of MIC values for each feature
        """
        try:
            from torchlogic.utils.policy.mic_e import compute_mic_matrix 
            
            mic_c_policy, _ = compute_mic_matrix(X, y, alpha=.45, c=6)
            mic_values = torch.tensor(mic_c_policy)
            
            self.logger.info(f"Computed MIC values for {len(mic_values)} features. "
                           f"Range: [{mic_values.min():.4f}, {mic_values.max():.4f}]")
            
            return mic_values
            
        except ImportError:
            self.logger.warning("minepy not available, using correlation-based proxy for MIC")
            return self._compute_correlation_proxy(X, y)
        except Exception as e:
            self.logger.warning(f"MIC computation failed: {e}, using correlation proxy")
            return self._compute_correlation_proxy(X, y)
    
    def _compute_correlation_proxy(self, X: pd.DataFrame, y: pd.DataFrame) -> torch.Tensor:
        """
        Fallback method using correlation as a proxy for MIC.
        """
        try:
            correlations = []
            y_values = y.values.reshape(-1)
            
            for col in X.columns:
                corr = np.corrcoef(X[col].values, y_values)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            
            # Transform correlations to be more MIC-like (non-linear relationships)
            corr_array = np.array(correlations)
            mic_proxy = corr_array ** 0.5  # Square root to emphasize stronger relationships
            
            return torch.tensor(mic_proxy, dtype=torch.float32)
            
        except Exception as e:
            self.logger.warning(f"Correlation proxy failed: {e}, using uniform values")
            return torch.ones(X.shape[1], dtype=torch.float32) / X.shape[1]
    
    def _create_integrated_model(self, X: pd.DataFrame):
        """This method is no longer needed - integrated into main fit()."""
        pass
    
    def _create_integrated_trainer(self):
        """This method is no longer needed - integrated into main fit().""" 
        pass
    
    def get_full_insights(self) -> Dict:
        """Get comprehensive insights from training and bandit analysis."""
        insights = {}
        
        # Training insights
        if hasattr(self.trainer, 'get_training_insights'):
            insights.update(self.trainer.get_training_insights())
        
        # Model insights
        if hasattr(self.model, 'get_layer_info'):
            insights['model_structure'] = self.model.get_layer_info()
        
        # Bandit insights
        if self.causal_tree_bandit:
            insights['bandit_final_state'] = {
                'feature_policies': {
                    f'head_{i}': policy.numpy().tolist() 
                    for i, policy in enumerate(self.causal_tree_bandit.feature_policy)
                },
                'discovered_patterns': {
                    'confounding': len(self.causal_tree_bandit.confounding_patterns),
                    'treatment_effect': len(self.causal_tree_bandit.treatment_effect_patterns),
                    'propensity': len(self.causal_tree_bandit.propensity_patterns)
                }
            }
        
        return insights
