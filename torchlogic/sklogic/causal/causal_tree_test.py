import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
import logging



class CausalTreeBanditPolicy:
    """
    Specialized tree bandit policy for causal inference tasks.
    Discovers patterns relevant to treatment effects and confounding.
    """
    
    def __init__(
        self,
        input_size: int,
        feature_names: List[str],
        treatment_col: str,
        ucb_scale: float = 1.96,
        tree_max_depth: int = 3,
        confounding_weight: float = 0.3,
        treatment_effect_weight: float = 0.7
    ):
        self.input_size = input_size
        self.feature_names = feature_names
        self.treatment_col = treatment_col
        self.ucb_scale = ucb_scale
        self.tree_max_depth = tree_max_depth
        self.confounding_weight = confounding_weight
        self.treatment_effect_weight = treatment_effect_weight
        
        # Causal-specific pattern tracking
        self.confounding_patterns = []
        self.treatment_effect_patterns = []
        self.propensity_patterns = []
        
        # Standard bandit components for 3 heads: y0, y1, g
        self.feature_policy = torch.ones(3, input_size) / input_size
        self.feature_rewards_history = {
            0: pd.DataFrame({'feature': range(input_size), 'reward': [0.0] * input_size}),  # y0
            1: pd.DataFrame({'feature': range(input_size), 'reward': [0.0] * input_size}),  # y1
            2: pd.DataFrame({'feature': range(input_size), 'reward': [0.0] * input_size})   # g
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def discover_causal_patterns(
        self, 
        X: np.ndarray, 
        t: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, List[Dict]]:
        """
        Discover patterns relevant to causal inference.
        """
        patterns = {
            'confounding': self._discover_confounding_patterns(X, t, y),
            'treatment_effect': self._discover_treatment_effect_patterns(X, t, y),
            'propensity': self._discover_propensity_patterns(X, t)
        }
        
        # Store for later use
        self.confounding_patterns = patterns['confounding']
        self.treatment_effect_patterns = patterns['treatment_effect']
        self.propensity_patterns = patterns['propensity']
        
        return patterns
    
    def _discover_confounding_patterns(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> List[Dict]:
        """Discover features that confound the treatment-outcome relationship."""
        confounding_patterns = []
        
        try:
            # Build tree to predict treatment
            treatment_tree = DecisionTreeRegressor(
                max_depth=self.tree_max_depth,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
            treatment_tree.fit(X, t)
            
            # Build tree to predict outcome
            outcome_tree = DecisionTreeRegressor(
                max_depth=self.tree_max_depth,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=43
            )
            outcome_tree.fit(X, y)
            
            # Get feature importance from both trees
            treatment_importance = treatment_tree.feature_importances_
            outcome_importance = outcome_tree.feature_importances_
            
            # Find features important for both (potential confounders)
            for i, (t_imp, y_imp) in enumerate(zip(treatment_importance, outcome_importance)):
                if t_imp > 0.05 and y_imp > 0.05:  # Both have reasonable importance
                    confounding_strength = min(t_imp, y_imp) * 2  # Harmonic mean-like score
                    pattern = {
                        'feature_idx': i,
                        'feature_name': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                        'treatment_importance': t_imp,
                        'outcome_importance': y_imp,
                        'confounding_strength': confounding_strength,
                        'score': confounding_strength
                    }
                    confounding_patterns.append(pattern)
            
            # Sort by confounding strength
            confounding_patterns.sort(key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error discovering confounding patterns: {e}")
        
        return confounding_patterns[:5]  # Top 5 confounders
    
    def _discover_treatment_effect_patterns(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> List[Dict]:
        """Discover patterns where treatment effects vary (heterogeneity)."""
        te_patterns = []
        
        try:
            # Split by treatment
            treated_mask = t == 1
            control_mask = t == 0
            
            if np.sum(treated_mask) < 10 or np.sum(control_mask) < 10:
                return te_patterns
            
            X_treated = X[treated_mask]
            y_treated = y[treated_mask]
            X_control = X[control_mask]
            y_control = y[control_mask]
            
            # Build trees for each group
            treated_tree = DecisionTreeRegressor(
                max_depth=self.tree_max_depth,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=44
            )
            treated_tree.fit(X_treated, y_treated)
            
            control_tree = DecisionTreeRegressor(
                max_depth=self.tree_max_depth,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=45
            )
            control_tree.fit(X_control, y_control)
            
            # Compare feature importances
            treated_importance = treated_tree.feature_importances_
            control_importance = control_tree.feature_importances_
            
            for i, (t_imp, c_imp) in enumerate(zip(treated_importance, control_importance)):
                importance_diff = abs(t_imp - c_imp)
                if importance_diff > 0.05:  # Significant difference in importance
                    pattern = {
                        'feature_idx': i,
                        'feature_name': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                        'treated_importance': t_imp,
                        'control_importance': c_imp,
                        'heterogeneity_strength': importance_diff,
                        'score': importance_diff
                    }
                    te_patterns.append(pattern)
            
            te_patterns.sort(key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error discovering treatment effect patterns: {e}")
        
        return te_patterns[:5]
    
    def _discover_propensity_patterns(self, X: np.ndarray, t: np.ndarray) -> List[Dict]:
        """Discover patterns that predict treatment assignment."""
        propensity_patterns = []
        
        try:
            propensity_tree = DecisionTreeRegressor(
                max_depth=self.tree_max_depth,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=46
            )
            propensity_tree.fit(X, t)
            
            importance = propensity_tree.feature_importances_
            
            for i, imp in enumerate(importance):
                if imp > 0.05:  # Reasonable importance for propensity
                    pattern = {
                        'feature_idx': i,
                        'feature_name': self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}',
                        'propensity_importance': imp,
                        'score': imp
                    }
                    propensity_patterns.append(pattern)
            
            propensity_patterns.sort(key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error discovering propensity patterns: {e}")
        
        return propensity_patterns[:5]
    
    def update_causal_policy(
        self, 
        patterns: Dict[str, List[Dict]], 
        dragon_performance: Dict[str, float]
    ):
        """
        Update bandit policy based on discovered causal patterns and DragonNet performance.
        
        Args:
            patterns: Dictionary with 'confounding', 'treatment_effect', 'propensity' patterns
            dragon_performance: Dict with 'factual_loss', 'propensity_loss', 'pehe_reg' etc.
        """
        
        # Calculate head-specific rewards based on pattern relevance
        head_rewards = {
            0: np.zeros(self.input_size),  # y0 (control outcomes)
            1: np.zeros(self.input_size),  # y1 (treated outcomes)
            2: np.zeros(self.input_size)   # g (propensity)
        }
        
        # Confounding patterns affect all heads
        for pattern in patterns.get('confounding', []):
            feat_idx = pattern['feature_idx']
            reward = pattern['score'] * self.confounding_weight
            head_rewards[0][feat_idx] += reward
            head_rewards[1][feat_idx] += reward
            head_rewards[2][feat_idx] += reward
        
        # Treatment effect patterns primarily affect outcome heads
        for pattern in patterns.get('treatment_effect', []):
            feat_idx = pattern['feature_idx']
            reward = pattern['score'] * self.treatment_effect_weight
            head_rewards[0][feat_idx] += reward
            head_rewards[1][feat_idx] += reward
        
        # Propensity patterns primarily affect propensity head
        for pattern in patterns.get('propensity', []):
            feat_idx = pattern['feature_idx']
            reward = pattern['score']
            head_rewards[2][feat_idx] += reward
        
        # Weight rewards by DragonNet performance
        factual_performance = 1.0 / (1.0 + dragon_performance.get('factual_loss', 1.0))
        propensity_performance = 1.0 / (1.0 + dragon_performance.get('propensity_loss', 1.0))
        
        head_rewards[0] *= factual_performance
        head_rewards[1] *= factual_performance
        head_rewards[2] *= propensity_performance
        
        # Update reward histories
        for head_idx in range(3):
            new_rewards_df = pd.DataFrame({
                'feature': range(self.input_size),
                'reward': head_rewards[head_idx]
            })
            self.feature_rewards_history[head_idx] = pd.concat([
                self.feature_rewards_history[head_idx],
                new_rewards_df
            ])
            
            # Update policy using UCB
            self._update_head_policy(head_idx)
    
    def _update_head_policy(self, head_idx: int):
        """Update feature selection policy for a specific head using UCB."""
        history_df = self.feature_rewards_history[head_idx]
        
        if len(history_df) == 0:
            return
        
        # Calculate UCB scores
        stats = history_df.groupby('feature')['reward'].agg(['mean', 'count', 'std']).fillna(0)
        total_counts = stats['count'].sum()
        
        ucb_scores = (
            stats['mean'] + 
            self.ucb_scale * np.sqrt(np.log(total_counts + 1) / (stats['count'] + 1))
        )
        
        # Convert to probabilities
        probabilities = softmax(ucb_scores.values)
        self.feature_policy[head_idx] = torch.tensor(probabilities, dtype=torch.float32)
    
    def sample_features_for_head(self, head_idx: int, n_features: int) -> np.ndarray:
        """Sample features for a specific DragonNet head."""
        try:
            probabilities = self.feature_policy[head_idx].numpy()
            return np.random.choice(
                self.input_size,
                size=n_features,
                replace=False,
                p=probabilities
            )
        except ValueError:
            # Fallback to uniform sampling
            return np.random.choice(self.input_size, size=n_features, replace=False)



