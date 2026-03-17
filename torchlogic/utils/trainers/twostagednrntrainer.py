import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchlogic.sklogic.causal.dragon_loss import DragonLoss
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import precision_score,roc_auc_score
import logging
import torch.nn.functional as F

class TwoPhaseDragonNRNTrainer:
    """
    Enhanced trainer with two-phase training:
    Phase 1: Pretrain propensity head (head_g) on treatment prediction
    Phase 2: Full model training with all heads
    """
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        alpha=None,
        lambda_reg=None,
        use_brier=False,
        lambda_tar=None,
        epochs=200,
        early_stopping_patience=20,
        device=None,
        loss_fn=None,
        balance_classes=False,
        clip_propensity=True,
        epsilon_init=0.1,
        causal_tree_bandit=None,
        bandit_update_frequency=10,
        # New two-phase parameters
        pretrain_g=True,
        g_pretrain_epochs=50,
        g_pretrain_lr=1e-3,
        g_lr_scale=0.1,  # Scale down g head LR in phase 2
        freeze_g_epochs=10,  # Epochs to freeze g in phase 2
        g_pretrain_patience=10
    ):
        self.model = model
        self.eps = epsilon_init
        self.alpha = alpha
        self.balance_classes = balance_classes
        self.lambda_reg = lambda_reg
        self.lambda_tar = lambda_tar
        self.use_brier = use_brier
        self.clip_propensity = clip_propensity
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Two-phase training parameters
        self.pretrain_g = pretrain_g
        self.g_pretrain_epochs = g_pretrain_epochs
        self.g_pretrain_lr = g_pretrain_lr
        self.g_lr_scale = g_lr_scale
        self.freeze_g_epochs = freeze_g_epochs
        self.g_pretrain_patience = g_pretrain_patience
        
        # Bandit integration
        self.causal_tree_bandit = causal_tree_bandit
        self.bandit_update_frequency = bandit_update_frequency
        self.training_iteration = 0
        
        # Loss function setup
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            from torchlogic.sklogic.causal.DragonNRNRegressor import DragonLoss
            self.loss_fn = DragonLoss(
                alpha=self.alpha or 0.5,
                lambda_reg=self.lambda_reg or 0.5,
                lambda_tar=self.lambda_tar or 0.0,
                use_brier=self.use_brier,
                clip_propensity=self.clip_propensity,
                balance_classes=self.balance_classes,
                epsilon_init=epsilon_init
            )
        
        self.model.to(self.device)
        
        if hasattr(self.loss_fn, 'parameters'):
            self.loss_fn.to(self.device)
            
        # Storage for performance tracking
        self.loss_history = []
        self.component_history = []
        self.g_pretrain_history = []
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def train_propensity_head(self, train_dl: DataLoader, val_dl: DataLoader = None):
        """
        Phase 1: Pretrain the propensity head (head_g) on treatment prediction.
        """
        self.logger.info(f"Starting Phase 1: Propensity head pretraining for {self.g_pretrain_epochs} epochs")
        
        # Freeze outcome heads
        for param in self.model.head_f0.parameters():
            param.requires_grad = False
        for param in self.model.head_f1.parameters():
            param.requires_grad = False
            
        # Create optimizer for propensity head + trunk
        g_params = list(self.model.shared_trunk.parameters()) + list(self.model.head_g.parameters())
        g_optimizer = torch.optim.AdamW(g_params, lr=self.g_pretrain_lr, weight_decay=1e-4)
        g_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            g_optimizer, T_0=max(1, self.g_pretrain_epochs // 4)
        )
        
        best_auc = 0.0
        patience = 0
        best_g_state = None
        
        for epoch in range(self.g_pretrain_epochs):
            self.model.train()
            
            epoch_loss = 0
            all_preds = []
            all_treatments = []
            
            for batch_idx, batch in enumerate(train_dl):
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                #g = batch['propensity'].to(self.device)  # Propensity labels
                
                # Forward pass through trunk and propensity head only
                trunk_output = self.model.shared_trunk(x)
                g_logits = self.model.head_g(trunk_output).squeeze(-1).squeeze(-1)
                g_preds = g_logits
                #g_calibrated = g
                
                #bce_loss = F.mse_loss(g_logits, g_calibrated)
                
                # Binary cross-entropy loss for treatment prediction
                bce_loss = F.binary_cross_entropy(g_preds, t.float()) #+ F.mse_loss(g_logits, g_calibrated)
                
                #g_optimizer.zero_grad()
                bce_loss.backward()
                g_optimizer.step()
                
                epoch_loss += bce_loss.item()
                
                # Store predictions for AUC calculation
                all_preds.extend(g_preds.detach().cpu().numpy())
                all_treatments.extend(t.cpu().numpy())
            
            epoch_loss /= len(train_dl)
            
            # Calculate training AUC
            try:
                train_auc = roc_auc_score(all_treatments, all_preds)
            except:
                train_auc = 0.5  # Fallback if AUC calculation fails
            
            if g_scheduler:
                g_scheduler.step()
            
            # Validation
            val_auc = 0.5
            if val_dl:
                val_auc, val_loss = self._evaluate_propensity_head(val_dl)
                
                self.logger.info(f"G-Pretrain Epoch {epoch+1}/{self.g_pretrain_epochs} - "
                               f"Train Loss: {epoch_loss:.6f}, Train AUC: {train_auc:.4f}, "
                               f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.4f}")
                
                # Early stopping based on AUC
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_g_state = {
                        'trunk': self.model.shared_trunk.state_dict().copy(),
                        'head_g': self.model.head_g.state_dict().copy()
                    }
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.g_pretrain_patience:
                        self.logger.info(f"Early stopping propensity pretraining at epoch {epoch+1} "
                                       f"with best AUC: {best_auc:.4f}")
                        break
            else:
                self.logger.info(f"G-Pretrain Epoch {epoch+1}/{self.g_pretrain_epochs} - "
                               f"Train Loss: {epoch_loss:.6f}, Train AUC: {train_auc:.4f}")
            
            # Store history
            self.g_pretrain_history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'lr': g_optimizer.param_groups[0]['lr']
            })
        
        # Load best propensity head weights
        if best_g_state:
            self.model.shared_trunk.load_state_dict(best_g_state['trunk'])
            self.model.head_g.load_state_dict(best_g_state['head_g'])
            self.logger.info(f"Loaded best propensity model with AUC: {best_auc:.4f}")
        
        # Unfreeze outcome heads for phase 2
        for param in self.model.head_f0.parameters():
            param.requires_grad = True
        for param in self.model.head_f1.parameters():
            param.requires_grad = True
            
        self.logger.info(f"Phase 1 completed. Best propensity AUC: {best_auc:.4f}")
        del g_optimizer, g_scheduler  # Clean up optimizer/scheduler to free memory
        return best_auc

    def _evaluate_propensity_head(self, dl: DataLoader) -> Tuple[float, float]:
        """Evaluate propensity head during pretraining."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_treatments = []
        
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                #g = batch['propensity'].to(self.device)
                
                trunk_output = self.model.shared_trunk(x)
                g_logits = self.model.head_g(trunk_output).squeeze(-1).squeeze(-1)
                
                bce_loss = F.binary_cross_entropy(g_logits, t.float())
                #mse_calibration = F.mse_loss(g_logits, g)
                
                
                total_loss += bce_loss.item()
                
                all_preds.extend(g_logits.cpu().numpy())
                all_treatments.extend(t.cpu().numpy())
        
        avg_loss = total_loss / len(dl)
        
        try:
            auc = roc_auc_score(all_treatments, all_preds)
        except:
            print("AUC calculation failed, returning 0.5")
            
        return auc, avg_loss

    


    def train_full_model(self, train_dl: DataLoader, val_dl: DataLoader = None, 
                        raw_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None):
        """
        Phase 2: Train the full model with all heads.
        """
        def deduplicate_param_groups(groups):
            seen = set()
            new_groups = []

            for group in groups:
                new_params = []
                for p in group['params']:
                    if id(p) not in seen:
                        seen.add(id(p))
                        new_params.append(p)
                if new_params:
                    group = {k: v for k, v in group.items()}
                    group['params'] = new_params
                    new_groups.append(group)
            return new_groups
        
        
        self.logger.info(f"Starting Phase 2: Full model training for {self.epochs} epochs")
        
        # Create new optimizer with different learning rates
        param_groups = []
        
        # Outcome heads get full learning rate
        outcome_params = list(self.model.head_f0.parameters()) + list(self.model.head_f1.parameters())
        param_groups.append({'params': outcome_params, 'lr': self.optimizer.defaults['lr']})
        
        # Propensity head gets scaled learning rate
        g_params = list(self.model.head_g.parameters())
        param_groups.append({'params': g_params, 'lr': self.optimizer.defaults['lr'] * self.g_lr_scale})
        
        # Trunk gets full learning rate
        trunk_params = list(self.model.shared_trunk.parameters())
        param_groups.append({'params': trunk_params, 'lr': self.optimizer.defaults['lr']})

        param_groups = [
            {'params': list(self.model.head_f0.parameters()), 'lr': self.optimizer.defaults['lr']},
            {'params': list(self.model.head_f1.parameters()), 'lr': self.optimizer.defaults['lr']},
            {'params': list(self.model.head_g.parameters()),  'lr': self.optimizer.defaults['lr'] * self.g_lr_scale},
            {'params': list(self.model.shared_trunk.parameters()), 'lr': self.optimizer.defaults['lr']},
        ]

        param_groups = deduplicate_param_groups(param_groups)
    
        phase2_optimizer = torch.optim.AdamW(param_groups, weight_decay=self.optimizer.defaults.get('weight_decay', 1e-4))
        phase2_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            phase2_optimizer, T_0=max(1, self.epochs // 4)
        )
        
        best_loss = float('inf')
        patience = 0
        best_state_dict = None
        
        for epoch in range(self.epochs):
            self.model.train()
            
            if hasattr(self.loss_fn, 'train'):
                self.loss_fn.train()
            
            # Optionally freeze propensity head for first few epochs
            freeze_g = epoch < self.freeze_g_epochs
            if freeze_g:
                for param in self.model.head_g.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.head_g.parameters():
                    param.requires_grad = True
            
            epoch_loss = 0
            epoch_components = []

            for batch_idx, batch in enumerate(train_dl):
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)  
                y = batch['outcome'].to(self.device)    

                # Forward pass
                y0_pred, y1_pred, g_preds = self.model(x)
                
                # Compute loss with components
                loss_components = self.loss_fn(y0_pred, y1_pred, g_preds, t, y, return_components=True)
                loss = loss_components['total_loss']
                
                # Store components for bandit updates
                epoch_components.append({
                    k: v.item() if hasattr(v, 'item') else v 
                    for k, v in loss_components.items()
                })

                phase2_optimizer.zero_grad()
                loss.backward()
                phase2_optimizer.step()

                epoch_loss += loss.item()
                self.training_iteration += 1
                
                # Bandit update during training
                if (self.causal_tree_bandit is not None and 
                    self.training_iteration % self.bandit_update_frequency == 0 and
                    raw_data is not None):
                    self._update_bandit_during_training(raw_data, epoch_components[-1])

            epoch_loss /= len(train_dl)
            
            # Store training history
            avg_components = self._average_components(epoch_components)
            self.loss_history.append(epoch_loss)
            self.component_history.append(avg_components)
            
            if phase2_scheduler:
                phase2_scheduler.step()

            # Validation and early stopping
            if val_dl:
                val_loss, val_components = self.evaluate_with_components(val_dl)
                freeze_status = " (G frozen)" if freeze_g else ""
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}{freeze_status} - "
                               f"Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
                self.logger.info(f"  Components - Factual: {avg_components['factual_loss']:.4f}, "
                               f"Prop: {avg_components['propensity_loss']:.4f}, "
                               f"PEHE: {avg_components['pehe_reg']:.4f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state_dict = self.model.state_dict().copy()
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1} with best val loss: {best_loss:.6f}")
                        break
            else:
                freeze_status = " (G frozen)" if freeze_g else ""
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}{freeze_status} - Train Loss: {epoch_loss:.6f}")

        if best_state_dict:
            self.model.load_state_dict(best_state_dict)
            self.logger.info(f"Loaded best model with validation loss: {best_loss:.6f}")

    def train(self, train_dl: DataLoader, val_dl: DataLoader = None, 
              raw_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None):
        """
        Main training method with two phases.
        """
        if self.pretrain_g:
            # Phase 1: Pretrain propensity head
            best_g_auc = self.train_propensity_head(train_dl, val_dl)
            
            # Phase 2: Train full model
            self.train_full_model(train_dl, val_dl, raw_data)
        else:
            # Single phase training (original approach)
            self.train_full_model(train_dl, val_dl, raw_data)

    def _update_bandit_during_training(self, raw_data: Tuple[pd.DataFrame, pd.DataFrame], 
                                     loss_components: Dict[str, float]):
        """Update bandit policy during training using real performance metrics."""
        try:
            X, y = raw_data
            X_cov = X.drop(columns=[self.causal_tree_bandit.treatment_col])
            t_arr = X[self.causal_tree_bandit.treatment_col].values.astype(np.float32)
            y_arr = y.values.astype(np.float32).reshape(-1)
            
            # Discover patterns with current data
            patterns = self.causal_tree_bandit.discover_causal_patterns(
                X_cov.values, t_arr, y_arr
            )
            
            # Use real performance metrics instead of mock data
            dragon_performance = {
                'factual_loss': loss_components['factual_loss'],
                'propensity_loss': loss_components['propensity_loss'],
                'pehe_reg': loss_components['pehe_reg']
            }
            
            # Update bandit policy
            self.causal_tree_bandit.update_causal_policy(patterns, dragon_performance)
            
        except Exception as e:
            self.logger.warning(f"Bandit update failed: {e}")

    def _average_components(self, component_list: List[Dict]) -> Dict[str, float]:
        """Average loss components across batches."""
        if not component_list:
            return {}
            
        avg_components = {}
        for key in component_list[0].keys():
            if key != 'total_loss':  # Skip total loss to avoid double counting
                values = [comp[key] for comp in component_list if key in comp]
                avg_components[key] = sum(values) / len(values) if values else 0.0
                
        return avg_components

    def evaluate_with_components(self, dl: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate model and return both total loss and components."""
        self.model.eval()
        
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
            
        loss_total = 0
        all_components = []
        
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)   

                y0_pred, y1_pred, g = self.model(x)
                
                loss_components = self.loss_fn(y0_pred, y1_pred, g, t, y, return_components=True)
                loss_total += loss_components['total_loss'].item()
                
                # Store components
                all_components.append({
                    k: v.item() if hasattr(v, 'item') else v 
                    for k, v in loss_components.items()
                })

        avg_loss = loss_total / len(dl)
        avg_components = self._average_components(all_components)
        
        return avg_loss, avg_components

    def get_training_insights(self) -> Dict:
        """Get comprehensive insights from both training phases."""
        insights = {
            'loss_history': self.loss_history,
            'component_history': self.component_history,
            'training_iterations': self.training_iteration,
            'g_pretrain_history': self.g_pretrain_history,
            'pretrain_enabled': self.pretrain_g
        }
        
        if self.causal_tree_bandit:
            insights['causal_patterns'] = {
                'confounding': self.causal_tree_bandit.confounding_patterns,
                'treatment_effect': self.causal_tree_bandit.treatment_effect_patterns,
                'propensity': self.causal_tree_bandit.propensity_patterns
            }
            
            insights['feature_policies'] = {
                f'head_{i}': policy.numpy().tolist() 
                for i, policy in enumerate(self.causal_tree_bandit.feature_policy)
            }
        
        return insights