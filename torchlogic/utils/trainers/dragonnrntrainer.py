
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchlogic.sklogic.causal.dragon_loss import DragonLoss


class DragonNRNTrainer:
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
        target_scaler=None,  # NEW: Add target scaler
        lambda_cal: float = 0.0,
        g_pretrain_epochs: int = 0,
        # Prune/grow style mask refresh controls (optional)
        perform_prune_plateau_count: int = 5,
        increase_prune_plateau_count: int = 0,
        bandit_policy: torch.Tensor = None,
        mic_retain_ratio: float = 0.9,
        mask_swap_frac: float = 0.1
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
        self.target_scaler = target_scaler  # Store target scaler
        self.lambda_cal = float(lambda_cal or 0.0)
        self.g_pretrain_epochs = int(g_pretrain_epochs or 0)
        # Prune/grow controls
        self.perform_prune_plateau_count = int(perform_prune_plateau_count or 0)
        self.increase_prune_plateau_count = int(increase_prune_plateau_count or 0)
        self.bandit_policy = bandit_policy
        self.mic_retain_ratio = float(mic_retain_ratio)
        self.mask_swap_frac = float(mask_swap_frac)
        self._prune_plateau_counter = 0
        
        # FIXED: Properly configure the loss function with instance parameters
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            # Use the instance parameters, not hardcoded values
            self.loss_fn = DragonLoss(
                alpha=self.alpha,
                lambda_reg=self.lambda_reg,
                lambda_tar=self.lambda_tar,
                use_brier=self.use_brier,
                clip_propensity=self.clip_propensity,
                balance_classes=self.balance_classes,
                epsilon_init=epsilon_init
            )
        
        self.model.to(self.device)
        
        # Move loss function to device if it has parameters
        if hasattr(self.loss_fn, 'parameters'):
            self.loss_fn.to(self.device)

        # Precompute differentiable inverse-transform parameters for MinMaxScaler
        self._y_min = None
        self._y_scale = None
        if self.target_scaler is not None and hasattr(self.target_scaler, 'min_') and hasattr(self.target_scaler, 'scale_'):
            # min_ and scale_ are shape (1,) for a single target
            import numpy as _np
            y_min = float(self.target_scaler.min_[0])
            y_scale = float(self.target_scaler.scale_[0]) if self.target_scaler.scale_[0] != 0 else 1.0
            self._y_min = torch.tensor(y_min, dtype=torch.float32, device=self.device)
            self._y_scale = torch.tensor(y_scale, dtype=torch.float32, device=self.device)

    def _inverse_transform_outcomes(self, y_tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable inverse transform from MinMax-scaled space to original.

        For MinMaxScaler with feature_range=(0,1): inverse_transform(x) = (x - min_) / scale_.
        If scaler params are not available, returns input as-is.
        """
        if self._y_min is None or self._y_scale is None:
            return y_tensor
        # Broadcast-safe affine transform without breaking graph
        return (y_tensor - self._y_min) / self._y_scale

    def train(self, train_dl: DataLoader, val_dl: DataLoader = None):
        best_loss = float('inf')
        patience = 0
        best_state_dict = None

        for epoch in range(self.epochs):
            self.model.train()

            # Optionally freeze outcome heads during g warmup
            if self.g_pretrain_epochs > 0:
                sym = getattr(self.model, 'symbolic_model', None)
                if sym is None:
                    sym = self.model
                if epoch == 0:
                    self._frozen_outcomes = False
                if epoch < self.g_pretrain_epochs and not getattr(self, '_frozen_outcomes', False):
                    for name in ['head_f0', 'head_f1']:
                        if hasattr(sym, name):
                            for p in getattr(sym, name).parameters():
                                p.requires_grad = False
                    self._frozen_outcomes = True
                if epoch == self.g_pretrain_epochs and getattr(self, '_frozen_outcomes', False):
                    for name in ['head_f0', 'head_f1']:
                        if hasattr(sym, name):
                            for p in getattr(sym, name).parameters():
                                p.requires_grad = True
                    self._frozen_outcomes = False
            
            # Ensure loss function is in training mode
            if hasattr(self.loss_fn, 'train'):
                self.loss_fn.train()
            
            epoch_loss = 0

            for batch in train_dl:
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device).view(-1)
                y = batch['outcome'].to(self.device).view(-1)

                # Forward pass - predictions in [0,1]
                y0_pred, y1_pred, g_preds = self.model(x)
                
                # Inverse transform predictions and targets to original scale (differentiable)
                y0_pred_original = self._inverse_transform_outcomes(y0_pred)
                y1_pred_original = self._inverse_transform_outcomes(y1_pred)
                y_original = self._inverse_transform_outcomes(y)
                
                # Compute loss in original units
                # Note: propensity head (g_preds) stays in [0,1] - it's already correct
                if self.g_pretrain_epochs > 0 and epoch < self.g_pretrain_epochs:
                    # Warmup: focus on propensity only
                    comps = self.loss_fn(y0_pred_original, y1_pred_original, g_preds, t, y_original, return_components=True)
                    loss = comps['propensity_loss']
                else:
                    loss = self.loss_fn(y0_pred_original, y1_pred_original, g_preds, t, y_original)

                # Optional calibration encouraging mean(g) ~= mean(t)
                if self.lambda_cal > 0.0:
                    g_mean = torch.clamp(g_preds.view(-1).mean(), 1e-4, 1 - 1e-4)
                    t_mean = t.mean()
                    cal_loss = F.binary_cross_entropy(g_mean, t_mean)
                    loss = loss + self.lambda_cal * cal_loss

                self.optimizer.zero_grad()
                loss.backward()
                
                # Optional gradient clipping for stability
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_dl)
            
            # Debug propensity predictions every 10 epochs
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(train_dl))
                    x_sample = sample_batch['features'].to(self.device)
                    t_sample = sample_batch['treatment'].to(self.device)
                    _, _, g_sample = self.model(x_sample)
                    print(f"  [Epoch {epoch+1}] Propensity stats - Min: {g_sample.min():.4f}, Max: {g_sample.max():.4f}, Mean: {g_sample.mean():.4f}, Treatment rate: {t_sample.mean():.4f}")
                self.model.train()
            
            # Step scheduler if provided
            if self.scheduler:
                self.scheduler.step()

            # Validation and early stopping
            if val_dl:
                val_loss = self.evaluate(val_dl)
                
                # Print loss components every 10 epochs
                if epoch % 10 == 0 and hasattr(self.loss_fn, '__call__'):
                    components = self.get_loss_components(val_dl, num_batches=1)
                    if components:
                        print(f"  [Loss Components] {', '.join([f'{k}: {v:.4f}' for k, v in components.items()])}")
                
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state_dict = self.model.state_dict().copy()
                    patience = 0
                    self._prune_plateau_counter = 0
                else:
                    patience += 1
                    # Prune/grow style refresh if configured and plateaued
                    if self.perform_prune_plateau_count > 0:
                        self._prune_plateau_counter += 1
                        if self._prune_plateau_counter >= self.perform_prune_plateau_count:
                            sym = getattr(self.model, 'symbolic_model', None)
                            if sym is None:
                                sym = self.model
                            try:
                                if hasattr(sym, 'refresh_input_mask') and self.bandit_policy is not None:
                                    print("[Mask Refresh] Applying MIC-guided input mask refresh...")
                                    sym.refresh_input_mask(
                                        policy_probs=self.bandit_policy,
                                        retain_ratio=self.mic_retain_ratio,
                                        swap_frac=self.mask_swap_frac
                                    )
                                    # After refresh, reset counters and optionally increase threshold
                                    self._prune_plateau_counter = 0
                                    self.perform_prune_plateau_count += self.increase_prune_plateau_count
                                else:
                                    print("[Mask Refresh] Skipped (no policy or refresh method)")
                            except Exception as e:
                                print(f"[Mask Refresh] Error: {e}")
                    if patience >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1} with best val loss: {best_loss:.6f}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_loss:.6f}")

        # Load best model if validation was used
        if best_state_dict:
            self.model.load_state_dict(best_state_dict)
            print(f"Loaded best model with validation loss: {best_loss:.6f}")

    def evaluate(self, dl: DataLoader):
        self.model.eval()
        
        # Ensure loss function is in evaluation mode
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
            
        loss_total = 0
        
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device).view(-1)
                y = batch['outcome'].to(self.device).view(-1)

                # Forward pass - predictions in [0,1]
                y0_pred, y1_pred, g = self.model(x)

                # Inverse transform to original scale (differentiable)
                y0_pred_original = self._inverse_transform_outcomes(y0_pred)
                y1_pred_original = self._inverse_transform_outcomes(y1_pred)
                y_original = self._inverse_transform_outcomes(y)

                # Compute loss in original units
                loss = self.loss_fn(y0_pred_original, y1_pred_original, g, t, y_original)
                loss_total += loss.item()

        return loss_total / len(dl)

    # Method to get loss components for debugging
    def get_loss_components(self, dl: DataLoader, num_batches=1):
        """Get detailed loss components for debugging."""
        self.model.eval()
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
            
        components_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(dl):
                if i >= num_batches:
                    break
                    
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device).view(-1)
                y = batch['outcome'].to(self.device).view(-1)

                # Forward pass - predictions in [0,1]
                y0_pred, y1_pred, g = self.model(x)
                
                # Inverse transform to original scale (differentiable)
                y0_pred_original = self._inverse_transform_outcomes(y0_pred)
                y1_pred_original = self._inverse_transform_outcomes(y1_pred)
                y_original = self._inverse_transform_outcomes(y)
                
                # Get loss components in original units
                components = self.loss_fn(y0_pred_original, y1_pred_original, g, t, y_original, return_components=True)
                components_list.append({k: v.item() if hasattr(v, 'item') else v 
                                      for k, v in components.items()})
        
        # Average components
        if components_list:
            avg_components = {}
            for key in components_list[0].keys():
                avg_components[key] = sum(comp[key] for comp in components_list) / len(components_list)
            return avg_components
        return None
