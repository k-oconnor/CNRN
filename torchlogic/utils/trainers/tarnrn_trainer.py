import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class TARNetLoss(nn.Module):
    def __init__(
        self,
        lambda_tar=0.0,
        lambda_pehe=0.0,
        balance_classes=False,
        eps=1e-6,
        epsilon_init=0.01,
        loss_on_original_scale=False,
    ):
        super().__init__()
        self.lambda_tar = lambda_tar
        self.lambda_pehe = lambda_pehe
        self.balance_classes = balance_classes
        self.eps = eps
        self.epsilon = nn.Parameter(torch.tensor(epsilon_init)) if lambda_tar > 0 else None
        self._warned_invalid_pehe = False
        self.loss_on_original_scale = bool(loss_on_original_scale)
        self.output_min = None
        self.output_max = None

    def set_output_scaler(self, output_min: float, output_max: float):
        self.output_min = float(output_min)
        self.output_max = float(output_max)

    def _inverse_scale(self, y_hat: torch.Tensor) -> torch.Tensor:
        if self.output_min is None or self.output_max is None:
            raise RuntimeError("Output scaler bounds were not set before original-scale loss was requested.")
        return torch.clamp(y_hat, 0.0, 1.0) * (self.output_max - self.output_min) + self.output_min

    def forward(self, y0_hat, y1_hat, g, t, y, return_components=False, y_original=None):
        y0 = y0_hat.view(-1,1); y1 = y1_hat.view(-1,1)
        t  = t.float().view(-1,1); y = y.view(-1,1)
        g  = g.view(-1,1)
        g = torch.clamp(g, self.eps, 1.0 - self.eps)

        # 1) Factual loss (scalar)
        q_t = t * y1 + (1 - t) * y0
        if self.loss_on_original_scale:
            if y_original is None:
                raise RuntimeError("Original-scale loss requested but y_original was not provided.")
            y_target = y_original.float().view(-1, 1)
            q_t_for_loss = self._inverse_scale(q_t)
            factual = F.mse_loss(q_t_for_loss, y_target, reduction="mean")
        else:
            factual = F.mse_loss(q_t, y, reduction="mean")

        # 3) Disable the previous observational 'PEHE' proxy. It only shrank treatment effects.
        pehe = torch.tensor(0.0, device=y.device)
        if self.lambda_pehe > 0 and not self._warned_invalid_pehe:
            print('Warning: lambda_pehe>0 requested, but the old observational proxy was invalid and has been disabled.')
            self._warned_invalid_pehe = True

        # 4) Targeted reg
        tar = torch.tensor(0.0, device=y.device)
        if self.lambda_tar > 0 and self.epsilon is not None:
            h = t / g - (1 - t) / (1 - g)
            y_pert = q_t + self.epsilon * h
            if self.loss_on_original_scale:
                y_target = y_original.float().view(-1, 1)
                tar = F.mse_loss(y_target, self._inverse_scale(y_pert), reduction="mean")
            else:
                tar = F.mse_loss(y, y_pert, reduction="mean")

        total = factual + self.lambda_pehe * pehe + self.lambda_tar * tar
        if return_components:
            return {"total": total, "factual": factual, "pehe": pehe, "tar": tar, "epsilon": float(self.epsilon.item()) if self.epsilon is not None else 0.0}
        return total


class TARNRNTrainer:
    """
    TARNet (Treatment-Agnostic Representation Network) Trainer
    
    Based on "Estimating individual treatment effect: generalization bounds and algorithms"
    by Shalit et al. (ICML 2017)
    
    TARNet uses a shared representation network followed by separate heads for 
    control and treatment outcomes. It trains on factual outcomes only.
    """
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        lambda_reg=0.0,
        lambda_tar = 0.0,
        epochs=200,
        early_stopping_patience=20,
        device=None,
        loss_fn=None,
        gradient_clip_norm=None,
        beta_l1: float = 0.0,
    ):
        """
        Initialize TARNet trainer
        
        Args:
            model: TARNet model with forward method returning (y0_pred, y1_pred, g_preds)
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            alpha: Weight for factual loss
            lambda_reg: L2 regularization weight
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            device: Training device
            loss_fn: Custom loss function (optional)
            gradient_clip_norm: Gradient clipping norm (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_clip_norm = gradient_clip_norm
        self.beta_l1 = float(beta_l1 or 0.0)
        self.beta_l1_eff = self.beta_l1
        
        # Configure loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = TARNetLoss(alpha=alpha, lambda_reg=lambda_reg)
        
        # Move model and loss to device
        self.model.to(self.device)
        if hasattr(self.loss_fn, 'parameters'):
            self.loss_fn.to(self.device)

    def train(self, train_dl: DataLoader, val_dl: DataLoader = None):
        """
        Train the TARNet model
        
        Args:
            train_dl: Training data loader
            val_dl: Validation data loader (optional)
        """
        best_loss = float('inf')
        patience = 0
        best_state_dict = None
        train_losses = []
        val_losses = []

        print(f"Starting TARNet training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        global_step = 0

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            if hasattr(self.loss_fn, 'train'):
                self.loss_fn.train()
            
            epoch_loss = 0
            num_batches = 0

            for batch in train_dl:
                self.beta_l1_eff = self.beta_l1

                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                y_original = batch.get('outcome_original')
                if y_original is not None:
                    y_original = y_original.to(self.device)
                g = batch['propensity'].to(self.device)

                # Forward pass
                y0_pred, y1_pred = self.model(x)
                
                # Compute loss
                task_loss = self.loss_fn(y0_pred, y1_pred, g, t, y, y_original=y_original)
                landauer_tax = (
                    sum(p.abs().sum() for p in self.model.parameters())
                    if self.beta_l1 > 0.0
                    else torch.tensor(0.0, device=self.device)
                )
                loss = task_loss + self.beta_l1_eff * landauer_tax

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping if specified
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.gradient_clip_norm
                    )
                
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

            # Average training loss
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Use validation loss if available, otherwise training loss
                    scheduler_metric = avg_train_loss
                    if val_dl is not None:
                        val_loss = self.evaluate(val_dl)
                        scheduler_metric = val_loss
                    self.scheduler.step(scheduler_metric)
                else:
                    self.scheduler.step()

            # Validation phase
            if val_dl:
                val_loss = self.evaluate(val_dl)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")
                
                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1} "
                              f"with best val loss: {best_loss:.6f}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}")

        # Load best model if validation was used
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f"Loaded best model with validation loss: {best_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_loss if val_dl else None
        }

    def evaluate(self, dl: DataLoader):
        """
        Evaluate the model on a dataset
        
        Args:
            dl: Data loader for evaluation
            
        Returns:
            Average loss on the dataset
        """
        self.model.eval()
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
            
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                y_original = batch.get('outcome_original')
                if y_original is not None:
                    y_original = y_original.to(self.device)
                g = batch['propensity'].to(self.device)

                # Forward pass
                y0_pred, y1_pred = self.model(x)

                # Compute loss
                loss = self.loss_fn(y0_pred, y1_pred, g, t, y, y_original=y_original)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
    
    def predict_individual_treatment_effects(self, dl: DataLoader):
        """
        Predict individual treatment effects (ITEs) for a dataset
        
        Args:
            dl: Data loader
            
        Returns:
            Dictionary with predictions and treatment effects
        """
        self.model.eval()
        
        all_y0_pred = []
        all_y1_pred = []
        all_ite_pred = []
        all_features = []
        all_treatments = []
        all_outcomes = []
        all_prop_pred = []
        
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                g = batch['propensity'].to(self.device)

                # Get predictions for both treatment and control
                y0_pred, y1_pred = self.model(x)
                
                # Calculate individual treatment effects
                ite_pred = y1_pred - y0_pred
                
                # Store results
                all_y0_pred.append(y0_pred.cpu())
                all_y1_pred.append(y1_pred.cpu())
                all_ite_pred.append(ite_pred.cpu())
                all_prop_pred.append(g.cpu())
                all_features.append(x.cpu())
                all_treatments.append(t.cpu())
                all_outcomes.append(y.cpu())
        
        return {
            'y0_pred': torch.cat(all_y0_pred, dim=0),
            'y1_pred': torch.cat(all_y1_pred, dim=0),
            'ite_pred': torch.cat(all_ite_pred, dim=0),
            'prop_pred': torch.cat(all_prop_pred, dim=0),
            'features': torch.cat(all_features, dim=0),
            'treatment': torch.cat(all_treatments, dim=0),
            'outcome': torch.cat(all_outcomes, dim=0)
        }
    
    def get_loss_components(self, dl: DataLoader, num_batches=1):
        """
        Get detailed loss components for debugging
        
        Args:
            dl: Data loader
            num_batches: Number of batches to analyze
            
        Returns:
            Dictionary with average loss components
        """
        self.model.eval()
        if hasattr(self.loss_fn, 'eval'):
            self.loss_fn.eval()
            
        components_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(dl):
                if i >= num_batches:
                    break
                    
                x = batch['features'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                y_original = batch.get('outcome_original')
                if y_original is not None:
                    y_original = y_original.to(self.device)
                g = batch['propensity'].to(self.device)

                # Get predictions
                y0_pred, y1_pred = self.model(x)
                
                # Get loss components
                components = self.loss_fn(
                    y0_pred, y1_pred, g, t, y,
                    y_original=y_original,
                    return_components=True
                )
                
                # Convert tensors to scalars
                components_dict = {}
                for k, v in components.items():
                    if hasattr(v, 'item'):
                        components_dict[k] = v.item()
                    else:
                        components_dict[k] = v
                components_list.append(components_dict)
        
        # Calculate averages
        if components_list:
            avg_components = {}
            for key in components_list[0].keys():
                if isinstance(components_list[0][key], (int, float)):
                    avg_components[key] = sum(comp[key] for comp in components_list) / len(components_list)
                else:
                    avg_components[key] = components_list[0][key]  # For non-numeric values
            return avg_components
        
        return None

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

