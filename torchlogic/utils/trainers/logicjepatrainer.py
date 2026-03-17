from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader


class LogicJEPALoss(nn.Module):
    def __init__(
            self,
            pred_weight: float = 1.0,
            var_weight: float = 25.0,
            cov_weight: float = 1.0,
            var_target: float = 1.0
    ):
        super().__init__()
        self.pred_weight = float(pred_weight)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.var_target = float(var_target)

    @staticmethod
    def _covariance_penalty(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / max(1, x.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        return (off_diag ** 2).sum() / x.shape[1]

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor, z_online: torch.Tensor, return_components: bool = False):
        pred = nn.functional.mse_loss(z_pred, z_target, reduction='mean')

        std_online = torch.sqrt(z_online.var(dim=0) + 1e-4)
        std_target = torch.sqrt(z_target.var(dim=0) + 1e-4)
        var = torch.relu(self.var_target - std_online).mean() + torch.relu(self.var_target - std_target).mean()

        cov = self._covariance_penalty(z_online) + self._covariance_penalty(z_target)

        total = self.pred_weight * pred + self.var_weight * var + self.cov_weight * cov
        if return_components:
            return {
                'total': total,
                'pred': pred,
                'var': var,
                'cov': cov
            }
        return total


class LogicJEPATrainer:
    def __init__(
            self,
            model,
            optimizer,
            scheduler=None,
            loss_fn=None,
            epochs: int = 200,
            early_stopping_patience: int = 20,
            ema_momentum: float = 0.99,
            ema_momentum_end: float = None,
            view_mask_prob: float = 0.15,
            view_mask_prob_end: float = None,
            view_noise_std: float = 0.01,
            beta_l1: float = 0.0,
            device=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else LogicJEPALoss()
        self.epochs = int(epochs)
        self.early_stopping_patience = int(early_stopping_patience)
        self.ema_momentum = float(ema_momentum)
        self.ema_momentum_end = float(ema_momentum if ema_momentum_end is None else ema_momentum_end)
        self.view_mask_prob = float(view_mask_prob)
        self.view_mask_prob_end = float(view_mask_prob if view_mask_prob_end is None else view_mask_prob_end)
        self.view_noise_std = float(view_noise_std)
        self.beta_l1 = float(beta_l1)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def _scheduled_value(self, start: float, end: float, epoch: int):
        if self.epochs <= 1:
            return float(end)
        alpha = min(1.0, max(0.0, float(epoch) / float(self.epochs - 1)))
        return float(start + alpha * (end - start))

    def _make_views(self, x: torch.Tensor, mask_prob: float):
        context_mask = (torch.rand_like(x) > mask_prob).float()
        target_mask = (torch.rand_like(x) > mask_prob).float()

        context_noise = torch.randn_like(x) * self.view_noise_std
        target_noise = torch.randn_like(x) * self.view_noise_std

        context_view = x * context_mask + context_noise
        target_view = x * target_mask + target_noise
        return context_view, target_view

    def _step(self, batch, ema_momentum: float, mask_prob: float):
        x = batch['features'].to(self.device)
        context_view, target_view = self._make_views(x, mask_prob=mask_prob)

        z_online, z_pred, z_target = self.model.forward_views(context_view, target_view)
        loss = self.loss_fn(z_pred=z_pred, z_target=z_target.detach(), z_online=z_online)

        if self.beta_l1 > 0.0:
            l1_norm = sum(p.abs().sum() for p in self.model.online_encoder.parameters())
            loss = loss + self.beta_l1 * l1_norm

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.update_target_encoder(momentum=ema_momentum)

        return loss.item()

    def evaluate(self, dl: DataLoader):
        self.model.eval()
        self.loss_fn.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(self.device)
                context_view, target_view = self._make_views(x, mask_prob=self.view_mask_prob_end)
                z_online, z_pred, z_target = self.model.forward_views(context_view, target_view)
                loss = self.loss_fn(z_pred=z_pred, z_target=z_target, z_online=z_online)
                total += float(loss.item())
                n += 1
        return total / max(1, n)

    def train(self, train_dl: DataLoader, val_dl: DataLoader = None):
        best_loss = float('inf')
        patience = 0
        best_state = None
        history = {'train_losses': [], 'val_losses': [], 'best_val_loss': None}

        if val_dl is None:
            val_dl = train_dl

        for epoch in range(self.epochs):
            self.model.train()
            self.loss_fn.train()
            ema_momentum = self._scheduled_value(self.ema_momentum, self.ema_momentum_end, epoch)
            mask_prob = self._scheduled_value(self.view_mask_prob, self.view_mask_prob_end, epoch)

            train_total = 0.0
            train_batches = 0
            for batch in train_dl:
                train_total += self._step(batch, ema_momentum=ema_momentum, mask_prob=mask_prob)
                train_batches += 1
            avg_train = train_total / max(1, train_batches)
            history['train_losses'].append(avg_train)

            val_loss = self.evaluate(val_dl)
            history['val_losses'].append(val_loss)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                best_state = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            history['best_val_loss'] = best_loss

        return history
