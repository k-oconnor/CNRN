from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TARNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.y0_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.y1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.shared(x)
        y0 = self.y0_head(z)
        y1 = self.y1_head(z)
        return y0, y1


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 40
    device: str = "cpu"
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.0
    balance_treatments: bool = False


def factual_mse_loss(
    y0: torch.Tensor,
    y1: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    balance_treatments: bool = False,
) -> torch.Tensor:
    pred = t * y1 + (1.0 - t) * y0
    sq_err = (pred - y) ** 2
    if not balance_treatments:
        return torch.mean(sq_err)

    treated = t >= 0.5
    control = ~treated
    treated_loss = sq_err[treated].mean() if treated.any() else None
    control_loss = sq_err[control].mean() if control.any() else None
    if treated_loss is None:
        return control_loss
    if control_loss is None:
        return treated_loss
    return 0.5 * (treated_loss + control_loss)


def fit_tarnet(
    X_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    t_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> TARNet:
    device = torch.device(config.device)
    model = TARNet(
        input_dim=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        dropout=config.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(t_train).float().reshape(-1, 1),
        torch.from_numpy(y_train).float().reshape(-1, 1),
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    Xv = torch.from_numpy(X_val).float().to(device)
    tv = torch.from_numpy(t_val).float().reshape(-1, 1).to(device)
    yv = torch.from_numpy(y_val).float().reshape(-1, 1).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    plateau = 0

    for _ in range(config.epochs):
        model.train()
        for xb, tb, yb in loader:
            xb = xb.to(device)
            tb = tb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            y0, y1 = model(xb)
            loss = factual_mse_loss(y0, y1, tb, yb, balance_treatments=config.balance_treatments)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            y0v, y1v = model(Xv)
            val_loss = factual_mse_loss(y0v, y1v, tv, yv, balance_treatments=config.balance_treatments).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            plateau = 0
        else:
            plateau += 1
            if plateau >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_tarnet(model: TARNet, X: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X).float().to(device)
        y0, y1 = model(xt)
    return y0.cpu().numpy().reshape(-1), y1.cpu().numpy().reshape(-1)
