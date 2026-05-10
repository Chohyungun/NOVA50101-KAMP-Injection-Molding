"""MLP implementation — PyTorch (강의 L11-15 범위).

Model  : MLP with configurable hidden layers, ReLU, Dropout
Loss   : BCEWithLogitsLoss (weighted for class imbalance)
Optim  : Adam
Loader : torch DataLoader with shuffle
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class MLPTrainer:
    """Thin wrapper for sklearn-style fit/predict_proba interface."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 256,
        random_state: int = 42,
    ):
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_: MLP | None = None
        self.input_dim_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPTrainer":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.input_dim_ = X.shape[1]
        self.model_ = MLP(self.input_dim_, self.hidden_dims, self.dropout)

        # class-weighted loss
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.random_state))

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None, "Call fit() first"
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model_(X_t)
            probs = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs, probs])


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
MLP_GRID: list[dict] = [
    {"hidden_dims": (64,),         "dropout": 0.3, "epochs": 30, "lr": 1e-3},
    {"hidden_dims": (128, 64),     "dropout": 0.3, "epochs": 30, "lr": 1e-3},
    {"hidden_dims": (256, 128, 64),"dropout": 0.3, "epochs": 30, "lr": 1e-3},
]

# Guidebook DNN approximation (§2.3): 3-layer, same architecture
GUIDEBOOK_DNN_PARAMS = {
    "hidden_dims": (128, 64, 32),
    "dropout": 0.0,   # guidebook does not mention Dropout
    "epochs": 50,
    "lr": 1e-3,
}


def make_mlp(params: dict) -> MLPTrainer:
    return MLPTrainer(**params)
