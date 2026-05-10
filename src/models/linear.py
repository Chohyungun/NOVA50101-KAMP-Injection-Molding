"""Linear / generative model configs for Phase 3 baseline."""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Hyperparameter grids  (each entry is one model variant)
# ---------------------------------------------------------------------------

LR_L1_GRID: list[dict] = [
    {"C": c, "penalty": "l1", "solver": "saga", "max_iter": 3000, "random_state": 42}
    for c in [0.01, 0.1, 1, 10]
]

LR_L2_GRID: list[dict] = [
    {"C": c, "penalty": "l2", "solver": "saga", "max_iter": 3000, "random_state": 42}
    for c in [0.01, 0.1, 1, 10]
]

LDA_GRID: list[dict] = [
    {"solver": "lsqr", "shrinkage": "auto"},
]

QDA_GRID: list[dict] = [
    {"reg_param": rp}
    for rp in [0.0, 0.01, 0.1]
]


def make_lr(params: dict) -> LogisticRegression:
    """Instantiate LR from a grid-entry dict."""
    return LogisticRegression(**params)


def make_lda(params: dict) -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(**params)


def make_qda(params: dict) -> QuadraticDiscriminantAnalysis:
    return QuadraticDiscriminantAnalysis(**params)
