"""Stacking 메타 학습기 — Phase 5.

Base learners: LR-L2, QDA, RF, MLP, GBM (Phase 3/4 best)
Meta-learner : LR-L2
OOF 예측으로만 메타 피처 생성 → data leakage 방지.
"""
from __future__ import annotations

from typing import Callable
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_base_configs() -> list[tuple[str, Callable]]:
    """Return (name, factory_fn) for each base learner."""
    from models.nn import make_mlp

    return [
        ("LR_L2",
         lambda: LogisticRegression(
             penalty="l2", C=10, solver="saga", max_iter=3000, random_state=42)),
        ("QDA",
         lambda: QuadraticDiscriminantAnalysis(reg_param=0.01)),
        ("RF",
         lambda: RandomForestClassifier(
             n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("GBM",
         lambda: GradientBoostingClassifier(
             n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ("MLP",
         lambda: make_mlp(
             {"hidden_dims": (256, 128, 64), "dropout": 0.3,
              "epochs": 30, "lr": 1e-3, "random_state": 42})),
    ]


def get_meta_learner() -> LogisticRegression:
    return LogisticRegression(penalty="l2", C=1.0, solver="saga",
                               max_iter=2000, random_state=42)
