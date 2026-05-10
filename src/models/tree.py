"""Tree / ensemble model configs for Phase 4."""
from __future__ import annotations

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

RF_GRID: list[dict] = [
    {"n_estimators": n, "max_depth": d, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
    for n in [100, 300, 500]
    for d in [None, 10]
]

ADABOOST_GRID: list[dict] = [
    {"n_estimators": n, "learning_rate": lr, "random_state": 42}
    for n in [50, 100, 200]
    for lr in [0.5, 1.0]
]

GBM_GRID: list[dict] = [
    {"n_estimators": n, "learning_rate": lr, "max_depth": d, "random_state": 42}
    for n in [100, 200]
    for lr in [0.05, 0.1]
    for d in [3, 5]
]


def make_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def make_adaboost(params: dict) -> AdaBoostClassifier:
    # class_weight on base estimator handles imbalance without SMOTE
    base = DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42)
    return AdaBoostClassifier(estimator=base, **params)


def make_gbm(params: dict) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(**params)
