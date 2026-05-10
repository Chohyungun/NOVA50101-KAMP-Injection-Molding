"""SVM model configs for Phase 3 baseline.

SVM uses class_weight='balanced' instead of SMOTE to avoid O(n²) cost
on the SMOTE-expanded training set (~12K → 45 fits × ~60s = impractical).
Training on original ~6.4K rows with class_weight is equivalent
and takes ~1.5 min total. Decision logged in decisions.md (D-009).
"""
from __future__ import annotations

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

SVM_LINEAR_GRID: list[dict] = [
    {"C": c} for c in [0.1, 1, 10]
]

SVM_RBF_GRID: list[dict] = [
    {"C": c, "gamma": g}
    for c in [0.1, 1, 10]
    for g in [0.001, 0.01, 0.1]
]


def make_svm_linear(params: dict, calibrate: bool = True):
    """LinearSVC with Platt calibration for probability output."""
    clf = LinearSVC(
        C=params["C"],
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    if calibrate:
        return CalibratedClassifierCV(clf, cv=3, method="sigmoid")
    return clf


def make_svm_rbf(params: dict) -> SVC:
    return SVC(
        kernel="rbf",
        C=params["C"],
        gamma=params["gamma"],
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
