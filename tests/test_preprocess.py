"""pytest — Phase 2 preprocessing validation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import load_raw, get_fold, generate_splits
from preprocess import (
    get_feature_cols,
    fit_transform_fold,
    get_scaler,
    get_resampler,
    ZERO_VAR_COLS,
)
from utils import set_seed

set_seed(42)


@pytest.fixture(scope="module")
def xy():
    df = load_raw("labeled_data")
    feat = get_feature_cols(df)
    X = df[feat].values
    y = df["PassOrFail"].values
    generate_splits(X, y)
    return X, y


def test_get_feature_cols_excludes_zero_var():
    df = load_raw("labeled_data")
    feat = get_feature_cols(df)
    for col in ZERO_VAR_COLS:
        assert col not in feat, f"{col} should be excluded"


def test_get_feature_cols_excludes_serial():
    df = load_raw("labeled_data")
    feat = get_feature_cols(df)
    assert "PART_FACT_SERIAL" not in feat


def test_get_feature_cols_count(xy):
    X, _ = xy
    assert 20 <= X.shape[1] <= 30, f"Expected ~25 features, got {X.shape[1]}"


def test_scaler_fit_on_train_only(xy):
    """Validate that scaler does NOT see val data during fit."""
    X, y = xy
    X_tr, X_va, y_tr, y_va = get_fold(0, X, y)
    X_tr_s, X_va_s, y_tr_r = fit_transform_fold(X_tr, X_va, y_tr, "standard", "none")

    # Scaled train mean should be ~0
    assert abs(X_tr_s.mean()) < 0.1, "Train mean after StandardScaler should be ~0"
    # Val mean will differ (scaled by train stats)
    assert X_tr_s.shape[1] == X_va_s.shape[1]


def test_smote_increases_minority_class(xy):
    X, y = xy
    X_tr, X_va, y_tr, y_va = get_fold(0, X, y)
    _, _, y_tr_r = fit_transform_fold(X_tr, X_va, y_tr, "standard", "smote")

    minority_before = (y_tr == 1).sum()
    minority_after  = (y_tr_r == 1).sum()
    assert minority_after > minority_before, "SMOTE should increase minority count"


def test_val_unchanged_after_smote(xy):
    """Validation set must not be touched by resampler."""
    X, y = xy
    X_tr, X_va, y_tr, y_va = get_fold(0, X, y)
    _, X_va_s, _ = fit_transform_fold(X_tr, X_va, y_tr, "standard", "smote")
    assert X_va_s.shape[0] == X_va.shape[0], "Val size must not change after SMOTE"


def test_preproc_ablation_csv_exists():
    result_path = Path("results/tables/preproc_ablation.csv")
    assert result_path.exists(), "preproc_ablation.csv must exist after ablation run"


def test_preproc_ablation_csv_completeness():
    import pandas as pd
    df = pd.read_csv("results/tables/preproc_ablation.csv")
    assert len(df) == 10, f"Expected 10 rows (2 scalers x 5 resamplers), got {len(df)}"
    for col in ["roc_auc_mean", "pr_auc_mean", "f1_mean"]:
        assert col in df.columns
        assert df[col].notna().all()
