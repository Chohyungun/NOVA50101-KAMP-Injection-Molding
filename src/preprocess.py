"""Preprocessing helpers — scaler, resampler, feature filtering.

All fit() calls must happen inside a CV fold to prevent data leakage.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

ScalerName = Literal["standard", "robust"]
ResamplerName = Literal["none", "class_weight", "smote", "adasyn", "undersample"]

# ---------------------------------------------------------------------------
# Zero-variance columns identified in EDA (labeled_data)
# ---------------------------------------------------------------------------
ZERO_VAR_COLS = [
    "Mold_Temperature_1", "Mold_Temperature_2",
    "Mold_Temperature_5", "Mold_Temperature_6",
    "Mold_Temperature_7", "Mold_Temperature_8",
    "Mold_Temperature_9", "Mold_Temperature_10",
    "Mold_Temperature_11", "Mold_Temperature_12",
    "Barrel_Temperature_7",  # near-zero: 2 unique values, mean≈0
]

# Metadata / non-feature columns to drop before modelling
META_COLS = [
    "_id", "TimeStamp", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL",
    "PART_NAME", "EQUIP_CD", "EQUIP_NAME", "Reason", "PART_NO",
    "ERR_FACT_QTY",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature column names, excluding label and zero-variance."""
    drop = set(ZERO_VAR_COLS + META_COLS + ["PassOrFail"])
    return [c for c in df.select_dtypes("number").columns if c not in drop]


def drop_zero_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Drop zero-variance and metadata columns; return feature-only DataFrame."""
    drop = set(ZERO_VAR_COLS + META_COLS)
    existing_drop = [c for c in drop if c in df.columns]
    return df.drop(columns=existing_drop, errors="ignore")


def get_scaler(name: ScalerName):
    """Return a fresh (unfitted) scaler instance."""
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler: {name!r}")


def get_resampler(name: ResamplerName):
    """Return a fresh (unfitted) resampler or None.

    'none' and 'class_weight' return None (no separate resampling step).
    'class_weight' is handled by passing class_weight='balanced' to the model.
    """
    if name in ("none", "class_weight"):
        return None
    if name == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=42, k_neighbors=5)
    if name == "adasyn":
        from imblearn.over_sampling import ADASYN
        return ADASYN(random_state=42)
    if name == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(random_state=42)
    raise ValueError(f"Unknown resampler: {name!r}")


def fit_transform_fold(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    y_tr: np.ndarray,
    scaler_name: ScalerName,
    resample_name: ResamplerName,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply scaler (fit on train) then resampler (train only).

    Returns (X_tr_out, X_va_out, y_tr_out).
    Scaler is fit on X_tr; resampler is fit on scaled X_tr.
    X_va is only transformed, never fit.
    """
    scaler = get_scaler(scaler_name)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    resampler = get_resampler(resample_name)
    if resampler is not None:
        X_tr_s, y_tr = resampler.fit_resample(X_tr_s, y_tr)

    return X_tr_s, X_va_s, y_tr
