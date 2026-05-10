"""Data loading, fold splitting, and split persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = _PROJECT_ROOT / "data" / "splits"

DatasetName = Literal[
    "labeled_data",
    "moldset_labeled",
    "unlabeled_data",
    "supervised_label_cn7",
    "moldset_labeled_cn7",
    "moldset_unlabeled_cn7",
    "moldset_labeled_rg3",
    "moldset_unlabeled_rg3",
]

# ---------------------------------------------------------------------------
# PassOrFail unification helpers
# ---------------------------------------------------------------------------
_PASSORFAIL_STR_MAP = {"Y": 0, "N": 1}  # labeled_data: Y=양품(0), N=불량(1)


def _unify_passorfail(df: pd.DataFrame) -> pd.DataFrame:
    """Convert PassOrFail to int (0=양품/pass, 1=불량/fail) in-place."""
    if "PassOrFail" not in df.columns:
        return df
    col = df["PassOrFail"]
    # Use sample value to detect string encoding (robust across pandas 1.x / 2.x)
    first_val = col.iloc[0] if len(col) > 0 else None
    if isinstance(first_val, str):
        df["PassOrFail"] = col.map(_PASSORFAIL_STR_MAP).astype(int)
    else:
        # numeric files already use 0=양품, 1=불량
        df["PassOrFail"] = col.astype(int)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(name: DatasetName) -> pd.DataFrame:
    """Load a raw CSV by dataset name.

    PassOrFail is normalised to int (0=양품, 1=불량) where present.
    """
    path = RAW_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    # Drop unnamed index column produced by some exports
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return _unify_passorfail(df)


def get_fold(
    fold: int,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_tr, X_va, y_tr, y_va) for the given fold (0-indexed).

    Uses pre-saved split indices when available; otherwise falls back to
    StratifiedKFold with random_state=42.
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    fold_path = SPLITS_DIR / f"fold_{fold}.npy"
    if fold_path.exists():
        indices = np.load(fold_path, allow_pickle=True).item()
        tr_idx, va_idx = indices["train"], indices["val"]
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(X_arr, y_arr))
        tr_idx, va_idx = splits[fold]

    return X_arr[tr_idx], X_arr[va_idx], y_arr[tr_idx], y_arr[va_idx]


def generate_splits(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> None:
    """Persist 5-fold StratifiedKFold indices to data/splits/.

    Idempotent: skips if fold_0.npy already exists.
    """
    fold_path = SPLITS_DIR / "fold_0.npy"
    if fold_path.exists():
        return

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (tr_idx, va_idx) in enumerate(skf.split(X_arr, y_arr)):
        np.save(
            SPLITS_DIR / f"fold_{i}.npy",
            {"train": tr_idx, "val": va_idx},
        )
    print(f"Saved 5-fold split indices to {SPLITS_DIR}")
