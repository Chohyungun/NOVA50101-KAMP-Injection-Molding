"""pytest — Phase 1 data validation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import (
    DatasetName,
    RAW_DIR,
    SPLITS_DIR,
    generate_splits,
    get_fold,
    load_raw,
)
from utils import set_seed

set_seed(42)

ALL_DATASETS: list[DatasetName] = [
    "labeled_data",
    "moldset_labeled",
    "unlabeled_data",
    "supervised_label_cn7",
    "moldset_labeled_cn7",
    "moldset_unlabeled_cn7",
    "moldset_labeled_rg3",
    "moldset_unlabeled_rg3",
]


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_load_raw_all_csvs(name: DatasetName) -> None:
    """All 8 CSV files must load without error."""
    df = load_raw(name)
    assert len(df) > 0, f"{name} loaded empty"


def test_labeled_data_passorfail_column() -> None:
    """labeled_data must contain PassOrFail column."""
    df = load_raw("labeled_data")
    assert "PassOrFail" in df.columns


def test_labeled_data_passorfail_binary() -> None:
    """PassOrFail must be normalised to {0, 1} integers."""
    df = load_raw("labeled_data")
    vals = set(df["PassOrFail"].unique())
    assert vals <= {0, 1}, f"Unexpected PassOrFail values: {vals}"


def test_total_row_count_near_886k() -> None:
    """Total rows across all 8 CSVs must be within 1% of 886,227."""
    total = sum(len(load_raw(n)) for n in ALL_DATASETS)
    assert abs(total - 886_227) / 886_227 < 0.01, f"Total rows: {total}"


def test_labeled_data_shape() -> None:
    """labeled_data must have at least 7000 rows and 20+ columns."""
    df = load_raw("labeled_data")
    assert df.shape[0] >= 7000
    assert df.shape[1] >= 20


def test_generate_and_get_fold_stratification() -> None:
    """5-fold stratification must keep class ratio within 1 pp across folds."""
    import pandas as pd

    df = load_raw("labeled_data")
    feat_cols = [
        c for c in df.select_dtypes("number").columns if c != "PassOrFail"
    ]
    X = df[feat_cols].values
    y = df["PassOrFail"].values

    generate_splits(X, y)

    global_rate = y.mean()
    for fold in range(5):
        _, _, _, y_va = get_fold(fold, X, y)
        fold_rate = y_va.mean()
        assert abs(fold_rate - global_rate) < 0.01, (
            f"Fold {fold}: val fail rate {fold_rate:.4f} deviates from "
            f"global {global_rate:.4f} by >{1:.0f}pp"
        )


def test_split_files_persisted() -> None:
    """fold_0.npy through fold_4.npy must exist after generate_splits."""
    df = load_raw("labeled_data")
    feat_cols = [
        c for c in df.select_dtypes("number").columns if c != "PassOrFail"
    ]
    X = df[feat_cols].values
    y = df["PassOrFail"].values
    generate_splits(X, y)

    for i in range(5):
        assert (SPLITS_DIR / f"fold_{i}.npy").exists(), f"fold_{i}.npy missing"
