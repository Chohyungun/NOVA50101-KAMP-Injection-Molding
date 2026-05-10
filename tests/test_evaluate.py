"""pytest — evaluate module validation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import compute_metrics, aggregate_fold_metrics


def _make_binary(n=200, fail_rate=0.1, seed=42):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fail_rate).astype(int)
    prob = rng.random(n)
    prob[y == 1] += 0.4  # bias toward fail class
    prob = np.clip(prob, 0, 1)
    return y, prob


def test_compute_metrics_keys():
    y, prob = _make_binary()
    m = compute_metrics(y, prob)
    assert "roc_auc" in m
    assert "pr_auc" in m
    assert "f1" in m
    assert "prec_at_rec90" in m
    assert "prec_at_rec99" in m


def test_compute_metrics_roc_range():
    y, prob = _make_binary()
    m = compute_metrics(y, prob)
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_compute_metrics_pr_range():
    y, prob = _make_binary()
    m = compute_metrics(y, prob)
    assert 0.0 <= m["pr_auc"] <= 1.0


def test_aggregate_fold_metrics_shape():
    folds = [compute_metrics(*_make_binary(seed=i)) for i in range(5)]
    agg = aggregate_fold_metrics(folds)
    for key in ["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"]:
        assert key in agg, f"Missing key: {key}"


def test_aggregate_std_non_negative():
    folds = [compute_metrics(*_make_binary(seed=i)) for i in range(5)]
    agg = aggregate_fold_metrics(folds)
    for k, v in agg.items():
        if k.endswith("_std"):
            assert v >= 0, f"{k} std should be non-negative"


def test_baseline_results_csv_exists():
    path = Path("results/tables/baseline_results.csv")
    assert path.exists(), "baseline_results.csv must exist"


def test_baseline_results_csv_six_models():
    import pandas as pd
    df = pd.read_csv("results/tables/baseline_results.csv")
    assert len(df) == 6, f"Expected 6 model rows, got {len(df)}"
    assert "roc_auc_mean" in df.columns
    assert df["roc_auc_mean"].min() > 0.5
