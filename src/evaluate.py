"""Evaluation metrics — ROC-AUC, PR-AUC, F1, Precision@Recall."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    recall_targets: tuple[float, ...] = (0.90, 0.99),
) -> dict[str, float]:
    """Compute standard evaluation metrics.

    Returns a dict with keys:
      roc_auc, pr_auc, f1,
      prec_at_rec_{int(t*100)} for each t in recall_targets.
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc":  float(average_precision_score(y_true, y_prob)),
        "f1":      float(f1_score(y_true, y_pred, zero_division=0)),
    }
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    for t in recall_targets:
        mask = rec >= t
        key = f"prec_at_rec{int(t * 100)}"
        metrics[key] = float(prec[mask][-1]) if mask.any() else 0.0
    return metrics


def aggregate_fold_metrics(
    fold_results: list[dict[str, float]],
) -> dict[str, float]:
    """Average and std across folds.

    Returns flat dict: {metric_mean: ..., metric_std: ...}
    """
    keys = fold_results[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in fold_results]
        out[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        out[f"{k}_std"]  = round(float(np.std(vals)),  4)
    return out
