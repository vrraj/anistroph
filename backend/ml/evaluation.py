"""Model evaluation — model-independent metrics for classification and regression.

Classification: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
Regression: MAE, RMSE, R², median absolute error, 95th percentile absolute error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate binary classification predictions.

    Args:
        y_true: ground-truth labels (0/1).
        y_proba: predicted probability of the positive class.
        threshold: decision threshold for class labels.

    Returns a metrics dict with ROC-AUC, PR-AUC, precision, recall, F1,
    confusion matrix, and the threshold used.
    """
    y_pred = (y_proba >= threshold).astype(int)

    # Guard against single-class edge cases.
    has_both = len(np.unique(y_true)) > 1

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if has_both else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if has_both else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    # Precision-recall curve points for plotting.
    # Downsample to at most 200 points — a PR curve with 200 points is visually
    # indistinguishable from one with 100K, but avoids bloating the model
    # registry (and MCP list_models responses) with multi-MB payloads.
    if has_both:
        p, r, t = precision_recall_curve(y_true, y_proba)
        max_points = 200
        if len(p) > max_points:
            step = len(p) // max_points
            p = p[::step]
            r = r[::step]
            t = t[::step]
        metrics["pr_curve"] = {
            "precision": p.tolist(),
            "recall": r.tolist(),
            "thresholds": t.tolist(),
        }

    return metrics


def best_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find the decision threshold that maximizes F1 score."""
    has_both = len(np.unique(y_true)) > 1
    if not has_both:
        return 0.5
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1s = 2 * p * r / (p + r + 1e-12)
    idx = int(np.nanargmax(f1s[:-1]))
    return float(t[idx]) if idx < len(t) else 0.5


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: float | None = None,
) -> dict[str, Any]:
    """Evaluate regression predictions.

    Args:
        y_true: ground-truth values.
        y_pred: predicted values.
        baseline_pred: if provided, compute baseline metrics using a
            constant predictor that always returns this value (e.g. the
            mean training yield).

    Returns a metrics dict with MAE, RMSE, R², median absolute error,
    95th percentile absolute error, and baseline comparison.
    """
    abs_errors = np.abs(y_true - y_pred)
    mse = float(mean_squared_error(y_true, y_pred))

    # MAPE — guard against zero actuals to avoid division-by-zero.
    non_zero_mask = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
        if non_zero_mask.any()
        else None
    )

    metrics: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
        "median_abs_error": float(np.median(abs_errors)),
        "p95_abs_error": float(np.percentile(abs_errors, 95)),
        "max_error": float(np.max(abs_errors)),
        "mean_prediction_error": float(np.mean(y_pred - y_true)),
    }

    if baseline_pred is not None:
        baseline_abs_errors = np.abs(y_true - baseline_pred)
        metrics["baseline"] = {
            "constant_value": float(baseline_pred),
            "mae": float(mean_absolute_error(y_true, np.full_like(y_true, baseline_pred))),
            "rmse": float(np.sqrt(mean_squared_error(y_true, np.full_like(y_true, baseline_pred)))),
            "r2": float(r2_score(y_true, np.full_like(y_true, baseline_pred))),
        }

    return metrics
