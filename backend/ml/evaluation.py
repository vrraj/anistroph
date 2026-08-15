"""Model evaluation — model-independent metrics for binary classification.

Emphasizes PR-AUC, recall, and precision for imbalanced/rare-event problems.
Supports configurable decision thresholds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
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
    if has_both:
        p, r, t = precision_recall_curve(y_true, y_proba)
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
