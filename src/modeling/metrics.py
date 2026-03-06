"""Metrics used by all experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true: Any, y_score: Any) -> tuple[float, float]:
    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = np.nan
    return auroc, pr_auc


def _specificity(y_true: Any, y_pred: Any) -> float:
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            return np.nan
        tn, fp, _, _ = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except Exception:
        return np.nan


def calculate_binary_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
    model_name: str,
    fold: int | None = None,
    completeness: float | None = None,
) -> dict[str, Any]:
    """Binary metrics block."""
    auroc, pr_auc = _safe_auc(y_true, y_score)
    metrics: dict[str, Any] = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": _specificity(y_true, y_pred),
        "AUROC": auroc,
        "PR-AUC": pr_auc,
    }
    if fold is not None:
        metrics["Fold"] = fold
    if completeness is not None:
        metrics["Completeness"] = completeness
    return metrics
