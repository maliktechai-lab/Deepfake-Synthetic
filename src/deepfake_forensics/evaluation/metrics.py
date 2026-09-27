"""
Evaluation metrics for deepfake forensics.

Implements: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR
"""

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute all binary classification metrics.

    Args:
        logits: Raw model outputs (logits), shape (N, 1) or (N,)
        labels: Ground truth labels (0 or 1), shape (N,) or (N, 1)
        threshold: Decision threshold for converting logits to predictions

    Returns:
        Dictionary with all metrics
    """
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Flatten
    logits = logits.flatten()
    labels = labels.flatten().astype(int)

    # Convert logits to probabilities (sigmoid)
    probs = 1 / (1 + np.exp(-logits))

    # Predictions at threshold
    preds = (probs >= threshold).astype(int)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    # Metrics
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        "pr_auc": average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,  # False Positive Rate
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,   # False Negative Rate
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "threshold": threshold,
    }

    return metrics


def compute_metrics_at_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute metrics across multiple thresholds for threshold analysis."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    logits = logits.flatten()
    labels = labels.flatten().astype(int)
    probs = 1 / (1 + np.exp(-logits))

    results = {
        "thresholds": thresholds.tolist(),
        "precision": [],
        "recall": [],
        "f1": [],
        "fpr": [],
        "fnr": [],
        "accuracy": [],
    }

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        results["precision"].append(precision_score(labels, preds, zero_division=0))
        results["recall"].append(recall_score(labels, preds, zero_division=0))
        results["f1"].append(f1_score(labels, preds, zero_division=0))
        results["fpr"].append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        results["fnr"].append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
        results["accuracy"].append(accuracy_score(labels, preds))

    return results


def find_best_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "f1",
) -> tuple[float, float]:
    """Find the threshold that maximizes a given metric."""
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    logits = logits.flatten()
    labels = labels.flatten().astype(int)
    probs = 1 / (1 + np.exp(-logits))

    thresholds = np.linspace(0.0, 1.0, 1001)
    best_thresh = 0.5
    best_score = -1.0

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(labels, preds, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(labels, preds)
        elif metric == "precision":
            score = precision_score(labels, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(labels, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    """Format metrics for logging."""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return " | ".join(lines)
