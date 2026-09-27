"""
Tests for evaluation metrics.
"""

import numpy as np
import pytest
import torch

from deepfake_forensics.evaluation.metrics import (
    compute_binary_metrics,
    compute_metrics_at_thresholds,
    find_best_threshold,
    format_metrics,
)


class TestMetrics:
    """Test metric computation."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        logits = torch.tensor([[10.0], [-10.0], [10.0], [-10.0]])  # Strong predictions
        labels = torch.tensor([[1], [0], [1], [0]])

        metrics = compute_binary_metrics(logits, labels, threshold=0.5)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["pr_auc"] == 1.0
        assert metrics["fpr"] == 0.0
        assert metrics["fnr"] == 0.0

    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        logits = torch.tensor([[-10.0], [10.0], [-10.0], [10.0]])  # Wrong predictions
        labels = torch.tensor([[1], [0], [1], [0]])

        metrics = compute_binary_metrics(logits, labels, threshold=0.5)

        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["roc_auc"] == 0.0

    def test_all_positive_predictions(self):
        """Test metrics when all predictions are positive."""
        logits = torch.tensor([[10.0], [10.0], [10.0], [10.0]])
        labels = torch.tensor([[1], [1], [0], [0]])

        metrics = compute_binary_metrics(logits, labels, threshold=0.5)

        assert metrics["tp"] == 2
        assert metrics["fp"] == 2
        assert metrics["tn"] == 0
        assert metrics["fn"] == 0
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 1.0
        assert metrics["fpr"] == 1.0

    def test_all_negative_predictions(self):
        """Test metrics when all predictions are negative."""
        logits = torch.tensor([[-10.0], [-10.0], [-10.0], [-10.0]])
        labels = torch.tensor([[1], [1], [0], [0]])

        metrics = compute_binary_metrics(logits, labels, threshold=0.5)

        assert metrics["tp"] == 0
        assert metrics["fp"] == 0
        assert metrics["tn"] == 2
        assert metrics["fn"] == 2
        assert metrics["recall"] == 0.0
        assert metrics["fnr"] == 1.0

    def test_single_class_labels(self):
        """Test metrics when labels only have one class."""
        logits = torch.randn(10, 1)
        labels = torch.zeros(10, 1)  # All authentic

        metrics = compute_binary_metrics(logits, labels, threshold=0.5)
        # ROC-AUC and PR-AUC should be 0 when only one class present
        assert metrics["roc_auc"] == 0.0
        assert metrics["pr_auc"] == 0.0

    def test_numpy_inputs(self):
        """Test that numpy arrays work as inputs."""
        logits = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        labels = np.array([[1], [0], [1], [0]])

        metrics = compute_binary_metrics(logits, labels)
        assert metrics["accuracy"] == 1.0

    def test_flatten_inputs(self):
        """Test that 1D and 2D inputs both work."""
        logits_1d = torch.tensor([10.0, -10.0, 10.0, -10.0])
        logits_2d = torch.tensor([[10.0], [-10.0], [10.0], [-10.0]])
        labels = torch.tensor([1, 0, 1, 0])

        m1 = compute_binary_metrics(logits_1d, labels)
        m2 = compute_binary_metrics(logits_2d, labels)

        assert m1["accuracy"] == m2["accuracy"]


class TestThresholdAnalysis:
    """Test threshold analysis functions."""

    def test_compute_metrics_at_thresholds(self):
        """Test computing metrics across thresholds."""
        logits = torch.tensor([2.0, 1.0, -1.0, -2.0])
        labels = torch.tensor([1, 1, 0, 0])

        results = compute_metrics_at_thresholds(logits, labels)

        assert "thresholds" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert len(results["thresholds"]) == 101
        assert len(results["precision"]) == 101

    def test_find_best_threshold(self):
        """Test finding best threshold for F1."""
        logits = torch.tensor([2.0, 1.0, -1.0, -2.0])
        labels = torch.tensor([1, 1, 0, 0])

        best_thresh, best_f1 = find_best_threshold(logits, labels, metric="f1")

        assert 0.0 <= best_thresh <= 1.0
        assert 0.0 <= best_f1 <= 1.0


class TestFormatMetrics:
    """Test metric formatting."""

    def test_format_metrics(self):
        """Test formatting metrics for logging."""
        metrics = {
            "accuracy": 0.95,
            "f1": 0.92,
            "roc_auc": 0.98,
            "count": 100,
        }
        formatted = format_metrics(metrics, prefix="val/")
        assert "val/accuracy: 0.9500" in formatted
        assert "val/f1: 0.9200" in formatted
        assert "val/roc_auc: 0.9800" in formatted
        assert "val/count: 100" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
