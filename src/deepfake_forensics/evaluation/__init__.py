"""
Evaluation module for deepfake forensics.
"""

from deepfake_forensics.evaluation.calibration import (
    Calibrator,
    NoCalibration,
    TemperatureScaling,
    get_calibrator,
)
from deepfake_forensics.evaluation.inference import evaluate, predict
from deepfake_forensics.evaluation.metrics import (
    compute_binary_metrics,
    compute_metrics_at_thresholds,
    find_best_threshold,
    format_metrics,
)

__all__ = [
    "compute_binary_metrics",
    "compute_metrics_at_thresholds",
    "find_best_threshold",
    "format_metrics",
    "predict",
    "evaluate",
    "Calibrator",
    "NoCalibration",
    "TemperatureScaling",
    "get_calibrator",
]
