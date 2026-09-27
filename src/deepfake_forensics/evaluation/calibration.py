"""
Calibration infrastructure for deepfake forensics.

This module provides the interface for calibration methods.
Sophisticated calibration is NOT part of S01 training - this is infrastructure only.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class Calibrator(ABC):
    """Abstract base class for calibration methods."""

    @abstractmethod
    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "Calibrator":
        """Fit calibrator on validation logits and labels."""
        pass

    @abstractmethod
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply calibration to logits, returning calibrated probabilities."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save calibrator state."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Calibrator":
        """Load calibrator from file."""
        pass


class NoCalibration(Calibrator):
    """Identity calibration - returns sigmoid(logits) as-is."""

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "NoCalibration":
        return self

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "NoCalibration":
        return cls()


class TemperatureScaling(Calibrator):
    """
    Temperature scaling calibration (Guo et al., 2017).

    Learns a single temperature parameter T > 0 to scale logits:
    calibrated_prob = sigmoid(logits / T)
    """

    def __init__(self, init_temperature: float = 1.0, max_iter: int = 100, lr: float = 0.01):
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.max_iter = max_iter
        self.lr = lr
        self._fitted = False

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> "TemperatureScaling":
        """Fit temperature using NLL minimization on validation set."""
        logits = logits.flatten()
        labels = labels.flatten().float()

        optimizer = torch.optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._fitted = True
        return self

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("Calibrator must be fitted before use")
        return torch.sigmoid(logits / self.temperature)

    def save(self, path: str) -> None:
        torch.save({"temperature": self.temperature.item()}, path)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaling":
        state = torch.load(path)
        calibrator = cls()
        calibrator.temperature = nn.Parameter(torch.tensor(state["temperature"]))
        calibrator._fitted = True
        return calibrator


def get_calibrator(name: str = "none", **kwargs) -> Calibrator:
    """Get calibrator by name."""
    calibrators = {
        "none": NoCalibration,
        "temperature": TemperatureScaling,
    }
    if name not in calibrators:
        raise ValueError(f"Unknown calibrator: {name}. Available: {list(calibrators.keys())}")
    return calibrators[name](**kwargs)


# Placeholder for future calibration methods
# class IsotonicRegression(Calibrator): ...
# class PlattScaling(Calibrator): ...
# class DirichletCalibration(Calibrator): ...
