"""
Reproducibility utilities for seeding and environment control.
"""

import contextlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # For PyTorch >= 1.12
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True)


def get_device(device_config: str = "auto") -> torch.device:
    """Get the appropriate torch device."""
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """Initialize worker seeds for DataLoader workers."""
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
