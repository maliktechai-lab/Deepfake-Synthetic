"""
Checkpointing utilities for model saving and loading.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from deepfake_forensics.utils.reproducibility import get_device


def _get_git_commit() -> str | None:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_environment_info() -> dict[str, Any]:
    """Collect environment information for reproducibility."""
    info = {
        "torch_version": torch.__version__,
        "torchvision_version": "unknown",
        "cuda_version": None,
        "python_version": sys.version.split()[0],
        "git_commit": _get_git_commit(),
    }
    
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except Exception:
        pass
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
    
    return info


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    config: DictConfig | None = None,
    experiment_id: str = "",
    manifest_hash: str = "",
    seed: int | None = None,
) -> None:
    """Save model checkpoint with full training state and environment metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "experiment_id": experiment_id,
        "manifest_hash": manifest_hash,
        "seed": seed,
        "environment": _get_environment_info(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = OmegaConf.to_container(config, resolve=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load model checkpoint and restore training state."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = device or get_device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "experiment_id": checkpoint.get("experiment_id", ""),
        "manifest_hash": checkpoint.get("manifest_hash", ""),
        "config": checkpoint.get("config", None),
    }


def load_model_only(
    path: str | Path,
    model: nn.Module,
    device: torch.device | None = None,
) -> None:
    """Load only model weights (for inference)."""
    device = device or get_device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume it's a raw state dict
        model.load_state_dict(checkpoint)


def get_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Get the latest checkpoint file by modification time."""
    path = Path(checkpoint_dir)
    if not path.exists():
        return None

    checkpoints = list(path.glob("*.pt")) + list(path.glob("*.pth"))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)
