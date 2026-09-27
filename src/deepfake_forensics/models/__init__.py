"""
Model construction utilities for deepfake forensics.
"""

from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch import nn

from deepfake_forensics.branches import SpatialBranch, create_branch
from deepfake_forensics.branches.contract import BranchOutput


class ForensicModel(nn.Module):
    """
    Main forensic model wrapper.

    For EXP-S01, this wraps the SpatialBranch.
    In the future, this will coordinate multiple branches.
    """

    def __init__(
        self,
        branch_name: str = "spatial",
        branch_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.branch_name = branch_name
        self.branch = create_branch(branch_name, **(branch_config or {}))

    def forward(self, x: torch.Tensor) -> BranchOutput:
        """Forward pass through the active branch."""
        return self.branch(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from the branch."""
        if hasattr(self.branch, "get_embedding"):
            return self.branch.get_embedding(x)
        # Fallback: run forward and return embedding
        return self.forward(x).embedding

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get classification logits from the branch."""
        if hasattr(self.branch, "get_logits"):
            return self.branch.get_logits(x)
        return self.forward(x).evidence_score


def build_model(config: DictConfig) -> ForensicModel:
    """
    Build model from configuration.

    Expected config structure:
    model:
      branch: "spatial"
      branch_config:
        pretrained: true
        embedding_dim: 256
        dropout: 0.1
    """
    model_config = config.get("model", {})
    branch_name = model_config.get("branch", "spatial")
    branch_config = model_config.get("branch_config", {})

    return ForensicModel(branch_name=branch_name, branch_config=branch_config)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
