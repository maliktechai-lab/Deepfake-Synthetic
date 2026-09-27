"""
Common branch contract for forensic evidence branches.
All branches must conform to this interface for heterogeneous evidence fusion.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BranchOutput:
    """
    Standardized output structure for all forensic branches.

    Attributes:
        embedding: Learned representation vector (e.g., 256-dim for spatial branch)
        evidence_score: Raw forensic evidence score (logit or probability-like)
        reliability: Branch reliability estimate [0, 1] - placeholder for now
        uncertainty: Branch uncertainty estimate - placeholder for now
        attribution: Spatial/temporal attribution map (optional)
        metadata: Additional branch-specific information
    """
    embedding: torch.Tensor
    evidence_score: torch.Tensor
    reliability: torch.Tensor
    uncertainty: torch.Tensor
    attribution: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tensor shapes and types."""
        if not isinstance(self.embedding, torch.Tensor):
            raise TypeError("embedding must be a torch.Tensor")
        if not isinstance(self.evidence_score, torch.Tensor):
            raise TypeError("evidence_score must be a torch.Tensor")
        if not isinstance(self.reliability, torch.Tensor):
            raise TypeError("reliability must be a torch.Tensor")
        if not isinstance(self.uncertainty, torch.Tensor):
            raise TypeError("uncertainty must be a torch.Tensor")
        if self.attribution is not None and not isinstance(self.attribution, torch.Tensor):
            raise TypeError("attribution must be a torch.Tensor or None")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "embedding": self.embedding.detach().cpu().numpy() if self.embedding is not None else None,
            "evidence_score": self.evidence_score.detach().cpu().numpy() if self.evidence_score is not None else None,
            "reliability": self.reliability.detach().cpu().numpy() if self.reliability is not None else None,
            "uncertainty": self.uncertainty.detach().cpu().numpy() if self.uncertainty is not None else None,
            "attribution": self.attribution.detach().cpu().numpy() if self.attribution is not None else None,
            "metadata": self.metadata,
        }


def create_default_branch_output(
    batch_size: int,
    embedding_dim: int,
    device: torch.device,
) -> BranchOutput:
    """Create a default BranchOutput with placeholder values."""
    return BranchOutput(
        embedding=torch.zeros(batch_size, embedding_dim, device=device),
        evidence_score=torch.zeros(batch_size, 1, device=device),
        reliability=torch.ones(batch_size, 1, device=device),  # Default: fully reliable
        uncertainty=torch.zeros(batch_size, 1, device=device),  # Default: no uncertainty
        attribution=None,
        metadata={},
    )
