"""
Branch factory for creating forensic branches by name.
"""

from typing import Dict, Type

from torch import nn

from deepfake_forensics.branches.contract import BranchOutput
from deepfake_forensics.branches.spatial import SpatialBranch

# Registry of available branches
BRANCH_REGISTRY: dict[str, type[nn.Module]] = {
    "spatial": SpatialBranch,
}


def create_branch(branch_name: str, **kwargs) -> nn.Module:
    """Create a branch instance by name."""
    if branch_name not in BRANCH_REGISTRY:
        raise ValueError(f"Unknown branch: {branch_name}. Available: {list(BRANCH_REGISTRY.keys())}")
    return BRANCH_REGISTRY[branch_name](**kwargs)


def list_branches() -> list:
    """List available branch names."""
    return list(BRANCH_REGISTRY.keys())


__all__ = [
    "BranchOutput",
    "SpatialBranch",
    "create_branch",
    "list_branches",
    "BRANCH_REGISTRY",
]
