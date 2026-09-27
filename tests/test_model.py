"""
Tests for model architecture and branch contract.
"""

import pytest
import torch
from omegaconf import OmegaConf

from deepfake_forensics.branches import SpatialBranch, create_branch, list_branches
from deepfake_forensics.branches.contract import BranchOutput, create_default_branch_output
from deepfake_forensics.models import ForensicModel, build_model, count_parameters


class TestBranchContract:
    """Test the BranchOutput contract."""

    def test_branch_output_creation(self):
        """Test creating a valid BranchOutput."""
        batch_size = 4
        embedding_dim = 256
        device = torch.device("cpu")

        output = create_default_branch_output(batch_size, embedding_dim, device)

        assert output.embedding.shape == (batch_size, embedding_dim)
        assert output.evidence_score.shape == (batch_size, 1)
        assert output.reliability.shape == (batch_size, 1)
        assert output.uncertainty.shape == (batch_size, 1)
        assert output.attribution is None
        assert isinstance(output.metadata, dict)

    def test_branch_output_validation(self):
        """Test that BranchOutput validates tensor types."""
        with pytest.raises(TypeError):
            BranchOutput(
                embedding="not a tensor",
                evidence_score=torch.zeros(1, 1),
                reliability=torch.ones(1, 1),
                uncertainty=torch.zeros(1, 1),
            )

    def test_branch_output_to_dict(self):
        """Test converting BranchOutput to dict."""
        output = create_default_branch_output(2, 128, torch.device("cpu"))
        d = output.to_dict()
        assert "embedding" in d
        assert "evidence_score" in d
        assert "reliability" in d
        assert "uncertainty" in d
        assert "metadata" in d


class TestSpatialBranch:
    """Test the SpatialBranch (ConvNeXt-Tiny) implementation."""

    def test_spatial_branch_creation(self):
        """Test creating a SpatialBranch."""
        branch = SpatialBranch(pretrained=False, embedding_dim=256, dropout=0.1)
        assert isinstance(branch, SpatialBranch)

    def test_spatial_branch_forward(self):
        """Test forward pass shape."""
        branch = SpatialBranch(pretrained=False, embedding_dim=256)
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)

        output = branch(x)

        assert isinstance(output, BranchOutput)
        assert output.embedding.shape == (batch_size, 256)
        assert output.evidence_score.shape == (batch_size, 1)
        assert output.reliability.shape == (batch_size, 1)
        assert output.uncertainty.shape == (batch_size, 1)
        assert output.metadata["branch"] == "spatial"
        assert output.metadata["backbone"] == "convnext_tiny"

    def test_spatial_branch_embedding_dim(self):
        """Test different embedding dimensions."""
        for dim in [128, 256, 512]:
            branch = SpatialBranch(pretrained=False, embedding_dim=dim)
            x = torch.randn(1, 3, 224, 224)
            output = branch(x)
            assert output.embedding.shape == (1, dim)

    def test_get_embedding(self):
        """Test extracting embedding only."""
        branch = SpatialBranch(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        emb = branch.get_embedding(x)
        assert emb.shape == (1, 256)

    def test_get_logits(self):
        """Test extracting logits only."""
        branch = SpatialBranch(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        logits = branch.get_logits(x)
        assert logits.shape == (1, 1)


class TestBranchFactory:
    """Test branch factory functions."""

    def test_list_branches(self):
        """Test listing available branches."""
        branches = list_branches()
        assert "spatial" in branches

    def test_create_branch(self):
        """Test creating branch by name."""
        branch = create_branch("spatial", pretrained=False)
        assert isinstance(branch, SpatialBranch)

    def test_create_unknown_branch(self):
        """Test creating unknown branch raises error."""
        with pytest.raises(ValueError, match="Unknown branch"):
            create_branch("nonexistent")


class TestForensicModel:
    """Test the ForensicModel wrapper."""

    def test_model_creation(self):
        """Test creating ForensicModel."""
        config = OmegaConf.create({
            "model": {
                "branch": "spatial",
                "branch_config": {"pretrained": False, "embedding_dim": 256}
            }
        })
        model = build_model(config)
        assert isinstance(model, ForensicModel)

    def test_model_forward(self):
        """Test model forward pass."""
        config = OmegaConf.create({
            "model": {
                "branch": "spatial",
                "branch_config": {"pretrained": False}
            }
        })
        model = build_model(config)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert isinstance(output, BranchOutput)
        assert output.embedding.shape == (2, 256)

    def test_count_parameters(self):
        """Test parameter counting."""
        config = OmegaConf.create({
            "model": {
                "branch": "spatial",
                "branch_config": {"pretrained": False}
            }
        })
        model = build_model(config)
        counts = count_parameters(model)
        assert "total" in counts
        assert "trainable" in counts
        assert "frozen" in counts
        assert counts["total"] == counts["trainable"] + counts["frozen"]
        assert counts["trainable"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
