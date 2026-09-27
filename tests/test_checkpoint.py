"""
Tests for checkpoint save/load functionality.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from deepfake_forensics.models import ForensicModel
from deepfake_forensics.training.checkpoint import (
    get_latest_checkpoint,
    load_checkpoint,
    load_model_only,
    save_checkpoint,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_save_load_checkpoint(self):
        """Test saving and loading a full checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

            # Save checkpoint
            save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=5,
                step=100,
                metrics={"val/f1": 0.85, "val/loss": 0.3},
                config=OmegaConf.create({"test": "config"}),
                experiment_id="test_exp",
                manifest_hash="abc123",
            )

            assert path.exists()

            # Load checkpoint
            model2 = SimpleModel()
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)

            info = load_checkpoint(path, model2, optimizer2, scheduler2)

            assert info["epoch"] == 5
            assert info["step"] == 100
            assert info["metrics"]["val/f1"] == 0.85
            assert info["experiment_id"] == "test_exp"
            assert info["manifest_hash"] == "abc123"

            # Check model weights match
            for p1, p2 in zip(model.parameters(), model2.parameters(), strict=False):
                assert torch.allclose(p1, p2)

    def test_load_model_only(self):
        """Test loading only model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_only.pt"

            model = SimpleModel()
            save_checkpoint(path=path, model=model, epoch=0)

            model2 = SimpleModel()
            load_model_only(path, model2)

            for p1, p2 in zip(model.parameters(), model2.parameters(), strict=False):
                assert torch.allclose(p1, p2)

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint raises error."""
        model = SimpleModel()
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pt", model)

    def test_get_latest_checkpoint(self):
        """Test finding latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # No checkpoints
            assert get_latest_checkpoint(dir_path) is None

            # Create some checkpoints
            (dir_path / "checkpoint_1.pt").write_text("dummy")
            (dir_path / "checkpoint_2.pt").write_text("dummy")

            # Make second one newer
            import time
            time.sleep(0.01)
            (dir_path / "checkpoint_3.pt").write_text("dummy")

            latest = get_latest_checkpoint(dir_path)
            assert latest.name == "checkpoint_3.pt"

    def test_forensic_model_checkpoint(self):
        """Test checkpointing with ForensicModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "forensic.pt"

            model = ForensicModel(branch_name="spatial", branch_config={"pretrained": False})
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                epoch=10,
                metrics={"val/f1": 0.92},
            )

            model2 = ForensicModel(branch_name="spatial", branch_config={"pretrained": False})
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

            info = load_checkpoint(path, model2, optimizer2)

            assert info["epoch"] == 10
            assert info["metrics"]["val/f1"] == 0.92

    def test_checkpoint_environment_metadata(self):
        """Test that checkpoint includes environment metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            model = SimpleModel()
            save_checkpoint(
                path=path,
                model=model,
                epoch=5,
                seed=42,
            )

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            
            assert "seed" in checkpoint
            assert checkpoint["seed"] == 42
            
            assert "environment" in checkpoint
            env = checkpoint["environment"]
            assert "torch_version" in env
            assert "torchvision_version" in env
            assert "python_version" in env
            assert "git_commit" in env  # May be None if not in git repo
            assert "cuda_version" in env  # May be None if no CUDA


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
