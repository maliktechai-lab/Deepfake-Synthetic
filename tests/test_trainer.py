"""
Tests for training loop, scheduler, and early stopping.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from deepfake_forensics.branches.contract import BranchOutput
from deepfake_forensics.models import ForensicModel
from deepfake_forensics.training import get_loss_fn, get_optimizer, get_scheduler, train_one_epoch
from deepfake_forensics.training.checkpoint import load_model_only, save_checkpoint


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class MockForensicModel(nn.Module):
    """Mock ForensicModel for testing train_one_epoch without full model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        # Return a BranchOutput-like object with evidence_score
        class Output:
            def __init__(self, logits):
                self.evidence_score = logits
                self.embedding = torch.zeros(x.size(0), 256)
                self.reliability = torch.ones(x.size(0), 1)
                self.uncertainty = torch.zeros(x.size(0), 1)
        return Output(self.linear(x))
    
    def train(self):
        super().train()
        return self
    
    def eval(self):
        super().eval()
        return self
    
    def parameters(self):
        return self.linear.parameters()


class TestScheduler:
    """Test scheduler behavior."""

    def test_cosine_scheduler_steps_per_epoch(self):
        """Test that CosineAnnealingLR steps once per epoch, not per batch."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        
        initial_lr = optimizer.param_groups[0]["lr"]
        
        # Simulate one epoch with 5 batches
        for _ in range(5):
            optimizer.step()
            # NOTE: train_one_epoch should NOT call scheduler.step() anymore
        
        # Step once per epoch (as the training loop should do)
        scheduler.step()
        
        # LR should have changed once
        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr != initial_lr
        
        # Step again for second epoch
        scheduler.step()
        newer_lr = optimizer.param_groups[0]["lr"]
        assert newer_lr != new_lr

    def test_scheduler_not_stepped_in_batch_loop(self):
        """Verify train_one_epoch doesn't step scheduler."""
        model = MockForensicModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Create dummy dataloader with proper batch structure (dict with "image" and "label")
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, n=10):
                self.images = torch.randn(n, 10)
                self.labels = torch.randint(0, 2, (n,)).float()  # 1D labels
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                return {
                    "image": self.images[idx],
                    "label": self.labels[idx],
                    "sample_id": f"sample_{idx}",
                    "media_id": f"media_{idx}",
                    "file_path": f"/fake/sample_{idx}.jpg",
                }
        
        dataset = DictDataset(10)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)
        
        initial_lr = optimizer.param_groups[0]["lr"]
        
        # Train one epoch - should NOT step scheduler
        train_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=0,
            gradient_clip=0.0,
        )
        
        # LR should be unchanged (scheduler not stepped in train_one_epoch)
        assert optimizer.param_groups[0]["lr"] == initial_lr


class TestEarlyStopping:
    """Test early stopping logic."""

    def test_early_stopping_config_structure(self):
        """Test that early stopping config has required fields."""
        config = OmegaConf.create({
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "patience": 5,
                    "min_delta": 1e-4,
                    "monitor": "val/f1"
                }
            },
            "checkpoint": {
                "mode": "max"
            }
        })
        
        early_stopping_config = config.training.early_stopping
        assert early_stopping_config.enabled is True
        assert early_stopping_config.patience == 5
        assert early_stopping_config.min_delta == 1e-4
        assert early_stopping_config.monitor == "val/f1"


class TestBestCheckpointLoading:
    """Test that best checkpoint is loaded for test evaluation."""

    def test_load_model_only_restores_weights(self):
        """Test that load_model_only restores model weights correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best.pt"
            
            model1 = ForensicModel(branch_name="spatial", branch_config={"pretrained": False})
            optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-4)
            
            # Save checkpoint
            save_checkpoint(
                path=path,
                model=model1,
                optimizer=optimizer,
                epoch=5,
                metrics={"val/f1": 0.95},
                seed=42,
            )
            
            # Create new model with different weights
            model2 = ForensicModel(branch_name="spatial", branch_config={"pretrained": False})
            
            # Verify weights are different before loading (use first conv layer)
            weights_before = model2.state_dict()["branch.backbone.features.0.0.weight"].clone()
            
            # Load best checkpoint
            load_model_only(path, model2)
            
            # Verify weights changed
            weights_after = model2.state_dict()["branch.backbone.features.0.0.weight"]
            assert not torch.allclose(weights_before, weights_after)
            
            # Verify it matches model1
            weights_original = model1.state_dict()["branch.backbone.features.0.0.weight"]
            assert torch.allclose(weights_after, weights_original)

    def test_checkpoint_saves_seed(self):
        """Test that seed is saved in checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            
            model = ForensicModel(branch_name="spatial", branch_config={"pretrained": False})
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                epoch=10,
                seed=12345,
            )
            
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["seed"] == 12345


class TestDatasetNoDoubleResize:
    """Test that dataset doesn't resize before transforms."""

    def test_dataset_does_not_resize_before_transform(self):
        """Test that DeepfakeDataset passes images to transform without resizing."""
        from PIL import Image
        import numpy as np
        from deepfake_forensics.data import Manifest, DeepfakeDataset
        from deepfake_forensics.data.transforms import get_eval_transforms
        
        # Create a temporary image larger than 224x224
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_image.jpg"
            # Create 500x500 image
            img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
            img.save(img_path)
            
            # Create manifest
            df = pd.DataFrame([{
                "sample_id": "test_001",
                "media_id": "media_001",
                "label": 0,
                "file_path": str(img_path),
                "split": "test",
            }])
            manifest = Manifest(dataframe=df)
            
            # Create dataset with eval transform (includes Resize)
            eval_transform = get_eval_transforms(target_size=(224, 224))
            dataset = DeepfakeDataset(manifest, transform=eval_transform, target_size=(224, 224))
            
            # Get item - should apply transform which includes Resize
            item = dataset[0]
            
            # Image should be tensor of shape (3, 224, 224)
            assert item["image"].shape == (3, 224, 224)
            assert isinstance(item["image"], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])