"""
Tests for reproducibility utilities.
"""

import random

import numpy as np
import pytest
import torch

from deepfake_forensics.utils.reproducibility import get_device, set_seed, worker_init_fn


class TestReproducibility:
    """Test reproducibility functions."""

    def test_set_seed_deterministic(self):
        """Test that setting seed produces deterministic results."""
        set_seed(42, deterministic=True)

        # Generate some random values
        torch_vals1 = torch.randn(10)
        np_vals1 = np.random.randn(10)
        py_vals1 = [random.random() for _ in range(10)]

        # Reset seed
        set_seed(42, deterministic=True)

        torch_vals2 = torch.randn(10)
        np_vals2 = np.random.randn(10)
        py_vals2 = [random.random() for _ in range(10)]

        # Should be identical
        assert torch.allclose(torch_vals1, torch_vals2)
        assert np.allclose(np_vals1, np_vals2)
        assert py_vals1 == py_vals2

    def test_get_device_auto(self):
        """Test auto device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        # Should be cuda, mps, or cpu
        assert device.type in ["cuda", "mps", "cpu"]

    def test_get_device_explicit(self):
        """Test explicit device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_worker_init_fn(self):
        """Test worker init function."""
        # Should not raise
        worker_init_fn(worker_id=0, base_seed=42)
        worker_init_fn(worker_id=1, base_seed=42)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
