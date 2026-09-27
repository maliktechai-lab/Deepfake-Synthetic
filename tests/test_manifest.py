"""
Tests for manifest validation and data loading.
"""


import numpy as np
import pandas as pd
import pytest

from deepfake_forensics.data import (
    REQUIRED_COLUMNS,
    Manifest,
)
from deepfake_forensics.data.dataset import create_datasets
from deepfake_forensics.data.transforms import get_eval_transforms, get_train_transforms


class TestManifestValidation:
    """Test manifest validation rules."""

    def create_valid_manifest_df(self, num_samples: int = 10) -> pd.DataFrame:
        """Create a valid manifest DataFrame for testing with split-exclusive media_ids."""
        np.random.seed(42)
        rows = []
        # Assign each media_id to exactly one split
        media_splits = {}
        for i in range(num_samples):
            media_id = f"media_{i // 3:03d}"
            if media_id not in media_splits:
                media_splits[media_id] = np.random.choice(["train", "val", "test"])
            split = media_splits[media_id]
            rows.append({
                "sample_id": f"sample_{i:04d}",
                "media_id": media_id,
                "label": np.random.choice([0, 1]),
                "file_path": f"/fake/path/sample_{i:04d}.jpg",
                "split": split,
            })
        return pd.DataFrame(rows)

    def test_valid_manifest(self):
        """Test that a valid manifest passes validation."""
        df = self.create_valid_manifest_df()
        manifest = Manifest(dataframe=df)
        assert len(manifest) == len(df)

    def test_missing_required_columns(self):
        """Test that missing required columns raise error."""
        df = pd.DataFrame({"sample_id": ["a"], "label": [0]})  # missing media_id, file_path, split
        with pytest.raises(ValueError, match="missing required columns"):
            Manifest(dataframe=df)

    def test_invalid_labels(self):
        """Test that invalid labels are rejected."""
        df = self.create_valid_manifest_df()
        df.loc[0, "label"] = 2  # Invalid label
        with pytest.raises(ValueError, match="Invalid labels"):
            Manifest(dataframe=df)

    def test_invalid_splits(self):
        """Test that invalid splits are rejected."""
        df = self.create_valid_manifest_df()
        df.loc[0, "split"] = "invalid"
        with pytest.raises(ValueError, match="Invalid splits"):
            Manifest(dataframe=df)

    def test_duplicate_sample_ids(self):
        """Test that duplicate sample_ids are rejected."""
        df = self.create_valid_manifest_df()
        df.loc[1, "sample_id"] = df.loc[0, "sample_id"]  # Duplicate
        with pytest.raises(ValueError, match="Duplicate sample_ids"):
            Manifest(dataframe=df)

    def test_empty_manifest(self):
        """Test that empty manifest is rejected."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        with pytest.raises(ValueError, match="empty"):
            Manifest(dataframe=df)

    def test_get_split(self):
        """Test filtering manifest by split."""
        df = self.create_valid_manifest_df(30)
        manifest = Manifest(dataframe=df)
        train_manifest = manifest.get_split("train")
        assert all(train_manifest.dataframe["split"] == "train")
        assert len(train_manifest) <= len(manifest)

    def test_manifest_hash(self):
        """Test manifest hash computation."""
        df = self.create_valid_manifest_df()
        manifest = Manifest(dataframe=df)
        hash1 = manifest.compute_hash()
        hash2 = manifest.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated SHA256

    def test_duplicate_file_paths(self):
        """Test that duplicate file_paths are rejected."""
        df = self.create_valid_manifest_df()
        df.loc[1, "file_path"] = df.loc[0, "file_path"]  # Duplicate
        with pytest.raises(ValueError, match="Duplicate file_paths"):
            Manifest(dataframe=df)

    def test_media_id_cross_split_leakage(self):
        """Test that media_id appearing in multiple splits fails."""
        df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "media_id": ["m1", "m1", "m2"],
            "label": [0, 1, 0],
            "file_path": ["/fake/s1.jpg", "/fake/s2.jpg", "/fake/s3.jpg"],
            "split": ["train", "test", "train"],  # m1 in train AND test
        })
        with pytest.raises(ValueError, match="media_id appears in multiple splits"):
            Manifest(dataframe=df)

    def test_identity_id_cross_split_leakage(self):
        """Test that identity_id appearing in multiple splits fails."""
        df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "media_id": ["m1", "m2", "m3"],
            "label": [0, 1, 0],
            "file_path": ["/fake/s1.jpg", "/fake/s2.jpg", "/fake/s3.jpg"],
            "split": ["train", "val", "train"],
            "identity_id": ["id1", "id1", "id2"],  # id1 in train AND val
        })
        with pytest.raises(ValueError, match="identity_id appears in multiple splits"):
            Manifest(dataframe=df)

    def test_sha256_cross_split_leakage(self):
        """Test that sha256 appearing in multiple splits fails."""
        df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "media_id": ["m1", "m2", "m3"],
            "label": [0, 1, 0],
            "file_path": ["/fake/s1.jpg", "/fake/s2.jpg", "/fake/s3.jpg"],
            "split": ["train", "test", "train"],
            "sha256": ["abc123", "abc123", "def456"],  # abc123 in train AND test
        })
        with pytest.raises(ValueError, match="sha256 appears in multiple splits"):
            Manifest(dataframe=df)

    def test_dummy_sha256_ignored(self):
        """Test that dummy hashes are ignored in leakage check."""
        df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "media_id": ["m1", "m2", "m3"],
            "label": [0, 1, 0],
            "file_path": ["/fake/s1.jpg", "/fake/s2.jpg", "/fake/s3.jpg"],
            "split": ["train", "test", "train"],
            "sha256": ["dummy_hash_000", "dummy_hash_000", "def456"],  # dummy hashes ignored
        })
        # Should NOT raise - dummy hashes are ignored
        manifest = Manifest(dataframe=df)
        assert len(manifest) == 3

    def test_valid_split_exclusivity(self):
        """Test that valid split-exclusive manifest passes."""
        df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3", "s4"],
            "media_id": ["m1", "m2", "m3", "m4"],
            "label": [0, 1, 0, 1],
            "file_path": ["/fake/s1.jpg", "/fake/s2.jpg", "/fake/s3.jpg", "/fake/s4.jpg"],
            "split": ["train", "train", "val", "test"],
            "identity_id": ["id1", "id2", "id3", "id4"],
            "sha256": ["abc1", "abc2", "abc3", "abc4"],
        })
        manifest = Manifest(dataframe=df)
        assert len(manifest) == 4


class TestDataset:
    """Test dataset creation and loading."""

    def create_valid_manifest_df(self, num_samples: int = 10) -> pd.DataFrame:
        """Create a valid manifest DataFrame for testing with split-exclusive media_ids."""
        np.random.seed(42)
        rows = []
        # Assign each media_id to exactly one split
        media_splits = {}
        for i in range(num_samples):
            media_id = f"media_{i // 3:03d}"
            if media_id not in media_splits:
                media_splits[media_id] = np.random.choice(["train", "val", "test"])
            split = media_splits[media_id]
            rows.append({
                "sample_id": f"sample_{i:04d}",
                "media_id": media_id,
                "label": np.random.choice([0, 1]),
                "file_path": f"/fake/path/sample_{i:04d}.jpg",
                "split": split,
            })
        return pd.DataFrame(rows)

    def test_dataset_requires_split(self):
        """Test that dataset requires explicit split column."""
        df = pd.DataFrame({
            "sample_id": ["a", "b"],
            "media_id": ["m1", "m2"],
            "label": [0, 1],
            "file_path": ["/fake/a.jpg", "/fake/b.jpg"],
            # Missing split column
        })
        with pytest.raises(ValueError, match="missing required columns"):
            Manifest(dataframe=df)

    def test_create_datasets(self):
        """Test creating train/val/test datasets."""
        df = self.create_valid_manifest_df(30)
        manifest = Manifest(dataframe=df)

        train_tf = get_train_transforms()
        eval_tf = get_eval_transforms()

        datasets = create_datasets(manifest, train_tf, eval_tf)

        assert "train" in datasets
        assert "val" in datasets
        assert "test" in datasets

        # At least train should have data
        assert datasets["train"] is not None
        assert len(datasets["train"]) > 0


class TestTransforms:
    """Test data transforms."""

    def test_train_transforms(self):
        """Test training transforms creation."""
        tf = get_train_transforms(hflip_prob=0.5, color_jitter=0.1)
        assert tf is not None

    def test_eval_transforms(self):
        """Test evaluation transforms creation."""
        tf = get_eval_transforms()
        assert tf is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
