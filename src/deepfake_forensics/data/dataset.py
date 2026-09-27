"""
PyTorch Dataset for deepfake forensics.
Loads aligned face crops from manifest file paths.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from deepfake_forensics.data.manifest import VALID_LABELS, Manifest


class DeepfakeDataset(Dataset):
    """
    Dataset for loading aligned face crops from a manifest.

    Expects manifest with columns: sample_id, media_id, label, file_path, split
    
    Note: Resize is handled by the transform pipeline. This dataset does NOT
    resize images; it only loads, converts to RGB, and applies transforms.
    """

    def __init__(
        self,
        manifest: Manifest,
        transform: Callable | None = None,
        target_size: tuple[int, int] = (224, 224),
        convert_rgb: bool = True,
    ) -> None:
        """
        Args:
            manifest: Validated Manifest object
            transform: Optional torchvision transforms (should include Resize)
            target_size: Target image size (H, W) - used for validation only
            convert_rgb: Whether to convert images to RGB
        """
        self.manifest = manifest
        self.transform = transform
        self.target_size = target_size
        self.convert_rgb = convert_rgb

        # Validate manifest has required split column
        if "split" not in manifest.dataframe.columns:
            raise ValueError("Manifest must contain explicit 'split' column. Never create random splits.")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest[idx]

        # Load image
        file_path = Path(row["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        image = Image.open(file_path)
        if self.convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms (resize is handled by transform pipeline)
        if self.transform:
            image = self.transform(image)

        # Label validation
        label = int(row["label"])
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label {label} for sample {row['sample_id']}")

        return {
            "sample_id": row["sample_id"],
            "media_id": row["media_id"],
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "file_path": str(file_path),
        }


def create_datasets(
    manifest: Manifest,
    train_transform: Callable | None = None,
    eval_transform: Callable | None = None,
    target_size: tuple[int, int] = (224, 224),
) -> dict[str, DeepfakeDataset]:
    """
    Create train/val/test datasets from a manifest with explicit splits.

    Args:
        manifest: Manifest with 'split' column containing 'train', 'val', 'test'
        train_transform: Transforms for training (with augmentation)
        eval_transform: Transforms for validation/test (no augmentation)
        target_size: Target image size

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    datasets = {}

    for split in ["train", "val", "test"]:
        split_manifest = manifest.get_split(split)
        if len(split_manifest) == 0:
            datasets[split] = None
            continue

        transform = train_transform if split == "train" else eval_transform
        datasets[split] = DeepfakeDataset(
            manifest=split_manifest,
            transform=transform,
            target_size=target_size,
        )

    return datasets
