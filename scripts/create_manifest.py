#!/usr/bin/env python3
"""
Create a template manifest CSV for deepfake forensics.

Usage:
    python scripts/create_manifest.py --output data/manifest.csv --num-samples 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_forensics.data import create_manifest_template


def parse_args():
    parser = argparse.ArgumentParser(description="Create a template manifest")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of dummy samples")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits")
    parser.add_argument("--with-files", action="store_true", help="Create dummy image files")
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = args.splits.split(",")

    # Create template
    df = create_manifest_template()

    # Generate dummy data
    np.random.seed(42)
    rows = []

    for i in range(args.num_samples):
        split = np.random.choice(splits, p=[0.7, 0.15, 0.15])
        label = np.random.choice([0, 1])
        media_id = f"media_{i // 5:04d}"  # 5 samples per media
        sample_id = f"sample_{i:06d}"
        dataset = np.random.choice(["FF++", "DFDC", "Celeb-DF"])
        identity_id = f"id_{np.random.randint(0, 100):03d}"
        manipulation_type = "Deepfakes" if label == 1 else "None"
        generator = "faceswap" if label == 1 else "None"

        row = {
            "sample_id": sample_id,
            "media_id": media_id,
            "label": label,
            "file_path": f"data/faces/{split}/{sample_id}.jpg",
            "split": split,
            "dataset": dataset,
            "identity_id": identity_id,
            "source_id": f"src_{np.random.randint(0, 10):02d}",
            "manipulation_type": manipulation_type,
            "generator": generator,
            "sha256": "dummy_hash_" + "0" * 58,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    # Create dummy image files if requested
    if args.with_files:
        from PIL import Image
        for split in splits:
            split_dir = output_path.parent / "faces" / split
            split_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            img_path = Path(row["file_path"])
            img_path.parent.mkdir(parents=True, exist_ok=True)
            # Create a dummy 224x224 RGB image
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(img_path)



if __name__ == "__main__":
    main()
