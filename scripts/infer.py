#!/usr/bin/env python3
"""
Inference script for trained models.

Usage:
    python scripts/infer.py --checkpoint checkpoints/exp_s01_spatial/best.pt --manifest data/test_manifest.csv
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_forensics.data import DeepfakeDataset, get_eval_transforms, load_manifest
from deepfake_forensics.evaluation import predict
from deepfake_forensics.models import build_model
from deepfake_forensics.training import load_model_only
from deepfake_forensics.utils.config import load_config
from deepfake_forensics.utils.reproducibility import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--config", type=str, default=None, help="Config used for training (optional)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu, mps")
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device(args.device)

    # Load manifest
    manifest = load_manifest(args.manifest)

    # Load config if provided, otherwise use defaults
    if args.config:
        config = load_config(args.config)
        target_size = tuple(config.data.target_size)
        normalize = config.data.eval_transforms.normalize
        branch_config = config.model.branch_config
    else:
        target_size = (224, 224)
        normalize = True
        branch_config = {}

    # Build model
    model = build_model(OmegaConf.create({
        "model": {
            "branch": "spatial",
            "branch_config": branch_config,
        }
    }))

    # Load weights
    load_model_only(args.checkpoint, model, device)
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    eval_transform = get_eval_transforms(target_size=target_size, normalize=normalize)
    dataset = DeepfakeDataset(manifest, transform=eval_transform, target_size=target_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Run inference
    predictions = predict(model, dataloader, device, threshold=args.threshold)

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    # Print summary


if __name__ == "__main__":
    main()
