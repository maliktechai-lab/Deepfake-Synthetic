#!/usr/bin/env python3
"""
Main training script for EXP-S01: Spatial Baseline.

Usage:
    python scripts/train.py --config configs/exp_s01_spatial.yaml
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_forensics.data import (
    create_datasets,
    get_eval_transforms,
    get_train_transforms,
    load_manifest,
)
from deepfake_forensics.evaluation import compute_binary_metrics, evaluate
from deepfake_forensics.models import build_model, count_parameters
from deepfake_forensics.training import (
    get_loss_fn,
    get_optimizer,
    get_scheduler,
    load_checkpoint,
    load_model_only,
    save_checkpoint,
    train_one_epoch,
    validate,
)
from deepfake_forensics.utils.config import load_config
from deepfake_forensics.utils.reproducibility import get_device, set_seed, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train spatial baseline (EXP-S01)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override seed and device if provided
    seed = args.seed or config.experiment.seed
    device_str = args.device or config.experiment.device

    # Set seed for reproducibility
    set_seed(seed)

    # Get device
    device = get_device(device_str)

    # Load manifest
    manifest = load_manifest(config.data.manifest_path)

    # Compute manifest hash for reproducibility tracking
    manifest_hash = manifest.compute_hash()

    # Create transforms
    train_transform = get_train_transforms(
        target_size=tuple(config.data.target_size),
        hflip_prob=config.data.train_transforms.hflip_prob,
        color_jitter=config.data.train_transforms.color_jitter,
        random_erase_prob=config.data.train_transforms.random_erase_prob,
        normalize=config.data.train_transforms.normalize,
    )

    eval_transform = get_eval_transforms(
        target_size=tuple(config.data.target_size),
        normalize=config.data.eval_transforms.normalize,
    )

    # Create datasets
    datasets = create_datasets(
        manifest=manifest,
        train_transform=train_transform,
        eval_transform=eval_transform,
        target_size=tuple(config.data.target_size),
    )

    # Create dataloaders with worker_init_fn for reproducibility
    dataloaders = {}
    for split in ["train", "val", "test"]:
        if datasets[split] is not None:
            dataloaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=config.training.batch_size,
                shuffle=(split == "train"),
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                drop_last=(split == "train"),
                worker_init_fn=lambda w, s=seed: worker_init_fn(w, s),
            )
        else:
            dataloaders[split] = None

    # Build model
    model = build_model(config).to(device)
    count_parameters(model)

    # Loss, optimizer, scheduler
    loss_fn = get_loss_fn(config.training.loss)
    optimizer = get_optimizer(
        model,
        optimizer_name=config.training.optimizer,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = get_scheduler(optimizer, config.training.scheduler, **config.training.scheduler_params)

    # Resume from checkpoint if provided
    start_epoch = 0
    if config.checkpoint.mode == "max":
        best_metric = -1.0
    else:
        best_metric = float("inf")
    if args.resume:
        ckpt_info = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = ckpt_info["epoch"] + 1
        best_metric = ckpt_info["metrics"].get(config.checkpoint.metric.replace("val/", ""), best_metric)

    # Setup checkpoint directory
    checkpoint_dir = Path(config.checkpoint.dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping state
    early_stopping_config = config.training.early_stopping
    early_stop_patience = early_stopping_config.patience if early_stopping_config.enabled else 0
    early_stop_counter = 0
    early_stop_min_delta = early_stopping_config.min_delta

    # Training loop
    for epoch in range(start_epoch, config.training.epochs):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            gradient_clip=config.training.gradient_clip,
        )

        # Validate
        val_metrics = {}
        if dataloaders["val"] is not None:
            val_results = validate(
                model=model,
                dataloader=dataloaders["val"],
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
            )
            val_metrics = compute_binary_metrics(
                val_results["logits"],
                val_results["labels"],
                threshold=config.evaluation.threshold,
            )
            val_metrics["loss"] = val_results["loss"]

        # Step scheduler once per epoch (for CosineAnnealingLR and similar)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        elif scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau steps with validation metric
            monitor_val = val_metrics.get(config.checkpoint.metric.replace("val/", ""), None)
            if monitor_val is not None:
                scheduler.step(monitor_val)

        # Print epoch summary
        if val_metrics:
            pass

        # Checkpointing
        monitor_metric = val_metrics.get(config.checkpoint.metric.replace("val/", ""), -1.0 if config.checkpoint.mode == "max" else float("inf"))
        is_best = False
        if config.checkpoint.mode == "max" and monitor_metric > best_metric + early_stop_min_delta:
            best_metric = monitor_metric
            is_best = True
            early_stop_counter = 0
        elif config.checkpoint.mode == "min" and monitor_metric < best_metric - early_stop_min_delta:
            best_metric = monitor_metric
            is_best = True
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if config.checkpoint.save_best and is_best:
            best_path = checkpoint_dir / "best.pt"
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={**train_metrics, **{f"val/{k}": v for k, v in val_metrics.items()}},
                config=config,
                experiment_id=config.experiment.name,
                manifest_hash=manifest_hash,
                seed=seed,
            )

        if config.checkpoint.save_last:
            last_path = checkpoint_dir / "last.pt"
            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={**train_metrics, **{f"val/{k}": v for k, v in val_metrics.items()}},
                config=config,
                experiment_id=config.experiment.name,
                manifest_hash=manifest_hash,
                seed=seed,
            )

        # Early stopping
        if early_stopping_config.enabled and early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stop_patience} epochs)")
            break

    # Final evaluation on test set using BEST checkpoint
    if dataloaders["test"] is not None:
        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            # Load best model for test evaluation
            load_model_only(best_path, model, device)
            print(f"Loaded best checkpoint from {best_path} for test evaluation")
        else:
            print("Warning: No best checkpoint found, using final model state for test evaluation")

        test_results = evaluate(
            model=model,
            dataloader=dataloaders["test"],
            device=device,
            loss_fn=loss_fn,
            threshold=config.evaluation.threshold,
        )

        for _key, _value in test_results["metrics"].items():
            pass

        # Save predictions
        pred_path = checkpoint_dir / "test_predictions.csv"
        test_results["predictions"].to_csv(pred_path, index=False)



if __name__ == "__main__":
    main()
