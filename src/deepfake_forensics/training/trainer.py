"""
Training utilities: loss, optimizer, scheduler, training loop.
"""

from collections.abc import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_forensics.branches.contract import BranchOutput
from deepfake_forensics.models import ForensicModel


def get_loss_fn(loss_name: str = "bce_with_logits", **kwargs) -> nn.Module:
    """Get loss function by name."""
    losses = {
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "bce": nn.BCELoss,
        "cross_entropy": nn.CrossEntropyLoss,
    }
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    return losses[loss_name](**kwargs)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs,
) -> Optimizer:
    """Get optimizer by name."""
    optimizers = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizers.keys())}")

    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    return optimizers[optimizer_name](params, lr=lr, weight_decay=weight_decay, **kwargs)


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    **kwargs,
) -> LRScheduler | None:
    """Get learning rate scheduler by name."""
    if scheduler_name == "none" or scheduler_name is None:
        return None

    schedulers = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "constant": torch.optim.lr_scheduler.ConstantLR,
    }

    if scheduler_name not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: {list(schedulers.keys())}")

    return schedulers[scheduler_name](optimizer, **kwargs)


def train_one_epoch(
    model: ForensicModel,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 0.0,
    log_fn: Callable | None = None,
) -> dict[str, float]:
    """Train for one epoch.
    
    Note: Learning rate scheduler stepping is handled by the caller (once per epoch),
    not inside this function. This ensures CosineAnnealingLR and similar schedulers
    advance once per epoch, not once per batch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        output: BranchOutput = model(images)
        logits = output.evidence_score

        # Compute loss
        loss = loss_fn(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        if log_fn:
            log_fn({"train/batch_loss": loss.item()}, step=epoch * len(dataloader) + num_samples)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return {"loss": avg_loss}


def validate(
    model: ForensicModel,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    log_fn: Callable | None = None,
) -> dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_samples = 0

    all_logits = []
    all_labels = []
    all_sample_ids = []

    with torch.no_grad():
        progress = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)
            sample_ids = batch["sample_id"]

            output: BranchOutput = model(images)
            logits = output.evidence_score

            loss = loss_fn(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_sample_ids.extend(sample_ids)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

    # Concatenate for metric computation
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        "loss": avg_loss,
        "logits": all_logits,
        "labels": all_labels,
        "sample_ids": all_sample_ids,
    }
