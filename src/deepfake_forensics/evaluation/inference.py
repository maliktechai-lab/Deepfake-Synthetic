"""
Inference and prediction utilities for evaluation.
"""

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_forensics.branches.contract import BranchOutput
from deepfake_forensics.evaluation.metrics import compute_binary_metrics
from deepfake_forensics.models import ForensicModel


@torch.no_grad()
def predict(
    model: ForensicModel,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run inference on a dataset and return predictions.

    Output columns: sample_id, label, manipulation_score, prediction
    """
    model.eval()

    all_sample_ids = []
    all_labels = []
    all_scores = []
    all_predictions = []

    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        sample_ids = batch["sample_id"]

        output: BranchOutput = model(images)
        logits = output.evidence_score

        # Convert logits to manipulation scores (sigmoid)
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        predictions = (scores >= threshold).astype(int)

        all_sample_ids.extend(sample_ids)
        all_labels.extend(labels)
        all_scores.extend(scores)
        all_predictions.extend(predictions)

    df = pd.DataFrame({
        "sample_id": all_sample_ids,
        "label": all_labels,
        "manipulation_score": all_scores,
        "prediction": all_predictions,
    })

    return df


@torch.no_grad()
def evaluate(
    model: ForensicModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Full evaluation: compute loss, metrics, and return predictions.

    Returns:
        Dict with 'metrics', 'predictions' (DataFrame), 'loss' (optional)
    """
    model.eval()

    all_logits = []
    all_labels = []
    all_sample_ids = []
    total_loss = 0.0
    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)
        sample_ids = batch["sample_id"]

        output: BranchOutput = model(images)
        logits = output.evidence_score

        if loss_fn is not None:
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_sample_ids.extend(sample_ids)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_binary_metrics(all_logits, all_labels, threshold=threshold)

    if loss_fn is not None:
        metrics["loss"] = total_loss / num_samples if num_samples > 0 else 0.0

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        "sample_id": all_sample_ids,
        "label": all_labels.flatten().numpy(),
        "manipulation_score": torch.sigmoid(all_logits).flatten().numpy(),
        "prediction": (torch.sigmoid(all_logits).flatten() >= threshold).int().numpy(),
    })

    return {
        "metrics": metrics,
        "predictions": predictions_df,
        "logits": all_logits,
        "labels": all_labels,
    }
