"""Training utilities for the U-Net model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Compute mean Dice coefficient across all classes.

    Args:
        pred: Model predictions (B, C, H, W) logits.
        target: Ground truth masks (B, H, W) with class indices.
        num_classes: Number of segmentation classes.

    Returns:
        Mean Dice score across all classes.
    """
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    dice_scores = []

    for cls in range(1, num_classes):  # Skip background (0)
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            continue
        dice = (2.0 * intersection) / (union + 1e-8)
        dice_scores.append(dice.item())

    return sum(dice_scores) / max(len(dice_scores), 1)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 7,
) -> dict[str, float]:
    """Train the model for one epoch.

    Returns:
        Dictionary with 'loss' and 'dice' metrics.
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks, num_classes)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 7,
) -> dict[str, float]:
    """Validate the model.

    Returns:
        Dictionary with 'loss' and 'dice' metrics.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, masks in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks, num_classes)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
    }
