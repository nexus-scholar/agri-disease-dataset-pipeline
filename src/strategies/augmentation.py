"""
Augmentation strategies for domain adaptation.

Includes:
- CutMix augmentation
- Standard augmentation pipeline
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from ..config import CutMixConfig


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Clip to valid region
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    config: CutMixConfig = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation to batch.

    Args:
        x: Input images [B, C, H, W]
        y: Labels [B]
        config: CutMix configuration

    Returns:
        (mixed_x, y_a, y_b, lam) for computing mixed loss
    """
    if config is None:
        config = CutMixConfig()

    if np.random.random() > config.probability:
        # No CutMix this batch
        return x, y, y, 1.0

    batch_size = x.size(0)

    # Sample lambda from beta distribution
    lam = np.random.beta(config.beta, config.beta)

    # Get random permutation
    index = torch.randperm(batch_size).to(x.device)

    # Get bbox
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # Create mixed images
    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda based on actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a = y
    y_b = y[index]

    return x_mixed, y_a, y_b, lam


def cutmix_criterion(
    criterion: nn.Module,
    outputs: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Compute mixed loss for CutMix."""
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


class CutMixTrainer:
    """
    Trainer with CutMix augmentation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: CutMixConfig = None,
        learning_rate: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config or CutMixConfig()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_epoch(self, loader):
        """Train for one epoch with CutMix."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Apply CutMix
            inputs_mixed, y_a, y_b, lam = cutmix_data(inputs, labels, self.config)

            self.optimizer.zero_grad()
            outputs = self.model(inputs_mixed)

            # Mixed loss
            loss = cutmix_criterion(self.criterion, outputs, y_a, y_b, lam)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            # For accuracy, use original labels with mixed inputs
            _, preds = outputs.max(1)
            correct += (lam * preds.eq(y_a).float() +
                       (1 - lam) * preds.eq(y_b).float()).sum().item()
            total += labels.size(0)

        return correct / total, total_loss / total

    def evaluate(self, loader):
        """Evaluate without CutMix."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        return correct / total, total_loss / total

