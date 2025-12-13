"""
Semi-supervised learning strategies for PDA experiments.

Implements FixMatch for combining labeled and unlabeled data during
active learning fine-tuning rounds.

Reference: https://arxiv.org/abs/2001.07685
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.transforms import WeakStrongTransform, get_train_transforms


@dataclass
class FixMatchConfig:
    """Configuration for FixMatch training."""
    batch_size: int = 4
    epochs: int = 5
    lr: float = 1e-3
    threshold: float = 0.95  # Confidence threshold for pseudo-labels
    lambda_u: float = 1.0    # Unsupervised loss weight
    mu: int = 5              # Ratio of unlabeled to labeled batch size


class SSLDataset(Dataset):
    """
    Dataset wrapper for semi-supervised learning.

    Wraps an ImageFolder-style dataset and applies specified transforms,
    supporting both labeled and unlabeled (weak/strong augmentation) modes.
    """

    def __init__(
        self,
        base_dataset,
        indices: List[int],
        transform=None,
        mode: str = 'labeled',
    ):
        """
        Args:
            base_dataset: Base ImageFolder dataset with .samples and .loader.
            indices: Indices into the base dataset.
            transform: Transform to apply (can be WeakStrongTransform for SSL).
            mode: 'labeled' or 'unlabeled'.
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, target = self.base_dataset.samples[real_idx]
        image = self.base_dataset.loader(path)

        if isinstance(self.transform, WeakStrongTransform):
            weak_img, strong_img = self.transform(image)
            return (weak_img, strong_img), target

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def train_fixmatch(
    model: nn.Module,
    labeled_indices: List[int],
    unlabeled_indices: List[int],
    base_dataset,
    device: torch.device,
    config: Optional[FixMatchConfig] = None,
) -> nn.Module:
    """
    Train model using FixMatch semi-supervised learning.

    Combines supervised loss on labeled data with consistency regularization
    on high-confidence pseudo-labels from unlabeled data.

    Args:
        model: Model to train.
        labeled_indices: Indices of labeled samples.
        unlabeled_indices: Indices of unlabeled samples (pool).
        base_dataset: Base ImageFolder dataset.
        device: Target device.
        config: FixMatch configuration.

    Returns:
        Trained model.
    """
    if config is None:
        config = FixMatchConfig()

    model.train()

    # Handle edge case: no unlabeled data
    if not unlabeled_indices:
        print("  [FixMatch] No unlabeled data; falling back to supervised training.")
        return _train_supervised(model, labeled_indices, base_dataset, device, config)

    # Create datasets
    train_transform = get_train_transforms()
    ssl_transform = WeakStrongTransform()

    labeled_ds = SSLDataset(base_dataset, labeled_indices, transform=train_transform)
    unlabeled_ds = SSLDataset(base_dataset, unlabeled_indices, transform=ssl_transform, mode='unlabeled')

    batch_size = min(config.batch_size, len(labeled_ds))
    if batch_size < 1:
        batch_size = 1

    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(labeled_ds) > batch_size,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=batch_size * config.mu,
        shuffle=True,
        drop_last=len(unlabeled_ds) > batch_size * config.mu,
    )

    if len(labeled_loader) == 0:
        print("  [FixMatch] Labeled loader empty; cannot train.")
        return model

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"  [FixMatch] |L|={len(labeled_ds)} |U|={len(unlabeled_ds)} | "
          f"batch={batch_size} (×{config.mu})")

    unlabeled_iter = iter(unlabeled_loader)
    total_steps = config.epochs * len(labeled_loader)
    step = 0

    for epoch in range(config.epochs):
        epoch_loss_x = 0.0
        epoch_loss_u = 0.0
        epoch_mask_rate = 0.0

        for images_x, targets_x in labeled_loader:
            # Get unlabeled batch
            try:
                (images_u_w, images_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                try:
                    (images_u_w, images_u_s), _ = next(unlabeled_iter)
                except StopIteration:
                    # Unlabeled loader is empty
                    images_u_w = images_u_s = None

            images_x = images_x.to(device)
            targets_x = targets_x.to(device)

            # Supervised loss
            logits_x = model(images_x)
            loss_x = criterion(logits_x, targets_x)

            # Unsupervised loss (if we have unlabeled data)
            loss_u = torch.tensor(0.0, device=device)
            mask_rate = 0.0

            if images_u_w is not None and images_u_s is not None:
                images_u_w = images_u_w.to(device)
                images_u_s = images_u_s.to(device)

                # Generate pseudo-labels from weak augmentation
                with torch.no_grad():
                    logits_u_w = model(images_u_w)
                    probs_u_w = torch.softmax(logits_u_w, dim=1)
                    max_probs, pseudo_targets = torch.max(probs_u_w, dim=1)
                    mask = max_probs.ge(config.threshold).float()
                    mask_rate = mask.mean().item()

                # Consistency loss on strong augmentation
                logits_u_s = model(images_u_s)
                loss_u = (F.cross_entropy(logits_u_s, pseudo_targets, reduction='none') * mask).mean()

            # Combined loss
            loss = loss_x + config.lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_x += loss_x.item()
            epoch_loss_u += loss_u.item()
            epoch_mask_rate += mask_rate
            step += 1

        n_batches = len(labeled_loader)
        if n_batches > 0:
            print(f"    Epoch {epoch+1}/{config.epochs}: "
                  f"L_x={epoch_loss_x/n_batches:.4f}, "
                  f"L_u={epoch_loss_u/n_batches:.4f}, "
                  f"mask={epoch_mask_rate/n_batches:.2%}")

    return model


def _train_supervised(
    model: nn.Module,
    labeled_indices: List[int],
    base_dataset,
    device: torch.device,
    config: FixMatchConfig,
) -> nn.Module:
    """Fallback supervised training when no unlabeled data is available."""
    train_transform = get_train_transforms()
    labeled_ds = SSLDataset(base_dataset, labeled_indices, transform=train_transform)

    batch_size = min(config.batch_size, len(labeled_ds))
    if batch_size < 1:
        return model

    loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config.epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

