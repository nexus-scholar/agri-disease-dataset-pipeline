"""
Image transforms for PDA experiments.

Provides consistent train/val/test transforms across all experiment scripts.
"""
from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_IMAGE_SIZE = 224


def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE, strong: bool = False) -> transforms.Compose:
    """
    Get training transforms with data augmentation.

    Args:
        image_size: Target image size (default 224 for ImageNet pretrained models).
        strong: If True, use AutoAugment (strong augmentation for Phase 2 experiments).

    Returns:
        Composed transform pipeline.
    """
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
    ]

    if strong:
        # Strong Augmentation (Phase 2) - AutoAugment ImageNet policy
        ops.append(transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET))
    else:
        # Weak Augmentation (Standard baseline)
        ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return transforms.Compose(ops)


def get_val_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size.

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> dict:
    """
    Get all transforms as a dictionary.

    Args:
        image_size: Target image size.

    Returns:
        Dictionary with 'train' and 'val' transforms.
    """
    return {
        'train': get_train_transforms(image_size),
        'val': get_val_transforms(image_size),
    }


class WeakStrongTransform:
    """
    Returns (weak, strong) augmentations for a single input image.

    Used for semi-supervised learning methods like FixMatch.
    """

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE):
        self.weak = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.strong = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
            transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, x):
        """Return (weak_augmented, strong_augmented) tuple."""
        weak_img = self.weak(x)
        strong_img = self.strong(x)
        return weak_img, strong_img


class CutMixTransform:
    """
    CutMix augmentation for mixing samples.

    Reference: https://arxiv.org/abs/1905.04899
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter for lambda sampling.
            prob: Probability of applying CutMix.
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images, labels):
        """
        Apply CutMix to a batch.

        Args:
            images: Batch of images (N, C, H, W).
            labels: Batch of labels (N,).

        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lam).
        """
        import torch
        import numpy as np

        if np.random.random() > self.prob:
            return images, labels, labels, 1.0

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation for mixing
        rand_index = torch.randperm(batch_size)
        labels_a = labels
        labels_b = labels[rand_index]

        # Get bounding box
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return images, labels_a, labels_b, lam

