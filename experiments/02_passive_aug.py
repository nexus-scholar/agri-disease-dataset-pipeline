#!/usr/bin/env python3
"""
Experiment 02: Passive Augmentation

This experiment tests whether strong data augmentation during training
can improve field robustness without using any field data.

Hypothesis: Simulating field-like variations (blur, noise, lighting) may help.

Key Augmentations:
- Geometric: flips, rotations
- Photometric: brightness, contrast, HSV shifts
- Noise: Gaussian noise, blur
- Masking: CoarseDropout (simulates occlusions)

Usage:
    python 02_passive_aug.py
    python 02_passive_aug.py --epochs 10
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import numpy as np

# Albumentations for advanced augmentation
try:
    import cv2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Using basic augmentation.")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    TrainingConfig, MODELS_DIR, PLANTVILLAGE_DIR, PLANTDOC_DIR,
    TOMATO_CLASSES, CLASS_NAME_MAPPING, ensure_dir
)
from src.utils import (
    get_transforms, CanonicalImageFolder,
    evaluate_accuracy, print_header, print_section, print_config,
    ExperimentLogger, Colors, progress_bar
)
from src.utils.device import get_device, set_seed
from src.models import create_model, save_model


# =============================================================================
# AUGMENTATION PIPELINE
# =============================================================================

def get_augmentation_pipeline(config: TrainingConfig):
    """
    Strong augmentation pipeline using Albumentations.

    These augmentations simulate field conditions:
    - Blur/noise: Camera quality, motion
    - Lighting: Sun, shadows, overcast
    - Occlusion: Overlapping leaves, debris
    """
    if not HAS_ALBUMENTATIONS:
        return None

    return {
        'train': A.Compose([
            A.Resize(config.image_size, config.image_size),

            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),

            # Photometric transforms (simulate field lighting)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),

            # Noise/blur (simulate camera quality)
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),

            # Masking (simulate occlusions)
            A.CoarseDropout(
                max_holes=8,
                max_height=20,
                max_width=20,
                p=0.2
            ),

            # Normalize and convert
            A.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std
            ),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std
            ),
            ToTensorV2()
        ])
    }


# =============================================================================
# AUGMENTED DATASET
# =============================================================================

class AugmentedImageFolder(datasets.ImageFolder):
    """
    ImageFolder with Albumentations augmentation support.

    Uses OpenCV for image loading (required by Albumentations).
    """

    def __init__(self, root: str, aug_transform=None, canonical_classes=None):
        super().__init__(root, transform=None)
        self.aug_transform = aug_transform

        if canonical_classes:
            self._apply_canonical(canonical_classes)

    def _apply_canonical(self, canonical_classes):
        """Align to canonical class indices."""
        canonical_map = {name: idx for idx, name in enumerate(canonical_classes)}
        original_classes = self.classes.copy()

        aligned = []
        for path, old_idx in self.samples:
            class_name = original_classes[old_idx]
            if class_name in canonical_map:
                new_idx = canonical_map[class_name]
                aligned.append((path, new_idx))

        self.samples = aligned
        self.targets = [s[1] for s in aligned]
        self.classes = canonical_classes
        self.class_to_idx = canonical_map

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        # Load with OpenCV
        image = cv2.imread(path)
        if image is None:
            # Fallback for corrupted images
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        if self.aug_transform:
            try:
                augmented = self.aug_transform(image=image)
                image = augmented['image']
            except Exception as e:
                # Fallback on transform failure
                print(f"  Warning: Augmentation failed on {path}: {e}")
                image = A.Resize(224, 224)(image=image)['image']
                image = A.Normalize()(image=image)['image']
                image = ToTensorV2()(image=image)['image']

        return image, target


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, config, device):
    """Train with progress tracking."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_acc = 0.0
    best_weights = None

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for inputs, labels in progress_bar(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validate
        val_acc = evaluate_accuracy(model, val_loader, device, desc="Validating") / 100

        print(f"Epoch {epoch+1}/{config.epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.state_dict().copy()

    # Restore best
    if best_weights:
        model.load_state_dict(best_weights)

    return model, best_acc


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 02: Passive Augmentation")

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-filter', type=str, default='tomato',
                        help='Filter classes (tomato or all)')

    return parser.parse_args()


def main():
    args = parse_args()

    print_header("Passive Augmentation", experiment_num=2)

    if not HAS_ALBUMENTATIONS:
        print(f"{Colors.RED}ERROR: albumentations package required{Colors.RESET}")
        print("Install with: pip install albumentations opencv-python")
        return

    set_seed(args.seed)
    device = get_device()
    logger = ExperimentLogger("exp02_passive_aug")

    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )

    print_section("Configuration")
    print_config(config)

    # Determine classes
    canonical_classes = TOMATO_CLASSES if args.class_filter == 'tomato' else None

    # ==========================================================================
    # DATA
    # ==========================================================================
    print_section("Loading Data with Strong Augmentation")

    aug_transforms = get_augmentation_pipeline(config)

    # Lab data with augmentation
    print(f"\n{Colors.BLUE}Lab Dataset:{Colors.RESET} PlantVillage (pv_p)")
    train_ds = AugmentedImageFolder(
        str(PLANTVILLAGE_DIR),
        aug_transform=aug_transforms['train'],
        canonical_classes=canonical_classes
    )

    # Split
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds_base = random_split(train_ds, [train_size, val_size])

    # Validation without augmentation (need to create separate)
    val_ds = AugmentedImageFolder(
        str(PLANTVILLAGE_DIR),
        aug_transform=aug_transforms['val'],
        canonical_classes=canonical_classes
    )
    # Use same indices as val_ds_base
    val_indices = val_ds_base.indices
    from torch.utils.data import Subset
    val_ds = Subset(val_ds, val_indices)

    # Field data
    print(f"\n{Colors.BLUE}Field Dataset:{Colors.RESET} PlantDoc (pd_p)")
    field_ds = AugmentedImageFolder(
        str(PLANTDOC_DIR),
        aug_transform=aug_transforms['val'],
        canonical_classes=canonical_classes
    )

    num_classes = len(canonical_classes) if canonical_classes else len(train_ds.dataset.classes)
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Field: {len(field_ds)}")

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    field_loader = DataLoader(field_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # ==========================================================================
    # MODEL & TRAINING
    # ==========================================================================
    print_section("Training with Augmentation")

    model = create_model(num_classes)
    model = model.to(device)

    model, best_val_acc = train_model(model, train_loader, val_loader, config, device)

    # ==========================================================================
    # EVALUATION
    # ==========================================================================
    print_section("Final Evaluation")

    lab_acc = evaluate_accuracy(model, val_loader, device, desc="Lab Validation")
    field_acc = evaluate_accuracy(model, field_loader, device, desc="Field Test")
    gap = lab_acc - field_acc

    print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
    print(f"  Lab Accuracy:   {Colors.GREEN}{lab_acc:.2f}%{Colors.RESET}")
    print(f"  Field Accuracy: {Colors.YELLOW}{field_acc:.2f}%{Colors.RESET}")
    print(f"  Gap: {gap:.2f}%")

    # Save
    ensure_dir(MODELS_DIR)
    model_path = MODELS_DIR / "passive_aug_model.pth"
    save_model(model, model_path)
    print(f"\n  Model saved to: {model_path}")

    # Log
    logger.log_results({
        'lab_accuracy': lab_acc,
        'field_accuracy': field_acc,
        'gap': gap,
    })

    print(f"\n{'=' * 60}")
    print(f"{Colors.BOLD}EXPERIMENT 02 COMPLETE{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"\nPassive augmentation achieved {field_acc:.1f}% field accuracy.")
    print(f"Compare with baseline (Exp 01) to see improvement.")


if __name__ == '__main__':
    main()

