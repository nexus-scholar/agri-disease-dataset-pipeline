#!/usr/bin/env python3
"""
Experiment 03: CutMix Augmentation

CutMix is a regularization technique that cuts patches from one training image
and pastes them onto another, mixing labels proportionally.

This helps the model:
1. Learn to focus on multiple regions of the image
2. Become more robust to partial occlusions
3. Improve generalization

Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers"

Usage:
    python 03_cutmix.py
    python 03_cutmix.py --cutmix-prob 0.7 --epochs 15
"""

import argparse
import sys
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    TrainingConfig, CutMixConfig, MODELS_DIR, PLANTVILLAGE_DIR, PLANTDOC_DIR,
    TOMATO_CLASSES, CLASS_NAME_MAPPING, ensure_dir
)
from src.utils import (
    get_transforms, CanonicalImageFolder,
    evaluate_accuracy, print_header, print_section, print_config,
    ExperimentLogger, Colors, progress_bar
)
from src.utils.device import get_device, set_seed
from src.models import create_model, save_model
from src.strategies.augmentation import cutmix_data, cutmix_criterion


# =============================================================================
# TRAINING WITH CUTMIX
# =============================================================================

def train_with_cutmix(model, train_loader, val_loader, config, cutmix_config, device):
    """
    Training loop with CutMix augmentation.

    Note: Training accuracy will appear lower due to mixed labels - this is normal.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_acc = 0.0
    best_weights = None

    for epoch in range(config.epochs):
        model.train()
        train_correct = 0
        train_total = 0

        for inputs, labels in progress_bar(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Apply CutMix
            inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, cutmix_config)

            outputs = model(inputs)

            # Mixed loss if CutMix was applied
            if lam < 1.0:
                loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Track accuracy
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validate (no CutMix)
        val_acc = evaluate_accuracy(model, val_loader, device, desc="Val") / 100

        print(f"Epoch {epoch+1}/{config.epochs} | Train: {train_acc:.4f} (mixed) | Val: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights:
        model.load_state_dict(best_weights)

    return model, best_acc


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 03: CutMix Augmentation")

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10,
                        help='CutMix needs more epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001)

    # CutMix specific
    parser.add_argument('--cutmix-prob', type=float, default=0.5,
                        help='Probability of applying CutMix')
    parser.add_argument('--cutmix-beta', type=float, default=1.0,
                        help='Beta parameter for CutMix')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-filter', type=str, default='tomato')

    return parser.parse_args()


def main():
    args = parse_args()

    print_header("CutMix Augmentation", experiment_num=3)

    set_seed(args.seed)
    device = get_device()
    logger = ExperimentLogger("exp03_cutmix")

    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )

    cutmix_config = CutMixConfig(
        probability=args.cutmix_prob,
        beta=args.cutmix_beta
    )

    canonical_classes = TOMATO_CLASSES if args.class_filter == 'tomato' else None

    print_section("Configuration")
    print_config(config)
    print(f"  cutmix_probability: {cutmix_config.probability}")
    print(f"  cutmix_beta: {cutmix_config.beta}")

    # ==========================================================================
    # DATA
    # ==========================================================================
    print_section("Loading Data")

    transforms_dict = get_transforms(config)

    # Lab data
    print(f"\n{Colors.BLUE}Lab Dataset:{Colors.RESET} PlantVillage (pv_p)")
    full_ds = CanonicalImageFolder(
        str(PLANTVILLAGE_DIR),
        canonical_classes=canonical_classes,
        transform=transforms_dict['train'],
        class_name_mapping=CLASS_NAME_MAPPING
    )

    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # Field data
    print(f"\n{Colors.BLUE}Field Dataset:{Colors.RESET} PlantDoc (pd_p)")
    field_ds = CanonicalImageFolder(
        str(PLANTDOC_DIR),
        canonical_classes=canonical_classes,
        transform=transforms_dict['val'],
        class_name_mapping=CLASS_NAME_MAPPING
    )

    num_classes = len(canonical_classes) if canonical_classes else len(full_ds.classes)
    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Field: {len(field_ds)}")

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    field_loader = DataLoader(field_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # ==========================================================================
    # TRAINING
    # ==========================================================================
    print_section("Training with CutMix")

    model = create_model(num_classes)
    model = model.to(device)

    model, best_val_acc = train_with_cutmix(
        model, train_loader, val_loader, config, cutmix_config, device
    )

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
    model_path = MODELS_DIR / "cutmix_model.pth"
    save_model(model, model_path)
    print(f"\n  Model saved to: {model_path}")

    logger.log_results({
        'lab_accuracy': lab_acc,
        'field_accuracy': field_acc,
        'gap': gap,
        'cutmix_prob': cutmix_config.probability,
    })

    print(f"\n{'=' * 60}")
    print(f"{Colors.BOLD}EXPERIMENT 03 COMPLETE{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"\nCutMix achieved {field_acc:.1f}% field accuracy.")


if __name__ == '__main__':
    main()

