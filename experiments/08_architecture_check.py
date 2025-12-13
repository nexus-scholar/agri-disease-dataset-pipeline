#!/usr/bin/env python3
"""
Experiment 08: Architecture Check (MobileViT)

Tests whether the generalization gap and hybrid approach are model-agnostic
by testing with MobileViT architecture (requires timm library).

Usage:
    python 08_architecture_check.py
    python 08_architecture_check.py --model mobilevit_s
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    TrainingConfig, MODELS_DIR, RESULTS_DIR,
    PLANTVILLAGE_DIR, PLANTDOC_DIR, TOMATO_CLASSES, CLASS_NAME_MAPPING, ensure_dir
)
from src.utils import (
    get_transforms, CanonicalImageFolder,
    evaluate_accuracy, print_header, print_section,
    Colors, progress_bar
)
from src.utils.device import get_device, set_seed
from src.models import create_model

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def create_mobilevit(model_name, num_classes, pretrained=True):
    """Create MobileViT model using timm."""
    if not HAS_TIMM:
        raise RuntimeError("timm library required. Install with: pip install timm")
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def create_model_by_name(model_type, num_classes, pretrained=True):
    """Create model by type name."""
    if model_type == 'mobilenetv3':
        return create_model(num_classes, pretrained)
    elif model_type.startswith('mobilevit'):
        return create_mobilevit(model_type, num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    """Train model and return best version."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for inputs, labels in progress_bar(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"  Epoch {epoch+1}: Val {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description="EXP-08: Architecture Check")
    parser.add_argument('--model', type=str, default='mobilenetv3',
                        choices=['mobilenetv3', 'mobilevit_s', 'mobilevit_xs'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-filter', type=str, default='tomato')
    args = parser.parse_args()

    print_header("ARCHITECTURE CHECK", 8)

    if args.model.startswith('mobilevit') and not HAS_TIMM:
        print(f"{Colors.RED}Error: timm not installed. Install with: pip install timm{Colors.RESET}")
        return 1

    device = get_device()
    set_seed(args.seed)

    config = TrainingConfig(batch_size=args.batch_size, epochs=args.epochs)
    canonical_classes = TOMATO_CLASSES if args.class_filter == 'tomato' else None
    num_classes = len(canonical_classes) if canonical_classes else 7

    print_section("Configuration")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")

    # Load data
    print_section("Loading Data")
    transforms_dict = get_transforms(config)

    lab_ds = CanonicalImageFolder(str(PLANTVILLAGE_DIR), canonical_classes, transforms_dict['train'], class_name_mapping=CLASS_NAME_MAPPING)
    train_size = int(0.8 * len(lab_ds))
    train_ds, val_ds = random_split(lab_ds, [train_size, len(lab_ds) - train_size])

    field_ds = CanonicalImageFolder(str(PLANTDOC_DIR), canonical_classes, transforms_dict['val'], class_name_mapping=CLASS_NAME_MAPPING)

    print(f"  Lab: {len(lab_ds)} | Field: {len(field_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    field_loader = DataLoader(field_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Train model
    print_section(f"Training {args.model}")
    model = create_model_by_name(args.model, num_classes)
    model, best_val = train_model(model, train_loader, val_loader, device, args.epochs)

    # Evaluate
    print_section("Evaluation")
    lab_acc = evaluate_accuracy(model, val_loader, device, "Lab")
    field_acc = evaluate_accuracy(model, field_loader, device, "Field")
    gap = lab_acc - field_acc

    print(f"\n{Colors.BOLD}Results for {args.model}:{Colors.RESET}")
    print(f"  Lab:   {Colors.GREEN}{lab_acc:.2f}%{Colors.RESET}")
    print(f"  Field: {Colors.RED}{field_acc:.2f}%{Colors.RESET}")
    print(f"  Gap:   {gap:.2f}%")

    print(f"\n{Colors.GREEN}Experiment 08 complete{Colors.RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

