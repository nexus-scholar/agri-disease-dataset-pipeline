#!/usr/bin/env python3
"""
Experiment 09: Multi-Crop Generalization

Tests whether the generalization gap and hybrid approach work across
different crop types beyond tomato.

Usage:
    python 09_multi_crop.py
    python 09_multi_crop.py --crops tomato,potato
"""

import argparse
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    TrainingConfig, ActiveLearningConfig, RESULTS_DIR,
    PLANTVILLAGE_DIR, PLANTDOC_DIR, CLASS_NAME_MAPPING, ensure_dir
)
from src.utils import (
    get_transforms, evaluate_accuracy, print_header, print_section,
    Colors, progress_bar
)
from src.utils.device import get_device, set_seed
from src.utils.uncertainty import compute_entropy
from src.models import create_model
from src.strategies.active_learning import fine_tune


def get_available_crops():
    """Get crops available in both datasets."""
    pv_crops = set(d.name for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir())
    pd_crops = set(d.name for d in PLANTDOC_DIR.iterdir() if d.is_dir())

    # Find crops with same prefix in both datasets
    common = []
    for pv in pv_crops:
        for pd in pd_crops:
            if pv.split('_')[0] == pd.split('_')[0]:
                common.append(pv.split('_')[0])

    return sorted(set(common))


def train_baseline(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    """Train baseline model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_accuracy(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def run_crop_experiment(crop_name, device, config, al_config):
    """Run experiment for a single crop."""
    transforms_dict = get_transforms(config)

    # Find matching folders
    pv_folders = [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir() and crop_name in d.name.lower()]
    pd_folders = [d for d in PLANTDOC_DIR.iterdir() if d.is_dir() and crop_name in d.name.lower()]

    if not pv_folders:
        return None, f"No lab data for {crop_name}"
    if not pd_folders:
        return None, f"No field data for {crop_name}"

    # Load datasets
    lab_ds = datasets.ImageFolder(str(pv_folders[0]), transform=transforms_dict['train'])
    field_ds = datasets.ImageFolder(str(pd_folders[0]), transform=transforms_dict['val'])

    if len(lab_ds) < 10 or len(field_ds) < 5:
        return None, f"Insufficient data for {crop_name}"

    num_classes = len(lab_ds.classes)

    # Split lab data
    train_size = int(0.8 * len(lab_ds))
    train_ds, val_ds = random_split(lab_ds, [train_size, len(lab_ds) - train_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    field_loader = DataLoader(field_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Train baseline
    model = create_model(num_classes)
    model, _ = train_baseline(model, train_loader, val_loader, device, epochs=config.epochs)

    # Evaluate
    lab_acc = evaluate_accuracy(model, val_loader, device)
    field_acc = evaluate_accuracy(model, field_loader, device)
    gap = lab_acc - field_acc

    return {
        'crop': crop_name,
        'num_classes': num_classes,
        'lab_images': len(lab_ds),
        'field_images': len(field_ds),
        'lab_accuracy': lab_acc,
        'field_accuracy': field_acc,
        'gap': gap
    }, None


def main():
    parser = argparse.ArgumentParser(description="EXP-09: Multi-Crop Generalization")
    parser.add_argument('--crops', type=str, default=None,
                        help='Comma-separated crops (default: auto-detect)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print_header("MULTI-CROP GENERALIZATION", 9)

    device = get_device()
    set_seed(args.seed)

    config = TrainingConfig(batch_size=args.batch_size, epochs=args.epochs)
    al_config = ActiveLearningConfig()

    # Get crops to test
    if args.crops:
        crops = [c.strip() for c in args.crops.split(',')]
    else:
        crops = get_available_crops()
        if not crops:
            crops = ['tomato', 'potato', 'pepper']  # defaults

    print_section("Configuration")
    print(f"  Crops to test: {crops}")
    print(f"  Epochs: {args.epochs}")

    # Run experiments
    print_section("Running Experiments")
    results = []

    for crop in crops:
        print(f"\n{Colors.CYAN}Testing: {crop}{Colors.RESET}")
        result, error = run_crop_experiment(crop, device, config, al_config)

        if error:
            print(f"  {Colors.YELLOW}Skipped: {error}{Colors.RESET}")
            continue

        results.append(result)
        print(f"  Lab: {result['lab_accuracy']:.2f}% | Field: {result['field_accuracy']:.2f}% | Gap: {result['gap']:.2f}%")

    if not results:
        print(f"{Colors.RED}No valid results{Colors.RESET}")
        return 1

    # Summary
    print_section("Results Summary")
    print(f"\n{'Crop':<15} | {'Lab':>8} | {'Field':>8} | {'Gap':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['crop']:<15} | {r['lab_accuracy']:>7.2f}% | {r['field_accuracy']:>7.2f}% | {r['gap']:>7.2f}%")

    avg_gap = np.mean([r['gap'] for r in results])
    print(f"\nAverage gap: {avg_gap:.2f}%")

    # Save results
    ensure_dir(RESULTS_DIR / "tables")
    csv_path = RESULTS_DIR / "tables" / "exp09_multi_crop.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to: {csv_path}")

    print(f"\n{Colors.GREEN}Experiment 09 complete{Colors.RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

