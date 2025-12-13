#!/usr/bin/env python3
"""
Experiment 01: Baseline Generalization Gap

This experiment establishes the fundamental problem:
Models trained on controlled "Lab" data (PlantVillage) fail on real-world "Field" data (PlantDoc).

Key Finding: ~60-70% accuracy drop from Lab to Field conditions.

Usage:
    python 01_baseline_gap.py                      # Default with tomato classes
    python 01_baseline_gap.py --epochs 10          # More training
    python 01_baseline_gap.py --use-processed      # Use processed pv_p/pd_p folders
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    TrainingConfig, MODELS_DIR, PROCESSED_DIR, PLANTVILLAGE_DIR, PLANTDOC_DIR,
    TOMATO_CLASSES, CLASS_NAME_MAPPING, ensure_dir
)
from src.utils import (
    get_transforms, FilteredImageFolder, CanonicalImageFolder,
    create_data_loaders, Trainer, evaluate_accuracy,
    print_header, print_section, print_config,
    ExperimentLogger, Colors
)
from src.utils.device import get_device, set_seed
from src.models import create_model, save_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Experiment 01: Baseline Generalization Gap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument('--use-processed', action='store_true', default=True,
                        help='Use processed pv_p/pd_p folders (default: True)')
    parser.add_argument('--class-filter', type=str, default='tomato',
                        help='Filter classes (tomato, all, or comma-separated list)')

    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the trained model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Optional path to save the trained baseline model')

    return parser.parse_args()


def _resolve_canonical_classes(filter_value: str) -> List[str]:
    """Resolve canonical classes via substring matching across lab/field folders."""
    tokens = [token.strip().lower() for token in filter_value.split(',') if token.strip()]
    matches = set()
    for token in tokens:
        for root in (PLANTVILLAGE_DIR, PLANTDOC_DIR):
            for child in root.iterdir():
                if child.is_dir() and token in child.name.lower():
                    matches.add(child.name)
    if not matches:
        raise ValueError(f"No classes matched filter '{filter_value}'.")
    return sorted(matches)


def main():
    args = parse_args()

    # ==========================================================================
    # SETUP
    # ==========================================================================
    print_header("Baseline Generalization Gap", experiment_num=1)

    set_seed(args.seed)
    device = get_device()
    logger = ExperimentLogger("exp01_baseline")

    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "baseline_model.pth"

    # Config
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )

    print_section("Configuration")
    print_config(config)

    # Determine class filter
    if args.class_filter.lower() == 'all':
        canonical_classes = None  # Use all available
        class_filter = None
    elif args.class_filter.lower() == 'tomato':
        canonical_classes = TOMATO_CLASSES
        class_filter = ['tomato']
    else:
        canonical_classes = _resolve_canonical_classes(args.class_filter)
        class_filter = canonical_classes

    print(f"  class_filter: {args.class_filter}")

    # ==========================================================================
    # DATA LOADING
    # ==========================================================================
    print_section("Loading Data")

    transforms_dict = get_transforms(config)

    # Lab data (PlantVillage - source domain)
    print(f"\n{Colors.BLUE}Lab Dataset:{Colors.RESET} PlantVillage (pv_p)")

    if canonical_classes:
        # Use canonical class alignment for consistent class indices
        lab_full = CanonicalImageFolder(
            str(PLANTVILLAGE_DIR),
            canonical_classes=canonical_classes,
            transform=transforms_dict['train'],
            class_name_mapping=CLASS_NAME_MAPPING
        )
    else:
        lab_full = FilteredImageFolder(
            str(PLANTVILLAGE_DIR),
            transform=transforms_dict['train'],
            class_filter=class_filter
        )

    # Split 80/20 for train/val
    train_size = int(0.8 * len(lab_full))
    val_size = len(lab_full) - train_size
    train_dataset, val_dataset = random_split(lab_full, [train_size, val_size])

    class_names = lab_full.classes
    num_classes = len(class_names)

    print(f"  Classes: {num_classes}")
    for i, name in enumerate(class_names):
        print(f"    {i}: {name}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Field data (PlantDoc - target domain)
    print(f"\n{Colors.BLUE}Field Dataset:{Colors.RESET} PlantDoc (pd_p)")

    if canonical_classes:
        field_dataset = CanonicalImageFolder(
            str(PLANTDOC_DIR),
            canonical_classes=canonical_classes,
            transform=transforms_dict['val'],
            class_name_mapping=CLASS_NAME_MAPPING
        )
    else:
        field_dataset = FilteredImageFolder(
            str(PLANTDOC_DIR),
            transform=transforms_dict['val'],
            class_filter=class_filter
        )
    print(f"  Test samples: {len(field_dataset)}")

    # Create loaders
    loaders = create_data_loaders(train_dataset, val_dataset, field_dataset, config)

    # ==========================================================================
    # MODEL
    # ==========================================================================
    print_section("Model")

    model = create_model(num_classes, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: MobileNetV3-Small")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # ==========================================================================
    # TRAINING
    # ==========================================================================
    print_section("Training")

    trainer = Trainer(model, device, config)
    model = trainer.train(loaders['train'], loaders['val'], epochs=config.epochs)

    print(f"\n{Colors.GREEN}Best validation accuracy: {trainer.metrics.best_val_acc:.4f}{Colors.RESET}")

    # ==========================================================================
    # EVALUATION
    # ==========================================================================
    print_section("Final Evaluation")

    lab_acc = evaluate_accuracy(model, loaders['val'], device, desc="Lab Validation")
    field_acc = evaluate_accuracy(model, loaders['test'], device, desc="Field Test")
    gap = lab_acc - field_acc

    print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
    print(f"  Lab Accuracy:   {Colors.GREEN}{lab_acc:.2f}%{Colors.RESET}")
    print(f"  Field Accuracy: {Colors.RED}{field_acc:.2f}%{Colors.RESET}")
    print(f"  {Colors.BOLD}Gap: {gap:.2f}%{Colors.RESET}")

    # ==========================================================================
    # SAVE
    # ==========================================================================
    if not args.no_save:
        print_section("Saving Model")

        ensure_dir(model_path.parent)
        save_model(model, model_path, metadata={
            'experiment': '01_baseline_gap',
            'lab_accuracy': lab_acc,
            'field_accuracy': field_acc,
            'gap': gap,
            'num_classes': num_classes,
            'class_names': class_names,
            'config': {
                'epochs': config.epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
            }
        })
        print(f"  Saved to: {model_path}")

    # Log results
    logger.log_results({
        'lab_accuracy': lab_acc,
        'field_accuracy': field_acc,
        'gap': gap,
        'num_classes': num_classes,
        'train_samples': len(train_dataset),
        'field_samples': len(field_dataset),
    })

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'=' * 60}")
    print(f"{Colors.BOLD}EXPERIMENT 01 COMPLETE{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"\nThe model trained on Lab data shows a {Colors.RED}{gap:.1f}%{Colors.RESET} accuracy drop on Field data.")
    print(f"This demonstrates the generalization gap that subsequent experiments aim to reduce.")
    print(f"\nNext: Run experiment 02 (passive augmentation) or 04 (active learning)")


if __name__ == '__main__':
    main()
