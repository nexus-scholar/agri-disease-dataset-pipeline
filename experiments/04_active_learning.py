#!/usr/bin/env python3
"""
Experiment 04: Active Learning Comparison

This experiment compares two sample selection strategies for active learning:
1. Random: Uniformly sample from unlabeled pool
2. Entropy: Select samples where model is most uncertain

Setup:
- Start with baseline model (from Exp 01)
- Field data split: 80% unlabeled pool, 20% held-out test
- Each round: select samples → fine-tune → evaluate

Key Insight: Entropy sampling shows an early "dip" because it selects hard samples
that confuse the model initially. This motivates the hybrid approach in Exp 05.

Usage:
    python 04_active_learning.py
    python 04_active_learning.py --budget-per-round 100 --num-rounds 3
"""

import argparse
import sys
from pathlib import Path
import copy

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ActiveLearningConfig, MODELS_DIR, PLANTDOC_DIR, TOMATO_CLASSES, CLASS_NAME_MAPPING
)
from src.utils import (
    get_transforms, CanonicalImageFolder,
    evaluate_accuracy, print_header, print_section, print_config,
    print_results_table, ExperimentLogger, Colors
)
from src.utils.device import get_device, set_seed
from src.models import load_model
from src.strategies.active_learning import select_samples, fine_tune


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation(strategy, baseline_path, pool_ds, test_ds, num_classes, al_config, device):
    """
    Run active learning simulation with given strategy.

    Returns:
        List of accuracies at each budget level
    """
    print(f"\n{Colors.BOLD}>>> Strategy: {strategy.upper()} <<<{Colors.RESET}")

    # Load fresh baseline model
    model = load_model(baseline_path, num_classes, device)

    # Setup
    test_loader = DataLoader(test_ds, batch_size=al_config.pool_batch_size, shuffle=False)
    pool_indices = list(range(len(pool_ds)))
    labeled_indices = []
    results = []

    # Initial accuracy (0 labels)
    acc = evaluate_accuracy(model, test_loader, device, desc="Initial")
    print(f"  0 labels: {acc:.2f}%")
    results.append(acc)

    # Active learning loop
    cumulative = 0
    for round_num, budget in enumerate(al_config.budget_rounds):
        cumulative += budget

        # Select samples
        new_indices, pool_indices = select_samples(
            strategy,
            pool_indices,
            budget,
            model=model,
            dataset=pool_ds,
            device=device,
            batch_size=al_config.pool_batch_size
        )
        labeled_indices.extend(new_indices)

        # Fine-tune on labeled set
        train_subset = Subset(pool_ds, labeled_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=al_config.train_batch_size,
            shuffle=True
        )
        model = fine_tune(model, train_loader, al_config, device)

        # Evaluate
        acc = evaluate_accuracy(model, test_loader, device, desc=f"Round {round_num+1}")
        print(f"  {cumulative} labels: {acc:.2f}%")
        results.append(acc)

    return results


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 04: Active Learning")

    parser.add_argument('--baseline-path', type=str, default=None,
                        help='Path to baseline model (default: data/models/baseline_model.pth)')

    # Active learning
    parser.add_argument('--budget-per-round', type=int, default=50)
    parser.add_argument('--num-rounds', type=int, default=4)
    parser.add_argument('--epochs-per-round', type=int, default=5)
    parser.add_argument('--fine-tune-lr', type=float, default=0.0001)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-filter', type=str, default='tomato')

    return parser.parse_args()


def main():
    args = parse_args()

    print_header("Active Learning Comparison", experiment_num=4)

    set_seed(args.seed)
    device = get_device()
    logger = ExperimentLogger("exp04_active_learning")

    # Config
    al_config = ActiveLearningConfig(
        budget_per_round=args.budget_per_round,
        num_rounds=args.num_rounds,
        epochs_per_round=args.epochs_per_round,
        fine_tune_lr=args.fine_tune_lr,
        random_seed=args.seed
    )

    canonical_classes = TOMATO_CLASSES if args.class_filter == 'tomato' else None

    print_section("Configuration")
    print_config(al_config)

    # Baseline model path
    baseline_path = Path(args.baseline_path) if args.baseline_path else MODELS_DIR / "baseline_model.pth"
    if not baseline_path.exists():
        print(f"{Colors.RED}Error: Baseline model not found: {baseline_path}{Colors.RESET}")
        print("Run experiment 01 first to create the baseline model.")
        return

    print(f"  baseline_model: {baseline_path}")

    # ==========================================================================
    # DATA
    # ==========================================================================
    print_section("Loading Field Data")

    transforms_dict = get_transforms()

    # Load dataset twice with different transforms
    print(f"\n{Colors.BLUE}Field Dataset:{Colors.RESET} PlantDoc (pd_p)")
    pool_ds = CanonicalImageFolder(
        str(PLANTDOC_DIR),
        canonical_classes=canonical_classes,
        transform=transforms_dict['train'],
        class_name_mapping=CLASS_NAME_MAPPING
    )
    test_ds = CanonicalImageFolder(
        str(PLANTDOC_DIR),
        canonical_classes=canonical_classes,
        transform=transforms_dict['val'],
        class_name_mapping=CLASS_NAME_MAPPING
    )

    num_classes = len(canonical_classes) if canonical_classes else len(pool_ds.classes)
    total_n = len(pool_ds)

    print(f"  Classes: {num_classes}")
    print(f"  Total field images: {total_n}")

    # Split: 80% pool, 20% test
    indices = np.arange(total_n)
    np.random.seed(al_config.random_seed)
    np.random.shuffle(indices)

    split = int(0.8 * total_n)
    pool_idx = indices[:split].tolist()
    test_idx = indices[split:].tolist()

    pool_subset = Subset(pool_ds, pool_idx)
    test_subset = Subset(test_ds, test_idx)

    print(f"  Pool (unlabeled): {len(pool_subset)}")
    print(f"  Test (held-out): {len(test_subset)}")

    # ==========================================================================
    # RUN SIMULATIONS
    # ==========================================================================
    print_section("Running Simulations")

    # Random strategy
    results_random = run_simulation(
        "random", baseline_path, pool_subset, test_subset,
        num_classes, al_config, device
    )

    # Entropy strategy
    results_entropy = run_simulation(
        "entropy", baseline_path, pool_subset, test_subset,
        num_classes, al_config, device
    )

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    print_section("Results Comparison")

    x_values = [0] + [sum(al_config.budget_rounds[:i+1]) for i in range(len(al_config.budget_rounds))]

    print_results_table(
        {"Random": results_random, "Entropy": results_entropy},
        x_values,
        "Final Results"
    )

    # Compute final difference
    final_diff = results_entropy[-1] - results_random[-1]

    print(f"\n{Colors.BOLD}Final Comparison:{Colors.RESET}")
    print(f"  Random:  {results_random[-1]:.2f}%")
    print(f"  Entropy: {results_entropy[-1]:.2f}%")
    print(f"  Difference: {final_diff:+.2f}%")

    logger.log_results({
        'random_final': results_random[-1],
        'entropy_final': results_entropy[-1],
        'difference': final_diff,
        'random_results': results_random,
        'entropy_results': results_entropy,
    })

    print(f"\n{'=' * 60}")
    print(f"{Colors.BOLD}EXPERIMENT 04 COMPLETE{Colors.RESET}")
    print(f"{'=' * 60}")

    if final_diff > 0:
        print(f"\nEntropy sampling outperforms random by {final_diff:.1f}%")
    else:
        print(f"\nRandom sampling outperforms entropy by {-final_diff:.1f}%")

    print("\nNote: Entropy often shows an early 'dip' due to selecting hard samples.")
    print("Experiment 05 addresses this with a hybrid warm-start approach.")


if __name__ == '__main__':
    main()

