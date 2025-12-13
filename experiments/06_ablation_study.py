#!/usr/bin/env python3
"""
Experiment 06: Ablation Study - Entropy/Random Ratio Tuning

This experiment systematically tests different ratios of entropy-based vs random
sampling to find the optimal exploration-exploitation balance.

Ratios Tested:
- 100% entropy (pure exploitation)
- 70% entropy / 30% random
- 50% entropy / 50% random (proposed hybrid)
- 30% entropy / 70% random
- 100% random (pure exploration)

Usage:
    python 06_ablation_study.py
    python 06_ablation_study.py --budget 200 --rounds 4
"""

import argparse
import sys
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ActiveLearningConfig, MODELS_DIR, RESULTS_DIR,
    PLANTVILLAGE_DIR, PLANTDOC_DIR, TOMATO_CLASSES, CLASS_NAME_MAPPING, ensure_dir
)
from src.utils import (
    get_transforms, CanonicalImageFolder,
    evaluate_accuracy, print_header, print_section,
    ExperimentLogger, Colors
)
from src.utils.device import get_device, set_seed
from src.utils.uncertainty import compute_entropy
from src.models import load_model
from src.strategies.active_learning import fine_tune


# Ratios to test: (entropy_fraction, random_fraction, name)
RATIOS_TO_TEST = [
    (1.0, 0.0, "entropy_100"),
    (0.7, 0.3, "entropy_70"),
    (0.5, 0.5, "hybrid_50"),
    (0.3, 0.7, "entropy_30"),
    (0.0, 1.0, "random_100"),
]


def select_samples_by_ratio(model, dataset, pool_indices, n, device,
                            entropy_fraction, random_fraction, batch_size=32):
    """Select samples using specified entropy/random ratio."""
    pool_indices = list(pool_indices)

    n_entropy = int(n * entropy_fraction)
    n_random = n - n_entropy

    selected = []

    if n_random > 0 and n_entropy > 0:
        # Mixed selection
        np.random.shuffle(pool_indices)
        random_selection = pool_indices[:n_random]
        remaining_pool = pool_indices[n_random:]

        if len(remaining_pool) > 0 and n_entropy > 0:
            pool_subset = Subset(dataset, remaining_pool)
            pool_loader = DataLoader(pool_subset, batch_size=batch_size, shuffle=False)
            uncertainties = compute_entropy(model, pool_loader, device)

            sorted_indices = np.argsort(uncertainties)[::-1]
            entropy_selection = [remaining_pool[i] for i in sorted_indices[:n_entropy]]
            remaining = [remaining_pool[i] for i in sorted_indices[n_entropy:]]
        else:
            entropy_selection = []
            remaining = remaining_pool

        selected = random_selection + entropy_selection

    elif n_entropy > 0:
        pool_subset = Subset(dataset, pool_indices)
        pool_loader = DataLoader(pool_subset, batch_size=batch_size, shuffle=False)
        uncertainties = compute_entropy(model, pool_loader, device)

        sorted_indices = np.argsort(uncertainties)[::-1]
        selected = [pool_indices[i] for i in sorted_indices[:n]]
        remaining = [pool_indices[i] for i in sorted_indices[n:]]

    else:
        np.random.shuffle(pool_indices)
        selected = pool_indices[:n]
        remaining = pool_indices[n:]

    return selected, remaining


def run_ablation_for_ratio(entropy_fraction, random_fraction, ratio_name,
                           base_model_path, field_dataset, test_loader,
                           device, al_config, num_classes, seed=42):
    """Run active learning experiment for a specific ratio."""

    print(f"\n{Colors.BOLD}Testing ratio: {ratio_name} "
          f"(entropy={entropy_fraction:.0%}, random={random_fraction:.0%}){Colors.RESET}")

    set_seed(seed)

    # Load fresh model
    model = load_model(base_model_path, num_classes, device)

    pool_indices = list(range(len(field_dataset)))
    labeled_indices = []

    accuracies = []
    budget_points = [0]

    # Initial accuracy
    initial_acc = evaluate_accuracy(model, test_loader, device, "Initial")
    accuracies.append(initial_acc)
    print(f"  Initial: {initial_acc:.2f}%")

    # Active learning loop
    for round_num in range(al_config.num_rounds):
        budget = al_config.budget_per_round

        selected, pool_indices = select_samples_by_ratio(
            model, field_dataset, pool_indices, budget, device,
            entropy_fraction, random_fraction, al_config.pool_batch_size
        )
        labeled_indices.extend(selected)

        train_subset = Subset(field_dataset, labeled_indices)
        train_loader = DataLoader(train_subset, batch_size=al_config.train_batch_size, shuffle=True)
        model = fine_tune(model, train_loader, al_config, device)

        acc = evaluate_accuracy(model, test_loader, device, f"Round {round_num+1}")
        accuracies.append(acc)
        budget_points.append(len(labeled_indices))

        print(f"  Round {round_num+1}: {len(labeled_indices)} labels -> {acc:.2f}%")

    return {
        'ratio_name': ratio_name,
        'entropy_fraction': entropy_fraction,
        'random_fraction': random_fraction,
        'budget_points': budget_points,
        'accuracies': accuracies,
        'final_accuracy': accuracies[-1]
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-06: Ablation Study")
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-filter', type=str, default='tomato')
    args = parser.parse_args()

    print_header("ABLATION STUDY: Entropy/Random Ratio Tuning", 6)

    device = get_device()
    set_seed(args.seed)

    al_config = ActiveLearningConfig(
        budget_per_round=args.budget,
        num_rounds=args.rounds,
        epochs_per_round=args.epochs,
        random_seed=args.seed
    )

    canonical_classes = TOMATO_CLASSES if args.class_filter == 'tomato' else None
    num_classes = len(canonical_classes) if canonical_classes else 7

    print_section("Configuration")
    print(f"  Budget per round: {al_config.budget_per_round}")
    print(f"  Number of rounds: {al_config.num_rounds}")
    print(f"  Total budget: {al_config.total_budget}")

    # Load data
    print_section("Loading Data")
    transforms_dict = get_transforms()

    field_dataset = CanonicalImageFolder(
        str(PLANTDOC_DIR),
        canonical_classes=canonical_classes,
        transform=transforms_dict['val'],
        class_name_mapping=CLASS_NAME_MAPPING
    )

    print(f"  Field dataset: {len(field_dataset)} images, {num_classes} classes")
    test_loader = DataLoader(field_dataset, batch_size=32, shuffle=False)

    # Check baseline
    base_model_path = MODELS_DIR / "baseline_model.pth"
    if not base_model_path.exists():
        print(f"{Colors.RED}Error: Run experiment 01 first to create baseline model{Colors.RESET}")
        return 1

    # Run ablation
    print_section("Running Ablation Study")
    all_results = []

    for entropy_frac, random_frac, name in RATIOS_TO_TEST:
        result = run_ablation_for_ratio(
            entropy_frac, random_frac, name,
            base_model_path, field_dataset, test_loader,
            device, al_config, num_classes, args.seed
        )
        all_results.append(result)

    # Save results
    print_section("Results Summary")

    results_dir = RESULTS_DIR / "tables"
    ensure_dir(results_dir)

    csv_path = results_dir / "exp06_ablation_study.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['ratio_name', 'entropy_fraction', 'final_accuracy']
        writer.writerow(header)
        for result in all_results:
            writer.writerow([result['ratio_name'], result['entropy_fraction'], result['final_accuracy']])

    print(f"\n{'Ratio':<15} | {'Final Acc':<10}")
    print("-" * 30)
    for result in all_results:
        print(f"{result['ratio_name']:<15} | {result['final_accuracy']:>8.2f}%")

    best = max(all_results, key=lambda x: x['final_accuracy'])
    print(f"\n{Colors.GREEN}Best: {best['ratio_name']} with {best['final_accuracy']:.2f}%{Colors.RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

