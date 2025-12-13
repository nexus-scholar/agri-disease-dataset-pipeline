#!/usr/bin/env python3
"""
Experiment 07: Seed Robustness Analysis

This experiment validates the statistical significance of the hybrid approach
by running experiments with multiple random seeds.

Design:
- Run hybrid strategy and random baseline with 5 different seeds
- Seeds: [42, 100, 2025, 777, 1234]
- Calculate mean, std, and 95% confidence intervals
- Perform paired t-test to compare methods

Usage:
    python 07_seed_robustness.py
    python 07_seed_robustness.py --seeds 42,100,2025,777,1234
"""

import argparse
import sys
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ActiveLearningConfig, MODELS_DIR, RESULTS_DIR,
    PLANTDOC_DIR, TOMATO_CLASSES, CLASS_NAME_MAPPING, ensure_dir
)
from src.utils import (
    get_transforms, CanonicalImageFolder,
    evaluate_accuracy, print_header, print_section, Colors
)
from src.utils.device import get_device, set_seed
from src.utils.uncertainty import compute_entropy
from src.models import load_model
from src.strategies.active_learning import fine_tune, select_samples


DEFAULT_SEEDS = [42, 100, 2025, 777, 1234]


def select_hybrid(model, dataset, pool_indices, n, device, batch_size=32):
    """Hybrid: 50% random + 50% entropy."""
    pool_indices = list(pool_indices)
    n_random = n // 2
    n_entropy = n - n_random

    np.random.shuffle(pool_indices)
    random_selection = pool_indices[:n_random]
    remaining = pool_indices[n_random:]

    if len(remaining) > 0 and n_entropy > 0:
        pool_subset = Subset(dataset, remaining)
        pool_loader = DataLoader(pool_subset, batch_size=batch_size, shuffle=False)
        uncertainties = compute_entropy(model, pool_loader, device)

        sorted_idx = np.argsort(uncertainties)[::-1]
        entropy_selection = [remaining[i] for i in sorted_idx[:n_entropy]]
        final_remaining = [remaining[i] for i in sorted_idx[n_entropy:]]
    else:
        entropy_selection = []
        final_remaining = remaining

    return random_selection + entropy_selection, final_remaining


def run_strategy(strategy_name, model_path, field_dataset, test_loader,
                 device, al_config, num_classes, seed):
    """Run one complete active learning experiment."""
    set_seed(seed)

    model = load_model(model_path, num_classes, device)
    pool_indices = list(range(len(field_dataset)))
    labeled_indices = []

    accuracies = []

    acc = evaluate_accuracy(model, test_loader, device)
    accuracies.append(acc)

    for round_num in range(al_config.num_rounds):
        budget = al_config.budget_per_round

        if strategy_name == "hybrid":
            selected, pool_indices = select_hybrid(
                model, field_dataset, pool_indices, budget, device
            )
        else:
            selected, pool_indices = select_samples(
                'random', pool_indices, budget
            )

        labeled_indices.extend(selected)

        train_subset = Subset(field_dataset, labeled_indices)
        train_loader = DataLoader(train_subset, batch_size=al_config.train_batch_size, shuffle=True)
        model = fine_tune(model, train_loader, al_config, device)

        acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(acc)

    return {
        'seed': seed,
        'strategy': strategy_name,
        'accuracies': accuracies,
        'final_accuracy': accuracies[-1]
    }


def compute_statistics(results_list):
    """Compute mean, std, and 95% CI."""
    finals = [r['final_accuracy'] for r in results_list]
    mean = np.mean(finals)
    std = np.std(finals)
    ci_95 = 1.96 * std / np.sqrt(len(finals))

    return {
        'final_mean': mean,
        'final_std': std,
        'final_ci_95': ci_95
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-07: Seed Robustness")
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--class-filter', type=str, default='tomato')
    args = parser.parse_args()

    print_header("SEED ROBUSTNESS ANALYSIS", 7)

    seeds = [int(s.strip()) for s in args.seeds.split(',')] if args.seeds else DEFAULT_SEEDS
    device = get_device()

    al_config = ActiveLearningConfig(
        budget_per_round=args.budget,
        num_rounds=args.rounds,
        epochs_per_round=args.epochs
    )

    canonical_classes = TOMATO_CLASSES if args.class_filter == 'tomato' else None
    num_classes = len(canonical_classes) if canonical_classes else 7

    print_section("Configuration")
    print(f"  Seeds: {seeds}")
    print(f"  Budget per round: {al_config.budget_per_round}")

    # Load data
    print_section("Loading Data")
    transforms_dict = get_transforms()

    field_dataset = CanonicalImageFolder(
        str(PLANTDOC_DIR),
        canonical_classes=canonical_classes,
        transform=transforms_dict['val'],
        class_name_mapping=CLASS_NAME_MAPPING
    )
    test_loader = DataLoader(field_dataset, batch_size=32, shuffle=False)
    print(f"  Field dataset: {len(field_dataset)} images")

    base_model_path = MODELS_DIR / "baseline_model.pth"
    if not base_model_path.exists():
        print(f"{Colors.RED}Error: Run experiment 01 first{Colors.RESET}")
        return 1

    # Run experiments
    print_section("Running Experiments")

    hybrid_results = []
    random_results = []

    for seed in seeds:
        print(f"\n{Colors.CYAN}Seed: {seed}{Colors.RESET}")

        print(f"  Running hybrid...")
        hybrid_result = run_strategy("hybrid", base_model_path, field_dataset,
                                     test_loader, device, al_config, num_classes, seed)
        hybrid_results.append(hybrid_result)
        print(f"    Final: {hybrid_result['final_accuracy']:.2f}%")

        print(f"  Running random...")
        random_result = run_strategy("random", base_model_path, field_dataset,
                                     test_loader, device, al_config, num_classes, seed)
        random_results.append(random_result)
        print(f"    Final: {random_result['final_accuracy']:.2f}%")

    # Statistics
    print_section("Results")

    hybrid_stats = compute_statistics(hybrid_results)
    random_stats = compute_statistics(random_results)

    print(f"\nHybrid: {hybrid_stats['final_mean']:.2f}% ± {hybrid_stats['final_std']:.2f}%")
    print(f"Random: {random_stats['final_mean']:.2f}% ± {random_stats['final_std']:.2f}%")

    improvement = hybrid_stats['final_mean'] - random_stats['final_mean']
    print(f"\nImprovement: {improvement:+.2f}%")

    if HAS_SCIPY:
        t_stat, p_value = stats.ttest_rel(
            [r['final_accuracy'] for r in hybrid_results],
            [r['final_accuracy'] for r in random_results]
        )
        print(f"p-value: {p_value:.6f}")
        if p_value < 0.05:
            print(f"{Colors.GREEN}Significant at p < 0.05{Colors.RESET}")

    # Save
    results_dir = RESULTS_DIR / "tables"
    ensure_dir(results_dir)

    csv_path = results_dir / "exp07_seed_robustness.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'strategy', 'final_accuracy'])
        for r in hybrid_results + random_results:
            writer.writerow([r['seed'], r['strategy'], r['final_accuracy']])

    print(f"\nResults saved to: {csv_path}")
    print(f"\n{Colors.GREEN}Experiment 07 complete{Colors.RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

