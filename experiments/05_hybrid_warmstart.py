#!/usr/bin/env python3
"""
Experiment 05: Hybrid Warm-Start Strategy

This experiment proposes a balanced sampling strategy that combines the benefits
of both random and entropy-based selection.

The Problem:
- Random: Good coverage but misses informative samples
- Entropy: Finds hard samples but causes early "dip" in accuracy

The Solution (Hybrid Warm-Start):
- Round 0: 50% Random + 50% Entropy (warm start)
- Subsequent rounds: Pure entropy sampling

Usage:
    python 05_hybrid_warmstart.py
    python 05_hybrid_warmstart.py --strategies random,entropy,hybrid
"""

import argparse
import sys
from pathlib import Path
import copy
import json
from typing import List

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ActiveLearningConfig, MODELS_DIR, PLANTDOC_DIR, TOMATO_CLASSES, CLASS_NAME_MAPPING,
    RESULTS_DIR, ensure_dir, PLANTVILLAGE_DIR
)
from src.utils import (
    get_transforms, CanonicalImageFolder,
    evaluate_accuracy, print_header, print_section, print_config,
    print_results_table, ExperimentLogger, Colors
)
from src.utils.device import get_device, set_seed
from src.utils.uncertainty import compute_entropy
from src.models import load_model
from src.strategies.active_learning import fine_tune, select_samples


# =============================================================================
# HYBRID SELECTION
# =============================================================================

def select_samples_hybrid(model, dataset, pool_indices, n, device, al_config, round_num):
    """
    Hybrid selection strategy.

    Round 0: 50% random + 50% entropy (warm start)
    Later rounds: Pure entropy
    """
    pool_indices = list(pool_indices)

    if round_num == 0:
        # WARM START: Split between random and entropy
        n_random = n // 2
        n_entropy = n - n_random

        print(f"    {Colors.CYAN}Warm start: {n_random} random + {n_entropy} entropy{Colors.RESET}")

        # Random selection
        np.random.shuffle(pool_indices)
        random_selection = pool_indices[:n_random]
        remaining_after_random = pool_indices[n_random:]

        # Entropy selection from remaining
        pool_subset = Subset(dataset, remaining_after_random)
        pool_loader = DataLoader(pool_subset, batch_size=al_config.pool_batch_size, shuffle=False)
        uncertainties = compute_entropy(model, pool_loader, device)

        sorted_indices = np.argsort(uncertainties)[::-1]
        sorted_pool = [remaining_after_random[i] for i in sorted_indices]

        entropy_selection = sorted_pool[:n_entropy]
        remaining = sorted_pool[n_entropy:]

        return random_selection + entropy_selection, remaining

    else:
        # Pure entropy for subsequent rounds
        return select_samples('entropy', pool_indices, n, model=model,
                             dataset=dataset, device=device, batch_size=al_config.pool_batch_size)


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation(strategy, baseline_path, pool_ds, test_ds, num_classes, al_config, device):
    """Run active learning simulation."""
    print(f"\n{Colors.BOLD}>>> Strategy: {strategy.upper()} <<<{Colors.RESET}")

    # Load fresh baseline
    model = load_model(baseline_path, num_classes, device)

    # Setup
    test_loader = DataLoader(test_ds, batch_size=al_config.pool_batch_size, shuffle=False)
    pool_indices = list(range(len(pool_ds)))
    labeled_indices = []
    results = []

    # Initial (0 labels)
    acc = evaluate_accuracy(model, test_loader, device, desc="Initial")
    print(f"  0 labels: {acc:.2f}%")
    results.append(acc)

    # Active learning loop
    cumulative = 0
    for round_num, budget in enumerate(al_config.budget_rounds):
        cumulative += budget

        # Select samples based on strategy
        if strategy == "hybrid":
            new_indices, pool_indices = select_samples_hybrid(
                model, pool_ds, pool_indices, budget, device, al_config, round_num
            )
        else:
            new_indices, pool_indices = select_samples(
                strategy, pool_indices, budget,
                model=model, dataset=pool_ds, device=device,
                batch_size=al_config.pool_batch_size
            )

        labeled_indices.extend(new_indices)

        # Fine-tune
        train_subset = Subset(pool_ds, labeled_indices)
        train_loader = DataLoader(train_subset, batch_size=al_config.train_batch_size, shuffle=True)
        model = fine_tune(model, train_loader, al_config, device)

        # Evaluate
        acc = evaluate_accuracy(model, test_loader, device, desc=f"Round {round_num+1}")
        print(f"  {cumulative} labels: {acc:.2f}%")
        results.append(acc)

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def _save_accuracy_plot(path: Path, x_values, results_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for name, values in results_dict.items():
        plt.plot(x_values, values, marker='o', label=name)
    plt.xlabel('Labeled Samples')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Active Learning Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# =============================================================================
# FILE LOADING
# =============================================================================

def _resolve_canonical_classes(filter_value: str) -> List[str]:
    """Resolve canonical classes across lab/field folders via substring matching."""
    tokens = [token.strip().lower() for token in filter_value.split(',') if token.strip()]
    matches = set()
    for token in tokens:
        for root in (PLANTDOC_DIR, PLANTVILLAGE_DIR):
            for child in root.iterdir():
                if child.is_dir() and token in child.name.lower():
                    matches.add(child.name)
    if not matches:
        raise ValueError(f"No classes matched filter '{filter_value}'.")
    return sorted(matches)


def _build_index_map(dataset):
    base = Path(dataset.root)
    return {Path(path).relative_to(base).as_posix(): idx for idx, (path, _) in enumerate(dataset.samples)}


def _load_split_indices(split_path: Path, pool_dataset, test_dataset):
    with open(split_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    pool_files = data['pool_files']
    test_files = data['test_files']

    pool_map = _build_index_map(pool_dataset)
    test_map = _build_index_map(test_dataset)

    pool_idx = [pool_map[f] for f in pool_files]
    test_idx = [test_map[f] for f in test_files]

    return pool_idx, test_idx


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 05: Hybrid Warm-Start")

    parser.add_argument('--baseline-path', type=str, default=None)
    parser.add_argument('--strategies', type=str, default='random,entropy,hybrid',
                        help='Comma-separated list of strategies to compare')

    parser.add_argument('--budget-per-round', type=int, default=50)
    parser.add_argument('--num-rounds', type=int, default=4)
    parser.add_argument('--epochs-per-round', type=int, default=5)
    parser.add_argument('--fine-tune-lr', type=float, default=0.0001)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-filter', type=str, default='tomato')
    parser.add_argument('--split-file', type=str, default=None,
                        help='JSON file containing pre-defined pool/test splits')
    parser.add_argument('--plot-path', type=str, default=None,
                        help='Optional path to save accuracy curve plot')

    return parser.parse_args()


def main():
    args = parse_args()

    print_header("Hybrid Warm-Start Strategy", experiment_num=5)

    set_seed(args.seed)
    device = get_device()
    logger = ExperimentLogger("exp05_hybrid")

    # Config
    al_config = ActiveLearningConfig(
        budget_per_round=args.budget_per_round,
        num_rounds=args.num_rounds,
        epochs_per_round=args.epochs_per_round,
        fine_tune_lr=args.fine_tune_lr,
        random_seed=args.seed
    )

    canonical_classes = TOMATO_CLASSES if args.class_filter == 'tomato' else _resolve_canonical_classes(args.class_filter)

    print_section("Configuration")
    print_config(al_config)

    strategies = [s.strip() for s in args.strategies.split(',')]
    print(f"  strategies: {strategies}")
    print(f"  class_filter: {args.class_filter}")

    # Baseline model
    baseline_path = Path(args.baseline_path) if args.baseline_path else MODELS_DIR / "baseline_model.pth"
    if not baseline_path.exists():
        print(f"{Colors.RED}Error: Baseline model not found: {baseline_path}{Colors.RESET}")
        print("Run experiment 01 first.")
        return

    # ==========================================================================
    # DATA
    # ==========================================================================
    print_section("Loading Data")

    transforms_dict = get_transforms()
    num_classes = len(canonical_classes) if canonical_classes else 7

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

    total_n = len(pool_ds)
    print(f"  Field images: {total_n}")

    # Split
    if args.split_file:
        pool_idx, test_idx = _load_split_indices(Path(args.split_file), pool_ds, test_ds)
    else:
        indices = np.arange(total_n)
        np.random.seed(al_config.random_seed)
        np.random.shuffle(indices)
        split = int(0.8 * total_n)
        pool_idx = indices[:split].tolist()
        test_idx = indices[split:].tolist()

    pool_subset = Subset(pool_ds, pool_idx)
    test_subset = Subset(test_ds, test_idx)

    print(f"  Pool: {len(pool_subset)} | Test: {len(test_subset)}")

    # ==========================================================================
    # RUN SIMULATIONS
    # ==========================================================================
    print_section("Running Strategy Comparisons")

    all_results = {}

    for strategy in strategies:
        results = run_simulation(
            strategy, baseline_path, pool_subset, test_subset,
            num_classes, al_config, device
        )
        all_results[strategy.capitalize()] = results

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    print_section("Final Comparison")

    x_values = [0] + [sum(al_config.budget_rounds[:i+1]) for i in range(len(al_config.budget_rounds))]

    print_results_table(all_results, x_values, "Strategy Comparison")

    # Find winner
    final_accs = {name: results[-1] for name, results in all_results.items()}
    winner = max(final_accs, key=final_accs.get)

    print(f"\n{Colors.BOLD}Final Accuracies:{Colors.RESET}")
    for name, acc in final_accs.items():
        marker = " ← BEST" if name == winner else ""
        color = Colors.GREEN if name == winner else ""
        reset = Colors.RESET if color else ""
        print(f"  {color}{name}: {acc:.2f}%{marker}{reset}")

    logger.log_results({
        'strategies': strategies,
        'winner': winner,
        'final_accuracies': final_accs,
    })

    plot_target = Path(args.plot_path) if args.plot_path else ensure_dir(RESULTS_DIR / 'figures') / f"exp05_{args.class_filter}_accuracy.png"
    _save_accuracy_plot(plot_target, x_values, all_results)
    print(f"\nSaved accuracy curve: {plot_target}")

    print(f"\n{'=' * 60}")
    print(f"{Colors.BOLD}EXPERIMENT 05 COMPLETE{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"\n{Colors.GREEN}Winner: {winner} with {final_accs[winner]:.2f}% accuracy{Colors.RESET}")


if __name__ == '__main__':
    main()
