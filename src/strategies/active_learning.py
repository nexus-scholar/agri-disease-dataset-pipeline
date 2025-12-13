"""
Active learning strategies for sample selection.

Implements:
- Random sampling
- Entropy-based uncertainty sampling
- Margin sampling
- Hybrid warm-start strategy
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from typing import List, Tuple, Callable

from ..config import ActiveLearningConfig
from ..utils.uncertainty import compute_entropy, compute_margin, compute_least_confidence
from ..utils.metrics import evaluate_accuracy, progress_bar


# =============================================================================
# SAMPLE SELECTION STRATEGIES
# =============================================================================

def select_random(
    pool_indices: List[int],
    n: int,
    **kwargs
) -> Tuple[List[int], List[int]]:
    """
    Random sampling strategy.

    Args:
        pool_indices: Available indices to sample from
        n: Number of samples to select

    Returns:
        (selected, remaining) index lists
    """
    pool = list(pool_indices)
    np.random.shuffle(pool)
    return pool[:n], pool[n:]


def select_entropy(
    pool_indices: List[int],
    n: int,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    **kwargs
) -> Tuple[List[int], List[int]]:
    """
    Entropy-based uncertainty sampling.

    Selects samples where model is most uncertain (highest entropy).

    Args:
        pool_indices: Available indices to sample from
        n: Number of samples to select
        model: Current model for uncertainty estimation
        dataset: Full dataset
        device: torch device
        batch_size: Batch size for uncertainty computation

    Returns:
        (selected, remaining) index lists
    """
    pool = list(pool_indices)

    # Create loader for pool
    pool_subset = Subset(dataset, pool)
    pool_loader = DataLoader(pool_subset, batch_size=batch_size, shuffle=False)

    # Compute uncertainties
    uncertainties = compute_entropy(model, pool_loader, device)

    # Select top-n most uncertain
    sorted_indices = np.argsort(uncertainties)[::-1]  # Descending
    sorted_pool = [pool[i] for i in sorted_indices]

    return sorted_pool[:n], sorted_pool[n:]


def select_margin(
    pool_indices: List[int],
    n: int,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 32,
    **kwargs
) -> Tuple[List[int], List[int]]:
    """
    Margin-based uncertainty sampling.

    Selects samples where difference between top 2 predictions is smallest.
    """
    pool = list(pool_indices)

    pool_subset = Subset(dataset, pool)
    pool_loader = DataLoader(pool_subset, batch_size=batch_size, shuffle=False)

    uncertainties = compute_margin(model, pool_loader, device)

    sorted_indices = np.argsort(uncertainties)[::-1]
    sorted_pool = [pool[i] for i in sorted_indices]

    return sorted_pool[:n], sorted_pool[n:]


STRATEGIES = {
    'random': select_random,
    'entropy': select_entropy,
    'margin': select_margin,
}


def select_samples(
    strategy: str,
    pool_indices: List[int],
    n: int,
    **kwargs
) -> Tuple[List[int], List[int]]:
    """
    Select samples using specified strategy.

    Args:
        strategy: 'random', 'entropy', or 'margin'
        pool_indices: Available indices
        n: Number to select
        **kwargs: Additional arguments for strategy (model, dataset, device, etc.)

    Returns:
        (selected, remaining) index lists
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGIES.keys())}")

    return STRATEGIES[strategy](pool_indices, n, **kwargs)


# =============================================================================
# FINE-TUNING
# =============================================================================

def fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    config: ActiveLearningConfig,
    device: torch.device
) -> nn.Module:
    """
    Fine-tune model on labeled samples.

    Args:
        model: Model to fine-tune
        train_loader: DataLoader with labeled samples
        config: Active learning configuration
        device: torch device

    Returns:
        Fine-tuned model
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.fine_tune_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs_per_round):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# =============================================================================
# ACTIVE LEARNING LOOP
# =============================================================================

class ActiveLearner:
    """
    Active learning controller.

    Manages the loop of: select -> label -> train -> evaluate
    """

    def __init__(
        self,
        model: nn.Module,
        pool_dataset: Dataset,
        test_dataset: Dataset,
        config: ActiveLearningConfig,
        device: torch.device
    ):
        self.model = model
        self.pool_dataset = pool_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device

        # State
        self.pool_indices = list(range(len(pool_dataset)))
        self.labeled_indices = []
        self.history = []

    def run(
        self,
        strategy: str = 'entropy',
        verbose: bool = True
    ) -> List[float]:
        """
        Run active learning simulation.

        Args:
            strategy: Sample selection strategy
            verbose: Print progress

        Returns:
            List of accuracies at each budget level
        """
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.pool_batch_size,
            shuffle=False
        )

        results = []

        # Initial accuracy (0 labels)
        acc = evaluate_accuracy(self.model, test_loader, self.device, desc="Initial")
        if verbose:
            print(f"  0 labels: {acc:.2f}%")
        results.append(acc)

        # Active learning loop
        cumulative = 0
        for round_num, budget in enumerate(self.config.budget_rounds):
            cumulative += budget

            # Select samples
            selected, self.pool_indices = select_samples(
                strategy,
                self.pool_indices,
                budget,
                model=self.model,
                dataset=self.pool_dataset,
                device=self.device,
                batch_size=self.config.pool_batch_size
            )
            self.labeled_indices.extend(selected)

            # Fine-tune on labeled set
            train_subset = Subset(self.pool_dataset, self.labeled_indices)
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.train_batch_size,
                shuffle=True
            )
            self.model = fine_tune(
                self.model, train_loader, self.config, self.device
            )

            # Evaluate
            acc = evaluate_accuracy(
                self.model, test_loader, self.device,
                desc=f"Round {round_num + 1}"
            )
            if verbose:
                print(f"  {cumulative} labels: {acc:.2f}%")
            results.append(acc)

        self.history = results
        return results

    def get_model(self) -> nn.Module:
        """Return the current model."""
        return self.model

    def reset(self, model: nn.Module):
        """Reset learner with a fresh model."""
        self.model = model
        self.pool_indices = list(range(len(self.pool_dataset)))
        self.labeled_indices = []
        self.history = []


# =============================================================================
# HYBRID WARM-START STRATEGY
# =============================================================================

def run_hybrid_warmstart(
    baseline_model: nn.Module,
    pool_dataset: Dataset,
    test_dataset: Dataset,
    config: ActiveLearningConfig,
    device: torch.device,
    warmup_rounds: int = 1,
    verbose: bool = True
) -> List[float]:
    """
    Hybrid warm-start strategy.

    Uses random sampling for initial rounds (warm-up), then switches to
    entropy-based selection. This avoids the "entropy dip" problem.

    Args:
        baseline_model: Pre-trained baseline model
        pool_dataset: Unlabeled pool dataset
        test_dataset: Test dataset
        config: Active learning configuration
        device: torch device
        warmup_rounds: Number of rounds to use random sampling
        verbose: Print progress

    Returns:
        List of accuracies at each budget level
    """
    model = copy.deepcopy(baseline_model).to(device)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.pool_batch_size,
        shuffle=False
    )

    pool_indices = list(range(len(pool_dataset)))
    labeled_indices = []
    results = []

    # Initial accuracy
    acc = evaluate_accuracy(model, test_loader, device, desc="Initial")
    if verbose:
        print(f"  0 labels: {acc:.2f}%")
    results.append(acc)

    # Active learning loop
    cumulative = 0
    for round_num, budget in enumerate(config.budget_rounds):
        cumulative += budget

        # Choose strategy based on round
        if round_num < warmup_rounds:
            strategy = 'random'
            phase = "WARM-UP"
        else:
            strategy = 'entropy'
            phase = "ENTROPY"

        if verbose:
            print(f"  Round {round_num + 1} ({phase}): ", end="")

        # Select samples
        selected, pool_indices = select_samples(
            strategy,
            pool_indices,
            budget,
            model=model,
            dataset=pool_dataset,
            device=device,
            batch_size=config.pool_batch_size
        )
        labeled_indices.extend(selected)

        # Fine-tune
        train_subset = Subset(pool_dataset, labeled_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=config.train_batch_size,
            shuffle=True
        )
        model = fine_tune(model, train_loader, config, device)

        # Evaluate
        acc = evaluate_accuracy(model, test_loader, device)
        if verbose:
            print(f"{cumulative} labels -> {acc:.2f}%")
        results.append(acc)

    return results

