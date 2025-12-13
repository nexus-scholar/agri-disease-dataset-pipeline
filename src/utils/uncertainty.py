"""
Uncertainty estimation utilities for active learning.

Provides:
- Entropy-based uncertainty
- Margin sampling
- Least confidence
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


def compute_entropy(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Compute entropy (uncertainty) for each sample.

    Entropy = -sum(p * log(p))
    Higher entropy = more uncertain = more informative

    Args:
        model: Neural network model
        loader: DataLoader (should be in order, no shuffle)
        device: torch device

    Returns:
        Array of entropy values for each sample
    """
    model.eval()
    entropies = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            # Entropy calculation
            log_probs = torch.log(probs + 1e-10)  # Numerical stability
            entropy = -(probs * log_probs).sum(dim=1)
            entropies.extend(entropy.cpu().numpy())

    return np.array(entropies)


def compute_margin(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Compute margin (difference between top 2 probabilities).

    Smaller margin = more uncertain

    Args:
        model: Neural network model
        loader: DataLoader
        device: torch device

    Returns:
        Array of margin values (negated so higher = more uncertain)
    """
    model.eval()
    margins = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            # Sort and get top 2
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]

            # Negate so higher = more uncertain
            margins.extend((-margin).cpu().numpy())

    return np.array(margins)


def compute_least_confidence(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Compute least confidence (1 - max probability).

    Higher = more uncertain

    Args:
        model: Neural network model
        loader: DataLoader
        device: torch device

    Returns:
        Array of uncertainty values
    """
    model.eval()
    uncertainties = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            max_probs, _ = probs.max(dim=1)
            uncertainty = 1.0 - max_probs
            uncertainties.extend(uncertainty.cpu().numpy())

    return np.array(uncertainties)


def get_predictions(model, loader: DataLoader, device: torch.device):
    """
    Get predictions, probabilities and true labels.

    Returns:
        (predictions, probabilities, labels) as numpy arrays
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = probs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

