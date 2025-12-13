"""
Training and evaluation metrics utilities.
"""

import copy
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..config import TrainingConfig, ensure_dir, LOGS_DIR

# Optional tqdm import
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def progress_bar(iterable, desc: str = "", total: int = None, leave: bool = True):
    """Wrapper for progress bar (uses tqdm if available)."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, leave=leave)
    return iterable


# =============================================================================
# TRAINING METRICS
# =============================================================================

@dataclass
class TrainingMetrics:
    """Track training metrics."""
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """Standard training loop with best model tracking."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: TrainingConfig = None
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config or TrainingConfig()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.metrics = TrainingMetrics()
        self.best_weights = None

    def train_epoch(self, loader: DataLoader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in progress_bar(loader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        return correct / total, total_loss / total

    def evaluate(self, loader: DataLoader):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        return correct / total, total_loss / total

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        verbose: bool = True
    ) -> nn.Module:
        """Full training loop."""
        epochs = epochs or self.config.epochs

        for epoch in range(epochs):
            # Train
            train_acc, train_loss = self.train_epoch(train_loader)
            val_acc, val_loss = self.evaluate(val_loader)

            # Track metrics
            self.metrics.train_acc.append(train_acc)
            self.metrics.val_acc.append(val_acc)
            self.metrics.train_loss.append(train_loss)
            self.metrics.val_loss.append(val_loss)

            # Track best
            if val_acc > self.metrics.best_val_acc:
                self.metrics.best_val_acc = val_acc
                self.metrics.best_epoch = epoch
                self.best_weights = copy.deepcopy(self.model.state_dict())

            # Progress
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        # Restore best weights
        if self.best_weights:
            self.model.load_state_dict(self.best_weights)

        return self.model


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating"
) -> float:
    """Calculate accuracy on a dataset (returns percentage)."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in progress_bar(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100


def compute_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, float]:
    """Compute per-class accuracy."""
    model.eval()
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            for pred, label in zip(preds, labels):
                class_name = class_names[label.item()]
                class_total[class_name] += 1
                if pred == label:
                    class_correct[class_name] += 1

    return {
        name: (class_correct[name] / class_total[name] * 100) if class_total[name] > 0 else 0.0
        for name in class_names
    }


# =============================================================================
# EXPERIMENT LOGGING
# =============================================================================

class ExperimentLogger:
    """Log experiment results."""

    def __init__(self, experiment_name: str, log_dir: Path = None):
        self.name = experiment_name
        self.log_dir = log_dir or LOGS_DIR
        ensure_dir(self.log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        self.results = {}
        self.start_time = time.time()

    def log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)

        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")

    def log_results(self, results: dict):
        """Log final results."""
        self.results.update(results)

        self.log("\n" + "=" * 40)
        self.log("FINAL RESULTS")
        self.log("=" * 40)

        for key, value in results.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.2f}%")
            else:
                self.log(f"  {key}: {value}")

        elapsed = time.time() - self.start_time
        self.log(f"\nTotal time: {elapsed:.1f}s")

    def save_json(self):
        """Save results as JSON."""
        json_path = self.log_file.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

