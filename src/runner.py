#!/usr/bin/env python3
"""
Unified CLI entry point for PDA experiments.

Supports:
- Baseline training (source-only)
- Active learning with multiple strategies (random, entropy, hybrid)
- Semi-supervised learning with FixMatch
- Multiple architectures (MobileNetV3, EfficientNet, MobileViT)

Usage:
    # Baseline training
    python -m src.runner --mode baseline --crop tomato --model mobilenetv3

    # Active learning with hybrid strategy
    python -m src.runner --mode active --crop potato --strategy hybrid --budget 10 --rounds 5

    # Active learning with FixMatch
    python -m src.runner --mode active --crop potato --strategy hybrid --use-fixmatch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_data_modules, get_train_transforms
from src.models import get_model, load_model, save_model
from src.strategies import train_fixmatch, FixMatchConfig
from src.strategies.fixmatch import SSLDataset
from src.utils.recorder import ExperimentRecorder, get_detailed_metrics


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def print_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, classes: list):
    """Print confusion matrix and classification report."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_indices = list(range(len(classes)))
    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)

    print("\n=== Confusion Matrix ===")
    print(cm)
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=classes, labels=label_indices, zero_division=0))


def compute_entropy(model: nn.Module, indices: list, dataset, device: torch.device) -> list:
    """Compute entropy-based uncertainty for sample indices."""
    import numpy as np
    from torch.utils.data import Subset

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    model.eval()
    entropies = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            entropies.extend(ent.cpu().numpy())

    return entropies


def select_samples(strategy: str, pool_indices: list, budget: int, model: nn.Module, dataset, device: torch.device, round_num: int = 0) -> list:
    """Select samples from pool using specified strategy."""
    import numpy as np

    if not pool_indices or budget <= 0:
        return []

    budget = min(budget, len(pool_indices))

    if strategy == 'random':
        selected = np.random.choice(pool_indices, budget, replace=False).tolist()

    elif strategy == 'entropy':
        entropies = compute_entropy(model, pool_indices, dataset, device)
        order = np.argsort(entropies)[::-1]
        selected = [pool_indices[i] for i in order[:budget]]

    elif strategy == 'hybrid':
        if round_num == 0:
            # First round: 70% entropy, 30% random
            n_entropy = int(0.7 * budget)
            n_random = budget - n_entropy

            entropies = compute_entropy(model, pool_indices, dataset, device)
            order = np.argsort(entropies)[::-1]
            entropy_sel = [pool_indices[i] for i in order[:n_entropy]]

            remaining = [idx for idx in pool_indices if idx not in entropy_sel]
            random_sel = np.random.choice(remaining, min(n_random, len(remaining)), replace=False).tolist()

            selected = entropy_sel + random_sel
        else:
            # Subsequent rounds: pure entropy
            selected = select_samples('entropy', pool_indices, budget, model, dataset, device, round_num)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return selected


def fine_tune_supervised(model: nn.Module, labeled_indices: list, dataset, device: torch.device, epochs: int, lr: float, batch_size: int = 8):
    """Fine-tune model on labeled samples using supervised learning."""
    train_transform = get_train_transforms()
    labeled_ds = SSLDataset(dataset, labeled_indices, transform=train_transform)

    if len(labeled_ds) == 0:
        return model

    loader = DataLoader(labeled_ds, batch_size=min(batch_size, len(labeled_ds)), shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified PDA Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment ID
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name for recording (auto-generated if not provided)')

    # Mode
    parser.add_argument('--mode', required=True, choices=['baseline', 'active'],
                        help='Experiment mode: baseline (source training) or active (AL/SSL)')

    # Data
    parser.add_argument('--crop', '--class-filter', dest='crop', default='tomato',
                        help='Crop filter (e.g., tomato, potato, pepper)')
    parser.add_argument('--split-file', default=None,
                        help='Path to JSON split file for pool/test indices')

    # Model
    parser.add_argument('--model', default='mobilenetv3',
                        choices=['mobilenetv3', 'efficientnet', 'mobilevit'],
                        help='Model architecture')
    parser.add_argument('--baseline-path', default='data/models/baselines/baseline.pth',
                        help='Path to save/load baseline model')

    # Active learning
    parser.add_argument('--strategy', default='random',
                        choices=['random', 'entropy', 'hybrid'],
                        help='Sample selection strategy')
    parser.add_argument('--use-fixmatch', action='store_true',
                        help='Enable FixMatch semi-supervised training')
    parser.add_argument('--budget', type=int, default=10,
                        help='Number of samples to label per round')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of active learning rounds')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs (baseline) or epochs per round (active)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--strong-aug', action='store_true',
                        help='Use strong augmentation (AutoAugment) for baseline training (Phase 2)')

    # Output
    parser.add_argument('--no-confusion', action='store_true',
                        help='Skip printing confusion matrix')

    return parser.parse_args()


def run_baseline(args, device):
    """Run baseline training on source (PlantVillage) data."""
    import time

    strong_tag = " | StrongAug" if getattr(args, 'strong_aug', False) else ""
    print(f"\n{'='*60}", flush=True)
    print(f"BASELINE TRAINING | Crop: {args.crop} | Model: {args.model}{strong_tag}", flush=True)
    print(f"{'='*60}", flush=True)

    # Initialize recorder
    t0 = time.time()
    strong_suffix = "_strong" if getattr(args, 'strong_aug', False) else ""
    exp_name = args.exp_name or f"baseline_{args.crop}_{args.model}{strong_suffix}"
    recorder = ExperimentRecorder(exp_name)
    recorder.save_config(args)
    print(f"  [Setup] Recorder init: {time.time() - t0:.1f}s", flush=True)

    # Load data
    t0 = time.time()
    data = load_data_modules(
        crop_filter=args.crop,
        batch_size=args.batch_size,
        seed=args.seed,
        split_file=args.split_file,
        use_strong_aug=getattr(args, 'strong_aug', False),
    )
    print(f"  [Setup] Data loading: {time.time() - t0:.1f}s", flush=True)

    # Save data splits for reproducibility (fast mode - indices only)
    t0 = time.time()
    recorder.save_splits(data, save_full_paths=False)
    print(f"  [Setup] Splits saved: {time.time() - t0:.1f}s", flush=True)

    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    num_classes = data['num_classes']
    classes = data['canonical_classes']

    # Create model
    model = get_model(args.model, num_classes, pretrained=True, device=device)
    print(f"  [Model] {args.model} with {num_classes} classes")

    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Training for {args.epochs} epochs...", flush=True)
    best_val_acc = 0.0

    num_batches = len(train_loader)
    report_every = max(1, num_batches // 4)  # Report 4 times per epoch

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Progress report within epoch
            if (batch_idx + 1) % report_every == 0 or batch_idx == num_batches - 1:
                pct = 100 * (batch_idx + 1) / num_batches
                print(f"    Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{num_batches} ({pct:.0f}%) | Loss: {loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        # Log epoch metrics
        recorder.log_epoch(epoch + 1, avg_loss, val_acc)
        print(f"    Epoch {epoch+1}/{args.epochs} DONE | Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%", flush=True)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), recorder.get_model_path("model_best.pth"))

    # Final evaluation on field (target) data
    field_acc, cm, report = get_detailed_metrics(model, test_loader, device, classes)
    recorder.save_final_evaluation(field_acc, cm, report)

    print(f"\n  FIELD ACCURACY: {field_acc:.2f}%", flush=True)

    # Save final model to both recorder dir and specified baseline path
    torch.save(model.state_dict(), recorder.get_model_path("model_final.pth"))
    save_model(model, args.baseline_path, metadata={
        'crop': args.crop,
        'model': args.model,
        'epochs': args.epochs,
        'field_acc': field_acc,
        'val_acc': best_val_acc,
        'experiment_dir': str(recorder.get_experiment_dir()),
    })
    print(f"  Saved baseline to {args.baseline_path}")
    print(f"  Full experiment log: {recorder.get_experiment_dir()}")

    # Confusion matrix
    if not args.no_confusion:
        print_confusion_matrix(model, test_loader, device, classes)

    return model, recorder


def run_active_learning(args, device):
    """Run active learning with optional FixMatch."""
    import time

    print(f"\n{'='*60}", flush=True)
    print(f"ACTIVE LEARNING | Crop: {args.crop} | Strategy: {args.strategy}", flush=True)
    print(f"Model: {args.model} | FixMatch: {args.use_fixmatch}", flush=True)
    print(f"{'='*60}", flush=True)

    # Initialize recorder
    t0 = time.time()
    fixmatch_tag = "_fixmatch" if args.use_fixmatch else ""
    exp_name = args.exp_name or f"active_{args.crop}_{args.model}_{args.strategy}{fixmatch_tag}"
    recorder = ExperimentRecorder(exp_name)
    recorder.save_config(args)
    print(f"  [Setup] Recorder init: {time.time() - t0:.1f}s", flush=True)

    # Load data
    t0 = time.time()
    data = load_data_modules(
        crop_filter=args.crop,
        batch_size=args.batch_size,
        seed=args.seed,
        split_file=args.split_file,
    )
    print(f"  [Setup] Data loading: {time.time() - t0:.1f}s", flush=True)

    # Save data splits for reproducibility (fast mode - indices only)
    t0 = time.time()
    recorder.save_splits(data, save_full_paths=False)
    print(f"  [Setup] Splits saved: {time.time() - t0:.1f}s", flush=True)

    test_loader = data['test_loader']
    pool_subset = data['pool_subset']
    num_classes = data['num_classes']
    classes = data['canonical_classes']
    target_dataset = data['target_dataset']

    # Load baseline model
    t0 = time.time()
    baseline_path = Path(args.baseline_path)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {baseline_path}\n"
            f"Run baseline training first: python -m src.runner --mode baseline --crop {args.crop} --model {args.model}"
        )

    model = load_model(baseline_path, args.model, num_classes, device)
    print(f"  [Setup] Model loaded: {time.time() - t0:.1f}s", flush=True)
    print(f"  [Model] Loaded {args.model} from {baseline_path}", flush=True)

    # Initialize pool indices
    pool_indices = list(pool_subset.indices)
    labeled_indices = []

    # Initial accuracy (0 labels) - Round 0
    acc, _, _ = get_detailed_metrics(model, test_loader, device, classes)
    recorder.log_al_round(0, 0, acc)
    print(f"\n  Round 0/{ args.rounds} (0 labels): {acc:.2f}%", flush=True)

    results = [acc]

    # Active learning loop
    for round_idx in range(args.rounds):
        # Select samples
        new_indices = select_samples(
            args.strategy,
            pool_indices,
            args.budget,
            model,
            target_dataset,
            device,
            round_num=round_idx,
        )

        # Update indices
        labeled_indices.extend(new_indices)
        pool_indices = [idx for idx in pool_indices if idx not in new_indices]

        # Train
        if args.use_fixmatch:
            config = FixMatchConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
            )
            model = train_fixmatch(
                model,
                labeled_indices,
                pool_indices,
                target_dataset,
                device,
                config,
            )
        else:
            model = fine_tune_supervised(
                model,
                labeled_indices,
                target_dataset,
                device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )

        # Evaluate and log
        acc, _, _ = get_detailed_metrics(model, test_loader, device, classes)
        recorder.log_al_round(round_idx + 1, len(labeled_indices), acc, extra={
            "new_indices": new_indices,
            "pool_remaining": len(pool_indices),
        })
        print(f"  Round {round_idx+1}/{args.rounds} ({len(labeled_indices)} labels): {acc:.2f}%", flush=True)
        results.append(acc)

    # Final detailed evaluation
    final_acc, cm, report = get_detailed_metrics(model, test_loader, device, classes)
    recorder.save_final_evaluation(final_acc, cm, report)

    # Save final model
    torch.save(model.state_dict(), recorder.get_model_path("model_final.pth"))

    # Save labeled indices for reproducibility
    recorder.save_artifact("labeled_indices", {
        "indices": labeled_indices,
        "total": len(labeled_indices),
    })

    # Final summary
    print(f"\n  FINAL ACCURACY: {final_acc:.2f}% with {len(labeled_indices)} labels")
    print(f"  Improvement: {results[-1] - results[0]:.2f}%")
    print(f"  Full experiment log: {recorder.get_experiment_dir()}")

    # Confusion matrix
    if not args.no_confusion:
        print_confusion_matrix(model, test_loader, device, classes)

    return model, results, recorder


def main():
    import time
    t_start = time.time()

    args = parse_args()

    # Set random seeds
    import numpy as np
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device - CUDA initialization can take several seconds on first call
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda_time = time.time() - t0

    # Enable cudnn benchmark for faster convolutions (input size is fixed at 224x224)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device} (init: {cuda_time:.1f}s, total startup: {time.time() - t_start:.1f}s)", flush=True)

    if args.mode == 'baseline':
        run_baseline(args, device)
    else:
        run_active_learning(args, device)


if __name__ == '__main__':
    main()

