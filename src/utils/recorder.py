"""
ExperimentRecorder: Forensic-level experiment logging.

Creates a complete record of every experiment run including:
- Configuration (hyperparameters, model config, seed)
- Data splits (exact file paths for train/val/pool/test)
- Metrics (per-epoch loss/acc, AL trajectory, confusion matrix)
- Model checkpoints

This enables full reproducibility and post-hoc analysis without re-running experiments.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


class ExperimentRecorder:
    """
    Forensic-level experiment recorder.

    Creates a timestamped directory for each experiment containing:
    - config.json: All hyperparameters
    - splits.json: Exact file paths for each data split
    - metrics.json: Training history, AL trajectory, final evaluation
    - model_best.pth: Best model checkpoint

    Usage:
        recorder = ExperimentRecorder("P1_Baseline_Tomato_MobileNet")
        recorder.save_config(args)
        recorder.save_splits(data_modules)

        for epoch in range(epochs):
            # ... training ...
            recorder.log_epoch(epoch, train_loss, val_acc)

        recorder.save_final_evaluation(acc, confusion_matrix, report)
        torch.save(model.state_dict(), recorder.get_model_path())
    """

    def __init__(
        self,
        experiment_id: str,
        output_root: Union[str, Path] = "results/experiments",
        include_timestamp: bool = True,
    ):
        """
        Initialize the recorder.

        Args:
            experiment_id: Unique identifier for this experiment.
            output_root: Root directory for experiment outputs.
            include_timestamp: Whether to append timestamp to directory name.
        """
        self.experiment_id = experiment_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if include_timestamp:
            dir_name = f"{experiment_id}_{self.timestamp}"
        else:
            dir_name = experiment_id

        self.exp_dir = Path(output_root) / dir_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics structure
        self.metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp": self.timestamp,
            "history": [],           # Epoch-wise training data
            "al_trajectory": [],     # Active Learning rounds
            "final_accuracy": None,
            "final_confusion": None,
            "final_report": None,
            "latency_stats": None,
            "data_stats": {},        # Dataset statistics
        }

        print(f"  [Recorder] Logging to: {self.exp_dir}")

    def save_config(self, args) -> Path:
        """
        Save experiment configuration.

        Args:
            args: argparse.Namespace or dict with experiment config.

        Returns:
            Path to saved config file.
        """
        config_path = self.exp_dir / "config.json"

        if hasattr(args, '__dict__'):
            config = vars(args).copy()
        else:
            config = dict(args)

        # Add metadata
        config['_recorder_timestamp'] = self.timestamp
        config['_recorder_experiment_id'] = self.experiment_id
        config['_python_version'] = sys.version
        config['_torch_version'] = torch.__version__
        config['_cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            config['_cuda_device'] = torch.cuda.get_device_name(0)

        # Convert Path objects to strings
        for k, v in config.items():
            if isinstance(v, Path):
                config[k] = str(v)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, default=str)

        print(f"  [Recorder] Config saved: {config_path}")
        return config_path

    def save_splits(self, data_modules: Dict[str, Any], save_full_paths: bool = False) -> Path:
        """
        Save data split information for reproducibility.

        Args:
            data_modules: Dictionary from load_data_modules() containing
                         train_loader, val_loader, pool_subset, test_subset, etc.
            save_full_paths: If True, save all file paths (slow for large datasets).
                            If False (default), save only indices and summary stats.

        Returns:
            Path to saved splits file.
        """
        print("  [Recorder] Snapshotting data splits...")

        def get_indices_only(dataset_or_loader) -> Dict[str, Any]:
            """Extract just indices and counts (fast)."""
            # Unwrap DataLoader
            if isinstance(dataset_or_loader, DataLoader):
                ds = dataset_or_loader.dataset
            else:
                ds = dataset_or_loader

            # Handle Subsets
            if isinstance(ds, Subset):
                indices = list(ds.indices)
                total = len(ds.dataset.samples) if hasattr(ds.dataset, 'samples') else len(ds.dataset)
                return {
                    "count": len(indices),
                    "indices": indices,
                    "total_in_source": total,
                }
            elif hasattr(ds, 'samples'):
                return {
                    "count": len(ds.samples),
                    "indices": list(range(len(ds.samples))),
                }
            else:
                return {"count": len(ds), "indices": []}

        def get_paths(dataset_or_loader) -> List[Dict[str, Any]]:
            """Extract file paths from a Dataset or DataLoader (slow for large datasets)."""
            # Unwrap DataLoader
            if isinstance(dataset_or_loader, DataLoader):
                ds = dataset_or_loader.dataset
            else:
                ds = dataset_or_loader

            # Handle Subsets (from random_split or manual selection)
            indices = None
            if isinstance(ds, Subset):
                indices = list(ds.indices)
                ds = ds.dataset  # Unwrap to underlying dataset

            # Now ds should be ImageFolder-like with .samples
            if not hasattr(ds, 'samples'):
                return [{"error": "Dataset does not have .samples attribute"}]

            paths = []
            sample_indices = indices if indices is not None else range(len(ds.samples))

            for i in sample_indices:
                path, label = ds.samples[i]
                entry = {
                    "path": str(path),
                    "label": int(label),
                }
                if hasattr(ds, 'classes') and label < len(ds.classes):
                    entry["classname"] = ds.classes[label]
                paths.append(entry)

            return paths

        splits = {}

        if save_full_paths:
            # Full paths (slow but complete reproducibility)
            print("  [Recorder] Saving full file paths (this may take a moment)...")
            if 'train_loader' in data_modules:
                splits['train'] = get_paths(data_modules['train_loader'])
            if 'val_loader' in data_modules:
                splits['val'] = get_paths(data_modules['val_loader'])
            if 'pool_subset' in data_modules:
                splits['pool'] = get_paths(data_modules['pool_subset'])
            if 'test_subset' in data_modules:
                splits['test'] = get_paths(data_modules['test_subset'])
        else:
            # Fast: indices only
            if 'train_loader' in data_modules:
                splits['train'] = get_indices_only(data_modules['train_loader'])
            if 'val_loader' in data_modules:
                splits['val'] = get_indices_only(data_modules['val_loader'])
            if 'pool_subset' in data_modules:
                splits['pool'] = get_indices_only(data_modules['pool_subset'])
            if 'test_subset' in data_modules:
                splits['test'] = get_indices_only(data_modules['test_subset'])

        # Add summary stats
        if save_full_paths:
            splits['_summary'] = {
                'train_count': len(splits.get('train', [])),
                'val_count': len(splits.get('val', [])),
                'pool_count': len(splits.get('pool', [])),
                'test_count': len(splits.get('test', [])),
                'full_paths_saved': True,
            }
        else:
            splits['_summary'] = {
                'train_count': splits.get('train', {}).get('count', 0),
                'val_count': splits.get('val', {}).get('count', 0),
                'pool_count': splits.get('pool', {}).get('count', 0),
                'test_count': splits.get('test', {}).get('count', 0),
                'full_paths_saved': False,
            }

        # Store in metrics too
        self.metrics['data_stats'] = splits['_summary']

        splits_path = self.exp_dir / "splits.json"
        with open(splits_path, 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2)

        print(f"  [Recorder] Splits saved: {splits_path}")
        print(f"             Train: {splits['_summary']['train_count']}, "
              f"Val: {splits['_summary']['val_count']}, "
              f"Pool: {splits['_summary']['pool_count']}, "
              f"Test: {splits['_summary']['test_count']}")

        return splits_path

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_acc: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metrics for a training epoch.

        Args:
            epoch: Epoch number (1-indexed recommended).
            train_loss: Average training loss for the epoch.
            val_acc: Validation accuracy (percentage).
            extra: Additional metrics to log (e.g., learning rate).
        """
        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_acc": float(val_acc),
            "timestamp": datetime.now().isoformat(),
        }

        if extra:
            entry.update(extra)

        self.metrics["history"].append(entry)
        self._flush_metrics()

    def log_al_round(
        self,
        round_num: int,
        num_labels: int,
        accuracy: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metrics for an Active Learning round.

        Args:
            round_num: Round number (0 = initial, before any labeling).
            num_labels: Total number of labeled samples after this round.
            accuracy: Test accuracy after this round (percentage).
            extra: Additional metrics (e.g., selected indices).
        """
        entry = {
            "round": round_num,
            "labels": int(num_labels),
            "accuracy": float(accuracy),
            "timestamp": datetime.now().isoformat(),
        }

        if extra:
            entry.update(extra)

        self.metrics["al_trajectory"].append(entry)
        self._flush_metrics()

    def save_final_evaluation(
        self,
        accuracy: float,
        confusion_matrix: np.ndarray,
        classification_report: Dict[str, Any],
    ) -> None:
        """
        Save final evaluation metrics.

        Args:
            accuracy: Final test accuracy (percentage).
            confusion_matrix: Numpy array of confusion matrix.
            classification_report: Dict from sklearn's classification_report.
        """
        self.metrics["final_accuracy"] = float(accuracy)
        self.metrics["final_confusion"] = confusion_matrix.tolist()
        self.metrics["final_report"] = classification_report
        self._flush_metrics()

        print(f"  [Recorder] Final accuracy: {accuracy:.2f}%")

    def save_latency_stats(self, stats: Dict[str, Any]) -> None:
        """Save latency/throughput statistics."""
        self.metrics["latency_stats"] = stats
        self._flush_metrics()

    def _flush_metrics(self) -> None:
        """Write metrics to disk immediately for crash recovery."""
        metrics_path = self.exp_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4, default=self._json_serializer)

    @staticmethod
    def _json_serializer(obj):
        """Handle non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, '__dict__'):
            return str(obj)
        return str(obj)

    def get_model_path(self, name: str = "model_best.pth") -> Path:
        """Get path for saving model checkpoint."""
        return self.exp_dir / name

    def get_experiment_dir(self) -> Path:
        """Get the experiment output directory."""
        return self.exp_dir

    def save_artifact(self, name: str, data: Any) -> Path:
        """
        Save an arbitrary artifact (JSON-serializable).

        Args:
            name: Filename (will add .json if not present).
            data: Data to save.

        Returns:
            Path to saved artifact.
        """
        if not name.endswith('.json'):
            name = f"{name}.json"

        path = self.exp_dir / name
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)

        return path

    def log_message(self, message: str) -> None:
        """Append a message to the experiment log."""
        log_path = self.exp_dir / "log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")


def get_detailed_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: List[str],
) -> tuple:
    """
    Get detailed evaluation metrics for final reporting.

    Args:
        model: Trained model.
        loader: Test data loader.
        device: Torch device.
        classes: List of class names.

    Returns:
        Tuple of (accuracy, confusion_matrix, classification_report_dict).
    """
    from sklearn.metrics import confusion_matrix, classification_report

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Force labels to match class list
    label_indices = list(range(len(classes)))

    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        labels=label_indices,
        output_dict=True,
        zero_division=0,
    )

    accuracy = report['accuracy'] * 100

    return accuracy, cm, report

