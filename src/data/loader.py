"""
Unified data loading module for PDA experiments.

Consolidates functionality from agri_refactor/data/loader.py and integrates
with src/config for consistent path and class management.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms
from ..config import (
    PLANTVILLAGE_DIR,
    PLANTDOC_DIR,
    CLASS_NAME_MAPPING,
)
from ..config.crop_configs import (
    get_crop_config,
    get_canonical_classes,
    get_source_to_canonical_mapping,
    get_target_to_canonical_mapping,
    print_crop_summary,
    CROP_CONFIGS,
)


class CanonicalImageFolder(ImageFolder):
    """
    ImageFolder with fixed class ordering across datasets.

    Ensures consistent label indices between PlantVillage (source) and
    PlantDoc (target) by mapping folder names to a canonical class list.
    """

    def __init__(
        self,
        root: Path,
        canonical_classes: List[str],
        transform=None,
        class_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            root: Dataset root directory containing class folders.
            canonical_classes: Ordered list of canonical class names.
            transform: Image transforms to apply.
            class_mapping: Optional dict mapping folder names to canonical names.
        """
        self.canonical_classes = sorted(canonical_classes)
        self.class_mapping = class_mapping or {}
        super().__init__(str(root), transform=transform)

        # Build target mapping from canonical classes
        target_map = {cls_name: idx for idx, cls_name in enumerate(self.canonical_classes)}

        # Filter and remap samples
        filtered = []
        for path, label in self.samples:
            class_name = self.classes[label]
            # Apply class name mapping if provided
            mapped_name = self.class_mapping.get(class_name, class_name)
            if mapped_name in target_map:
                filtered.append((path, target_map[mapped_name]))

        self.samples = filtered
        self.targets = [s[1] for s in filtered]
        self.classes = list(self.canonical_classes)
        self.class_to_idx = target_map
        self.imgs = self.samples

        # Track samples per class for diagnostics
        self.samples_per_class = {cls: 0 for cls in self.canonical_classes}
        for _, label in filtered:
            cls_name = self.canonical_classes[label]
            self.samples_per_class[cls_name] += 1

        missing = [cls for cls, count in self.samples_per_class.items() if count == 0]
        if missing:
            print(f"  [Data] Warning: Missing classes on disk: {missing}")

    def get_class_counts(self) -> Dict[str, int]:
        """Return per-class sample counts."""
        return dict(self.samples_per_class)


def resolve_classes_for_crop(
    crop_filter: str,
    source_dir: Optional[Path] = None,
    target_dir: Optional[Path] = None,
    use_crop_config: bool = True,
) -> List[str]:
    """
    Resolve canonical class names for a given crop filter.

    If use_crop_config is True (default), uses the predefined crop configurations
    which handle label mismatches between PlantVillage and PlantDoc.

    Args:
        crop_filter: Crop name to filter (e.g., 'tomato', 'potato', 'pepper').
                     Can also be comma-separated for multiple crops.
        source_dir: PlantVillage directory (default from config).
        target_dir: PlantDoc directory (default from config).
        use_crop_config: If True, use predefined crop configs for semantic mapping.

    Returns:
        Sorted list of canonical class names matching the filter.
    """
    # Check if this is a known crop with predefined config
    crops = [c.strip().lower() for c in crop_filter.split(",") if c.strip()]

    if use_crop_config and all(c in CROP_CONFIGS for c in crops):
        # Use predefined canonical classes
        canonical = get_canonical_classes(crop_filter)
        return canonical

    # Fallback: scan directories (legacy behavior)
    source_dir = source_dir or PLANTVILLAGE_DIR
    target_dir = target_dir or PLANTDOC_DIR

    tokens = crops
    matches = set()

    for root in (source_dir, target_dir):
        if not Path(root).exists():
            continue
        for folder in Path(root).iterdir():
            if folder.is_dir() and any(tok in folder.name.lower() for tok in tokens):
                # Apply canonical mapping if available
                canonical = CLASS_NAME_MAPPING.get(folder.name, folder.name)
                matches.add(canonical)

    if not matches:
        raise ValueError(f"No classes matched filter '{crop_filter}' in {source_dir} or {target_dir}")

    return sorted(matches)


def load_split_indices(
    dataset: ImageFolder,
    split_path: Path,
    base_dir: Path,
) -> Tuple[List[int], List[int]]:
    """
    Load pool and test indices from a JSON split file.

    Args:
        dataset: The ImageFolder dataset to index into.
        split_path: Path to JSON file with 'pool_files' and 'test_files' keys.
        base_dir: Base directory for resolving relative paths in the split file.

    Returns:
        Tuple of (pool_indices, test_indices).
    """
    with open(split_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    # Build mapping from relative paths to dataset indices
    rel_to_idx = {}
    for idx, (path, _) in enumerate(dataset.samples):
        try:
            rel_path = Path(path).relative_to(base_dir).as_posix()
            rel_to_idx[rel_path] = idx
        except ValueError:
            # Path not under base_dir, try just the filename parts
            parts = Path(path).parts[-2:]  # class/filename
            rel_to_idx["/".join(parts)] = idx

    pool = [rel_to_idx[p] for p in data.get('pool_files', []) if p in rel_to_idx]
    test = [rel_to_idx[p] for p in data.get('test_files', []) if p in rel_to_idx]

    return pool, test


def create_canonical_datasets(
    crop_filter: str,
    source_dir: Optional[Path] = None,
    target_dir: Optional[Path] = None,
    train_transform=None,
    val_transform=None,
    class_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[CanonicalImageFolder, CanonicalImageFolder, List[str]]:
    """
    Create source and target datasets with canonical class alignment.

    Args:
        crop_filter: Crop name to filter classes.
        source_dir: PlantVillage directory.
        target_dir: PlantDoc directory.
        train_transform: Transform for training data.
        val_transform: Transform for validation/test data.
        class_mapping: Optional mapping from folder names to canonical names.

    Returns:
        Tuple of (source_dataset, target_dataset, canonical_classes).
    """
    source_dir = Path(source_dir or PLANTVILLAGE_DIR)
    target_dir = Path(target_dir or PLANTDOC_DIR)

    canonical_classes = resolve_classes_for_crop(crop_filter, source_dir, target_dir)
    print(f"  [Data] Canonical classes ({len(canonical_classes)}): {canonical_classes}")

    # Use default transforms if not provided
    if train_transform is None:
        train_transform = get_train_transforms()
    if val_transform is None:
        val_transform = get_val_transforms()

    # Create datasets
    source_dataset = CanonicalImageFolder(
        source_dir,
        canonical_classes,
        transform=train_transform,
        class_mapping=class_mapping,
    )

    target_dataset = CanonicalImageFolder(
        target_dir,
        canonical_classes,
        transform=val_transform,
        class_mapping=class_mapping,
    )

    print(f"  [Data] Source samples: {len(source_dataset)}, Target samples: {len(target_dataset)}")

    return source_dataset, target_dataset, canonical_classes


def load_data_modules(
    crop_filter: str,
    batch_size: int = 16,
    seed: int = 42,
    split_file: Optional[str] = None,
    train_val_split: float = 0.8,
    pool_test_split: float = 0.8,
    source_dir: Optional[Path] = None,
    target_dir: Optional[Path] = None,
    num_workers: int = -1,  # -1 = auto-detect
    verbose: bool = True,
    use_strong_aug: bool = False,
) -> Dict[str, Any]:
    """
    Load all data modules needed for PDA experiments.

    This is the main entry point for experiment scripts.


    Args:
        crop_filter: Crop name to filter classes (e.g., 'tomato', 'potato').
                     Can be comma-separated for multiple crops.
        batch_size: Batch size for data loaders.
        seed: Random seed for reproducibility.
        split_file: Optional path to JSON split file for pool/test indices.
        train_val_split: Fraction of source data for training (rest for validation).
        pool_test_split: Fraction of target data for pool (rest for test).
        source_dir: PlantVillage directory override.
        target_dir: PlantDoc directory override.
        num_workers: Number of data loader workers.
        verbose: Print diagnostic information.
        use_strong_aug: If True, use AutoAugment for training transforms (Phase 2).

    Returns:
        Dictionary with keys:
            - train_loader: DataLoader for source training data
            - val_loader: DataLoader for source validation data
            - pool_subset: Subset of target data for AL pool
            - test_subset: Subset of target data for evaluation
            - test_loader: DataLoader for test data
            - num_classes: Number of classes
            - canonical_classes: List of class names
            - source_dataset: Full source dataset
            - target_dataset: Full target dataset
            - crop_config: CropConfig object (if single crop)
    """
    import os
    import platform
    import time
    import torch  # Import torch at the start to avoid UnboundLocalError

    source_dir = Path(source_dir or PLANTVILLAGE_DIR)
    target_dir = Path(target_dir or PLANTDOC_DIR)

    t_start = time.time()

    # Auto-detect num_workers if not specified
    if num_workers < 0:
        if platform.system() == 'Windows':
            # Windows has issues with multiprocessing in DataLoader
            num_workers = 0
        else:
            # Linux/Colab: use more workers for A100 throughput
            # 8 workers with prefetch_factor=4 should saturate most GPUs
            num_workers = min(8, os.cpu_count() or 4)
        if verbose:
            print(f"  [Data] Auto-detected num_workers={num_workers} ({platform.system()})", flush=True)

    # Get transforms (with optional strong augmentation)
    train_transform = get_train_transforms(strong=use_strong_aug)
    val_transform = get_val_transforms()

    # Check if we have predefined crop configs
    crops = [c.strip().lower() for c in crop_filter.split(",") if c.strip()]
    use_crop_config = all(c in CROP_CONFIGS for c in crops)

    if use_crop_config:
        # Use semantic label mapping
        canonical_classes = get_canonical_classes(crop_filter)
        source_mapping = get_source_to_canonical_mapping(crop_filter)
        target_mapping = get_target_to_canonical_mapping(crop_filter)

        # Filter out None mappings (classes to exclude)
        source_mapping = {k: v for k, v in source_mapping.items() if v is not None}
        target_mapping = {k: v for k, v in target_mapping.items() if v is not None}

        if verbose:
            print_crop_summary(crop_filter)
    else:
        # Legacy behavior
        canonical_classes = resolve_classes_for_crop(crop_filter, source_dir, target_dir, use_crop_config=False)
        source_mapping = {}
        target_mapping = {}

    if verbose:
        print(f"  [Data] Config resolved: {time.time() - t_start:.1f}s", flush=True)

    # Create datasets with canonical class alignment
    t0 = time.time()
    source_train = CanonicalImageFolder(
        source_dir,
        canonical_classes,
        transform=train_transform,
        class_mapping=source_mapping,
    )
    if verbose:
        print(f"  [Data] Source train loaded: {time.time() - t0:.1f}s ({len(source_train)} samples)", flush=True)

    t0 = time.time()
    source_val = CanonicalImageFolder(
        source_dir,
        canonical_classes,
        transform=val_transform,
        class_mapping=source_mapping,
    )
    if verbose:
        print(f"  [Data] Source val loaded: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    target_dataset = CanonicalImageFolder(
        target_dir,
        canonical_classes,
        transform=val_transform,
        class_mapping=target_mapping,
    )
    if verbose:
        print(f"  [Data] Target loaded: {time.time() - t0:.1f}s ({len(target_dataset)} samples)", flush=True)

    num_classes = len(canonical_classes)

    if verbose:
        print(f"  [Data] Canonical classes ({num_classes}): {canonical_classes}", flush=True)
        print(f"  [Data] Source samples: {len(source_train)}, Target samples: {len(target_dataset)}", flush=True)

    # Split source into train/val
    generator = torch.Generator().manual_seed(seed)
    train_len = int(train_val_split * len(source_train))
    val_len = len(source_train) - train_len
    train_indices, val_indices = random_split(
        range(len(source_train)), [train_len, val_len], generator=generator
    )

    train_subset = Subset(source_train, list(train_indices))
    val_subset = Subset(source_val, list(val_indices))

    # Optimize DataLoader settings for GPU training
    pin_memory = torch.cuda.is_available() and num_workers > 0
    persistent = num_workers > 0  # Keep workers alive between epochs
    prefetch = 4 if num_workers > 0 else None  # Prefetch more batches

    # Common DataLoader kwargs for performance
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent,
    }
    if prefetch is not None:
        loader_kwargs['prefetch_factor'] = prefetch

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        shuffle=False,
        **loader_kwargs,
    )

    # Split target into pool/test
    if split_file and Path(split_file).exists():
        print(f"  [Data] Using split file: {split_file}")
        pool_indices, test_indices = load_split_indices(
            target_dataset, Path(split_file), target_dir
        )
    else:
        indices = np.arange(len(target_dataset))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_idx = int(pool_test_split * len(indices))
        pool_indices = indices[:split_idx].tolist()
        test_indices = indices[split_idx:].tolist()

    pool_subset = Subset(target_dataset, pool_indices)
    test_subset = Subset(target_dataset, test_indices)

    test_loader = DataLoader(
        test_subset,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"  [Data] Train: {len(train_subset)}, Val: {len(val_subset)}, Pool: {len(pool_subset)}, Test: {len(test_subset)}", flush=True)

    # Get crop config if available
    crop_config = None
    if use_crop_config and len(crops) == 1:
        crop_config = get_crop_config(crops[0])

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'pool_subset': pool_subset,
        'test_subset': test_subset,
        'test_loader': test_loader,
        'num_classes': num_classes,
        'canonical_classes': canonical_classes,
        'source_dataset': source_train,
        'target_dataset': target_dataset,
        'crop_config': crop_config,
    }

