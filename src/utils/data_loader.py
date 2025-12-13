"""
Data loading utilities for the PhD Active Adaptation experiments.

Provides:
- Custom dataset classes for filtered/aligned data loading
- Transform functions
- DataLoader creation utilities
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..config import TrainingConfig


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms(config: TrainingConfig = None) -> Dict[str, transforms.Compose]:
    """Get standard train/val transforms with ImageNet normalization."""
    if config is None:
        config = TrainingConfig()

    return {
        'train': transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std)
        ]),
    }


def get_augmented_transforms(config: TrainingConfig = None) -> Dict[str, transforms.Compose]:
    """Get transforms with stronger augmentation."""
    if config is None:
        config = TrainingConfig()

    return {
        'train': transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std)
        ]),
    }


# =============================================================================
# DATASET PATH UTILITIES
# =============================================================================

def find_dataset_path(root: Path, folder_name: str) -> Tuple[Path, Optional[Path]]:
    """
    Find dataset directory, handling both train/val split and flat structures.

    Returns:
        (train_path, val_path) - val_path is None for flat structures
    """
    root = Path(root)
    base = root / folder_name

    if not base.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    train_sub = base / "train"
    val_sub = base / "val"

    if train_sub.is_dir() and val_sub.is_dir():
        return train_sub, val_sub
    elif train_sub.is_dir():
        return train_sub, None
    else:
        return base, None


# =============================================================================
# CUSTOM DATASET CLASSES
# =============================================================================

class FilteredImageFolder(datasets.ImageFolder):
    """
    ImageFolder that filters to specific classes.

    Supports both exact matching and partial matching (e.g., 'tomato' matches
    'tomato_early_blight').
    """

    def __init__(self, root: str, transform=None, class_filter: List[str] = None):
        super().__init__(root, transform=transform)

        if class_filter:
            self._apply_filter(class_filter)

    def _apply_filter(self, class_filter: List[str]):
        """Filter samples to only include matching classes."""
        original_classes = self.classes.copy()

        # Try exact match first, then partial match
        matched = [c for c in original_classes if c in class_filter]
        if not matched:
            matched = [
                c for c in original_classes
                if any(f.lower() in c.lower() for f in class_filter)
            ]

        if not matched:
            raise ValueError(
                f"No classes matched filter {class_filter}. "
                f"Available: {original_classes[:5]}..."
            )

        # Rebuild class mapping
        self.classes = matched
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Filter samples
        filtered = []
        for path, old_idx in self.samples:
            class_name = original_classes[old_idx]
            if class_name in self.classes:
                new_idx = self.class_to_idx[class_name]
                filtered.append((path, new_idx))

        self.samples = filtered
        self.targets = [s[1] for s in self.samples]


class CanonicalImageFolder(datasets.ImageFolder):
    """
    ImageFolder that aligns class indices to a canonical schema.

    This ensures that class indices are consistent across datasets,
    even if some classes are missing or have different names.

    Args:
        root: Path to dataset folder
        canonical_classes: List of canonical class names for alignment
        transform: Image transform to apply
        class_name_mapping: Dict mapping source class names to canonical names
                           e.g., {'tomato_two_spotted_spider_mites': 'tomato_spider_mites'}
    """

    def __init__(self, root: str, canonical_classes: List[str], transform=None,
                 class_name_mapping: Dict[str, str] = None):
        super().__init__(root, transform=transform)

        # Build canonical mapping
        canonical_map = {name: idx for idx, name in enumerate(canonical_classes)}

        # Store mapping for reference
        self.class_name_mapping = class_name_mapping or {}
        self.mapped_classes = {}  # Track which classes were mapped

        # Remap samples
        aligned = []
        found_classes = set()

        for path, old_idx in self.samples:
            class_name = self.classes[old_idx]

            # Apply class name mapping if provided
            mapped_name = self.class_name_mapping.get(class_name, class_name)

            if mapped_name in canonical_map:
                new_idx = canonical_map[mapped_name]
                aligned.append((path, new_idx))
                found_classes.add(mapped_name)

                # Track mapping
                if class_name != mapped_name:
                    self.mapped_classes[class_name] = mapped_name

        # Update state
        self.samples = aligned
        self.targets = [s[1] for s in aligned]
        self.classes = canonical_classes
        self.class_to_idx = canonical_map

        # Report mapping info
        if self.mapped_classes:
            print(f"  Class mappings applied:")
            for src, dst in self.mapped_classes.items():
                print(f"    {src} -> {dst}")

        missing = set(canonical_classes) - found_classes
        if missing:
            print(f"  Note: {len(missing)} classes missing from this dataset: {sorted(missing)}")


# =============================================================================
# DATA LOADER CREATION
# =============================================================================

def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset = None,
    config: TrainingConfig = None
) -> Dict[str, DataLoader]:
    """Create DataLoaders with proper configuration."""
    if config is None:
        config = TrainingConfig()

    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available()
        ),
    }

    if test_dataset:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return loaders


def load_tomato_datasets(
    lab_dir: Path,
    field_dir: Path,
    canonical_classes: List[str],
    config: TrainingConfig = None,
    class_name_mapping: Dict[str, str] = None
) -> Tuple[Dataset, Dataset]:
    """
    Load lab and field datasets with aligned class indices.

    Args:
        lab_dir: Path to lab (PlantVillage) processed data
        field_dir: Path to field (PlantDoc) processed data
        canonical_classes: List of class names for alignment
        config: Training configuration
        class_name_mapping: Dict mapping source class names to canonical names

    Returns:
        (lab_dataset, field_dataset) with aligned class indices
    """
    if config is None:
        config = TrainingConfig()

    transforms_dict = get_transforms(config)

    lab_dataset = CanonicalImageFolder(
        str(lab_dir),
        canonical_classes=canonical_classes,
        transform=transforms_dict['train'],
        class_name_mapping=class_name_mapping
    )

    field_dataset = CanonicalImageFolder(
        str(field_dir),
        canonical_classes=canonical_classes,
        transform=transforms_dict['val'],
        class_name_mapping=class_name_mapping
    )

    return lab_dataset, field_dataset

