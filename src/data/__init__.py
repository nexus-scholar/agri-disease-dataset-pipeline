"""
Data module for PDA experiments.

Provides unified data loading, transforms, and augmentation utilities.
"""
from .loader import (
    CanonicalImageFolder,
    create_canonical_datasets,
    load_split_indices,
    load_data_modules,
    resolve_classes_for_crop,
)
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_transforms,
    WeakStrongTransform,
    CutMixTransform,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_IMAGE_SIZE,
)
from .augmentations import WeakStrongTransform as WeakStrongAug

__all__ = [
    # Loader
    'CanonicalImageFolder',
    'create_canonical_datasets',
    'load_split_indices',
    'load_data_modules',
    'resolve_classes_for_crop',
    # Transforms
    'get_train_transforms',
    'get_val_transforms',
    'get_transforms',
    'WeakStrongTransform',
    'CutMixTransform',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'DEFAULT_IMAGE_SIZE',
    # Legacy
    'WeakStrongAug',
]

