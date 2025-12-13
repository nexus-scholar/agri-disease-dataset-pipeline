"""
Strategies module for PhD Active Adaptation experiments.

Provides active learning sample selection, augmentation strategies,
and semi-supervised learning (FixMatch) for domain adaptation.
"""

from .augmentation import (
    cutmix_data,
    cutmix_criterion,
    rand_bbox,
    CutMixTrainer,
)

from .active_learning import (
    select_samples,
    select_random,
    select_entropy,
    select_margin,
    fine_tune,
    ActiveLearner,
    run_hybrid_warmstart,
    STRATEGIES,
)

from .fixmatch import (
    train_fixmatch,
    FixMatchConfig,
    SSLDataset,
)

__all__ = [
    # Augmentation
    'cutmix_data',
    'cutmix_criterion',
    'rand_bbox',
    'CutMixTrainer',

    # Active learning
    'select_samples',
    'select_random',
    'select_entropy',
    'select_margin',
    'fine_tune',
    'ActiveLearner',
    'run_hybrid_warmstart',
    'STRATEGIES',

    # Semi-supervised (FixMatch)
    'train_fixmatch',
    'FixMatchConfig',
    'SSLDataset',
]

