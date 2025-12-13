"""
PhD Active Adaptation - Source Package

A modular framework for plant disease classification with active learning
and domain adaptation.

Modules:
- config: Configuration and hyperparameters
- data: Data loading and transforms
- models: Neural network architectures
- strategies: Active learning and augmentation strategies
- utils: Metrics and utilities
"""

from . import config
from . import data
from . import models
from . import strategies
from . import utils

# Convenient imports
from .config import (
    TrainingConfig,
    ActiveLearningConfig,
    CutMixConfig,
    ExperimentConfig,
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
    PLANTVILLAGE_DIR,
    PLANTDOC_DIR,
    TOMATO_CLASSES,
    ensure_dir,
)

from .data import (
    load_data_modules,
    create_canonical_datasets,
    CanonicalImageFolder,
    resolve_classes_for_crop,
    get_transforms,
    get_train_transforms,
    get_val_transforms,
)

from .models import (
    get_model,
    get_edge_model,
    create_model,
    create_mobilenetv3,
    load_model,
    save_model,
    get_model_info,
    SUPPORTED_MODELS,
)

from .strategies import (
    select_samples,
    ActiveLearner,
    train_fixmatch,
    FixMatchConfig,
    run_hybrid_warmstart,
)

from .utils import (
    FilteredImageFolder,
    create_data_loaders,
    Trainer,
    evaluate_accuracy,
    ExperimentLogger,
    Colors,
    print_header,
    print_section,
    print_config,
)

from .utils.device import (
    get_device,
    set_seed,
)

__version__ = "1.0.0"

__all__ = [
    # Modules
    'config',
    'data',
    'models',
    'strategies',
    'utils',

    # Config
    'TrainingConfig',
    'ActiveLearningConfig',
    'CutMixConfig',
    'ExperimentConfig',
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'RESULTS_DIR',
    'LOGS_DIR',
    'PROCESSED_DIR',
    'PLANTVILLAGE_DIR',
    'PLANTDOC_DIR',
    'TOMATO_CLASSES',
    'ensure_dir',

    # Data
    'load_data_modules',
    'create_canonical_datasets',
    'CanonicalImageFolder',
    'resolve_classes_for_crop',
    'get_transforms',
    'get_train_transforms',
    'get_val_transforms',

    # Models
    'get_model',
    'get_edge_model',
    'create_model',
    'create_mobilenetv3',
    'load_model',
    'save_model',
    'get_model_info',
    'SUPPORTED_MODELS',

    # Strategies
    'select_samples',
    'ActiveLearner',
    'train_fixmatch',
    'FixMatchConfig',
    'run_hybrid_warmstart',

    # Utils
    'FilteredImageFolder',
    'create_data_loaders',
    'Trainer',
    'evaluate_accuracy',
    'ExperimentLogger',
    'Colors',
    'print_header',
    'print_section',
    'print_config',
    'get_device',
    'set_seed',
]

