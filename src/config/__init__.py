"""
Configuration module for PhD Active Adaptation experiments.
"""

from .settings import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
    PLANTVILLAGE_DIR,
    PLANTDOC_DIR,
    DEFAULT_DATASET_ROOT,
    ensure_dir,

    # Configs
    TrainingConfig,
    ActiveLearningConfig,
    CutMixConfig,
    ExperimentConfig,

    # Class lists
    TOMATO_CLASSES,
    TOMATO_CLASSES_EXTENDED,
    TOMATO_CLASSES_EXACT,
    PLANTVILLAGE_CLASSES,
    PLANTDOC_CLASSES,
    PLANTVILLAGE_TOMATO_CLASSES,
    PLANTDOC_TOMATO_CLASSES,

    # Class name mapping
    CLASS_NAME_MAPPING,
    PLANTDOC_TO_CANONICAL,
    PLANTVILLAGE_TO_CANONICAL,
)

from .crop_configs import (
    CropConfig,
    get_crop_config,
    get_canonical_classes,
    get_source_to_canonical_mapping,
    get_target_to_canonical_mapping,
    print_crop_summary,
    CROP_CONFIGS,
    TOMATO_CONFIG,
    POTATO_CONFIG,
    PEPPER_CONFIG,
)

__all__ = [
    # Paths
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'RESULTS_DIR',
    'LOGS_DIR',
    'PROCESSED_DIR',
    'PLANTVILLAGE_DIR',
    'PLANTDOC_DIR',
    'DEFAULT_DATASET_ROOT',
    'ensure_dir',

    # Configs
    'TrainingConfig',
    'ActiveLearningConfig',
    'CutMixConfig',
    'ExperimentConfig',

    # Class lists
    'TOMATO_CLASSES',
    'TOMATO_CLASSES_EXTENDED',
    'TOMATO_CLASSES_EXACT',
    'PLANTVILLAGE_CLASSES',
    'PLANTDOC_CLASSES',
    'PLANTVILLAGE_TOMATO_CLASSES',
    'PLANTDOC_TOMATO_CLASSES',

    # Class name mapping
    'CLASS_NAME_MAPPING',
    'PLANTDOC_TO_CANONICAL',
    'PLANTVILLAGE_TO_CANONICAL',

    # Crop configurations
    'CropConfig',
    'get_crop_config',
    'get_canonical_classes',
    'get_source_to_canonical_mapping',
    'get_target_to_canonical_mapping',
    'print_crop_summary',
    'CROP_CONFIGS',
    'TOMATO_CONFIG',
    'POTATO_CONFIG',
    'PEPPER_CONFIG',
]

