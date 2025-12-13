"""
Utilities module for PhD Active Adaptation experiments.
"""

from .data_loader import (
    get_transforms,
    get_augmented_transforms,
    find_dataset_path,
    FilteredImageFolder,
    CanonicalImageFolder,
    create_data_loaders,
    load_tomato_datasets,
)

from .metrics import (
    TrainingMetrics,
    Trainer,
    evaluate_accuracy,
    compute_class_accuracy,
    ExperimentLogger,
    progress_bar,
)

from .uncertainty import (
    compute_entropy,
    compute_margin,
    compute_least_confidence,
    get_predictions,
)

from .console import (
    Colors,
    print_header,
    print_section,
    print_config,
    print_results_table,
    print_success,
    print_error,
    print_warning,
    print_info,
)

from .device import (
    get_device,
    set_seed,
    get_memory_stats,
    clear_memory,
)

from .recorder import (
    ExperimentRecorder,
    get_detailed_metrics,
)

__all__ = [
    # Data loading
    'get_transforms',
    'get_augmented_transforms',
    'find_dataset_path',
    'FilteredImageFolder',
    'CanonicalImageFolder',
    'create_data_loaders',
    'load_tomato_datasets',

    # Metrics and training
    'TrainingMetrics',
    'Trainer',
    'evaluate_accuracy',
    'compute_class_accuracy',
    'ExperimentLogger',
    'progress_bar',

    # Uncertainty
    'compute_entropy',
    'compute_margin',
    'compute_least_confidence',
    'get_predictions',

    # Console
    'Colors',
    'print_header',
    'print_section',
    'print_config',
    'print_results_table',
    'print_success',
    'print_error',
    'print_warning',
    'print_info',

    # Device
    'get_device',
    'set_seed',
    'get_memory_stats',
    'clear_memory',

    # Experiment Recording
    'ExperimentRecorder',
    'get_detailed_metrics',
]
