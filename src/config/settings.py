"""
Configuration module for the PhD Active Adaptation experiments.

Contains dataclasses for hyperparameters and path configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths (relative to project root)
_MODULE_DIR = Path(__file__).parent.resolve()
_SRC_DIR = _MODULE_DIR.parent
PROJECT_ROOT = _SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"

# Dataset paths - processed data
PROCESSED_DIR = DATA_DIR / "processed" / "dataset"
PLANTVILLAGE_DIR = PROCESSED_DIR / "PlantVillage_processed"
PLANTDOC_DIR = PROCESSED_DIR / "PlantDoc_processed"

# Legacy dataset path (one level up from project)
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "dataset"


def ensure_dir(path: Path) -> Path:
    """Create directory if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# HYPERPARAMETER CONFIGURATIONS
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 0.001
    num_workers: int = 0  # 0 for Windows compatibility
    image_size: int = 224

    # ImageNet normalization
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclass
class ActiveLearningConfig:
    """Active learning hyperparameters."""
    fine_tune_lr: float = 0.0001
    epochs_per_round: int = 5
    budget_per_round: int = 50
    num_rounds: int = 4
    pool_batch_size: int = 32
    train_batch_size: int = 8
    random_seed: int = 42

    @property
    def total_budget(self) -> int:
        return self.budget_per_round * self.num_rounds

    @property
    def budget_rounds(self) -> List[int]:
        return [self.budget_per_round] * self.num_rounds


@dataclass
class CutMixConfig:
    """CutMix augmentation hyperparameters."""
    probability: float = 0.5
    beta: float = 1.0


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    cutmix: CutMixConfig = field(default_factory=CutMixConfig)

    # Experiment settings
    seed: int = 42
    experiment_name: str = "experiment"

    # Dataset settings
    class_filter: List[str] = None
    lab_folder: str = "PlantVillage_processed"
    field_folder: str = "PlantDoc_processed"


# =============================================================================
# TOMATO DISEASE CLASSES (Common across datasets)
# =============================================================================

# Class name mapping: PlantDoc name -> Canonical (PlantVillage) name
# This maps different naming conventions to a single canonical name
PLANTDOC_TO_CANONICAL = {
    'tomato_two_spotted_spider_mites': 'tomato_spider_mites',
    'tomato_yellow_virus': 'tomato_yellow_curl_virus',
}

# PlantVillage name -> Canonical name (for consistency)
PLANTVILLAGE_TO_CANONICAL = {
    'tomato_spider_mites_two_spotted_spider_mite': 'tomato_spider_mites',
}

# All mappings combined (source name -> canonical name)
CLASS_NAME_MAPPING = {
    # PlantDoc names
    'tomato_two_spotted_spider_mites': 'tomato_spider_mites',
    'tomato_yellow_virus': 'tomato_yellow_curl_virus',
    # PlantVillage names
    'tomato_spider_mites_two_spotted_spider_mite': 'tomato_spider_mites',
}

# Extended tomato classes (9 classes) - includes mapped classes
# Use this for experiments with class mapping enabled
TOMATO_CLASSES_EXTENDED = [
    'tomato_bacterial_spot',
    'tomato_early_blight',
    'tomato_healthy',
    'tomato_late_blight',
    'tomato_mold',
    'tomato_mosaic_virus',
    'tomato_septoria_spot',
    'tomato_spider_mites',      # Mapped from both datasets
    'tomato_yellow_curl_virus', # Mapped from PlantDoc's tomato_yellow_virus
]

# Original 7 classes with EXACT name match (no mapping needed)
TOMATO_CLASSES_EXACT = [
    'tomato_bacterial_spot',
    'tomato_early_blight',
    'tomato_healthy',
    'tomato_late_blight',
    'tomato_mold',
    'tomato_mosaic_virus',
    'tomato_septoria_spot',
]

# Default: Use extended classes with mapping
TOMATO_CLASSES = TOMATO_CLASSES_EXTENDED

# All tomato classes in PlantVillage (10 classes)
PLANTVILLAGE_TOMATO_CLASSES = [
    'tomato_bacterial_spot',
    'tomato_early_blight',
    'tomato_healthy',
    'tomato_late_blight',
    'tomato_mold',
    'tomato_mosaic_virus',
    'tomato_septoria_spot',
    'tomato_spider_mites_two_spotted_spider_mite',  # No exact match in PlantDoc
    'tomato_target_spot',                            # Not in PlantDoc
    'tomato_yellow_curl_virus',                      # PlantDoc uses: tomato_yellow_virus
]

# All tomato classes in PlantDoc (9 classes)
PLANTDOC_TOMATO_CLASSES = [
    'tomato_bacterial_spot',
    'tomato_early_blight',
    'tomato_healthy',
    'tomato_late_blight',
    'tomato_mold',
    'tomato_mosaic_virus',
    'tomato_septoria_spot',
    'tomato_two_spotted_spider_mites',  # PlantVillage uses: tomato_spider_mites_two_spotted_spider_mite
    'tomato_yellow_virus',               # PlantVillage uses: tomato_yellow_curl_virus
]

# All classes available in PlantVillage processed data (15 classes)
PLANTVILLAGE_CLASSES = [
    'pepper_bell_bacterial_spot',
    'pepper_bell_healthy',
    'potato_early_blight',
    'potato_healthy',
    'potato_late_blight',
    'tomato_bacterial_spot',
    'tomato_early_blight',
    'tomato_healthy',
    'tomato_late_blight',
    'tomato_mold',
    'tomato_mosaic_virus',
    'tomato_septoria_spot',
    'tomato_spider_mites_two_spotted_spider_mite',
    'tomato_target_spot',
    'tomato_yellow_curl_virus',
]

# All classes available in PlantDoc processed data (29 classes)
PLANTDOC_CLASSES = [
    'apple_healthy',
    'apple_rust',
    'apple_scab',
    'blueberry_healthy',
    'cherry_healthy',
    'corn_blight',
    'corn_gray_spot',
    'corn_rust',
    'grape_black_rot',
    'grape_healthy',
    'peach_healthy',
    'pepper_bell',
    'pepper_bell_spot',
    'potato_early_blight',
    'potato_late_blight',
    'raspberry_healthy',
    'soybean_healthy',
    'squash_powdery_mildew',
    'strawberry_healthy',
    'tomato_bacterial_spot',
    'tomato_early_blight',
    'tomato_healthy',
    'tomato_late_blight',
    'tomato_mold',
    'tomato_mosaic_virus',
    'tomato_septoria_spot',
    'tomato_two_spotted_spider_mites',
    'tomato_yellow_virus',
]

