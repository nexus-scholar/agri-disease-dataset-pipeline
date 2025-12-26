"""
Hyperparameters and Configuration.
"""
from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    PROJECT_ROOT = "D:/experiments/dataset-processing"
    DATA_ROOT = f"{PROJECT_ROOT}/data/processed/dataset"
    PLANT_VILLAGE_DIR = f"{DATA_ROOT}/PlantVillage_processed"
    PLANT_DOC_DIR = f"{DATA_ROOT}/PlantDoc_processed"
    
    # Model
    MODEL_ARCH = "mobilenet_v3_small"
    NUM_CLASSES = 10 # 9 Diseases + 1 Healthy (Source Domain)
    
    # Training
    IMAGE_SIZE = 224
    # Reduced batch sizes for 4GB GPU
    BATCH_SIZE_SOURCE = 32
    BATCH_SIZE_TARGET_LABELED = 16
    BATCH_SIZE_TARGET_UNLABELED = 8 # 8 * 7 = 56 images
    
    LEARNING_RATE = 0.03
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    EPOCHS = 5
    LR_DECAY_EPOCHS = [30, 50]
    LR_DECAY_GAMMA = 0.1
    
    # Active Learning
    AL_BUDGET = 50
    
    # FixMatch
    FIXMATCH_THRESHOLD = 0.95
    FIXMATCH_LAMBDA_U = 1.0
    FIXMATCH_MU = 7 # Ratio of unlabeled to labeled data
    
    # Reproducibility
    SEED = 42

cfg = Config()
