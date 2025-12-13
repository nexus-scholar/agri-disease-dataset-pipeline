# Experiments

This folder contains the core experiment scripts for the Active Domain Adaptation research.

## Overview

The experiments follow a progression from establishing the baseline generalization gap 
to implementing and evaluating strategies to reduce it.

| Script | Purpose | Key Finding |
|--------|---------|-------------|
| `01_baseline_gap.py` | Establish Lab→Field accuracy drop | ~60-70% gap |
| `02_passive_aug.py` | Test strong data augmentation | Helps slightly |
| `03_cutmix.py` | Test CutMix regularization | Better regularization |
| `04_active_learning.py` | Compare Random vs Entropy | Entropy has "dip" |
| `05_hybrid_warmstart.py` | Proposed balanced strategy | **Best results** |

## Quick Start

```bash
# Run all experiments in sequence
python 01_baseline_gap.py --class-name Tomato
python 02_passive_aug.py --class-name Tomato
python 03_cutmix.py --class-name Tomato
python 04_active_learning.py --class-name Tomato
python 05_hybrid_warmstart.py --class-name Tomato
```

## Common Module

All experiments share a common module (`common.py`) that provides:

- **Configuration classes**: `TrainingConfig`, `ActiveLearningConfig`, `CutMixConfig`
- **Data utilities**: `FilteredImageFolder`, `CanonicalImageFolder`, transforms
- **Model utilities**: `create_model`, `load_model`, `save_model`
- **Training utilities**: `Trainer`, `evaluate_accuracy`
- **Logging**: `ExperimentLogger`, colored console output

### Key Configuration

```python
from common import TrainingConfig, ActiveLearningConfig

# Training defaults
config = TrainingConfig(
    batch_size=16,
    epochs=5,
    learning_rate=0.001,
    num_workers=0  # Windows compatibility
)

# Active learning defaults
al_config = ActiveLearningConfig(
    budget_per_round=50,
    num_rounds=4,
    epochs_per_round=5,
    fine_tune_lr=0.0001
)
```

---

## Experiment 01: Baseline Gap

**Purpose**: Establish the fundamental problem - models trained on controlled "Lab" data fail on "Field" data.

```bash
python 01_baseline_gap.py
python 01_baseline_gap.py --class-name Tomato --epochs 10
```

**Arguments**:
- `--dataset-root`: Root directory for datasets
- `--lab-folder`: Lab dataset (default: PlantVillage)
- `--field-folder`: Field dataset (default: PlantDoc)
- `--class-name`: Filter to specific class (e.g., "Tomato")
- `--batch-size`, `--epochs`, `--lr`: Training hyperparameters

**Output**: `data/models/baseline_model.pth`

**Expected Results**:
- Lab accuracy: ~95%
- Field accuracy: ~25-35%
- **Gap: ~60-70%**

---

## Experiment 02: Passive Augmentation

**Purpose**: Test if strong augmentation can improve field robustness without field data.

```bash
python 02_passive_aug.py --class-name Tomato
```

**Augmentation Pipeline** (Albumentations):
- Geometric: flips, rotations
- Photometric: brightness, contrast, HSV
- Noise: Gaussian noise, blur
- Masking: CoarseDropout

**Expected Results**:
- Lab accuracy: ~90% (slightly lower)
- Field accuracy: ~30-40% (slight improvement)

---

## Experiment 03: CutMix

**Purpose**: Test CutMix regularization - mixing patches between images.

```bash
python 03_cutmix.py --class-name Tomato --epochs 10
python 03_cutmix.py --cutmix-prob 0.7 --cutmix-beta 1.0
```

**CutMix Parameters**:
- `--cutmix-prob`: Probability of applying CutMix (default: 0.5)
- `--cutmix-beta`: Beta distribution parameter (default: 1.0)

**Note**: Training accuracy appears lower due to mixed labels - this is normal.

---

## Experiment 04: Active Learning

**Purpose**: Compare Random vs Entropy-based sample selection.

```bash
python 04_active_learning.py --class-name Tomato
python 04_active_learning.py --budget-per-round 100 --num-rounds 3
```

**Requires**: `baseline_model.pth` from Experiment 01

**Strategies**:
- **Random**: Uniformly sample from unlabeled pool
- **Entropy**: Select most uncertain samples

**Key Insight**: Entropy shows early "dip" because hard samples confuse the model initially.

---

## Experiment 05: Hybrid Warm-Start

**Purpose**: Proposed balanced strategy combining Random + Entropy.

```bash
python 05_hybrid_warmstart.py --class-name Tomato
python 05_hybrid_warmstart.py --strategies random,entropy,hybrid
```

**The Hybrid Strategy**:
- Round 0: 50% Random + 50% Entropy (warm start)
- Later rounds: Pure entropy sampling

**Expected Results**:

| Strategy | 0 labels | 50 | 100 | 150 | 200 |
|----------|----------|-----|-----|-----|-----|
| Random   | 26.17%   | 25.50% | 33.56% | 41.61% | 44.97% |
| Entropy  | 26.17%   | 26.17% | 28.19% | 40.27% | 42.95% |
| **Hybrid** | 26.17% | 26.17% | 34.23% | 38.93% | **46.31%** |

---

## Project Structure

```
experiments/
├── common.py              # Shared utilities and configuration
├── 01_baseline_gap.py     # Baseline experiment
├── 02_passive_aug.py      # Passive augmentation
├── 03_cutmix.py           # CutMix augmentation
├── 04_active_learning.py  # Active learning comparison
├── 05_hybrid_warmstart.py # Proposed hybrid strategy
└── README.md              # This file
```

## Output Locations

- **Models**: `data/models/`
- **Logs**: `results/logs/`
- **Figures**: `results/figures/`

## Hardware Requirements

- **GPU**: 4GB+ VRAM (tested on GTX 1650)
- **RAM**: 8GB+
- **Storage**: ~2GB for datasets

## Common Issues

1. **"baseline_model.pth not found"**
   - Run experiment 01 first

2. **CUDA out of memory**
   - Reduce `--batch-size` to 8

3. **Windows freeze on multiprocessing**
   - `num_workers=0` is already set in common.py

4. **Albumentations import error**
   - Run: `pip install albumentations opencv-python`

## Reproducibility

All experiments use:
- Fixed random seed (default: 42)
- Deterministic data splits
- Saved model checkpoints

To reproduce exact results:
```bash
python 01_baseline_gap.py --seed 42 --class-name Tomato
```

