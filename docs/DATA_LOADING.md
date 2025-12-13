# Data Loading and Splitting Documentation

This document explains how data is loaded, mapped, and split for the PDA (Partial Domain Adaptation) experiments.

## Overview

The framework handles two datasets with different naming conventions:
- **PlantVillage** (Source/Lab): High-quality controlled environment images
- **PlantDoc** (Target/Field): Real-world field images with noise and variations

The key challenge is that class names don't always match between datasets, and some classes exist in only one dataset.

---

## Dataset Statistics

### Summary Table

| Crop | Source Classes | Target Classes | Canonical Classes | Domain Type |
|------|----------------|----------------|-------------------|-------------|
| **Tomato** | 10 | 9 | 9 | Standard DA |
| **Potato** | 3 | 2 | 3 | **Partial DA** |
| **Pepper** | 2 | 2 | 2 | Full Alignment |

### Sample Counts

| Crop | Source Total | Source Mapped | Target Total | Target Mapped |
|------|--------------|---------------|--------------|---------------|
| **Tomato** | 32,022 | 29,214 | 1,491 | 1,491 |
| **Potato** | 4,304 | 4,304 | 442 | 442 |
| **Pepper** | 4,949 | 4,949 | 264 | 264 |
| **Combined** | 41,275 | 38,467 | 2,197 | 2,197 |

---

## Tomato Configuration (9 Classes)

### Class Mapping

| Canonical Name | PlantVillage Folder | Samples | PlantDoc Folder | Samples |
|----------------|---------------------|---------|-----------------|---------|
| `tomato_bacterial_spot` | `tomato_bacterial_spot` | 4,254 | `tomato_bacterial_spot` | 220 |
| `tomato_early_blight` | `tomato_early_blight` | 2,000 | `tomato_early_blight` | 176 |
| `tomato_healthy` | `tomato_healthy` | 3,182 | `tomato_healthy` | 126 |
| `tomato_late_blight` | `tomato_late_blight` | 3,818 | `tomato_late_blight` | 222 |
| `tomato_leaf_mold` | `tomato_mold` | 1,904 | `tomato_mold` | 182 |
| `tomato_mosaic_virus` | `tomato_mosaic_virus` | 746 | `tomato_mosaic_virus` | 108 |
| `tomato_septoria_spot` | `tomato_septoria_spot` | 3,542 | `tomato_septoria_spot` | 301 |
| `tomato_spider_mites` | `tomato_spider_mites_two_spotted_spider_mite` | 3,352 | `tomato_two_spotted_spider_mites` | 4 |
| `tomato_yellow_leaf_curl` | `tomato_yellow_curl_virus` | 6,416 | `tomato_yellow_virus` | 152 |

### Excluded Classes

| PlantVillage Class | Samples | Reason |
|--------------------|---------|--------|
| `tomato_target_spot` | 2,808 | Does not exist in PlantDoc (*Corynespora cassiicola*) |

### Notes on Mappings

| Mapping | Biological Justification |
|---------|-------------------------|
| `tomato_mold` → `tomato_leaf_mold` | Scientifically accurate name for *Passalora fulva* |
| `tomato_spider_mites_two_spotted_spider_mite` → `tomato_spider_mites` | Same pest: *Tetranychus urticae* (Two-Spotted Spider Mite) |
| `tomato_yellow_curl_virus` → `tomato_yellow_leaf_curl` | Tomato Yellow Leaf Curl Virus (TYLCV) |

---

## Potato Configuration (3 Classes) - **Partial Domain Adaptation**

This is the key PDA scenario where `potato_healthy` exists in source but NOT in target.

### Class Mapping

| Canonical Name | PlantVillage Folder | Samples | PlantDoc Folder | Samples |
|----------------|---------------------|---------|-----------------|---------|
| `potato_early_blight` | `potato_early_blight` | 2,000 | `potato_early_blight` | 232 |
| `potato_healthy` | `potato_healthy` | 304 | *(missing)* | 0 |
| `potato_late_blight` | `potato_late_blight` | 2,000 | `potato_late_blight` | 210 |

### PDA Scenario

```
Source (PlantVillage):  [early_blight] [healthy] [late_blight]
                              ↓            ↓           ↓
Target (PlantDoc):      [early_blight]    ❌     [late_blight]
```

The model is trained on 3 classes but evaluated on data containing only 2 classes.
This tests the model's ability to avoid **negative transfer** (predicting the missing "healthy" class).

---

## Pepper Configuration (2 Classes)

Full alignment with name mapping required.

### Class Mapping

| Canonical Name | PlantVillage Folder | Samples | PlantDoc Folder | Samples |
|----------------|---------------------|---------|-----------------|---------|
| `pepper_bell_bacterial_spot` | `pepper_bell_bacterial_spot` | 1,994 | `pepper_bell_spot` | 142 |
| `pepper_bell_healthy` | `pepper_bell_healthy` | 2,955 | `pepper_bell` | 122 |

Note: PlantDoc uses shortened names (`pepper_bell` instead of `pepper_bell_healthy`).

---

## Data Splitting Strategy

### Baseline Training (Source Only)

```
PlantVillage (Source)
├── Train: 80%
└── Val: 20%

PlantDoc (Target)
├── Pool: 80% (for Active Learning)
└── Test: 20% (for evaluation)
```

### Active Learning Splits

```
Source Dataset (PlantVillage)
├── Train Set: 80% of source
│   └── Used for initial training
└── Val Set: 20% of source
    └── Used for model selection

Target Dataset (PlantDoc)
├── Pool Set: 80% of target
│   └── Unlabeled data for AL queries
│   └── Selected samples move to labeled set
└── Test Set: 20% of target
    └── Held out - NEVER used for training
    └── Final evaluation only
```

### Split Ratios by Crop

| Crop | Train (Source) | Val (Source) | Pool (Target) | Test (Target) |
|------|----------------|--------------|---------------|---------------|
| **Tomato** | 20,690 | 5,172 | 1,190 | 297 |
| **Potato** | 3,443 | 861 | 354 | 88 |
| **Pepper** | 3,959 | 990 | 211 | 53 |

*Based on 80/20 splits with default seed=42*

---

## Data Loading Pipeline

### 1. Class Resolution

```python
from src.config.crop_configs import get_canonical_classes, get_source_to_canonical_mapping

# Get canonical classes for a crop
classes = get_canonical_classes("tomato")
# ['tomato_bacterial_spot', 'tomato_early_blight', ...]

# Get mapping from folder names to canonical names
mapping = get_source_to_canonical_mapping("tomato")
# {'tomato_yellow_curl_virus': 'tomato_yellow_leaf_curl', ...}
```

### 2. Dataset Loading

```python
from src.data import load_data_modules

data = load_data_modules(
    crop_filter="tomato",      # or "potato", "pepper", "tomato,potato,pepper"
    batch_size=16,
    seed=42,
    train_val_split=0.8,       # 80% train, 20% val for source
    pool_test_split=0.8,       # 80% pool, 20% test for target
)

# Returns dictionary with:
# - train_loader: DataLoader for source training
# - val_loader: DataLoader for source validation
# - pool_subset: Subset for AL queries (unlabeled target)
# - test_subset: Subset for final evaluation
# - test_loader: DataLoader for test evaluation
# - num_classes: Number of canonical classes
# - canonical_classes: List of class names
```

### 3. Label Mapping Flow

```
Disk Folder Name → Source/Target Mapping → Canonical Label → Integer Index
     ↓                    ↓                      ↓              ↓
"tomato_yellow_curl"  →  source_map  →  "tomato_yellow_leaf"  →  7
"tomato_yellow_virus" →  target_map  →  "tomato_yellow_leaf"  →  7
```

---

## Active Learning Data Flow

### Round 0 (Initial)
```
Pool Set: [img1, img2, img3, ..., imgN]  (all unlabeled)
Labeled Set: []  (empty)
```

### Round 1 (After query)
```
Pool Set: [img3, img4, ..., imgN]  (remaining unlabeled)
Labeled Set: [img1, img2]  (budget=10 selected by strategy)
```

### Round K (Final)
```
Pool Set: [remaining images]
Labeled Set: [img1, img2, ..., img50]  (budget × rounds)
```

### Strategy Options

| Strategy | Description | Best For |
|----------|-------------|----------|
| `random` | Uniform random selection | Baseline comparison |
| `entropy` | Select highest uncertainty | Exploitation |
| `hybrid` | 70% entropy + 30% random | **Recommended** (avoids cold start) |

---

## Reproducibility

### Seed Control

All random operations use the same seed:
- Train/Val split
- Pool/Test split  
- Model initialization
- Data augmentation
- AL sampling

```python
data = load_data_modules(crop_filter="potato", seed=42)
```

### Split Recording

Every experiment records exact splits in `splits.json`:

```json
{
  "train": [
    {"path": "PlantVillage/.../img001.jpg", "label": 0, "classname": "potato_early_blight"},
    ...
  ],
  "val": [...],
  "pool": [...],
  "test": [...]
}
```

---

## Validation

Run before experiments:

```bash
python validate_data.py
```

Output:
```
======================================================================
VALIDATING: TOMATO
======================================================================
[SOURCE: PlantVillage]
  Classes on disk (10):
    tomato_bacterial_spot: 4254 samples -> tomato_bacterial_spot
    ...
    tomato_yellow_curl_virus: 6416 samples -> tomato_yellow_leaf_curl
    
[CANONICAL CLASSES]
  Defined (8): ['tomato_bacterial_spot', ...]
  Source covers: ['tomato_bacterial_spot', ...]
  Target covers: ['tomato_bacterial_spot', ...]

[OK] Configuration matches data on disk
```

---

## Common Issues

### 1. Class Mismatch Warning

```
Warning: Missing classes on disk: ['tomato_spider_mites']
```

**Cause**: Class is in config but not on disk (excluded intentionally).
**Solution**: This is expected for excluded classes.

### 2. Zero Samples in Target

```
potato_healthy: 0 samples in target
```

**Cause**: This is the PDA scenario - class doesn't exist in target.
**Solution**: Expected behavior. Model should learn NOT to predict this class.

### 3. Label Index Mismatch

**Cause**: Source and target using different label indices.
**Solution**: Both datasets map to canonical names first, then to consistent indices.

---

## File Locations

| File | Description |
|------|-------------|
| `src/config/crop_configs.py` | Class mappings and configurations |
| `src/data/loader.py` | Data loading and splitting logic |
| `validate_data.py` | Validation script |
| `data/processed/dataset/PlantVillage_processed/` | Source dataset |
| `data/processed/dataset/PlantDoc_processed/` | Target dataset |

