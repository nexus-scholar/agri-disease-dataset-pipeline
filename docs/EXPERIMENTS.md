# Experiment Protocol

Complete guide for running the Partial Domain Adaptation (PDA) experiments for agricultural disease detection.

## Overview

This experiment suite validates our Active Learning + Semi-Supervised Learning approach for bridging the Lab-to-Field generalization gap in plant disease detection.

**Total: 26 experiments across 5 phases (~9 hours on GPU)**

| Phase | Name | Experiments | Scientific Goal |
|-------|------|-------------|-----------------|
| **P1** | Baselines | 12 | Quantify generalization gap across architectures |
| **P2** | Strong Augmentation | 3 | Prove augmentation alone doesn't solve the gap |
| **P3** | AL Strategy Ablation | 4 | Justify Hybrid (70/30) over Random/Entropy |
| **P4** | FixMatch + PDA | 3 | Core contribution - SSL eliminates negative transfer |
| **P5** | Architecture Benchmark | 4 | Compare MobileNetV3, EfficientNet, MobileViT |

---

## Quick Start

```bash
# List all experiments
python colab_experiment_runner.py --list

# Dry run (preview without executing)
python colab_experiment_runner.py --all --dry-run

# Run specific phase
python colab_experiment_runner.py --phase 1

# Run all phases
python colab_experiment_runner.py --all
```

---

## Data Configuration

### Semantic Label Adapter

The system uses `src/config/crop_configs.py` to handle label mismatches between PlantVillage (source/lab) and PlantDoc (target/field).

| Crop | Classes | Source-Only (PDA) | Notes |
|------|---------|-------------------|-------|
| **Tomato** | 7 | 3 excluded | spider_mites, target_spot, yellow_curl differ in naming |
| **Potato** | 3 | 1 (healthy) | **PDA scenario** - healthy missing in field |
| **Pepper** | 2 | 0 | Full alignment with label mapping |

### Label Mappings

**PlantDoc → Canonical:**
- `pepper_bell` → `pepper_bell_healthy`
- `pepper_bell_spot` → `pepper_bell_bacterial_spot`
- `tomato_two_spotted_spider_mites` → EXCLUDED (only 2 samples)
- `tomato_yellow_virus` → EXCLUDED (naming differs)

---

## Phase 1: Baseline Generalization Gap

**Goal:** Prove models trained on Lab data fail on Field data regardless of architecture.

| ID | Experiment | Model | Crop | Time |
|----|------------|-------|------|------|
| P1-01 | Baseline | MobileNetV3 | Tomato | 10 min |
| P1-02 | Baseline | EfficientNet | Tomato | 10 min |
| P1-03 | Baseline | MobileViT | Tomato | 15 min |
| P1-04 | Baseline | MobileNetV3 | Potato | 10 min |
| P1-05 | Baseline | EfficientNet | Potato | 10 min |
| P1-06 | Baseline | MobileViT | Potato | 15 min |
| P1-07 | Baseline | MobileNetV3 | Pepper | 10 min |
| P1-08 | Baseline | EfficientNet | Pepper | 10 min |
| P1-09 | Baseline | MobileViT | Pepper | 15 min |
| P1-10 | Baseline | MobileNetV3 | All | 15 min |
| P1-11 | Baseline | EfficientNet | All | 15 min |
| P1-12 | Baseline | MobileViT | All | 20 min |

```bash
# Example
python run_experiment.py --mode baseline --model mobilenetv3 --crop tomato \
    --baseline-path data/models/baselines/tomato_mobilenetv3_base.pth \
    --epochs 10 --lr 0.001 --exp-name P1_01_baseline_tomato_mobilenetv3
```

**Expected Output:** Table I - "Generalization Gap Across Architectures"

---

## Phase 2: Strong Augmentation (Passive Intervention)

**Goal:** Prove AutoAugment doesn't close the generalization gap.

| ID | Experiment | Crop | Time |
|----|------------|------|------|
| P2-01 | StrongAug | Tomato | 15 min |
| P2-02 | StrongAug | Potato | 15 min |
| P2-03 | StrongAug | Pepper | 15 min |

```bash
# Example
python run_experiment.py --mode baseline --model mobilenetv3 --crop potato \
    --strong-aug --baseline-path data/models/baselines/potato_strong_base.pth \
    --epochs 10 --lr 0.001 --exp-name P2_02_strongaug_potato
```

**Expected Result:** Augmentation may hurt performance (especially on Potato PDA case).

---

## Phase 3: Active Learning Strategy Ablation

**Goal:** Justify the Hybrid (70% Entropy / 30% Random) strategy.

| ID | Experiment | Strategy | Crop | Time |
|----|------------|----------|------|------|
| P3-01 | AL Random | Random | Tomato | 20 min |
| P3-02 | AL Entropy | Entropy | Tomato | 20 min |
| P3-03 | AL Hybrid | Hybrid | Tomato | 20 min |
| P3-04 | AL Hybrid | Hybrid | Potato | 20 min |

```bash
# Example - Hybrid strategy
python run_experiment.py --mode active --model mobilenetv3 --crop tomato \
    --strategy hybrid --baseline-path data/models/baselines/tomato_mobilenetv3_base.pth \
    --budget 10 --rounds 5 --epochs 5 --lr 0.0001 --exp-name P3_03_AL_hybrid_tomato
```

**Expected Results:**
- Random: Inefficient (slow improvement slope)
- Entropy: "Cold Start" problem (early dips)
- Hybrid: Best of both worlds

---

## Phase 4: FixMatch + Partial Domain Adaptation

**Goal:** Prove FixMatch (SSL) eliminates negative transfer in the PDA scenario.

| ID | Experiment | Crop | Time |
|----|------------|------|------|
| P4-01 | FixMatch | Potato | 35 min |
| P4-02 | FixMatch | Tomato | 35 min |
| P4-03 | FixMatch | Pepper | 35 min |

```bash
# Example - THE KEY EXPERIMENT
python run_experiment.py --mode active --model mobilenetv3 --crop potato \
    --strategy hybrid --use-fixmatch \
    --baseline-path data/models/baselines/potato_mobilenetv3_base.pth \
    --budget 10 --rounds 5 --epochs 15 --lr 0.001 --exp-name P4_01_fixmatch_potato
```

**Expected Result:** Potato confusion matrix shows 0 predictions for "Healthy" class (missing in field).

---

## Phase 5: Architecture Benchmark

**Goal:** Compare EfficientNet and MobileViT with our method.

| ID | Experiment | Model | Crop | Time |
|----|------------|-------|------|------|
| P5-01 | Benchmark | EfficientNet | Potato | 40 min |
| P5-02 | Benchmark | MobileViT | Potato | 40 min |
| P5-03 | Benchmark | EfficientNet | Tomato | 40 min |
| P5-04 | Benchmark | MobileViT | Tomato | 40 min |

```bash
# Example - MobileViT (lower LR for transformers)
python run_experiment.py --mode active --model mobilevit --crop potato \
    --strategy hybrid --use-fixmatch \
    --baseline-path data/models/baselines/potato_mobilevit_base.pth \
    --budget 10 --rounds 5 --epochs 15 --lr 0.0005 --exp-name P5_02_mobilevit_potato
```

**Expected Results:**
- EfficientNet: May fail (negative transfer)
- MobileViT: Succeeds but slower inference

---

## CLI Reference

### Main Runner (`run_experiment.py`)

```bash
python run_experiment.py [OPTIONS]

Required:
  --mode {baseline,active}    Experiment mode

Data:
  --crop CROP                 Crop filter (tomato, potato, pepper, or comma-separated)
  --split-file PATH           JSON split file for pool/test indices

Model:
  --model {mobilenetv3,efficientnet,mobilevit}
  --baseline-path PATH        Path to save/load baseline model

Active Learning:
  --strategy {random,entropy,hybrid}
  --use-fixmatch              Enable FixMatch semi-supervised training
  --budget N                  Samples to label per round (default: 10)
  --rounds N                  AL rounds (default: 5)

Training:
  --epochs N                  Training epochs (default: 10)
  --batch-size N              Batch size (default: 16)
  --lr FLOAT                  Learning rate (default: 0.001)
  --seed N                    Random seed (default: 42)
  --strong-aug                Use AutoAugment (Phase 2)

Output:
  --exp-name NAME             Experiment name for recording
  --no-confusion              Skip confusion matrix printing
```

### Batch Runner (`colab_experiment_runner.py`)

```bash
python colab_experiment_runner.py [OPTIONS]

  --list                      List all experiments
  --phase {1,2,3,4,5}         Run specific phase
  --all                       Run all phases
  --dry-run                   Preview without executing
  --output-dir PATH           Output directory (default: results/experiments)
```

---

## Output Structure

Each experiment creates a forensic record:

```
results/experiments/{exp_name}_{timestamp}/
├── config.json           # All hyperparameters
├── splits.json           # Exact file paths for reproducibility
├── metrics.json          # Loss curves, AL trajectory, confusion matrix
├── model_best.pth        # Best checkpoint (by validation accuracy)
├── model_final.pth       # Final checkpoint
└── labeled_indices.json  # (AL only) Which samples were labeled
```

---

## Paper Artifacts

| Section | Artifact | Source |
|---------|----------|--------|
| Intro | "Generalization Gap > 40%" | Phase 1 |
| Methods | "Hybrid Strategy" Rationale | Phase 3 (P3-03 > P3-02) |
| Methods | "Partial Domain Adaptation" | Phase 1 (Potato healthy missing) |
| Results | Figure 3 (Main Result) | Phase 4 vs Phase 3 |
| Results | Figure 4 (Confusion Matrix) | Phase 4 (P4-01) |
| Results | Table VII (Architecture Bench) | Phase 5 |
| Discussion | "EfficientNet Failure" | Phase 5 (P5-01) |
| Discussion | "Edge Viability" | Latency benchmark |

