# Partial Domain Adaptation for Agricultural Edge AI

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-red.svg" alt="PyTorch 2.1+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> **Bridging the Lab-to-Field Generalization Gap with Active Learning and Semi-Supervised Learning**

## Overview

This repository implements a **Partial Domain Adaptation (PDA)** framework for plant disease detection, combining:
- **Active Learning** with Hybrid sampling (70% Entropy / 30% Random)
- **FixMatch** semi-supervised learning for handling missing classes
- Support for **MobileNetV3**, **EfficientNet**, and **MobileViT** architectures

## Key Features

- 🔬 **Semantic Label Mapping** - Handles class mismatches between PlantVillage (lab) and PlantDoc (field)
- 🎯 **PDA-aware Training** - Handles "phantom classes" present in source but missing in target
- 📊 **Forensic Experiment Logging** - Full reproducibility with config, splits, and metrics recording
- ⚡ **Edge-optimized** - Designed for deployment on resource-constrained devices

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/dataset-processing.git
cd dataset-processing
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Place datasets in `data/raw/dataset/`:
- `plantvillage.zip`
- `plantdoc.zip`

```bash
python process_datasets.py --datasets plantvillage plantdoc
```

### 2. Run Experiments

```bash
# List all 26 experiments
python colab_experiment_runner.py --list

# Run Phase 1 (Baselines)
python colab_experiment_runner.py --phase 1

# Run all experiments (~9 hours on GPU)
python colab_experiment_runner.py --all
```

### 3. Single Experiment

```bash
# Baseline training
python run_experiment.py --mode baseline --model mobilenetv3 --crop tomato

# Active Learning with FixMatch
python run_experiment.py --mode active --model mobilenetv3 --crop potato \
    --strategy hybrid --use-fixmatch --budget 10 --rounds 5
```

## Project Structure

```
├── src/                    # Core ML modules
│   ├── config/             # Configuration and crop mappings
│   ├── data/               # Data loading and transforms
│   ├── models/             # Model factory (MobileNet, EfficientNet, ViT)
│   ├── strategies/         # Active learning and FixMatch
│   └── utils/              # Metrics, recording, console utilities
├── experiments/            # Individual experiment scripts
├── pipeline/               # Dataset processing modules
├── data/
│   ├── raw/                # Place zip files here
│   ├── processed/          # Generated outputs
│   └── models/baselines/   # Trained model checkpoints
├── results/
│   └── experiments/        # Experiment logs and metrics
├── docs/                   # Documentation
│   ├── EXPERIMENTS.md      # Full experiment protocol
│   ├── datasets.md         # Dataset information
│   └── pipeline_overview.md
├── run_experiment.py       # CLI entry point
└── colab_experiment_runner.py  # Batch experiment runner
```

## Experiment Protocol

| Phase | Name | Experiments | Purpose |
|-------|------|-------------|---------|
| **P1** | Baselines | 12 | Measure generalization gap |
| **P2** | Strong Aug | 3 | Prove augmentation isn't enough |
| **P3** | AL Ablation | 4 | Justify Hybrid strategy |
| **P4** | FixMatch | 3 | Core SSL contribution |
| **P5** | Architecture | 4 | Benchmark models |

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the complete protocol.

## Supported Crops

| Crop | Classes | PDA Scenario |
|------|---------|--------------|
| Tomato | 7 | No (standard DA) |
| Potato | 3 | **Yes** (healthy missing in field) |
| Pepper | 2 | No (full alignment) |

## CLI Reference

```bash
python run_experiment.py --help

Options:
  --mode {baseline,active}              Experiment mode
  --model {mobilenetv3,efficientnet,mobilevit}
  --crop CROP                           Crop filter
  --strategy {random,entropy,hybrid}    AL strategy
  --use-fixmatch                        Enable FixMatch SSL
  --strong-aug                          Use AutoAugment
  --budget N                            Samples per AL round
  --rounds N                            Number of AL rounds
  --exp-name NAME                       Experiment identifier
```

## Citation

```bibtex
@article{author2025pda,
  title={Partial Domain Adaptation for Agricultural Edge AI: 
         Bridging the Lab-to-Field Gap with Active Learning},
  author={[Author Names]},
  journal={[Journal]},
  year={2025}
}
```

## License

Code: MIT License  
Datasets: Refer to original authors ([PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset))

