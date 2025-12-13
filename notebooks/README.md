# 📓 Jupyter Notebooks

Interactive notebooks for running experiments without command-line knowledge.

## Available Notebooks

| Notebook | Purpose | Difficulty |
|----------|---------|------------|
| **01_run_all_experiments.ipynb** | Complete walkthrough of all experiments | Beginner |
| **02_quick_experiment.ipynb** | Run individual experiments quickly | Beginner |
| **03_visualize_results.ipynb** | Create publication-ready figures | Intermediate |

---

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
# From project root
cd notebooks
jupyter notebook
```

Then open `01_run_all_experiments.ipynb` and follow the instructions.

### Option 2: JupyterLab

```bash
jupyter lab
```

### Option 3: VS Code

1. Open VS Code
2. Install "Jupyter" extension
3. Open any `.ipynb` file
4. Click "Run All" or run cells individually

---

## 📓 Notebook Descriptions

### 01_run_all_experiments.ipynb

**Full experiment pipeline with detailed explanations.**

This notebook:
- Sets up the environment automatically
- Runs all 5 experiments in sequence
- Explains what each experiment does
- Shows results with visualizations
- Saves figures to `results/figures/`

Best for: **First-time users** who want to understand the full research pipeline.

### 02_quick_experiment.ipynb

**Run individual experiments with custom settings.**

This notebook:
- Lets you choose which experiment to run (1-5)
- Allows customization of parameters
- Runs experiments using subprocess (cleaner output)
- Quick and simple interface

Best for: **Repeated experimentation** with different settings.

### 03_visualize_results.ipynb

**Create publication-ready figures from results.**

This notebook:
- Takes your experiment results as input
- Generates high-quality figures
- Saves PNG (for presentations) and PDF (for papers)
- Includes summary tables

Best for: **Paper writing** and preparing visualizations.

---

## ⚙️ Configuration Options

### Common Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLASS_NAME` | "Tomato" | Filter to specific crop |
| `EPOCHS` | 5 | Training epochs |
| `BATCH_SIZE` | 16 | Batch size (reduce if memory error) |

### Active Learning Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BUDGET_PER_ROUND` | 50 | Samples labeled per round |
| `NUM_ROUNDS` | 4 | Number of AL rounds |

### Available Class Filters

- `"Tomato"` - Tomato diseases
- `"Apple"` - Apple diseases
- `"Grape"` - Grape diseases
- `"Corn"` - Corn diseases
- `None` - All classes (slower but more comprehensive)

---

## 💡 Tips for Supervisors

### If you get an error:

1. **"Module not found"**: Run the Setup cell first
2. **"CUDA out of memory"**: Reduce `BATCH_SIZE` to 8 or 4
3. **"baseline_model.pth not found"**: Run Experiment 01 first
4. **Kernel crashes**: Restart kernel and reduce batch size

### To run faster:

- Reduce `EPOCHS` to 3
- Reduce `NUM_ROUNDS` to 2
- Use a smaller class filter

### To get better results:

- Increase `EPOCHS` to 10-15
- Run the full dataset (`CLASS_NAME = None`)
- Use more `NUM_ROUNDS` with smaller `BUDGET_PER_ROUND`

---

## 📊 Expected Results

After running all experiments, you should see:

| Experiment | Lab Acc | Field Acc | Gap |
|------------|---------|-----------|-----|
| 01 Baseline | ~95-99% | ~25-30% | ~65-70% |
| 02 Passive Aug | ~88-92% | ~30-35% | ~55-60% |
| 03 CutMix | ~85-90% | ~32-38% | ~50-55% |

| AL Strategy | 0 labels | 200 labels |
|-------------|----------|------------|
| Random | ~26% | ~45% |
| Entropy | ~26% | ~43% |
| **Hybrid** | ~26% | **~46%** |

---

## 📁 Output Files

Notebooks save results to:

```
phd-active-adaptation-v1/
├── data/models/
│   └── baseline_model.pth    # Trained baseline model
├── results/
│   ├── figures/
│   │   ├── Figure1_Generalization_Gap.png
│   │   ├── Figure2_Active_Learning.png
│   │   └── notebook_results.png
│   └── logs/
│       └── exp01_baseline_*.log
```

---

## 🆘 Getting Help

If something doesn't work:

1. Check that all required packages are installed (`pip install -r requirements.txt`)
2. Make sure you're using Python 3.8+
3. Verify the dataset folder exists and contains PlantVillage/PlantDoc
4. Try restarting the Jupyter kernel

