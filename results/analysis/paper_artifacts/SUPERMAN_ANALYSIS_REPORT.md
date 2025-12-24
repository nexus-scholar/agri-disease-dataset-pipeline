# 🦸 COMPREHENSIVE EXPERIMENT ANALYSIS REPORT
## Plant Disease Classification: Lab-to-Field Domain Adaptation

**Generated:** December 14, 2025  
**Total Experiments:** 26  
**Total Compute Time:** 2.6 hours (GPU on Colab)

---

## 📊 EXECUTIVE SUMMARY

This report analyzes experiments across 5 phases testing domain adaptation techniques for plant disease classification, moving from controlled lab imagery to real-world field conditions.

### Key Numbers at a Glance
| Metric | Value |
|--------|-------|
| Average Lab Accuracy | **99.8%** |
| Average Field Accuracy (baseline) | **45.8%** |
| Average Domain Gap | **54.0%** |
| Best Final Field Accuracy | **81.5%** (Pepper + FixMatch) |
| Maximum Improvement | **+20.0%** (Tomato with EfficientNet + FixMatch) |

---

## 1️⃣ DOMAIN SHIFT ANALYSIS (Phase 1)

### The Problem: Catastrophic Performance Drop

Models trained on pristine lab images suffer massive accuracy drops when deployed on field images:

| Crop | Lab Accuracy | Field Accuracy | Gap | Severity |
|------|-------------|----------------|-----|----------|
| **Tomato** | 99.6% | 24.0% | **75.6%** | 🔴 SEVERE |
| **Potato** | 99.7% | 54.1% | **45.6%** | 🟡 MODERATE |
| **Pepper** | 100.0% | 59.3% | **40.7%** | 🟢 MILD |

### Architecture Performance (Baseline)

| Architecture | Tomato | Potato | Pepper | Avg |
|--------------|--------|--------|--------|-----|
| MobileNetV3 | 22.0% | 57.8% | 74.1% | **51.3%** |
| EfficientNet | 19.3% | 44.4% | 40.7% | 34.8% |
| MobileViT | 30.7% | 60.0% | 63.0% | **51.2%** |

**Key Insight:** MobileViT shows best average baseline performance, but the best per-crop varies.

---

## 2️⃣ STRONG AUGMENTATION ANALYSIS (Phase 2)

Testing aggressive data augmentation (RandAugment, CutOut, etc.):

| Crop | Baseline | +StrongAug | Delta | Result |
|------|----------|------------|-------|--------|
| Tomato | 22.0% | 24.0% | **+2.0%** | ✅ Helps |
| Potato | 57.8% | 62.2% | **+4.4%** | ✅ Helps |
| Pepper | 74.1% | 59.3% | **-14.8%** | ❌ Hurts |

**Key Insight:** Strong augmentation helps on harder domains (tomato, potato) but can hurt on simpler ones (pepper) - suggesting potential over-regularization.

---

## 3️⃣ ACTIVE LEARNING STRATEGIES (Phase 3)

Comparing selection strategies with 50 labeled target samples:

| Strategy | Tomato | vs Baseline |
|----------|--------|-------------|
| Random | 19.3% | -2.7% |
| Entropy | 21.3% | -0.7% |
| **Hybrid** | **24.0%** | **+2.0%** |

**Key Insight:** Hybrid strategy (combining entropy + diversity) outperforms pure uncertainty sampling. Random sampling actually hurts!

---

## 4️⃣ SEMI-SUPERVISED LEARNING - FixMatch (Phase 4)

FixMatch combines consistency regularization with pseudo-labeling:

| Crop | Baseline | +FixMatch | Improvement |
|------|----------|-----------|-------------|
| Tomato | 22.0% | 27.3% | **+5.3%** |
| Potato | 57.8% | 60.0% | **+2.2%** |
| Pepper | 74.1% | 81.5% | **+7.4%** |

**Key Insight:** FixMatch provides consistent improvements across all crops, with the largest absolute gain on pepper.

---

## 5️⃣ ARCHITECTURE COMPARISON WITH FIXMATCH (Phase 5)

Which architecture benefits most from FixMatch?

| Architecture | Tomato | Potato |
|--------------|--------|--------|
| MobileNetV3 | 27.3% | 60.0% |
| **EfficientNet** | **42.0%** | **60.0%** |
| MobileViT | 37.3% | 57.8% |

**Key Insight:** EfficientNet + FixMatch achieves the best results on tomato, nearly **doubling** performance from baseline (19.3% → 42.0%)!

---

## 6️⃣ PER-CLASS PERFORMANCE ANALYSIS

### Consistently Hard Classes (Tomato)

Classes with average F1 < 0.3 across all experiments:

| Class | Avg F1 | Analysis |
|-------|--------|----------|
| `tomato_spider_mites` | **0.000** | Never correctly classified! |
| `tomato_mosaic_virus` | 0.022 | Almost never detected |
| `tomato_bacterial_spot` | 0.104 | Very poor |
| `tomato_leaf_mold` | 0.110 | Very poor |
| `tomato_healthy` | 0.235 | Poor |
| `tomato_yellow_leaf_curl` | 0.253 | Poor |

**Key Insight:** Some classes show zero recall across all methods - these require additional investigation (possible label noise or extreme domain shift for these specific diseases).

### Best Class Performance

| Class | Best F1 | Achieved By |
|-------|---------|-------------|
| `tomato_healthy` | 0.621 | MobileViT + FixMatch |
| `tomato_leaf_mold` | 0.606 | EfficientNet + FixMatch |
| `tomato_early_blight` | 0.538 | EfficientNet + FixMatch |
| `tomato_late_blight` | 0.541 | MobileViT + FixMatch |

---

## 7️⃣ ACTIVE LEARNING EFFICIENCY

Label efficiency (% accuracy gain per labeled sample):

| Method | Crop | Efficiency |
|--------|------|------------|
| **EfficientNet + FixMatch** | Tomato | **0.453%/label** |
| **EfficientNet + FixMatch** | Potato | **0.389%/label** |
| MobileNetV3 + FixMatch | Pepper | 0.185%/label |
| MobileViT + FixMatch | Tomato | 0.133%/label |
| AL Hybrid (no FixMatch) | Tomato | 0.040%/label |

**Key Insight:** EfficientNet with FixMatch is the most label-efficient approach, gaining nearly 0.5% accuracy per labeled sample!

---

## 8️⃣ COMPUTATIONAL COST ANALYSIS

### Time by Phase

| Phase | Experiments | Avg Time | Total Time |
|-------|-------------|----------|------------|
| P1 (Baselines) | 12 | 6.1 min | 73.7 min |
| P2 (Strong Aug) | 3 | 2.6 min | 7.7 min |
| P3 (AL) | 4 | 3.0 min | 12.2 min |
| P4 (FixMatch) | 3 | 9.3 min | 27.8 min |
| P5 (Arch Bench) | 4 | 8.7 min | 34.6 min |

### Cost-Efficiency Ranking

| Method | Accuracy/Minute |
|--------|-----------------|
| Strong Aug (P2) | 2.36%/min |
| FixMatch (P4) | 1.89%/min |
| Arch Benchmark (P5) | 1.21%/min |
| AL Only (P3) | -2.68%/min ❌ |

**Key Insight:** Strong augmentation provides the best improvement per compute minute, but FixMatch provides larger absolute gains.

---

## 🏆 FINAL RECOMMENDATIONS

### Per-Crop Best Configuration

| Crop | Architecture | Method | Field Accuracy | Improvement |
|------|--------------|--------|----------------|-------------|
| **Tomato** | EfficientNet | FixMatch + Hybrid AL | **42.0%** | +20.0% |
| **Potato** | MobileNetV3 | Strong Aug | **62.2%** | +4.4% |
| **Pepper** | MobileNetV3 | FixMatch + Hybrid AL | **81.5%** | +7.4% |

### Methodology Ranking

1. **FixMatch + EfficientNet** - Best for hard domains (tomato), massive improvements
2. **FixMatch + MobileNetV3** - Best for simpler domains (pepper), efficient
3. **Strong Augmentation** - Quick win, good for moderate domains
4. **Hybrid AL** - Better than random/entropy, essential for FixMatch setup
5. **Pure AL without SSL** - Not recommended, often hurts performance

### Technical Recommendations

1. **Always use FixMatch** when unlabeled field data is available
2. **Choose architecture based on domain complexity:**
   - Complex (tomato, 9 classes): EfficientNet
   - Simple (pepper, 2 classes): MobileNetV3
3. **Hybrid AL strategy** over entropy or random for sample selection
4. **Be cautious with strong augmentation** on simple/easy domains
5. **Investigate zero-F1 classes** - may indicate data quality issues

---

## 📁 Generated Artifacts

### Tables (CSV)
- `TABLE_1_baseline_gap.csv` - Lab vs Field accuracy
- `TABLE_2_method_comparison.csv` - All methods head-to-head
- `TABLE_3_architecture.csv` - Architecture benchmark
- `table_al_trajectories.csv` - Learning curves data
- `table_fixmatch.csv` - FixMatch improvements
- `full_data.csv` - Complete dataset for further analysis

### Figures (PNG/PDF)
- `FIGURE_1_gap_barchart` - Generalization gap visualization
- `FIGURE_2_method_comparison` - Method comparison bar chart
- `FIGURE_3_al_trajectories` - Active learning curves
- `FIGURE_4_architecture_heatmap` - Architecture performance heatmap

---

## 📈 Future Work Suggestions

1. **Address zero-F1 classes:** Investigate `tomato_spider_mites`, `tomato_mosaic_virus`
2. **Try more AL rounds:** Current 5 rounds may not be enough
3. **Increase labeled budget:** Test with 100, 200 labels
4. **Domain-specific augmentation:** Agricultural-aware augmentation
5. **Ensemble methods:** Combine best models per class
6. **Source-free adaptation:** Reduce reliance on source data
7. **Test on other datasets:** Validate findings on PlantDoc, other field datasets

---

*Report generated by `comprehensive_analysis.py`*

