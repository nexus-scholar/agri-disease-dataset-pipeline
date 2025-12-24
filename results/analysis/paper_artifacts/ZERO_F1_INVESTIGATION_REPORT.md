# 🔬 ZERO-F1 CLASS INVESTIGATION REPORT

## Executive Summary

Investigation into why certain tomato disease classes achieve F1=0 across all experiments revealed **TWO DISTINCT ROOT CAUSES**:

| Class | Root Cause | Fixable? |
|-------|------------|----------|
| `tomato_spider_mites` | **DATA SCARCITY** (4 samples) | ❌ No |
| `tomato_mosaic_virus` | **EXTREME DOMAIN SHIFT** | ⚠️ Partially |
| `tomato_leaf_mold` | **DOMAIN SHIFT** | ✅ Yes (with FixMatch) |

---

## 1. Data Scarcity Issue: `tomato_spider_mites`

### The Problem
```
PlantDoc (target domain) sample counts:
  tomato_spider_mites: 4 samples  ← ONLY 4!
  tomato_mosaic_virus: 108 samples
  tomato_healthy: 126 samples
  ... other classes: 150-301 samples
```

### What Happens
- With 80/20 pool/test split of 4 samples → **0-1 samples in test set**
- Across ALL experiments: `support=0` (no samples to evaluate!)
- **This is NOT a model failure** - there's simply no data to test on

### Evidence
```
MobileNetV3 Baseline: tomato_spider_mites - support=0 (NO SAMPLES IN TEST SET!)
EfficientNet Baseline: tomato_spider_mites - support=0 (NO SAMPLES IN TEST SET!)
MobileViT Baseline: tomato_spider_mites - support=0 (NO SAMPLES IN TEST SET!)
Strong Aug: tomato_spider_mites - support=0 (NO SAMPLES IN TEST SET!)
```

### Solution Options
1. **EXCLUDE from evaluation** - Report as "insufficient data"
2. **Use stratified sampling** - Guarantee at least 1 sample per class in test
3. **Collect more data** - Need at least 20 samples minimum
4. **Cross-validation** - Leave-one-out for rare classes

### Recommendation
```python
# In crop_configs.py or evaluation code:
EVALUATION_EXCLUSIONS = ["tomato_spider_mites"]  # Insufficient target data
```

---

## 2. Domain Shift Issue: `tomato_mosaic_virus`

### The Problem
Despite having **108 samples in PlantDoc (14 in test)**, the class is NEVER correctly predicted.

### Confusion Pattern (MobileNetV3 Baseline)
```
True samples: 14
Times predicted: 0  ← NEVER predicts this class!
Correct: 0

Misclassified as:
  → tomato_early_blight: 8 times (57%)
  → tomato_late_blight: 4 times (29%)
  → tomato_healthy: 1 time
  → tomato_septoria_spot: 1 time
```

### Root Cause: Visual Domain Shift
The symptoms of Tomato Mosaic Virus look VERY different between lab and field:

| Lab (PlantVillage) | Field (PlantDoc) |
|-------------------|------------------|
| Clear mottled pattern | Yellowing & curling |
| Controlled lighting | Variable lighting |
| Clean backgrounds | Cluttered backgrounds |
| Single leaf focus | Whole plant/multiple leaves |

The field symptoms (yellowing) get confused with `early_blight` which also causes yellowing.

### Improvement with FixMatch
```
Class                          Baseline   FixMatch   EN+FixMatch
tomato_mosaic_virus              0.000      0.222       0.000
```

**MobileNetV3 + FixMatch improves to F1=0.222!** But EfficientNet regresses - suggesting the class needs specific attention.

### Solution Options
1. **Class-weighted loss** - Increase weight for mosaic_virus during training
2. **More labeled samples** - Prioritize this class in active learning
3. **Fine-grained model** - Attention mechanisms for subtle virus patterns
4. **Separate binary classifier** - Train dedicated mosaic vs non-mosaic

---

## 3. Recoverable Domain Shift: `tomato_leaf_mold`

### The Problem
Baseline: F1=0.000 (never detected)

### Confusion Pattern
```
True samples: 21
Predicted as leaf_mold: 1 time
Correct: 0

Misclassified as:
  → tomato_early_blight: 11 times (52%)
  → tomato_late_blight: 8 times (38%)
  → tomato_septoria_spot: 2 times
```

### Recovery with Domain Adaptation
```
Class                          Baseline   FixMatch   EN+FixMatch
tomato_leaf_mold                 0.000      0.000       0.606 📈
```

**EfficientNet + FixMatch achieves F1=0.606!** This is a massive improvement.

### Why It Works
- EfficientNet has better feature extraction for subtle fungal patterns
- FixMatch pseudo-labeling helps calibrate features to target domain
- The class IS distinguishable - it just needs proper domain adaptation

---

## 4. Attractor Class Problem: `tomato_early_blight`

### The Problem
The model over-predicts `early_blight` - it's an "attractor class":

```
Prediction Distribution:
  tomato_early_blight: predicted 77/150 = 51.3%  ← MAJOR BIAS
  tomato_late_blight: predicted 38/150 = 25.3%
  All other classes: predicted 35/150 = 23.4%
```

### Why This Happens
1. **Visual similarity** - Yellowing is common symptom
2. **Class imbalance in source** - PlantVillage has imbalanced classes
3. **Easier to classify** - Clearer visual boundaries in lab data

### Solution
- Temperature scaling for calibrated predictions
- Focal loss to reduce easy-example dominance
- Balanced sampling during training

---

## 5. Summary: Class-by-Class Improvement

| Class | Baseline | FixMatch | EN+FM | Status |
|-------|----------|----------|-------|--------|
| `tomato_bacterial_spot` | 0.071 | 0.061 | **0.340** | 📈 Improving |
| `tomato_early_blight` | 0.264 | 0.182 | **0.538** | 📈 Improving |
| `tomato_healthy` | 0.091 | 0.077 | **0.529** | 📈 Improving |
| `tomato_late_blight` | 0.377 | 0.483 | **0.500** | 📈 Improving |
| `tomato_leaf_mold` | 0.000 | 0.000 | **0.606** | 📈 **RECOVERED!** |
| `tomato_mosaic_virus` | 0.000 | **0.222** | 0.000 | ⚠️ Needs attention |
| `tomato_septoria_spot` | 0.269 | **0.400** | 0.360 | ✅ Improving |
| `tomato_spider_mites` | 0.000 | 0.000 | 0.000 | ❌ **No data** |
| `tomato_yellow_leaf_curl` | 0.235 | 0.390 | **0.462** | 📈 Improving |

---

## 6. Actionable Recommendations

### Immediate Actions

1. **Exclude `tomato_spider_mites` from evaluation**
   ```python
   # Only 4 samples in PlantDoc - statistically meaningless
   if class_name == "tomato_spider_mites":
       continue  # Skip in evaluation
   ```

2. **Use macro F1 with exclusions** for paper reporting
   ```python
   valid_classes = [c for c in classes if c != "tomato_spider_mites"]
   macro_f1 = np.mean([f1_scores[c] for c in valid_classes])
   ```

3. **Add stratified sampling** for future experiments
   ```python
   from sklearn.model_selection import StratifiedShuffleSplit
   ```

### Future Work

1. **Class-weighted training** for `tomato_mosaic_virus`
2. **Collect more PlantDoc samples** for spider_mites
3. **Ensemble model** - MobileNetV3+FixMatch for virus, EfficientNet+FixMatch for rest
4. **Per-class domain adaptation** - Different strategies per class

---

## 7. Paper Discussion Points

### What to Report
- The 54% average domain gap
- Per-class F1 scores showing heterogeneous performance
- `spider_mites` exclusion due to data scarcity (transparent reporting)
- Recovery of `leaf_mold` with domain adaptation (success story)
- Persistent challenge with `mosaic_virus` (honest limitation)

### Framing
> "Not all classes are equally affected by domain shift. While some classes like `tomato_leaf_mold` can be recovered through semi-supervised domain adaptation (F1: 0→0.61), others like `tomato_mosaic_virus` present fundamental visual domain shifts that current methods cannot fully address (F1: 0→0.22). Furthermore, `tomato_spider_mites` could not be evaluated due to data scarcity in the target domain (n=4)."

---

*Investigation completed: December 14, 2025*
*Script: `investigate_zero_f1.py`*

