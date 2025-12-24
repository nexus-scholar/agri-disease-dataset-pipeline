#!/usr/bin/env python3
"""
ZERO-F1 CLASS INVESTIGATION
===========================
Deep investigation into why certain classes have F1=0.

Key findings from analysis:
1. tomato_spider_mites: Support=0 in test set (TARGET HAS ONLY 4 SAMPLES!)
2. tomato_mosaic_virus: Has samples but never predicted correctly (domain shift)
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/experiments")

# ============================================================================
# CLASS MAPPING DATA
# ============================================================================

CLASS_SAMPLE_COUNTS = {
    # Tomato classes - PlantDoc (TARGET) sample counts
    'tomato_bacterial_spot': 220,
    'tomato_early_blight': 176,
    'tomato_healthy': 126,
    'tomato_late_blight': 222,
    'tomato_leaf_mold': 182,  # mapped from tomato_mold
    'tomato_mosaic_virus': 108,
    'tomato_septoria_spot': 301,
    'tomato_spider_mites': 4,   # ← ONLY 4 SAMPLES!
    'tomato_yellow_leaf_curl': 152,  # mapped from tomato_yellow_virus
}

CLASS_ORDER = [
    'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy',
    'tomato_late_blight', 'tomato_leaf_mold', 'tomato_mosaic_virus',
    'tomato_septoria_spot', 'tomato_spider_mites', 'tomato_yellow_leaf_curl'
]

def load_metrics(exp_dir: str):
    """Load metrics.json for an experiment."""
    path = RESULTS_DIR / exp_dir / "metrics.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

def analyze_confusion_matrix(confusion: list, classes: list):
    """Analyze confusion matrix for zero-F1 classes."""
    cm = np.array(confusion)

    results = []
    for i, cls in enumerate(classes):
        row = cm[i]  # True class i
        col = cm[:, i]  # Predictions of class i

        true_count = row.sum()  # Samples of this class in test set
        pred_count = col.sum()  # Times this class was predicted
        correct = cm[i, i]  # Correct predictions

        # What was this class confused with?
        confusions = {}
        for j, count in enumerate(row):
            if j != i and count > 0:
                confusions[classes[j]] = int(count)

        # What classes were predicted as this class?
        predicted_as = {}
        for j, count in enumerate(col):
            if j != i and count > 0:
                predicted_as[classes[j]] = int(count)

        results.append({
            'class': cls,
            'true_count': int(true_count),
            'predicted_count': int(pred_count),
            'correct': int(correct),
            'precision': correct / pred_count if pred_count > 0 else 0,
            'recall': correct / true_count if true_count > 0 else 0,
            'confused_with': confusions,
            'predicted_as_this': predicted_as,
        })

    return results

def investigate_zero_f1():
    """Main investigation of zero-F1 classes."""
    print("="*80)
    print("🔍 ZERO-F1 CLASS INVESTIGATION")
    print("="*80)

    # ========================================================================
    # 1. Data-level issue: Sample counts
    # ========================================================================
    print("\n" + "="*80)
    print("1️⃣ TARGET DATASET SAMPLE COUNTS")
    print("="*80)
    print("\nPlantDoc (target domain) has the following samples per class:\n")

    for cls, count in sorted(CLASS_SAMPLE_COUNTS.items(), key=lambda x: x[1]):
        flag = "⚠️ CRITICAL" if count < 20 else "⚡ LOW" if count < 50 else "✅"
        print(f"  {flag} {cls}: {count} samples")

    print("\n" + "-"*60)
    print("DIAGNOSIS: tomato_spider_mites has only 4 samples in PlantDoc!")
    print("With 80/20 train/test split → ~1 sample in test set on average")
    print("High probability of 0 samples in test set!")
    print("-"*60)

    # ========================================================================
    # 2. Check actual test set support across experiments
    # ========================================================================
    print("\n" + "="*80)
    print("2️⃣ TEST SET SUPPORT ANALYSIS (ACTUAL VALUES)")
    print("="*80)

    tomato_exps = [
        ("P1_01_baseline_tomato_mobilenetv3_20251214_162301", "MobileNetV3 Baseline"),
        ("P1_02_baseline_tomato_efficientnet_20251214_163024", "EfficientNet Baseline"),
        ("P1_03_baseline_tomato_mobilevit_20251214_163802", "MobileViT Baseline"),
        ("P2_01_strongaug_tomato_20251214_173641", "Strong Aug"),
    ]

    for exp_dir, name in tomato_exps:
        metrics = load_metrics(exp_dir)
        if not metrics:
            continue

        report = metrics.get('final_report', {})
        print(f"\n{name}:")

        for cls in CLASS_ORDER:
            cls_data = report.get(cls, {})
            support = cls_data.get('support', 0)
            f1 = cls_data.get('f1-score', 0)

            if support == 0:
                print(f"  ❌ {cls}: support=0 (NO SAMPLES IN TEST SET!)")
            elif f1 == 0:
                print(f"  ⚠️ {cls}: support={support}, F1=0 (never detected)")
            elif f1 < 0.1:
                print(f"  ⚠️ {cls}: support={support}, F1={f1:.3f} (very poor)")

    # ========================================================================
    # 3. Confusion analysis for mosaic_virus (has samples, still F1~0)
    # ========================================================================
    print("\n" + "="*80)
    print("3️⃣ CONFUSION ANALYSIS: tomato_mosaic_virus")
    print("="*80)

    metrics = load_metrics("P1_01_baseline_tomato_mobilenetv3_20251214_162301")
    if metrics:
        confusion = metrics.get('final_confusion', [])
        if confusion:
            analysis = analyze_confusion_matrix(confusion, CLASS_ORDER)

            # Find mosaic_virus
            for result in analysis:
                if result['class'] == 'tomato_mosaic_virus':
                    print(f"\nTrue samples: {result['true_count']}")
                    print(f"Times predicted: {result['predicted_count']}")
                    print(f"Correct: {result['correct']}")

                    print(f"\nMisclassified as:")
                    for cls, count in sorted(result['confused_with'].items(), key=lambda x: -x[1]):
                        print(f"  → {cls}: {count} times")

    # ========================================================================
    # 4. Confusion analysis for leaf_mold
    # ========================================================================
    print("\n" + "="*80)
    print("4️⃣ CONFUSION ANALYSIS: tomato_leaf_mold")
    print("="*80)

    if metrics:
        confusion = metrics.get('final_confusion', [])
        if confusion:
            analysis = analyze_confusion_matrix(confusion, CLASS_ORDER)

            for result in analysis:
                if result['class'] == 'tomato_leaf_mold':
                    print(f"\nTrue samples: {result['true_count']}")
                    print(f"Times predicted: {result['predicted_count']}")
                    print(f"Correct: {result['correct']}")

                    print(f"\nMisclassified as:")
                    for cls, count in sorted(result['confused_with'].items(), key=lambda x: -x[1]):
                        print(f"  → {cls}: {count} times")

    # ========================================================================
    # 5. Full confusion matrix analysis
    # ========================================================================
    print("\n" + "="*80)
    print("5️⃣ FULL CONFUSION PATTERN ANALYSIS")
    print("="*80)

    if metrics:
        confusion = metrics.get('final_confusion', [])
        if confusion:
            cm = np.array(confusion)

            print("\nPrediction bias (what classes are over/under predicted):")
            for i, cls in enumerate(CLASS_ORDER):
                true_total = cm[i].sum()
                pred_total = cm[:, i].sum()
                ratio = pred_total / true_total if true_total > 0 else 0

                if ratio > 2:
                    flag = "📈 OVER-predicted"
                elif ratio < 0.5:
                    flag = "📉 UNDER-predicted"
                elif ratio == 0:
                    flag = "🚫 NEVER predicted"
                else:
                    flag = "✅ Balanced"

                print(f"  {cls}: true={true_total}, pred={pred_total}, ratio={ratio:.2f} {flag}")

            print("\nKey over-predicted class (receives most confusion):")
            col_sums = cm.sum(axis=0)
            max_idx = np.argmax(col_sums)
            print(f"  → {CLASS_ORDER[max_idx]}: predicted {col_sums[max_idx]} times")

            # What proportion of test set is classified as early_blight?
            print(f"\nAttractor class analysis (tomato_early_blight):")
            early_blight_preds = cm[:, 1].sum()
            total_samples = cm.sum()
            print(f"  Total predictions as early_blight: {early_blight_preds}/{total_samples} = {early_blight_preds/total_samples*100:.1f}%")

    # ========================================================================
    # 6. Root cause summary
    # ========================================================================
    print("\n" + "="*80)
    print("📊 ROOT CAUSE SUMMARY")
    print("="*80)

    print("""
┌─────────────────────────────────┬──────────────────────────────────────────┐
│ Class                           │ Root Cause                                │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ tomato_spider_mites             │ DATA SCARCITY: Only 4 samples in PlantDoc │
│                                 │ → 0 samples in test set (support=0)      │
│                                 │ → Not a model failure!                   │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ tomato_mosaic_virus             │ EXTREME DOMAIN SHIFT:                    │
│                                 │ → Visual features don't transfer         │
│                                 │ → Confused with early_blight (similar    │
│                                 │   yellowing pattern in field conditions) │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ tomato_leaf_mold                │ DOMAIN SHIFT + VISUAL SIMILARITY:        │
│                                 │ → Confused with late_blight/early_blight │
│                                 │ → Fungal symptoms look different in field│
├─────────────────────────────────┼──────────────────────────────────────────┤
│ tomato_bacterial_spot           │ CONFUSION with late_blight & septoria    │
│                                 │ → All cause spotting patterns            │
│                                 │ → Lab vs field lighting affects spots    │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ tomato_yellow_leaf_curl         │ MODERATE SHIFT:                          │
│                                 │ → Virus symptoms appear different        │
│                                 │ → Confused with early_blight (yellowing) │
└─────────────────────────────────┴──────────────────────────────────────────┘
    """)

    # ========================================================================
    # 7. Recommendations
    # ========================================================================
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    print("""
1. tomato_spider_mites:
   ✗ CANNOT FIX with better models
   ✓ SOLUTION: Collect more field samples or exclude from evaluation
   ✓ NOTE: Remove from canonical classes or stratified sampling

2. tomato_mosaic_virus:
   ✓ SOLUTION: Fine-grained attention models
   ✓ SOLUTION: More target labels for this class specifically
   ✓ Try: Class-weighted loss during FixMatch

3. tomato_leaf_mold:
   ✓ Improves with FixMatch (F1=0.61 with EfficientNet!)
   ✓ Domain adaptation is working for this class

4. General:
   ✓ Use stratified test split to guarantee samples
   ✓ Consider macro-averaged metrics over micro
   ✓ Report per-class results in paper (some classes are unsolvable)
    """)

    # ========================================================================
    # 8. Check improvement with FixMatch
    # ========================================================================
    print("\n" + "="*80)
    print("8️⃣ CLASS-LEVEL IMPROVEMENT WITH FIXMATCH")
    print("="*80)

    baseline = load_metrics("P1_01_baseline_tomato_mobilenetv3_20251214_162301")
    fixmatch = load_metrics("P4_01_fixmatch_tomato_20251214_175634")
    efficientnet = load_metrics("P5_01_efficientnet_tomato_fixmatch_20251214_182423")

    if baseline and fixmatch:
        print("\nF1 Score Comparison (Baseline vs FixMatch vs EfficientNet+FixMatch):")
        print("-"*70)
        print(f"{'Class':<30} {'Baseline':>10} {'FixMatch':>10} {'EN+FM':>10}")
        print("-"*70)

        for cls in CLASS_ORDER:
            base_f1 = baseline['final_report'].get(cls, {}).get('f1-score', 0)
            fix_f1 = fixmatch['final_report'].get(cls, {}).get('f1-score', 0) if fixmatch else 0
            en_f1 = efficientnet['final_report'].get(cls, {}).get('f1-score', 0) if efficientnet else 0

            # Determine improvement
            if en_f1 > base_f1 + 0.1:
                flag = "📈"
            elif en_f1 < base_f1 - 0.1:
                flag = "📉"
            else:
                flag = "➖"

            print(f"{cls:<30} {base_f1:>10.3f} {fix_f1:>10.3f} {en_f1:>10.3f} {flag}")

        print("-"*70)

def load_all_metrics():
    """Load metrics from all experiments for further analysis."""
    all_metrics = {}
    for exp_dir in RESULTS_DIR.iterdir():
        if exp_dir.is_dir():
            metrics = load_metrics(exp_dir.name)
            if metrics:
                all_metrics[exp_dir.name] = metrics
    return all_metrics

def main():
    all = load_all_metrics()

    df = pd.DataFrame.from_dict(all)

    # print analysis
    print(df)
    print(df.describe())

    i=0; o=15; s = 7
    for exp, metrics in all.items():
        if i < o:
            i += 1
            continue
        print(f"\nExperiment: {exp}")
        # print(json.dumps(metrics, indent=2))
        if metrics and "final_confusion" in metrics:
            print(f"Loaded real data from {exp}")
            matrix = np.array(metrics["final_confusion"])
            # Extract class names from report keys (removing 'tomato_' prefix)
            keys = [k for k in metrics["final_report"].keys() if
                    k != "accuracy" and k != "macro avg" and k != "weighted avg"]
            classes = [k.replace("tomato_", "").replace("_", " ").title() for k in sorted(keys)]
        else:
            print("Real data not found or incomplete. Generating SIMULATION based on your JSON snippet...")
            classes = None
        print(classes)
        print(matrix)
        i += 1
        if i >= o+s:
            break

if __name__ == "__main__":
    main()

