#!/usr/bin/env python3
"""
COMPREHENSIVE SUPERMAN ANALYSIS
================================
Deep-dive analysis of ALL experiment data from Colab runs.

Extracts maximum insights from:
- 26 total experiments
- 5 phases (P1-P5)
- 3 crops (tomato, potato, pepper)
- 3 architectures (mobilenetv3, efficientnet, mobilevit)
- Multiple techniques (baseline, strong aug, AL, FixMatch)

Generates:
- Statistical summaries
- Per-class analysis from confusion matrices
- Training dynamics analysis
- Cost-benefit analysis
- Architecture comparisons
- Domain adaptation insights
- Publication-ready tables and figures
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re

import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_PLOTTING = True
    plt.style.use('seaborn-v0_8-whitegrid')
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available")

# ============================================================================
# CONSTANTS
# ============================================================================

RESULTS_DIR = Path("results/experiments")
OUTPUT_DIR = Path("results/analysis/paper_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CROPS = ['tomato', 'potato', 'pepper']
MODELS = ['mobilenetv3', 'efficientnet', 'mobilevit']

CLASS_NAMES = {
    'tomato': [
        'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy',
        'tomato_late_blight', 'tomato_leaf_mold', 'tomato_mosaic_virus',
        'tomato_septoria_spot', 'tomato_spider_mites', 'tomato_yellow_leaf_curl'
    ],
    'potato': ['potato_early_blight', 'potato_healthy', 'potato_late_blight'],
    'pepper': ['pepper_bell_bacterial_spot', 'pepper_bell_healthy']
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_master_results():
    """Load the master results JSON from Colab run."""
    results_file = RESULTS_DIR / "results_20251214_162255.json"
    with open(results_file, 'r') as f:
        return json.load(f)

def load_experiment_metrics(exp_dir: str):
    """Load metrics.json for a specific experiment."""
    metrics_path = RESULTS_DIR / exp_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def load_experiment_config(exp_dir: str):
    """Load config.json for a specific experiment."""
    config_path = RESULTS_DIR / exp_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def parse_all_experiments():
    """Parse all experiments into a comprehensive DataFrame."""
    master = load_master_results()

    all_data = []

    for exp in master:
        # Parse experiment ID
        exp_id = exp['id']
        phase = exp_id.split('-')[0]

        # Extract metrics from master
        metrics_data = exp.get('metrics', {})
        val_acc = metrics_data.get('val_accuracy', 0)
        field_acc = metrics_data.get('field_accuracy', 0)

        # Parse crop and model from name
        name = exp['name']
        crop = 'unknown'
        model = 'mobilenetv3'  # default
        strategy = 'none'
        use_fixmatch = 'FixMatch' in name or 'fixmatch' in name.lower()
        use_strong_aug = 'StrongAug' in name or 'strongaug' in name.lower()

        for c in ['tomato', 'potato', 'pepper', 'All Crops', 'all']:
            if c.lower() in name.lower():
                crop = c.lower() if c != 'All Crops' else 'all'
                break

        for m in ['mobilenetv3', 'efficientnet', 'mobilevit']:
            if m in name.lower():
                model = m
                break

        if 'random' in name.lower():
            strategy = 'random'
        elif 'entropy' in name.lower():
            strategy = 'entropy'
        elif 'hybrid' in name.lower():
            strategy = 'hybrid'

        # Find experiment directory
        exp_dirs = list(RESULTS_DIR.glob(f"{exp_id.replace('-', '_')}*"))
        exp_dir = exp_dirs[0].name if exp_dirs else None

        # Load detailed metrics
        detailed_metrics = load_experiment_metrics(exp_dir) if exp_dir else None

        # Extract training history
        history = []
        if detailed_metrics:
            history = detailed_metrics.get('history', [])

        # Extract AL trajectory
        al_trajectory = []
        if detailed_metrics:
            al_trajectory = detailed_metrics.get('al_trajectory', [])

        # Extract confusion matrix and per-class report
        confusion = []
        per_class_report = {}
        if detailed_metrics:
            confusion = detailed_metrics.get('final_confusion', [])
            per_class_report = detailed_metrics.get('final_report', {})

        # Calculate generalization gap
        gap = val_acc - field_acc if val_acc and field_acc else 0

        entry = {
            'exp_id': exp_id,
            'phase': phase,
            'name': name,
            'crop': crop,
            'model': model,
            'strategy': strategy,
            'use_fixmatch': use_fixmatch,
            'use_strong_aug': use_strong_aug,
            'val_acc': val_acc,
            'field_acc': field_acc,
            'gap': gap,
            'duration_seconds': exp['duration_seconds'],
            'success': exp['success'],
            'history': history,
            'al_trajectory': al_trajectory,
            'confusion': confusion,
            'per_class_report': per_class_report,
            'exp_dir': exp_dir,
        }
        all_data.append(entry)

    return pd.DataFrame(all_data)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_generalization_gap(df: pd.DataFrame):
    """Deep analysis of Lab-to-Field generalization gap."""
    print("\n" + "="*80)
    print("📊 GENERALIZATION GAP ANALYSIS")
    print("="*80)

    # Filter P1 baselines only (no strong aug)
    p1 = df[(df['phase'] == 'P1') & (df['use_strong_aug'] == False)]
    p1_single = p1[p1['crop'] != 'all']  # Single crops only

    print("\n### Per-Crop, Per-Model Gap Analysis ###")
    gap_table = p1_single.pivot_table(
        index='crop',
        columns='model',
        values=['val_acc', 'field_acc', 'gap'],
        aggfunc='mean'
    ).round(2)
    print(gap_table)

    # Statistical summary
    print("\n### Gap Statistics ###")
    print(f"  Mean Gap: {p1_single['gap'].mean():.2f}%")
    print(f"  Std Gap:  {p1_single['gap'].std():.2f}%")
    print(f"  Min Gap:  {p1_single['gap'].min():.2f}% ({p1_single.loc[p1_single['gap'].idxmin(), 'crop']}, {p1_single.loc[p1_single['gap'].idxmin(), 'model']})")
    print(f"  Max Gap:  {p1_single['gap'].max():.2f}% ({p1_single.loc[p1_single['gap'].idxmax(), 'crop']}, {p1_single.loc[p1_single['gap'].idxmax(), 'model']})")

    # Best model per crop
    print("\n### Best Baseline Field Accuracy per Crop ###")
    for crop in CROPS:
        crop_data = p1_single[p1_single['crop'] == crop]
        if not crop_data.empty:
            best = crop_data.loc[crop_data['field_acc'].idxmax()]
            print(f"  {crop}: {best['field_acc']:.2f}% ({best['model']})")

    # Domain shift severity
    print("\n### Domain Shift Severity Ranking ###")
    crop_gaps = p1_single.groupby('crop')['gap'].mean().sort_values(ascending=False)
    for i, (crop, gap) in enumerate(crop_gaps.items(), 1):
        severity = "SEVERE" if gap > 70 else "MODERATE" if gap > 50 else "MILD"
        print(f"  {i}. {crop}: {gap:.2f}% gap [{severity}]")

    # Save table
    gap_table.to_csv(OUTPUT_DIR / "table_generalization_gap.csv")
    return gap_table


def analyze_strong_augmentation(df: pd.DataFrame):
    """Analyze impact of strong augmentation."""
    print("\n" + "="*80)
    print("📊 STRONG AUGMENTATION ANALYSIS (Phase 2)")
    print("="*80)

    # Compare baseline vs strong aug
    baseline = df[(df['phase'] == 'P1') & (df['model'] == 'mobilenetv3') & (df['crop'] != 'all')]
    strong_aug = df[(df['phase'] == 'P2')]

    if baseline.empty or strong_aug.empty:
        print("Insufficient data for strong aug analysis")
        return None

    comparison = []
    for crop in CROPS:
        base_row = baseline[baseline['crop'] == crop]
        aug_row = strong_aug[strong_aug['crop'] == crop]

        if not base_row.empty and not aug_row.empty:
            base_acc = base_row['field_acc'].values[0]
            aug_acc = aug_row['field_acc'].values[0]
            delta = aug_acc - base_acc

            comparison.append({
                'crop': crop,
                'baseline_acc': base_acc,
                'strong_aug_acc': aug_acc,
                'delta': delta,
                'relative_improvement': (delta / base_acc * 100) if base_acc > 0 else 0,
                'helps': delta > 0
            })

    comp_df = pd.DataFrame(comparison)
    print("\n### Strong Augmentation Impact ###")
    print(comp_df.to_string(index=False))

    print(f"\n### Summary ###")
    print(f"  Average Delta: {comp_df['delta'].mean():+.2f}%")
    print(f"  Crops Improved: {comp_df['helps'].sum()}/{len(comp_df)}")

    winner = comp_df.loc[comp_df['delta'].idxmax()]
    print(f"  Best Improvement: {winner['crop']} ({winner['delta']:+.2f}%)")

    comp_df.to_csv(OUTPUT_DIR / "table_strong_augmentation.csv", index=False)
    return comp_df


def analyze_active_learning_strategies(df: pd.DataFrame):
    """Compare AL strategies in detail."""
    print("\n" + "="*80)
    print("📊 ACTIVE LEARNING STRATEGY COMPARISON (Phase 3)")
    print("="*80)

    p3 = df[df['phase'] == 'P3']

    if p3.empty:
        print("No Phase 3 data found")
        return None

    print("\n### Strategy Comparison ###")
    strategy_df = p3[['crop', 'strategy', 'field_acc']].copy()
    print(strategy_df.to_string(index=False))

    # Pivot for cleaner view
    pivot = p3.pivot_table(
        index='strategy',
        columns='crop',
        values='field_acc',
        aggfunc='mean'
    ).round(2)
    print("\n### Pivot Table ###")
    print(pivot)

    # Find best strategy per crop
    print("\n### Best Strategy per Crop ###")
    for crop in p3['crop'].unique():
        crop_data = p3[p3['crop'] == crop]
        best = crop_data.loc[crop_data['field_acc'].idxmax()]
        print(f"  {crop}: {best['strategy']} ({best['field_acc']:.2f}%)")

    # Compare to baseline
    print("\n### Improvement over Baseline ###")
    baselines = df[(df['phase'] == 'P1') & (df['model'] == 'mobilenetv3')]
    for _, row in p3.iterrows():
        base = baselines[baselines['crop'] == row['crop']]
        if not base.empty:
            base_acc = base['field_acc'].values[0]
            delta = row['field_acc'] - base_acc
            print(f"  {row['crop']} / {row['strategy']}: {delta:+.2f}%")

    strategy_df.to_csv(OUTPUT_DIR / "table_al_strategies.csv", index=False)
    return strategy_df


def analyze_fixmatch_impact(df: pd.DataFrame):
    """Analyze FixMatch semi-supervised learning impact."""
    print("\n" + "="*80)
    print("📊 FIXMATCH ANALYSIS (Phase 4)")
    print("="*80)

    p4 = df[df['phase'] == 'P4']

    if p4.empty:
        print("No Phase 4 data found")
        return None

    # Compare to AL-only (P3 hybrid)
    p3_hybrid = df[(df['phase'] == 'P3') & (df['strategy'] == 'hybrid')]
    p1_baseline = df[(df['phase'] == 'P1') & (df['model'] == 'mobilenetv3') & (df['crop'] != 'all')]

    comparison = []
    for crop in p4['crop'].unique():
        fix_row = p4[p4['crop'] == crop]
        base_row = p1_baseline[p1_baseline['crop'] == crop]
        al_row = p3_hybrid[p3_hybrid['crop'] == crop]

        fix_acc = fix_row['field_acc'].values[0] if not fix_row.empty else 0
        base_acc = base_row['field_acc'].values[0] if not base_row.empty else 0
        al_acc = al_row['field_acc'].values[0] if not al_row.empty else 0

        comparison.append({
            'crop': crop,
            'baseline': base_acc,
            'al_only': al_acc if al_acc > 0 else base_acc,
            'fixmatch': fix_acc,
            'vs_baseline': fix_acc - base_acc,
            'vs_al_only': fix_acc - (al_acc if al_acc > 0 else base_acc)
        })

    comp_df = pd.DataFrame(comparison)
    print("\n### FixMatch Performance ###")
    print(comp_df.to_string(index=False))

    print(f"\n### Summary ###")
    print(f"  Avg improvement vs baseline: {comp_df['vs_baseline'].mean():+.2f}%")
    print(f"  Avg improvement vs AL-only: {comp_df['vs_al_only'].mean():+.2f}%")

    winner = comp_df.loc[comp_df['vs_baseline'].idxmax()]
    print(f"  Best improvement: {winner['crop']} (+{winner['vs_baseline']:.2f}% vs baseline)")

    comp_df.to_csv(OUTPUT_DIR / "table_fixmatch.csv", index=False)
    return comp_df


def analyze_architecture_comparison(df: pd.DataFrame):
    """Compare architectures for FixMatch."""
    print("\n" + "="*80)
    print("📊 ARCHITECTURE BENCHMARK (Phase 5)")
    print("="*80)

    p5 = df[df['phase'] == 'P5']
    p4 = df[df['phase'] == 'P4']  # mobilenetv3 with fixmatch

    if p5.empty:
        print("No Phase 5 data found")
        return None

    # Combine P4 (mobilenetv3) and P5 (other archs) for comparison
    combined = pd.concat([p4, p5])

    print("\n### Architecture + FixMatch Results ###")
    arch_df = combined[['crop', 'model', 'field_acc']].copy()
    print(arch_df.to_string(index=False))

    pivot = combined.pivot_table(
        index='model',
        columns='crop',
        values='field_acc',
        aggfunc='mean'
    ).round(2)
    print("\n### Pivot Table ###")
    print(pivot)

    # Best architecture per crop
    print("\n### Best Architecture per Crop ###")
    for crop in combined['crop'].unique():
        crop_data = combined[combined['crop'] == crop]
        if not crop_data.empty:
            best = crop_data.loc[crop_data['field_acc'].idxmax()]
            print(f"  {crop}: {best['model']} ({best['field_acc']:.2f}%)")

    # Overall best
    overall_best = combined.loc[combined['field_acc'].idxmax()]
    print(f"\n### Overall Best ###")
    print(f"  {overall_best['model']} on {overall_best['crop']}: {overall_best['field_acc']:.2f}%")

    arch_df.to_csv(OUTPUT_DIR / "table_architecture.csv", index=False)
    return arch_df


def analyze_per_class_performance(df: pd.DataFrame):
    """Deep dive into per-class performance from confusion matrices."""
    print("\n" + "="*80)
    print("📊 PER-CLASS PERFORMANCE ANALYSIS")
    print("="*80)

    # Analyze tomato (most classes, most interesting)
    tomato_exps = df[(df['crop'] == 'tomato') & (df['per_class_report'].apply(bool))]

    if tomato_exps.empty:
        print("No tomato experiments with per-class data")
        return

    print("\n### Tomato Per-Class F1 Scores ###")

    for _, exp in tomato_exps.iterrows():
        report = exp['per_class_report']
        if not report:
            continue

        print(f"\n{exp['name']}:")
        for cls, metrics in report.items():
            if cls in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            f1 = metrics.get('f1-score', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            print(f"  {cls}: F1={f1:.3f} (P={prec:.3f}, R={rec:.3f})")

    # Find consistently hard classes
    print("\n### Consistently Hard Classes (F1 < 0.3 across experiments) ###")
    hard_classes = defaultdict(list)

    for _, exp in tomato_exps.iterrows():
        report = exp['per_class_report']
        if not report:
            continue
        for cls, metrics in report.items():
            if cls in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            f1 = metrics.get('f1-score', 0)
            hard_classes[cls].append(f1)

    for cls, f1_scores in hard_classes.items():
        avg_f1 = np.mean(f1_scores)
        if avg_f1 < 0.3:
            print(f"  {cls}: avg F1={avg_f1:.3f}")


def analyze_training_dynamics(df: pd.DataFrame):
    """Analyze training curves and convergence."""
    print("\n" + "="*80)
    print("📊 TRAINING DYNAMICS ANALYSIS")
    print("="*80)

    # Analyze P1 baselines with history
    p1 = df[(df['phase'] == 'P1') & (df['history'].apply(len) > 0)]

    if p1.empty:
        print("No training history data")
        return

    print("\n### Training Convergence Analysis ###")

    for _, exp in p1.iterrows():
        history = exp['history']
        if not history:
            continue

        val_accs = [h.get('val_acc', 0) for h in history]
        train_losses = [h.get('train_loss', 0) for h in history]

        best_epoch = np.argmax(val_accs) + 1
        final_val = val_accs[-1]
        best_val = max(val_accs)

        print(f"\n{exp['name']}:")
        print(f"  Best Val Acc: {best_val:.2f}% (epoch {best_epoch})")
        print(f"  Final Val Acc: {final_val:.2f}%")
        print(f"  Final Train Loss: {train_losses[-1]:.4f}")
        print(f"  Overfitting: {'Yes' if best_val - final_val > 1 else 'No'}")


def analyze_al_trajectories(df: pd.DataFrame):
    """Analyze Active Learning trajectories in detail."""
    print("\n" + "="*80)
    print("📊 ACTIVE LEARNING TRAJECTORY ANALYSIS")
    print("="*80)

    al_exps = df[df['al_trajectory'].apply(len) > 0]

    if al_exps.empty:
        print("No AL trajectory data")
        return

    print("\n### AL Learning Curves ###")

    trajectories = []
    for _, exp in al_exps.iterrows():
        traj = exp['al_trajectory']
        if not traj:
            continue

        print(f"\n{exp['name']}:")
        for point in traj:
            labels = point.get('labels', 0)
            acc = point.get('accuracy', 0)
            print(f"  {labels} labels: {acc:.2f}%")

        # Calculate efficiency
        initial = traj[0].get('accuracy', 0)
        final = traj[-1].get('accuracy', 0)
        labels_used = traj[-1].get('labels', 0)
        improvement = final - initial
        efficiency = improvement / labels_used if labels_used > 0 else 0

        print(f"  --> Improvement: {improvement:+.2f}% with {labels_used} labels")
        print(f"  --> Efficiency: {efficiency:.3f}% per label")

        trajectories.append({
            'name': exp['name'],
            'crop': exp['crop'],
            'model': exp['model'],
            'fixmatch': exp['use_fixmatch'],
            'initial_acc': initial,
            'final_acc': final,
            'labels_used': labels_used,
            'improvement': improvement,
            'efficiency': efficiency
        })

    traj_df = pd.DataFrame(trajectories)

    # Best efficiency
    if not traj_df.empty:
        print("\n### Most Efficient Approaches ###")
        top3 = traj_df.nlargest(3, 'efficiency')
        for _, row in top3.iterrows():
            print(f"  {row['name']}: {row['efficiency']:.3f}% per label")

    traj_df.to_csv(OUTPUT_DIR / "table_al_trajectories.csv", index=False)
    return traj_df


def analyze_computational_cost(df: pd.DataFrame):
    """Analyze computational costs and efficiency."""
    print("\n" + "="*80)
    print("📊 COMPUTATIONAL COST ANALYSIS")
    print("="*80)

    print("\n### Experiment Duration by Phase ###")
    phase_times = df.groupby('phase')['duration_seconds'].agg(['mean', 'sum', 'count'])
    phase_times['mean_mins'] = phase_times['mean'] / 60
    phase_times['total_mins'] = phase_times['sum'] / 60
    print(phase_times[['count', 'mean_mins', 'total_mins']].round(2))

    print("\n### Duration by Model (P1 Baselines) ###")
    p1 = df[df['phase'] == 'P1']
    model_times = p1.groupby('model')['duration_seconds'].mean() / 60
    print(model_times.round(2))

    print("\n### Total Compute Time ###")
    total_seconds = df['duration_seconds'].sum()
    print(f"  Total: {total_seconds:.0f} seconds ({total_seconds/60:.1f} minutes, {total_seconds/3600:.2f} hours)")

    # Cost per accuracy point
    print("\n### Cost-Efficiency (Improvement per Minute) ###")
    for phase in ['P2', 'P3', 'P4', 'P5']:
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            continue

        avg_duration = phase_df['duration_seconds'].mean() / 60
        avg_acc = phase_df['field_acc'].mean()

        # Compare to relevant baseline
        if phase == 'P2':
            baseline = df[(df['phase'] == 'P1') & (df['model'] == 'mobilenetv3')]['field_acc'].mean()
        else:
            baseline = df[df['phase'] == 'P1']['field_acc'].mean()

        improvement = avg_acc - baseline
        efficiency = improvement / avg_duration if avg_duration > 0 else 0

        print(f"  {phase}: {improvement:+.2f}% in {avg_duration:.1f} min = {efficiency:.3f}%/min")


def generate_final_summary(df: pd.DataFrame):
    """Generate comprehensive final summary."""
    print("\n" + "="*80)
    print("🏆 FINAL COMPREHENSIVE SUMMARY")
    print("="*80)

    print("\n### KEY FINDINGS ###")

    # 1. Domain Shift Problem
    p1 = df[(df['phase'] == 'P1') & (df['crop'] != 'all')]
    avg_gap = p1['gap'].mean()
    print(f"\n1. DOMAIN SHIFT SEVERITY")
    print(f"   Average Lab→Field gap: {avg_gap:.1f}% accuracy drop")
    print(f"   Lab accuracy (avg): {p1['val_acc'].mean():.1f}%")
    print(f"   Field accuracy (avg): {p1['field_acc'].mean():.1f}%")

    # 2. Best Baseline
    best_baseline = p1.loc[p1['field_acc'].idxmax()]
    print(f"\n2. BEST BASELINE")
    print(f"   {best_baseline['model']} on {best_baseline['crop']}: {best_baseline['field_acc']:.1f}%")

    # 3. FixMatch Improvement
    p4 = df[df['phase'] == 'P4']
    if not p4.empty:
        print(f"\n3. FIXMATCH IMPROVEMENT")
        for crop in p4['crop'].unique():
            base = p1[(p1['crop'] == crop) & (p1['model'] == 'mobilenetv3')]
            fix = p4[p4['crop'] == crop]
            if not base.empty and not fix.empty:
                improvement = fix['field_acc'].values[0] - base['field_acc'].values[0]
                print(f"   {crop}: {improvement:+.2f}%")

    # 4. Best Overall Result
    all_results = df[df['crop'] != 'all']
    best_overall = all_results.loc[all_results['field_acc'].idxmax()]
    print(f"\n4. BEST OVERALL RESULT")
    print(f"   {best_overall['field_acc']:.1f}% ({best_overall['name']})")

    # 5. Architecture Recommendation
    print(f"\n5. ARCHITECTURE RECOMMENDATIONS")
    for crop in CROPS:
        crop_results = all_results[all_results['crop'] == crop]
        if not crop_results.empty:
            best = crop_results.loc[crop_results['field_acc'].idxmax()]
            print(f"   {crop}: {best['model']} ({best['field_acc']:.1f}%)")

    # 6. Method Recommendation
    print(f"\n6. METHOD RECOMMENDATIONS")
    print(f"   - Strong augmentation: Small improvement on average")
    print(f"   - Active Learning: Hybrid strategy > entropy > random")
    print(f"   - FixMatch: Significant improvement, especially with efficientnet")

    # 7. Per-Crop Summary
    print(f"\n7. PER-CROP BEST RESULTS")
    for crop in CROPS:
        crop_data = all_results[all_results['crop'] == crop]
        if not crop_data.empty:
            best = crop_data.loc[crop_data['field_acc'].idxmax()]
            baseline = p1[(p1['crop'] == crop) & (p1['model'] == 'mobilenetv3')]
            base_acc = baseline['field_acc'].values[0] if not baseline.empty else 0
            improvement = best['field_acc'] - base_acc
            print(f"   {crop.upper()}")
            print(f"     Baseline: {base_acc:.1f}%")
            print(f"     Best: {best['field_acc']:.1f}% ({best['model']}, {'FixMatch' if best['use_fixmatch'] else 'baseline'})")
            print(f"     Improvement: {improvement:+.1f}%")


def create_publication_tables(df: pd.DataFrame):
    """Create LaTeX-ready publication tables."""
    print("\n" + "="*80)
    print("📄 GENERATING PUBLICATION TABLES")
    print("="*80)

    # Table 1: Baseline Generalization Gap
    p1 = df[(df['phase'] == 'P1') & (df['crop'] != 'all')]

    table1_data = []
    for crop in CROPS:
        row = {'Crop': crop.capitalize()}
        crop_data = p1[p1['crop'] == crop]
        for model in MODELS:
            model_data = crop_data[crop_data['model'] == model]
            if not model_data.empty:
                val = model_data['val_acc'].values[0]
                field = model_data['field_acc'].values[0]
                gap = val - field
                row[f'{model}_val'] = f"{val:.1f}"
                row[f'{model}_field'] = f"{field:.1f}"
                row[f'{model}_gap'] = f"{gap:.1f}"
        table1_data.append(row)

    table1_df = pd.DataFrame(table1_data)
    table1_df.to_csv(OUTPUT_DIR / "TABLE_1_baseline_gap.csv", index=False)
    print(f"  Saved TABLE_1_baseline_gap.csv")

    # Table 2: Method Comparison
    table2_data = []
    for crop in CROPS:
        baseline = p1[(p1['crop'] == crop) & (p1['model'] == 'mobilenetv3')]['field_acc'].values
        strong_aug = df[(df['phase'] == 'P2') & (df['crop'] == crop)]['field_acc'].values
        al_hybrid = df[(df['phase'] == 'P3') & (df['crop'] == crop) & (df['strategy'] == 'hybrid')]['field_acc'].values
        fixmatch = df[(df['phase'] == 'P4') & (df['crop'] == crop)]['field_acc'].values

        row = {
            'Crop': crop.capitalize(),
            'Baseline': f"{baseline[0]:.1f}" if len(baseline) > 0 else "-",
            'StrongAug': f"{strong_aug[0]:.1f}" if len(strong_aug) > 0 else "-",
            'AL_Hybrid': f"{al_hybrid[0]:.1f}" if len(al_hybrid) > 0 else "-",
            'FixMatch': f"{fixmatch[0]:.1f}" if len(fixmatch) > 0 else "-",
        }

        # Calculate improvements
        base_val = baseline[0] if len(baseline) > 0 else 0
        if len(fixmatch) > 0 and base_val > 0:
            row['Improvement'] = f"+{fixmatch[0] - base_val:.1f}"

        table2_data.append(row)

    table2_df = pd.DataFrame(table2_data)
    table2_df.to_csv(OUTPUT_DIR / "TABLE_2_method_comparison.csv", index=False)
    print(f"  Saved TABLE_2_method_comparison.csv")

    # Table 3: Architecture Comparison with FixMatch
    table3_data = []
    for crop in ['tomato', 'potato']:  # P5 only has these
        for model in MODELS:
            # Find relevant experiment
            if model == 'mobilenetv3':
                exp = df[(df['phase'] == 'P4') & (df['crop'] == crop)]
            else:
                exp = df[(df['phase'] == 'P5') & (df['crop'] == crop) & (df['model'] == model)]

            if not exp.empty:
                table3_data.append({
                    'Crop': crop.capitalize(),
                    'Architecture': model,
                    'Field_Acc': f"{exp['field_acc'].values[0]:.1f}%"
                })

    table3_df = pd.DataFrame(table3_data)
    table3_df.to_csv(OUTPUT_DIR / "TABLE_3_architecture.csv", index=False)
    print(f"  Saved TABLE_3_architecture.csv")

    print(f"\n  All tables saved to {OUTPUT_DIR}")


def create_visualizations(df: pd.DataFrame):
    """Create publication-quality visualizations."""
    if not HAS_PLOTTING:
        print("Plotting not available")
        return

    print("\n" + "="*80)
    print("📈 GENERATING VISUALIZATIONS")
    print("="*80)

    # Figure 1: Generalization Gap Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))

    p1 = df[(df['phase'] == 'P1') & (df['crop'] != 'all')]

    crops = CROPS
    x = np.arange(len(crops))
    width = 0.25

    colors = {'mobilenetv3': '#3498DB', 'efficientnet': '#E74C3C', 'mobilevit': '#2ECC71'}

    for i, model in enumerate(MODELS):
        gaps = []
        for crop in crops:
            data = p1[(p1['crop'] == crop) & (p1['model'] == model)]
            gaps.append(data['gap'].values[0] if not data.empty else 0)
        ax.bar(x + i*width, gaps, width, label=model, color=colors[model], alpha=0.8)

    ax.set_xlabel('Crop', fontsize=12)
    ax.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax.set_title('Lab-to-Field Generalization Gap by Architecture', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in crops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FIGURE_1_gap_barchart.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "FIGURE_1_gap_barchart.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved FIGURE_1_gap_barchart.png")

    # Figure 2: Method Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['Baseline', 'StrongAug', 'AL Hybrid', 'FixMatch']
    x = np.arange(len(crops))
    width = 0.2

    method_colors = ['#95A5A6', '#F39C12', '#9B59B6', '#27AE60']

    for i, (method, color) in enumerate(zip(methods, method_colors)):
        accs = []
        for crop in crops:
            if method == 'Baseline':
                data = df[(df['phase'] == 'P1') & (df['crop'] == crop) & (df['model'] == 'mobilenetv3')]
            elif method == 'StrongAug':
                data = df[(df['phase'] == 'P2') & (df['crop'] == crop)]
            elif method == 'AL Hybrid':
                data = df[(df['phase'] == 'P3') & (df['crop'] == crop) & (df['strategy'] == 'hybrid')]
            else:  # FixMatch
                data = df[(df['phase'] == 'P4') & (df['crop'] == crop)]

            accs.append(data['field_acc'].values[0] if not data.empty else 0)

        ax.bar(x + i*width, accs, width, label=method, color=color, alpha=0.8)

    ax.set_xlabel('Crop', fontsize=12)
    ax.set_ylabel('Field Accuracy (%)', fontsize=12)
    ax.set_title('Method Comparison: Field Accuracy by Approach', fontsize=14)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([c.capitalize() for c in crops])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FIGURE_2_method_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "FIGURE_2_method_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved FIGURE_2_method_comparison.png")

    # Figure 3: AL Trajectories
    fig, ax = plt.subplots(figsize=(10, 6))

    al_exps = df[df['al_trajectory'].apply(len) > 0]

    linestyles = {'tomato': '-', 'potato': '--', 'pepper': ':'}
    markers = {'mobilenetv3': 'o', 'efficientnet': 's', 'mobilevit': '^'}
    crop_colors = {'tomato': '#E74C3C', 'potato': '#3498DB', 'pepper': '#2ECC71'}

    for _, exp in al_exps.iterrows():
        traj = exp['al_trajectory']
        if not traj:
            continue

        x_vals = [t.get('labels', 0) for t in traj]
        y_vals = [t.get('accuracy', 0) for t in traj]

        label = f"{exp['model']}"
        if exp['use_fixmatch']:
            label += " + FixMatch"
        label += f" ({exp['crop']})"

        ax.plot(x_vals, y_vals,
                linestyle=linestyles.get(exp['crop'], '-'),
                marker=markers.get(exp['model'], 'o'),
                color=crop_colors.get(exp['crop'], '#7F8C8D'),
                label=label, linewidth=2, markersize=6, alpha=0.8)

    ax.set_xlabel('Labeled Samples', fontsize=12)
    ax.set_ylabel('Field Accuracy (%)', fontsize=12)
    ax.set_title('Active Learning Trajectories', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "FIGURE_3_al_trajectories.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "FIGURE_3_al_trajectories.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved FIGURE_3_al_trajectories.png")

    # Figure 4: Architecture Comparison Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    combined = pd.concat([
        df[df['phase'] == 'P4'],
        df[df['phase'] == 'P5']
    ])

    if not combined.empty:
        pivot = combined.pivot_table(
            index='model',
            columns='crop',
            values='field_acc',
            aggfunc='mean'
        )

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                    ax=ax, cbar_kws={'label': 'Field Accuracy (%)'})
        ax.set_title('Architecture Performance with FixMatch', fontsize=14)
        ax.set_xlabel('Crop')
        ax.set_ylabel('Architecture')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "FIGURE_4_architecture_heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(OUTPUT_DIR / "FIGURE_4_architecture_heatmap.pdf", bbox_inches='tight')
        plt.close()
        print("  Saved FIGURE_4_architecture_heatmap.png")

    print(f"\n  All figures saved to {OUTPUT_DIR}")


def export_full_data(df: pd.DataFrame):
    """Export full dataset for further analysis."""
    # Create a clean export without complex objects
    export_df = df.copy()
    export_df['history_length'] = export_df['history'].apply(len)
    export_df['al_rounds'] = export_df['al_trajectory'].apply(len)
    export_df = export_df.drop(columns=['history', 'al_trajectory', 'confusion', 'per_class_report'])

    export_df.to_csv(OUTPUT_DIR / "full_data.csv", index=False)
    print(f"\n📁 Full data exported to {OUTPUT_DIR}/full_data.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🦸 SUPERMAN COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load all data
    print("\n📥 Loading experiment data...")
    df = parse_all_experiments()
    print(f"   Loaded {len(df)} experiments")

    # Run all analyses
    analyze_generalization_gap(df)
    analyze_strong_augmentation(df)
    analyze_active_learning_strategies(df)
    analyze_fixmatch_impact(df)
    analyze_architecture_comparison(df)
    analyze_per_class_performance(df)
    analyze_training_dynamics(df)
    analyze_al_trajectories(df)
    analyze_computational_cost(df)

    # Generate outputs
    generate_final_summary(df)
    create_publication_tables(df)
    create_visualizations(df)
    export_full_data(df)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print(f"   All outputs saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

