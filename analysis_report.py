#!/usr/bin/env python3
"""
Data Harvester - Experiment Analysis & Report Generator

Automatically crawls experiment results and generates:
- Table I: Generalization Gap across architectures
- Table II: Strong Augmentation comparison
- Table III: AL Strategy comparison
- Table IV: FixMatch results
- Table V: Architecture benchmark
- Learning curve plots
- Confusion matrix heatmaps

Usage:
    python analysis_report.py
    python analysis_report.py --results-dir results/experiments
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plotting disabled.")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_RESULTS_DIR = Path("results/experiments")
DEFAULT_OUTPUT_DIR = Path("results/analysis")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_experiments(results_dir: Path) -> pd.DataFrame:
    """
    Crawl results directory and load all experiment data.
    """
    data = []

    if not results_dir.exists():
        print(f"[ERROR] {results_dir} not found.")
        return pd.DataFrame()

    print(f"[SCAN] Scanning {results_dir}...")

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        metrics_path = exp_dir / "metrics.json"
        config_path = exp_dir / "config.json"

        if not metrics_path.exists():
            print(f"  [WARN] Skipping {exp_dir.name}: missing metrics.json")
            continue
        if not config_path.exists():
            print(f"  [WARN] Skipping {exp_dir.name}: missing config.json")
            continue

        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [WARN] Skipping {exp_dir.name}: JSON parse error - {e}")
            continue

        # Extract experiment ID and phase
        exp_name = config.get('exp_name', exp_dir.name)
        phase = extract_phase(exp_name)

        # Get accuracies
        final_acc = metrics.get('final_accuracy', 0.0)

        # Get best validation accuracy from training history
        history = metrics.get('history', [])
        best_val = max([h.get('val_acc', 0) for h in history], default=0.0)

        # Get AL trajectory info
        trajectory = metrics.get('al_trajectory', [])
        final_labels = trajectory[-1].get('labels', 0) if trajectory else 0

        entry = {
            'exp_dir': exp_dir.name,
            'exp_name': exp_name,
            'phase': phase,
            'crop': config.get('crop', 'unknown'),
            'model': config.get('model', 'unknown'),
            'mode': config.get('mode', 'unknown'),
            'strategy': config.get('strategy', 'none'),
            'use_fixmatch': config.get('use_fixmatch', False),
            'strong_aug': config.get('strong_aug', False),
            'epochs': config.get('epochs', 0),
            'lr': config.get('lr', 0),
            'seed': config.get('seed', 42),
            'lab_acc': best_val,
            'field_acc': final_acc,
            'gap': best_val - final_acc,
            'final_labels': final_labels,
            'trajectory': trajectory,
            'confusion': metrics.get('final_confusion', []),
            'report': metrics.get('final_report', {}),
        }
        data.append(entry)
        print(f"  [OK] Loaded: {exp_name}")

    df = pd.DataFrame(data)
    print(f"\n[DONE] Loaded {len(df)} experiments total.")
    return df


def extract_phase(exp_name: str) -> str:
    """Extract phase from experiment name (P1, P2, etc.)."""
    if exp_name.startswith('P1'):
        return 'P1'
    elif exp_name.startswith('P2'):
        return 'P2'
    elif exp_name.startswith('P3'):
        return 'P3'
    elif exp_name.startswith('P4'):
        return 'P4'
    elif exp_name.startswith('P5'):
        return 'P5'
    else:
        return 'unknown'


# =============================================================================
# TABLE GENERATORS
# =============================================================================

def generate_table_1_baselines(df: pd.DataFrame, output_dir: Path):
    """Generate Table I: Generalization Gap (Phase 1 Baselines)."""
    print("\n" + "="*60)
    print("TABLE I: GENERALIZATION GAP (Phase 1 Baselines)")
    print("="*60)

    p1 = df[(df['phase'] == 'P1') & (df['strong_aug'] == False)]

    if p1.empty:
        print("No Phase 1 baseline data found.")
        return None

    summary = p1[['crop', 'model', 'lab_acc', 'field_acc', 'gap']].copy()
    summary = summary.sort_values(['crop', 'model'])

    print("\n" + summary.to_string(index=False, float_format='%.2f'))

    pivot = p1.pivot_table(
        index='crop',
        columns='model',
        values=['lab_acc', 'field_acc', 'gap'],
        aggfunc='mean'
    ).round(2)

    print("\n--- Pivot Table ---")
    print(pivot)

    summary.to_csv(output_dir / "Table_I_Baselines.csv", index=False)
    pivot.to_csv(output_dir / "Table_I_Baselines_Pivot.csv")
    print(f"\n[SAVED] {output_dir}/Table_I_Baselines.csv")

    return summary


def generate_table_2_strong_aug(df: pd.DataFrame, output_dir: Path):
    """Generate Table II: Strong Augmentation Comparison (Phase 2)."""
    print("\n" + "="*60)
    print("TABLE II: STRONG AUGMENTATION COMPARISON (Phase 2)")
    print("="*60)

    p1 = df[(df['phase'] == 'P1') & (df['model'] == 'mobilenetv3')]
    p2 = df[df['phase'] == 'P2']

    if p1.empty or p2.empty:
        print("Insufficient data for strong aug comparison.")
        return None

    base = p1[['crop', 'field_acc']].rename(columns={'field_acc': 'baseline_acc'})
    strong = p2[['crop', 'field_acc']].rename(columns={'field_acc': 'strong_aug_acc'})

    merged = pd.merge(base, strong, on='crop')
    merged['delta'] = merged['strong_aug_acc'] - merged['baseline_acc']
    merged['helps'] = merged['delta'] > 0

    print("\n" + merged.to_string(index=False, float_format='%.2f'))

    merged.to_csv(output_dir / "Table_II_StrongAug.csv", index=False)
    print(f"\n[SAVED] {output_dir}/Table_II_StrongAug.csv")

    return merged


def generate_table_3_al_strategies(df: pd.DataFrame, output_dir: Path):
    """Generate Table III: Active Learning Strategy Comparison (Phase 3)."""
    print("\n" + "="*60)
    print("TABLE III: AL STRATEGY COMPARISON (Phase 3)")
    print("="*60)

    p3 = df[df['phase'] == 'P3']

    if p3.empty:
        print("No Phase 3 data found.")
        return None

    summary = p3[['crop', 'strategy', 'field_acc', 'final_labels']].copy()
    summary = summary.sort_values(['crop', 'field_acc'], ascending=[True, False])

    print("\n" + summary.to_string(index=False, float_format='%.2f'))

    summary.to_csv(output_dir / "Table_III_AL_Strategies.csv", index=False)
    print(f"\n[SAVED] {output_dir}/Table_III_AL_Strategies.csv")

    return summary


def generate_table_4_fixmatch(df: pd.DataFrame, output_dir: Path):
    """Generate Table IV: FixMatch Results (Phase 4)."""
    print("\n" + "="*60)
    print("TABLE IV: FIXMATCH RESULTS (Phase 4)")
    print("="*60)

    p3 = df[(df['phase'] == 'P3') & (df['strategy'] == 'hybrid')]
    p4 = df[df['phase'] == 'P4']

    if p3.empty or p4.empty:
        print("Insufficient data for FixMatch comparison.")
        return None

    al_only = p3[['crop', 'field_acc']].rename(columns={'field_acc': 'AL_only'})
    fixmatch = p4[['crop', 'field_acc']].rename(columns={'field_acc': 'AL_FixMatch'})

    merged = pd.merge(al_only, fixmatch, on='crop', how='outer')
    merged['improvement'] = merged['AL_FixMatch'] - merged['AL_only']

    print("\n" + merged.to_string(index=False, float_format='%.2f'))

    merged.to_csv(output_dir / "Table_IV_FixMatch.csv", index=False)
    print(f"\n[SAVED] {output_dir}/Table_IV_FixMatch.csv")

    return merged


def generate_table_5_architecture(df: pd.DataFrame, output_dir: Path):
    """Generate Table V: Architecture Benchmark (Phase 5)."""
    print("\n" + "="*60)
    print("TABLE V: ARCHITECTURE BENCHMARK (Phase 5)")
    print("="*60)

    p5 = df[df['phase'] == 'P5']

    if p5.empty:
        print("No Phase 5 data found.")
        return None

    p4_mobile = df[(df['phase'] == 'P4') & (df['model'] == 'mobilenetv3')]

    combined = pd.concat([p4_mobile, p5])
    summary = combined[['crop', 'model', 'field_acc']].copy()
    summary = summary.sort_values(['crop', 'field_acc'], ascending=[True, False])

    print("\n" + summary.to_string(index=False, float_format='%.2f'))

    pivot = summary.pivot_table(index='model', columns='crop', values='field_acc').round(2)
    print("\n--- Pivot ---")
    print(pivot)

    summary.to_csv(output_dir / "Table_V_Architecture.csv", index=False)
    pivot.to_csv(output_dir / "Table_V_Architecture_Pivot.csv")
    print(f"\n[SAVED] {output_dir}/Table_V_Architecture.csv")

    return summary


# =============================================================================
# PLOT GENERATORS
# =============================================================================

def plot_learning_curves(df: pd.DataFrame, output_dir: Path):
    """Plot Active Learning trajectories."""
    if not HAS_PLOTTING:
        print("Plotting not available (matplotlib not installed)")
        return

    print("\n" + "="*60)
    print("PLOTTING: Learning Curves")
    print("="*60)

    al_df = df[df['mode'] == 'active']
    al_df = al_df[al_df['trajectory'].apply(len) > 0]

    if al_df.empty:
        print("No AL experiments with trajectory data found.")
        return

    plt.figure(figsize=(12, 7))

    colors = {
        'tomato': '#E74C3C',
        'potato': '#3498DB',
        'pepper': '#2ECC71',
    }

    for _, row in al_df.iterrows():
        traj = row['trajectory']
        if not traj:
            continue

        x = [t.get('labels', 0) for t in traj]
        y = [t.get('accuracy', 0) for t in traj]

        label = f"{row['model']}"
        if row['use_fixmatch']:
            label += " + FixMatch"
        label += f" ({row['crop']})"

        linestyle = '-' if row['use_fixmatch'] else '--'
        marker = 'o' if 'mobilenet' in row['model'] else ('s' if 'efficient' in row['model'] else '^')
        color = colors.get(row['crop'], '#7F8C8D')

        plt.plot(x, y, label=label, linestyle=linestyle, marker=marker,
                color=color, markersize=6, linewidth=2)

    plt.xlabel("Labeled Samples", fontsize=12)
    plt.ylabel("Field Accuracy (%)", fontsize=12)
    plt.title("Active Learning Trajectories", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / "Figure_Learning_Curves.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "Figure_Learning_Curves.pdf", bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {output_dir}/Figure_Learning_Curves.png")


def plot_generalization_gap(df: pd.DataFrame, output_dir: Path):
    """Plot generalization gap bar chart."""
    if not HAS_PLOTTING:
        return

    print("\n" + "="*60)
    print("PLOTTING: Generalization Gap")
    print("="*60)

    p1 = df[(df['phase'] == 'P1') & (df['strong_aug'] == False)]

    if p1.empty:
        print("No Phase 1 data for gap plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    crops = p1['crop'].unique()
    models = p1['model'].unique()
    x = np.arange(len(crops))
    width = 0.25

    for i, model in enumerate(models):
        model_data = p1[p1['model'] == model]
        gaps = [model_data[model_data['crop'] == c]['gap'].values[0]
                if len(model_data[model_data['crop'] == c]) > 0 else 0
                for c in crops]
        ax.bar(x + i*width, gaps, width, label=model)

    ax.set_xlabel('Crop')
    ax.set_ylabel('Generalization Gap (%)')
    ax.set_title('Lab-to-Field Generalization Gap by Architecture')
    ax.set_xticks(x + width)
    ax.set_xticklabels(crops)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "Figure_Gap_BarChart.png", dpi=300)
    plt.close()

    print(f"[SAVED] {output_dir}/Figure_Gap_BarChart.png")


def plot_confusion_matrix(df: pd.DataFrame, output_dir: Path, exp_name: str = None):
    """Plot confusion matrix for a specific experiment."""
    if not HAS_PLOTTING:
        return

    if exp_name:
        exp = df[df['exp_name'] == exp_name]
    else:
        exp = df[(df['phase'] == 'P4') & (df['crop'] == 'potato')]

    if exp.empty:
        print("No experiment found for confusion matrix.")
        return

    row = exp.iloc[0]
    cm = row['confusion']

    if not cm:
        print("No confusion matrix data.")
        return

    print(f"\nPlotting confusion matrix for: {row['exp_name']}")

    cm_array = np.array(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"Confusion Matrix: {row['exp_name']}")
    plt.tight_layout()

    filename = f"Figure_ConfusionMatrix_{row['crop']}_{row['model']}.png"
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

    print(f"[SAVED] {output_dir}/{filename}")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate a comprehensive summary report."""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)

    report = []
    report.append("# Experiment Summary Report\n")
    report.append(f"Total experiments: {len(df)}\n")

    report.append("\n## Experiments by Phase\n")
    phase_counts = df['phase'].value_counts().sort_index()
    for phase, count in phase_counts.items():
        report.append(f"- {phase}: {count} experiments\n")

    report.append("\n## Experiments by Crop\n")
    crop_counts = df['crop'].value_counts()
    for crop, count in crop_counts.items():
        report.append(f"- {crop}: {count} experiments\n")

    report.append("\n## Best Field Accuracy by Crop\n")
    for crop in df['crop'].unique():
        crop_df = df[df['crop'] == crop]
        best = crop_df.loc[crop_df['field_acc'].idxmax()]
        report.append(f"- {crop}: {best['field_acc']:.2f}% ({best['exp_name']})\n")

    report.append("\n## Key Findings\n")

    if not df[df['phase'] == 'P1'].empty:
        p1 = df[df['phase'] == 'P1']
        max_gap = p1.loc[p1['gap'].idxmax()]
        report.append(f"- Largest generalization gap: {max_gap['gap']:.2f}% ({max_gap['crop']}, {max_gap['model']})\n")

    p3_hybrid = df[(df['phase'] == 'P3') & (df['strategy'] == 'hybrid') & (df['crop'] == 'potato')]
    p4_potato = df[(df['phase'] == 'P4') & (df['crop'] == 'potato')]
    if not p3_hybrid.empty and not p4_potato.empty:
        improvement = p4_potato['field_acc'].values[0] - p3_hybrid['field_acc'].values[0]
        report.append(f"- FixMatch improvement on Potato: {improvement:+.2f}%\n")

    report_text = ''.join(report)
    print(report_text)

    with open(output_dir / "SUMMARY_REPORT.md", 'w') as f:
        f.write(report_text)

    print(f"[SAVED] {output_dir}/SUMMARY_REPORT.md")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Analysis & Report Generator")
    parser.add_argument('--results-dir', type=str, default=str(DEFAULT_RESULTS_DIR),
                        help='Path to experiments directory')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Path for output files')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("EXPERIMENT DATA HARVESTER")
    print("="*60)

    df = load_all_experiments(results_dir)

    if df.empty:
        print("\n[ERROR] No experiments found. Run some experiments first!")
        return

    generate_table_1_baselines(df, output_dir)
    generate_table_2_strong_aug(df, output_dir)
    generate_table_3_al_strategies(df, output_dir)
    generate_table_4_fixmatch(df, output_dir)
    generate_table_5_architecture(df, output_dir)

    if not args.no_plots and HAS_PLOTTING:
        plot_learning_curves(df, output_dir)
        plot_generalization_gap(df, output_dir)
        plot_confusion_matrix(df, output_dir)

    generate_summary_report(df, output_dir)

    df_export = df.drop(columns=['trajectory', 'confusion', 'report'])
    df_export.to_csv(output_dir / "full_experiment_data.csv", index=False)
    print(f"\n[SAVED] Full data exported to {output_dir}/full_experiment_data.csv")

    print("\n" + "="*60)
    print("[DONE] ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

