#!/usr/bin/env python3
"""
Google Colab Experiment Runner for PDA Research

This script is designed to run the full experiment suite on Google Colab Pro.
It packages all experiments and manages GPU resources efficiently.

Usage on Colab:
    1. Upload your dataset to Google Drive
    2. Mount Drive and set DATA_ROOT
    3. Run this script

Local usage:
    python colab_experiment_runner.py --dry-run  # Preview experiments
    python colab_experiment_runner.py --phase 1   # Run Phase 1 only
    python colab_experiment_runner.py --all       # Run all phases
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

# Detect environment
IN_COLAB = 'google.colab' in sys.modules if 'google' in sys.modules else False


@dataclass
class ExperimentRun:
    """Single experiment configuration."""
    id: str
    name: str
    phase: int
    command: List[str]
    description: str = ""
    expected_time_minutes: int = 10
    priority: int = 1  # Lower = higher priority
    

@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    id: str
    name: str
    success: bool
    start_time: str
    end_time: str
    duration_seconds: float
    output: str = ""
    error: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# EXPERIMENT DEFINITIONS - FINAL PROTOCOL
# =============================================================================

def get_phase1_experiments() -> List[ExperimentRun]:
    """
    Phase 1: Baseline Generalization Gap (12 experiments)

    Scientific Goal: Prove that models trained on Lab data fail on Field data,
    regardless of architecture (CNN vs Transformer) or size.
    """
    experiments = []
    
    # Single crop baselines (3 crops × 3 models = 9)
    crops = ["tomato", "potato", "pepper"]
    models = ["mobilenetv3", "efficientnet", "mobilevit"]
    
    idx = 0
    for crop in crops:
        for model in models:
            idx += 1
            exp_name = f"P1_{idx:02d}_baseline_{crop}_{model}"
            # Larger batch size for A100 (128 for light models, 64 for ViT)
            batch_size = 64 if model == "mobilevit" else 128
            experiments.append(ExperimentRun(
                id=f"P1-{idx:02d}",
                name=f"Baseline {crop.title()} {model}",
                phase=1,
                command=[
                    "python", "run_experiment.py",
                    "--mode", "baseline",
                    "--model", model,
                    "--crop", crop,
                    "--baseline-path", f"data/models/baselines/{crop}_{model}_base.pth",
                    "--epochs", "10",
                    "--lr", "0.001",
                    "--batch-size", str(batch_size),
                    "--exp-name", exp_name,
                    "--no-confusion",
                ],
                description=f"Baseline: {crop} with {model} - measure generalization gap",
                expected_time_minutes=5 if model == "mobilevit" else 3,  # Much faster with A100
                priority=1,
            ))
    
    # All crops combined (1 combined × 3 models = 3)
    for model in models:
        idx += 1
        exp_name = f"P1_{idx:02d}_baseline_all_{model}"
        experiments.append(ExperimentRun(
            id=f"P1-{idx:02d}",
            name=f"Baseline All Crops {model}",
            phase=1,
            command=[
                "python", "run_experiment.py",
                "--mode", "baseline",
                "--model", model,
                "--crop", "tomato,potato,pepper",
                "--baseline-path", f"data/models/baselines/all_{model}_base.pth",
                "--epochs", "10",
                "--lr", "0.001",
                "--exp-name", exp_name,
                "--no-confusion",
            ],
            description=f"Baseline: All crops combined with {model}",
            expected_time_minutes=20 if model == "mobilevit" else 15,
            priority=2,
        ))
    
    return experiments


def get_phase2_experiments() -> List[ExperimentRun]:
    """
    Phase 2: Passive Interventions - Strong Augmentation (3 experiments)

    Scientific Goal: Prove that simply adding Strong Augmentation (AutoAugment)
    during source training does NOT close the generalization gap.
    This justifies the need for Active Learning.
    """
    experiments = []
    crops = ["tomato", "potato", "pepper"]

    for idx, crop in enumerate(crops, 1):
        exp_name = f"P2_{idx:02d}_strongaug_{crop}"
        experiments.append(ExperimentRun(
            id=f"P2-{idx:02d}",
            name=f"StrongAug {crop.title()}",
            phase=2,
            command=[
                "python", "run_experiment.py",
                "--mode", "baseline",
                "--model", "mobilenetv3",
                "--crop", crop,
                "--strong-aug",  # The key flag for Phase 2
                "--baseline-path", f"data/models/baselines/{crop}_strong_base.pth",
                "--epochs", "10",
                "--lr", "0.001",
                "--batch-size", "128",
                "--exp-name", exp_name,
                "--no-confusion",
            ],
            description=f"Passive: Strong Augmentation on {crop} - test if aug closes gap",
            expected_time_minutes=4,  # Faster with A100
            priority=1,
        ))

    return experiments


def get_phase3_experiments() -> List[ExperimentRun]:
    """
    Phase 3: Active Learning Strategy Ablation (4 experiments)

    Scientific Goal: Justify the Hybrid (70/30) strategy by showing:
    1. Random is inefficient (slow slope)
    2. Entropy suffers "Cold Start" (dips in early rounds)
    3. Hybrid stabilizes start and exploits decision boundary
    """
    experiments = []
    
    # Tomato Strategies (3 experiments)
    strategies = ["random", "entropy", "hybrid"]
    for idx, strategy in enumerate(strategies, 1):
        exp_name = f"P3_{idx:02d}_AL_{strategy}_tomato"
        experiments.append(ExperimentRun(
            id=f"P3-{idx:02d}",
            name=f"AL {strategy.title()} (Tomato)",
            phase=3,
            command=[
                "python", "run_experiment.py",
                "--mode", "active",
                "--model", "mobilenetv3",
                "--crop", "tomato",
                "--strategy", strategy,
                "--baseline-path", "data/models/baselines/tomato_mobilenetv3_base.pth",
                "--budget", "10",
                "--rounds", "5",
                "--epochs", "5",
                "--lr", "0.0001",
                "--batch-size", "64",
                "--exp-name", exp_name,
                "--no-confusion",
            ],
            description=f"Ablation: {strategy} strategy on Tomato",
            expected_time_minutes=10,
            priority=1,
        ))
    
    # Potato Hybrid Confirmation (1 experiment)
    experiments.append(ExperimentRun(
        id="P3-04",
        name="AL Hybrid (Potato)",
        phase=3,
        command=[
            "python", "run_experiment.py",
            "--mode", "active",
            "--model", "mobilenetv3",
            "--crop", "potato",
            "--strategy", "hybrid",
            "--baseline-path", "data/models/baselines/potato_mobilenetv3_base.pth",
            "--budget", "10",
            "--rounds", "5",
            "--epochs", "5",
            "--lr", "0.0001",
            "--batch-size", "64",
            "--exp-name", "P3_04_AL_hybrid_potato",
            "--no-confusion",
        ],
        description="Verify Hybrid strategy works on Potato PDA case",
        expected_time_minutes=8,
        priority=1,
    ))

    return experiments


def get_phase4_experiments() -> List[ExperimentRun]:
    """
    Phase 4: The Main Contribution - FixMatch & PDA (3 experiments)

    Scientific Goal: Prove that adding FixMatch (SSL) to the Active Learning loop
    eliminates Negative Transfer in PDA scenarios.

    Tomato is the PRIMARY testbed because:
    - 746 target samples (enough for meaningful AL)
    - 9 canonical classes (complex task)
    - PDA: source has 'target_spot' (2,808 samples) missing in target

    Potato/Pepper are secondary validation with smaller target pools.
    """
    experiments = []
    # Order by importance: Tomato first (primary), then Potato, then Pepper
    crops = ["tomato", "potato", "pepper"]

    for idx, crop in enumerate(crops, 1):
        exp_name = f"P4_{idx:02d}_fixmatch_{crop}"

        # Adjust budget based on target pool size
        # Tomato: 746 samples -> budget 10 per round is fine
        # Potato: 222 samples -> budget 10 per round
        # Pepper: 132 samples -> budget 8 per round (smaller pool)
        budget = 8 if crop == "pepper" else 10

        experiments.append(ExperimentRun(
            id=f"P4-{idx:02d}",
            name=f"FixMatch ({crop.title()})",
            phase=4,
            command=[
                "python", "run_experiment.py",
                "--mode", "active",
                "--model", "mobilenetv3",
                "--crop", crop,
                "--strategy", "hybrid",
                "--use-fixmatch",  # The key flag
                "--baseline-path", f"data/models/baselines/{crop}_mobilenetv3_base.pth",
                "--budget", str(budget),
                "--rounds", "5",
                "--epochs", "15",
                "--lr", "0.001",
                "--batch-size", "64",
                "--exp-name", exp_name,
                "--no-confusion",
            ],
            description=f"{'PRIMARY: ' if crop == 'tomato' else ''}Hybrid + FixMatch on {crop} (PDA)",
            expected_time_minutes=15,  # Faster with A100
            priority=1,
        ))

    return experiments


def get_phase5_experiments() -> List[ExperimentRun]:
    """
    Phase 5: Architecture Benchmark - Efficiency vs Robustness (4 experiments)

    Scientific Goal: Apply winning method (Hybrid + FixMatch) to EfficientNet and MobileViT.
    Focus on TOMATO as primary testbed (largest target pool for reliable AL).
    Expected: EfficientNet may suffer from Negative Transfer, MobileViT succeeds but slow.
    """
    experiments = []

    # Focus on Tomato as primary benchmark crop (746 target samples)
    # (model, crop, lr, batch_size)
    configs = [
        ("efficientnet", "tomato", 0.001, 96),    # EfficientNet can handle larger batches
        ("mobilevit", "tomato", 0.0005, 48),      # ViT needs smaller batches (memory)
        ("efficientnet", "potato", 0.001, 96),   # Secondary validation
        ("mobilevit", "potato", 0.0005, 48),      # Secondary validation
    ]

    for idx, (model, crop, lr, batch_size) in enumerate(configs, 1):
        exp_name = f"P5_{idx:02d}_{model}_{crop}_fixmatch"
        budget = 10 if crop == "tomato" else 8  # Smaller budget for smaller pools
        experiments.append(ExperimentRun(
            id=f"P5-{idx:02d}",
            name=f"Benchmark {model} ({crop.title()})",
            phase=5,
            command=[
                "python", "run_experiment.py",
                "--mode", "active",
                "--model", model,
                "--crop", crop,
                "--strategy", "hybrid",
                "--use-fixmatch",
                "--baseline-path", f"data/models/baselines/{crop}_{model}_base.pth",
                "--budget", str(budget),
                "--rounds", "5",
                "--epochs", "15",
                "--lr", str(lr),
                "--batch-size", str(batch_size),
                "--exp-name", exp_name,
                "--no-confusion",
            ],
            description=f"{'PRIMARY: ' if crop == 'tomato' else ''}SOTA {model} on {crop} + FixMatch",
            expected_time_minutes=20 if model == "mobilevit" else 15,
            priority=1,
        ))

    return experiments


def get_all_experiments() -> List[ExperimentRun]:
    """Get all experiments across all phases (26 total)."""
    return (
        get_phase1_experiments() +   # 12 experiments
        get_phase2_experiments() +   # 3 experiments
        get_phase3_experiments() +   # 4 experiments
        get_phase4_experiments() +   # 3 experiments
        get_phase5_experiments()     # 4 experiments
    )


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def create_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Create a text-based progress bar."""
    if total == 0:
        return "[" + "=" * width + "]"
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    percent = 100 * current / total
    return f"[{bar}] {percent:.0f}%"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


class ExperimentRunner:
    """Manages experiment execution and logging."""
    
    def __init__(self, output_dir: Path = Path("results/experiments")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_epoch = 0
        self._total_epochs = 0
        self._current_round = 0
        self._total_rounds = 0
        self._start_time = None

    def run_experiment(self, exp: ExperimentRun, dry_run: bool = False) -> ExperimentResult:
        """Run a single experiment with real-time progress display."""
        print(f"\n{'='*60}")
        print(f"[{exp.id}] {exp.name}")
        print(f"{'='*60}")
        print(f"Description: {exp.description}")
        print(f"Command: {' '.join(exp.command)}")
        print(f"Expected time: {exp.expected_time_minutes} minutes")
        
        start_time = datetime.now()
        
        if dry_run:
            print("[DRY RUN] Skipping actual execution")
            return ExperimentResult(
                id=exp.id,
                name=exp.name,
                success=True,
                start_time=start_time.isoformat(),
                end_time=start_time.isoformat(),
                duration_seconds=0,
                output="[DRY RUN]",
            )
        
        try:
            print(f"\nStarting at {start_time.strftime('%H:%M:%S')}...")
            print("-" * 60)

            # Set start time for progress tracking
            self._start_time = start_time

            # Set up environment for unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                exp.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                env=env,  # Use unbuffered environment
            )

            output_lines = []
            current_epoch = 0
            current_round = 0

            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break

                line = line.rstrip()
                output_lines.append(line)

                # Parse and display progress information
                line_lower = line.lower()

                # Detect epoch progress
                if 'epoch' in line_lower:
                    self._print_progress(line, 'EPOCH')
                # Detect AL round progress
                elif 'round' in line_lower:
                    self._print_progress(line, 'ROUND')
                # Detect accuracy updates
                elif 'acc' in line_lower or 'accuracy' in line_lower:
                    self._print_progress(line, 'METRIC')
                # Detect loss updates
                elif 'loss' in line_lower:
                    self._print_progress(line, 'LOSS')
                # Detect data loading info
                elif '[data]' in line_lower:
                    self._print_progress(line, 'DATA')
                # Detect model info
                elif '[model]' in line_lower:
                    self._print_progress(line, 'MODEL')
                # Show any warnings or errors
                elif 'warning' in line_lower or 'error' in line_lower:
                    self._print_progress(line, 'WARN')
                # Show phase/stage markers
                elif '===' in line or '---' in line:
                    print(line)

            process.wait()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            success = process.returncode == 0
            full_output = '\n'.join(output_lines)

            # Parse metrics from output
            metrics = self._parse_metrics(full_output)

            exp_result = ExperimentResult(
                id=exp.id,
                name=exp.name,
                success=success,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                output=full_output,
                error="" if success else "Process failed",
                metrics=metrics,
            )
            
            print("-" * 60)
            status = "[OK] SUCCESS" if success else "[FAIL] FAILED"
            print(f"{status} in {duration:.1f}s ({duration/60:.1f} min)")

            if metrics:
                print("Final Metrics:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
            
            return exp_result
            
        except subprocess.TimeoutExpired:
            process.kill()
            end_time = datetime.now()
            print("[FAIL] TIMEOUT")
            return ExperimentResult(
                id=exp.id,
                name=exp.name,
                success=False,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=(end_time - start_time).total_seconds(),
                error="TIMEOUT",
            )
        except Exception as e:
            end_time = datetime.now()
            print(f"[FAIL] ERROR: {e}")
            return ExperimentResult(
                id=exp.id,
                name=exp.name,
                success=False,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=(end_time - start_time).total_seconds(),
                error=str(e),
            )
    
    def _print_progress(self, line: str, prefix: str, start_time: datetime = None):
        """Print a progress line with formatting and elapsed time."""
        import re

        # Color codes for terminal (works in Colab too)
        colors = {
            'EPOCH': '\033[94m',  # Blue
            'ROUND': '\033[95m',  # Magenta
            'METRIC': '\033[92m', # Green
            'LOSS': '\033[93m',   # Yellow
            'DATA': '\033[96m',   # Cyan
            'MODEL': '\033[96m',  # Cyan
            'WARN': '\033[91m',   # Red
        }
        reset = '\033[0m'
        bold = '\033[1m'

        color = colors.get(prefix, '')

        # Calculate elapsed time if available
        elapsed_str = ""
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            elapsed_str = f" [{format_duration(elapsed)}]"

        # Format the line based on type
        if prefix == 'EPOCH':
            # Extract epoch info and show progress bar if possible
            match = re.search(r'epoch\s*(\d+)(?:/(\d+))?', line.lower())
            if match:
                current = int(match.group(1))
                total = int(match.group(2)) if match.group(2) else 0
                if total > 0:
                    bar = create_progress_bar(current, total, width=20)
                    print(f"{color}{bold}[{prefix}]{reset}{elapsed_str} Epoch {current}/{total} {bar}")
                else:
                    print(f"{color}{bold}[{prefix}]{reset}{elapsed_str} {line}")
            else:
                print(f"{color}[{prefix}]{reset}{elapsed_str} {line}")
        elif prefix == 'ROUND':
            # Extract round info for AL
            match = re.search(r'round\s*(\d+)(?:/(\d+))?', line.lower())
            if match:
                current = int(match.group(1))
                total = int(match.group(2)) if match.group(2) else 5
                bar = create_progress_bar(current, total, width=15)
                print(f"{color}{bold}[{prefix}]{reset}{elapsed_str} Round {current}/{total} {bar}")
            else:
                print(f"{color}{bold}[{prefix}]{reset}{elapsed_str} {line}")
        elif prefix == 'METRIC':
            # Highlight accuracy values
            match = re.search(r'(\d+\.?\d*)\s*%', line)
            if match:
                acc = float(match.group(1))
                # Color based on accuracy level
                if acc >= 80:
                    acc_color = '\033[92m'  # Green
                elif acc >= 50:
                    acc_color = '\033[93m'  # Yellow
                else:
                    acc_color = '\033[91m'  # Red
                print(f"{color}[{prefix}]{reset}{elapsed_str} {line.replace(match.group(0), f'{acc_color}{bold}{match.group(0)}{reset}')}")
            else:
                print(f"{color}[{prefix}]{reset}{elapsed_str} {line}")
        elif prefix == 'LOSS':
            # Only show loss lines (they're important)
            print(f"{color}[{prefix}]{reset}{elapsed_str} {line}")
        elif prefix == 'WARN':
            print(f"{color}{bold}[{prefix}]{reset} {line}")
        else:
            print(f"{color}[{prefix}]{reset}{elapsed_str} {line}")

    def _parse_metrics(self, output: str) -> Dict[str, Any]:
        """Extract metrics from experiment output."""
        metrics = {}
        
        # Look for accuracy patterns
        for line in output.split('\n'):
            line_lower = line.lower()
            if 'field accuracy' in line_lower or 'final accuracy' in line_lower:
                try:
                    # Extract percentage
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*%', line)
                    if match:
                        metrics['field_accuracy'] = float(match.group(1))
                except:
                    pass
            elif 'val acc' in line_lower or 'val:' in line_lower:
                try:
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*%', line)
                    if match:
                        metrics['val_accuracy'] = float(match.group(1))
                except:
                    pass
        
        return metrics
    
    def run_phase(self, phase: int, dry_run: bool = False) -> List[ExperimentResult]:
        """Run all experiments in a phase."""
        phase_funcs = {
            1: get_phase1_experiments,
            2: get_phase2_experiments,
            3: get_phase3_experiments,
            4: get_phase4_experiments,
            5: get_phase5_experiments,
        }
        
        if phase not in phase_funcs:
            raise ValueError(f"Invalid phase: {phase}. Valid: 1-5")

        experiments = phase_funcs[phase]()
        print(f"\n{'#'*60}")
        print(f"# PHASE {phase}: {len(experiments)} experiments")
        print(f"{'#'*60}")
        
        results = []
        for exp in experiments:
            result = self.run_experiment(exp, dry_run)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_all(self, dry_run: bool = False, phases: Optional[List[int]] = None):
        """Run all experiments."""
        phases = phases or [1, 2, 3, 4, 5]

        total_time = sum(
            exp.expected_time_minutes 
            for exp in get_all_experiments() 
            if exp.phase in phases
        )
        
        print(f"\n{'#'*60}")
        print(f"# PDA EXPERIMENT SUITE")
        print(f"# Run ID: {self.run_id}")
        print(f"# Phases: {phases}")
        print(f"# Estimated total time: {total_time} minutes ({total_time/60:.1f} hours)")
        print(f"{'#'*60}")
        
        for phase in phases:
            self.run_phase(phase, dry_run)
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to JSON."""
        results_file = self.output_dir / f"results_{self.run_id}.json"
        
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def print_summary(self):
        """Print summary of all results."""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        failed = total - success
        
        print(f"Total: {total}, Success: {success}, Failed: {failed}")
        
        if failed > 0:
            print("\nFailed experiments:")
            for r in self.results:
                if not r.success:
                    print(f"  - [{r.id}] {r.name}: {r.error[:100]}")
        
        # Print metrics table
        print("\nResults by experiment:")
        print("-" * 60)
        for r in self.results:
            status = "✓" if r.success else "✗"
            acc = r.metrics.get('field_accuracy', 'N/A')
            if isinstance(acc, float):
                acc = f"{acc:.2f}%"
            print(f"  {status} [{r.id}] {r.name}: {acc}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="PDA Experiment Runner for Colab")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5],
                        help='Run specific phase only')
    parser.add_argument('--all', action='store_true',
                        help='Run all phases')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview experiments without running')
    parser.add_argument('--list', action='store_true',
                        help='List all experiments')
    parser.add_argument('--generate-notebook', action='store_true',
                        help='Generate Colab notebook')
    parser.add_argument('--output-dir', type=str, default='results/experiments',
                        help='Output directory for results')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list:
        print("\nALL EXPERIMENTS:")
        print("=" * 60)
        for exp in get_all_experiments():
            print(f"  [{exp.id}] {exp.name} ({exp.expected_time_minutes} min)")
        
        total_time = sum(e.expected_time_minutes for e in get_all_experiments())
        print(f"\nTotal estimated time: {total_time} minutes ({total_time/60:.1f} hours)")
        return
    
    runner = ExperimentRunner(Path(args.output_dir))
    
    if args.phase:
        runner.run_phase(args.phase, args.dry_run)
        runner.save_results()
        runner.print_summary()
    elif args.all:
        runner.run_all(args.dry_run)
    else:
        # Default: dry run to show what would happen
        print("No action specified. Use --phase N, --all, or --dry-run")
        print("Use --list to see all experiments")
        print("Use --help for more options")


if __name__ == '__main__':
    main()

