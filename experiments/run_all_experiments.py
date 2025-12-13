#!/usr/bin/env python3
"""
Run All Experiments

This script runs all experiments in sequence and generates a report.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --experiments 1,2,3
    python run_all_experiments.py --quick  # Reduced settings for testing
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, ensure_dir
from src.utils import print_header, print_section, Colors


EXPERIMENTS = {
    1: {
        'name': 'Baseline Gap',
        'script': '01_baseline_gap.py',
        'description': 'Establish generalization gap',
        'default_args': ['--epochs', '5'],
        'quick_args': ['--epochs', '2']
    },
    2: {
        'name': 'Passive Augmentation',
        'script': '02_passive_aug.py',
        'description': 'Test strong augmentation',
        'default_args': ['--epochs', '5'],
        'quick_args': ['--epochs', '2']
    },
    3: {
        'name': 'CutMix',
        'script': '03_cutmix.py',
        'description': 'Test CutMix regularization',
        'default_args': ['--epochs', '10'],
        'quick_args': ['--epochs', '3']
    },
    4: {
        'name': 'Active Learning',
        'script': '04_active_learning.py',
        'description': 'Compare random vs entropy sampling',
        'default_args': ['--budget-per-round', '50', '--num-rounds', '4'],
        'quick_args': ['--budget-per-round', '20', '--num-rounds', '2']
    },
    5: {
        'name': 'Hybrid Warm-Start',
        'script': '05_hybrid_warmstart.py',
        'description': 'Test hybrid sampling strategy',
        'default_args': ['--budget-per-round', '50', '--num-rounds', '4'],
        'quick_args': ['--budget-per-round', '20', '--num-rounds', '2']
    },
    6: {
        'name': 'Ablation Study',
        'script': '06_ablation_study.py',
        'description': 'Test different entropy/random ratios',
        'default_args': ['--budget', '50', '--rounds', '4'],
        'quick_args': ['--budget', '20', '--rounds', '2']
    },
    7: {
        'name': 'Seed Robustness',
        'script': '07_seed_robustness.py',
        'description': 'Statistical validation with multiple seeds',
        'default_args': ['--budget', '50', '--rounds', '4'],
        'quick_args': ['--budget', '20', '--rounds', '2']
    },
    8: {
        'name': 'Architecture Check',
        'script': '08_architecture_check.py',
        'description': 'Test with different architectures',
        'default_args': ['--epochs', '10'],
        'quick_args': ['--epochs', '3']
    },
    9: {
        'name': 'Multi-Crop',
        'script': '09_multi_crop.py',
        'description': 'Test across different crops',
        'default_args': ['--epochs', '10'],
        'quick_args': ['--epochs', '3']
    },
    10: {
        'name': 'Operational Cost',
        'script': '10_operational_cost.py',
        'description': 'Measure computational costs',
        'default_args': ['--batch-sizes', '1,8,16,32'],
        'quick_args': ['--batch-sizes', '1,16']
    }
}


def run_experiment(exp_num, quick=False):
    """Run a single experiment."""
    exp = EXPERIMENTS[exp_num]
    script_path = Path(__file__).parent / exp['script']

    if not script_path.exists():
        print(f"{Colors.RED}Script not found: {script_path}{Colors.RESET}")
        return False, 0

    args = exp['quick_args'] if quick else exp['default_args']
    cmd = [sys.executable, str(script_path)] + args

    print(f"\n{Colors.CYAN}Running: {' '.join(cmd)}{Colors.RESET}\n")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"{Colors.RED}Experiment {exp_num} failed with code {e.returncode}{Colors.RESET}")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run All Experiments")
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma-separated experiment numbers (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Use reduced settings for quick testing')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip experiments 1-3 (baseline)')
    args = parser.parse_args()

    print_header("RUN ALL EXPERIMENTS")

    # Determine experiments to run
    if args.experiments:
        exp_nums = [int(e.strip()) for e in args.experiments.split(',')]
    elif args.skip_baseline:
        exp_nums = list(range(4, 11))
    else:
        exp_nums = list(range(1, 11))

    print_section("Experiments to Run")
    for num in exp_nums:
        if num in EXPERIMENTS:
            exp = EXPERIMENTS[num]
            print(f"  {num:02d}. {exp['name']}: {exp['description']}")

    if args.quick:
        print(f"\n{Colors.YELLOW}Quick mode: Using reduced settings{Colors.RESET}")

    # Run experiments
    results = {}
    total_start = time.time()

    for exp_num in exp_nums:
        if exp_num not in EXPERIMENTS:
            print(f"{Colors.YELLOW}Unknown experiment: {exp_num}{Colors.RESET}")
            continue

        exp = EXPERIMENTS[exp_num]
        print_section(f"Experiment {exp_num:02d}: {exp['name']}")

        success, elapsed = run_experiment(exp_num, args.quick)
        results[exp_num] = {'success': success, 'time': elapsed, 'name': exp['name']}

    total_time = time.time() - total_start

    # Summary
    print_section("Summary")

    print(f"\n{'Exp':>4} | {'Name':<25} | {'Status':<10} | {'Time':>10}")
    print("-" * 60)

    for exp_num, result in results.items():
        status = f"{Colors.GREEN}OK{Colors.RESET}" if result['success'] else f"{Colors.RED}FAIL{Colors.RESET}"
        time_str = f"{result['time']:.1f}s"
        print(f"{exp_num:>4} | {result['name']:<25} | {status:<10} | {time_str:>10}")

    success_count = sum(1 for r in results.values() if r['success'])
    print(f"\n{success_count}/{len(results)} experiments completed successfully")
    print(f"Total time: {total_time:.1f}s")

    # Save summary
    ensure_dir(RESULTS_DIR / "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / "logs" / f"run_all_{timestamp}.txt"

    with open(summary_path, 'w') as f:
        f.write(f"Run All Experiments - {datetime.now()}\n")
        f.write(f"Quick mode: {args.quick}\n\n")
        for exp_num, result in results.items():
            status = "OK" if result['success'] else "FAIL"
            f.write(f"Exp {exp_num:02d}: {result['name']} - {status} ({result['time']:.1f}s)\n")
        f.write(f"\nTotal: {success_count}/{len(results)} in {total_time:.1f}s\n")

    print(f"\nSummary saved to: {summary_path}")

    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

