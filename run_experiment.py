#!/usr/bin/env python3
"""
Convenience wrapper for the unified experiment runner.

Usage:
    python run_experiment.py --mode baseline --crop tomato --model mobilenetv3
    python run_experiment.py --mode active --crop potato --strategy hybrid --use-fixmatch
"""
import sys
from pathlib import Path

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runner import main

if __name__ == '__main__':
    main()

