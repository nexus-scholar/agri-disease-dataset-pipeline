#!/usr/bin/env python3
"""
Experiment 10: Operational Cost Analysis

Measures practical deployment feasibility by analyzing computational costs
and inference times.

Measurements:
- Model inference time at different batch sizes
- Entropy calculation overhead
- Memory usage
- FPS estimation

Usage:
    python 10_operational_cost.py
    python 10_operational_cost.py --batch-sizes 1,8,16,32
"""

import argparse
import sys
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    TrainingConfig, RESULTS_DIR, PLANTVILLAGE_DIR, TOMATO_CLASSES, ensure_dir
)
from src.utils import (
    get_transforms, CanonicalImageFolder,
    print_header, print_section, Colors
)
from src.utils.device import get_device, set_seed
from src.models import create_model, get_num_parameters

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

DEFAULT_BATCH_SIZES = [1, 8, 16, 32]
NUM_WARMUP = 10
NUM_TIMING = 50


def get_memory_usage():
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    return None


def get_gpu_memory():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return None


def profile_inference(model, batch_size, device, image_size=224):
    """Profile model inference time."""
    model.eval()
    input_tensor = torch.randn(batch_size, 3, image_size, image_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Time inference
    times = []
    with torch.no_grad():
        for _ in range(NUM_TIMING):
            start = time.perf_counter()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    mean_time = np.mean(times)
    return {
        'batch_size': batch_size,
        'mean_ms': mean_time * 1000,
        'std_ms': np.std(times) * 1000,
        'fps': batch_size / mean_time,
        'latency_per_image_ms': mean_time * 1000 / batch_size
    }


def profile_entropy(model, loader, device, num_batches=20):
    """Profile entropy calculation time."""
    model.eval()

    times = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= num_batches:
                break

            inputs = inputs.to(device)

            start = time.perf_counter()
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

    mean_time = np.mean(times)
    batch_size = inputs.size(0) if len(times) > 0 else 1

    return {
        'total_time_ms': mean_time * 1000,
        'time_per_sample_ms': mean_time * 1000 / batch_size
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-10: Operational Cost")
    parser.add_argument('--batch-sizes', type=str, default=None,
                        help='Comma-separated batch sizes')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print_header("OPERATIONAL COST ANALYSIS", 10)

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')] if args.batch_sizes else DEFAULT_BATCH_SIZES

    device = get_device()
    set_seed(args.seed)

    # Load model
    print_section("Model Info")
    num_classes = len(TOMATO_CLASSES)
    model = create_model(num_classes)
    model = model.to(device)
    model.eval()

    total_params = get_num_parameters(model)
    trainable_params = get_num_parameters(model, trainable_only=True)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (approx): {total_params * 4 / 1024 / 1024:.2f} MB")

    # Memory usage
    print_section("Memory Usage")
    cpu_mem = get_memory_usage()
    gpu_mem = get_gpu_memory()

    if cpu_mem:
        print(f"  CPU Memory: {cpu_mem:.1f} MB")
    if gpu_mem:
        print(f"  GPU Memory: {gpu_mem:.1f} MB")

    # Inference profiling
    print_section("Inference Profiling")
    print(f"\n{'Batch':>8} | {'Mean (ms)':>10} | {'FPS':>10} | {'Per Image':>12}")
    print("-" * 50)

    inference_results = []
    for batch_size in batch_sizes:
        result = profile_inference(model, batch_size, device)
        inference_results.append(result)
        print(f"{batch_size:>8} | {result['mean_ms']:>10.2f} | {result['fps']:>10.1f} | {result['latency_per_image_ms']:>10.2f} ms")

    # Entropy profiling
    print_section("Entropy Calculation Overhead")

    transforms_dict = get_transforms()
    dataset = CanonicalImageFolder(str(PLANTVILLAGE_DIR), TOMATO_CLASSES, transforms_dict['val'])
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    entropy_result = profile_entropy(model, loader, device)
    print(f"  Time per batch: {entropy_result['total_time_ms']:.2f} ms")
    print(f"  Time per sample: {entropy_result['time_per_sample_ms']:.3f} ms")

    # Save results
    print_section("Summary")

    ensure_dir(RESULTS_DIR / "tables")
    csv_path = RESULTS_DIR / "tables" / "exp10_operational_cost.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=inference_results[0].keys())
        writer.writeheader()
        writer.writerows(inference_results)

    print(f"  Results saved to: {csv_path}")

    # Key metrics summary
    best_throughput = max(r['fps'] for r in inference_results)
    best_latency = min(r['latency_per_image_ms'] for r in inference_results)

    print(f"\n{Colors.BOLD}Key Metrics:{Colors.RESET}")
    print(f"  Best throughput: {best_throughput:.1f} FPS")
    print(f"  Best latency: {best_latency:.2f} ms/image")
    print(f"  Entropy overhead: {entropy_result['time_per_sample_ms']:.3f} ms/sample")

    print(f"\n{Colors.GREEN}Experiment 10 complete{Colors.RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

