#!/usr/bin/env python3
"""
Data Validation Script

Validates that the crop configurations match the actual data on disk.
Run this BEFORE conducting expensive experiments!

Usage:
    python validate_data.py
    python validate_data.py --crop tomato
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.crop_configs import (
    CROP_CONFIGS,
    get_crop_config,
    get_canonical_classes,
    get_source_to_canonical_mapping,
    get_target_to_canonical_mapping,
)

# Paths
PLANTVILLAGE_DIR = Path("data/processed/dataset/PlantVillage_processed")
PLANTDOC_DIR = Path("data/processed/dataset/PlantDoc_processed")


def get_classes_on_disk(dataset_dir: Path, crop_filter: str = None) -> dict:
    """
    Get all class folders on disk with sample counts.

    Args:
        dataset_dir: Path to dataset directory
        crop_filter: Optional crop name to filter (e.g., "tomato")

    Returns:
        Dict mapping class name to sample count
    """
    if not dataset_dir.exists():
        print(f"[ERROR] Directory not found: {dataset_dir}")
        return {}

    classes = {}
    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue

        # Filter by crop if specified
        if crop_filter and not folder.name.lower().startswith(crop_filter.lower()):
            continue

        # Count images
        count = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png"))) + len(list(folder.glob("*.JPG")))
        classes[folder.name] = count

    return classes


def validate_crop(crop_name: str) -> dict:
    """
    Validate a single crop configuration against actual data.

    Returns:
        Dict with validation results
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING: {crop_name.upper()}")
    print(f"{'='*70}")

    config = get_crop_config(crop_name)
    results = {
        'crop': crop_name,
        'issues': [],
        'warnings': [],
        'source_classes': {},
        'target_classes': {},
    }

    # Get actual classes on disk
    source_on_disk = get_classes_on_disk(PLANTVILLAGE_DIR, crop_name)
    target_on_disk = get_classes_on_disk(PLANTDOC_DIR, crop_name)

    results['source_classes'] = source_on_disk
    results['target_classes'] = target_on_disk

    # === SOURCE (PlantVillage) VALIDATION ===
    print(f"\n[SOURCE: PlantVillage]")
    print(f"  Classes on disk ({len(source_on_disk)}):")
    for cls, count in sorted(source_on_disk.items()):
        mapping = config.source_mapping.get(cls, "NOT IN CONFIG")
        status = "-> " + (mapping if mapping else "EXCLUDED")
        print(f"    {cls}: {count} samples {status}")

    # Check for classes on disk not in config
    for cls in source_on_disk:
        if cls not in config.source_mapping:
            results['issues'].append(f"SOURCE: '{cls}' on disk but NOT in config mapping!")

    # Check for classes in config not on disk
    for cls in config.source_mapping:
        if cls not in source_on_disk:
            results['warnings'].append(f"SOURCE: '{cls}' in config but NOT on disk")

    # === TARGET (PlantDoc) VALIDATION ===
    print(f"\n[TARGET: PlantDoc]")
    print(f"  Classes on disk ({len(target_on_disk)}):")
    for cls, count in sorted(target_on_disk.items()):
        mapping = config.target_mapping.get(cls, "NOT IN CONFIG")
        status = "-> " + (mapping if mapping else "EXCLUDED")
        print(f"    {cls}: {count} samples {status}")

    # Check for classes on disk not in config
    for cls in target_on_disk:
        if cls not in config.target_mapping:
            results['issues'].append(f"TARGET: '{cls}' on disk but NOT in config mapping!")

    # Check for classes in config not on disk
    for cls in config.target_mapping:
        if cls not in target_on_disk:
            results['warnings'].append(f"TARGET: '{cls}' in config but NOT on disk")

    # === CANONICAL CLASSES ===
    print(f"\n[CANONICAL CLASSES]")
    print(f"  Defined ({len(config.canonical_classes)}): {config.canonical_classes}")

    # Check source coverage
    source_mapped = [cls for cls, mapped in config.source_mapping.items()
                     if mapped and cls in source_on_disk]
    source_canonical = set(config.source_mapping[cls] for cls in source_mapped
                          if config.source_mapping[cls])

    # Check target coverage
    target_mapped = [cls for cls, mapped in config.target_mapping.items()
                     if mapped and cls in target_on_disk]
    target_canonical = set(config.target_mapping[cls] for cls in target_mapped
                          if config.target_mapping[cls])

    print(f"  Source covers: {sorted(source_canonical)}")
    print(f"  Target covers: {sorted(target_canonical)}")

    # Check for canonical classes not covered
    for canonical in config.canonical_classes:
        if canonical not in source_canonical:
            results['issues'].append(f"CANONICAL: '{canonical}' not available in source!")
        if canonical not in target_canonical and canonical not in config.source_only_classes:
            results['warnings'].append(f"CANONICAL: '{canonical}' not available in target (expected for PDA)")

    # === SAMPLE COUNTS ===
    print(f"\n[SAMPLE COUNTS]")
    total_source = sum(source_on_disk.values())
    total_target = sum(target_on_disk.values())

    # Only count mapped classes
    mapped_source = sum(count for cls, count in source_on_disk.items()
                       if config.source_mapping.get(cls))
    mapped_target = sum(count for cls, count in target_on_disk.items()
                       if config.target_mapping.get(cls))

    print(f"  Source total: {total_source} ({mapped_source} after mapping)")
    print(f"  Target total: {total_target} ({mapped_target} after mapping)")

    # === ISSUES SUMMARY ===
    if results['issues']:
        print(f"\n[CRITICAL ISSUES]")
        for issue in results['issues']:
            print(f"  [ERROR] {issue}")

    if results['warnings']:
        print(f"\n[WARNINGS]")
        for warning in results['warnings']:
            print(f"  [WARN] {warning}")

    if not results['issues'] and not results['warnings']:
        print(f"\n[OK] Configuration matches data on disk")

    return results


def validate_all_crops():
    """Validate all configured crops."""
    print("="*70)
    print("DATA VALIDATION REPORT")
    print(f"PlantVillage: {PLANTVILLAGE_DIR}")
    print(f"PlantDoc: {PLANTDOC_DIR}")
    print("="*70)

    all_results = {}
    all_issues = []
    all_warnings = []

    for crop_name in CROP_CONFIGS:
        results = validate_crop(crop_name)
        all_results[crop_name] = results
        all_issues.extend(results['issues'])
        all_warnings.extend(results['warnings'])

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for crop_name, results in all_results.items():
        source_count = sum(results['source_classes'].values())
        target_count = sum(results['target_classes'].values())
        config = get_crop_config(crop_name)
        status = "[OK]" if not results['issues'] else "[ERR]"
        print(f"  {status} {crop_name}: {config.num_classes} canonical classes | "
              f"Source: {source_count} | Target: {target_count}")

    print(f"\nTotal Issues: {len(all_issues)}")
    print(f"Total Warnings: {len(all_warnings)}")

    if all_issues:
        print("\n[ACTION REQUIRED] Fix these issues before running experiments:")
        for issue in all_issues:
            print(f"  - {issue}")

    return all_results


def generate_recommended_config(crop_name: str):
    """Generate recommended configuration based on actual data."""
    print(f"\n{'='*70}")
    print(f"RECOMMENDED CONFIG FOR: {crop_name.upper()}")
    print(f"{'='*70}")

    source_on_disk = get_classes_on_disk(PLANTVILLAGE_DIR, crop_name)
    target_on_disk = get_classes_on_disk(PLANTDOC_DIR, crop_name)

    print(f"\n# Source classes (PlantVillage):")
    print(f"{crop_name.upper()}_SOURCE_CLASSES = [")
    for cls in sorted(source_on_disk.keys()):
        print(f'    "{cls}",  # {source_on_disk[cls]} samples')
    print("]")

    print(f"\n# Target classes (PlantDoc):")
    print(f"{crop_name.upper()}_TARGET_CLASSES = [")
    for cls in sorted(target_on_disk.keys()):
        print(f'    "{cls}",  # {target_on_disk[cls]} samples')
    print("]")

    # Find intersection (exact matches)
    source_set = set(source_on_disk.keys())
    target_set = set(target_on_disk.keys())
    intersection = source_set & target_set

    print(f"\n# Exact matches (intersection):")
    print(f"# {sorted(intersection)}")

    print(f"\n# Source only:")
    print(f"# {sorted(source_set - target_set)}")

    print(f"\n# Target only:")
    print(f"# {sorted(target_set - source_set)}")


def main():
    parser = argparse.ArgumentParser(description="Validate data configuration")
    parser.add_argument('--crop', type=str, help='Validate specific crop only')
    parser.add_argument('--recommend', type=str, help='Generate recommended config for crop')
    args = parser.parse_args()

    if args.recommend:
        generate_recommended_config(args.recommend)
    elif args.crop:
        validate_crop(args.crop)
    else:
        validate_all_crops()


if __name__ == "__main__":
    main()

