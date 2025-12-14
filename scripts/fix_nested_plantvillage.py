#!/usr/bin/env python3
"""
Fix nested folder structure in PlantVillage dataset.

The original PlantVillage dataset sometimes has a nested structure:
  PlantVillage_processed/PlantVillage/tomato_bacterial_spot/...

This script detects and flattens it to:
  PlantVillage_processed/tomato_bacterial_spot/...

Usage:
    python scripts/fix_nested_plantvillage.py
    python scripts/fix_nested_plantvillage.py --data-dir /path/to/data
"""
import argparse
import shutil
from pathlib import Path


def detect_nested_structure(dataset_dir: Path) -> tuple[bool, Path | None]:
    """
    Detect if the dataset has a nested folder structure.

    Returns:
        (is_nested, inner_folder_path)
    """
    # Check if there's only one subfolder and it looks like a container
    subfolders = [f for f in dataset_dir.iterdir() if f.is_dir()]

    # If there's exactly one subfolder and it contains class folders
    if len(subfolders) == 1:
        inner = subfolders[0]
        inner_subfolders = [f for f in inner.iterdir() if f.is_dir()]

        # Check if inner folder contains class-like folders (with underscores)
        class_like = [f for f in inner_subfolders if '_' in f.name]

        if len(class_like) > 0:
            print(f"  Detected nested structure: {dataset_dir.name}/{inner.name}/")
            return True, inner

    return False, None


def flatten_nested_structure(dataset_dir: Path, inner_folder: Path, dry_run: bool = False):
    """
    Flatten nested folder structure by moving inner contents up one level.
    """
    print(f"\n  Flattening {inner_folder} -> {dataset_dir}")

    # Get all items in the inner folder
    items = list(inner_folder.iterdir())

    for item in items:
        dest = dataset_dir / item.name

        if dest.exists():
            print(f"    [SKIP] {item.name} already exists in destination")
            continue

        if dry_run:
            print(f"    [DRY RUN] Would move: {item.name}")
        else:
            print(f"    Moving: {item.name}")
            shutil.move(str(item), str(dest))

    # Remove the now-empty inner folder
    if not dry_run:
        try:
            inner_folder.rmdir()
            print(f"  Removed empty folder: {inner_folder.name}")
        except OSError:
            print(f"  [WARN] Could not remove {inner_folder.name} (not empty)")


def fix_dataset(data_dir: Path, dry_run: bool = False):
    """Fix nested structure in PlantVillage dataset."""
    pv_dir = data_dir / "processed" / "dataset" / "PlantVillage_processed"

    if not pv_dir.exists():
        print(f"PlantVillage directory not found: {pv_dir}")
        return False

    print(f"Checking {pv_dir}...")

    is_nested, inner = detect_nested_structure(pv_dir)

    if is_nested:
        flatten_nested_structure(pv_dir, inner, dry_run=dry_run)
        print("\n[OK] Structure fixed!")
        return True
    else:
        print("  Structure is already flat (correct)")
        return False


def count_samples(data_dir: Path):
    """Count samples per class in PlantVillage."""
    pv_dir = data_dir / "processed" / "dataset" / "PlantVillage_processed"

    if not pv_dir.exists():
        print(f"Directory not found: {pv_dir}")
        return

    print(f"\nSample counts in {pv_dir.name}:")
    total = 0

    for folder in sorted(pv_dir.iterdir()):
        if folder.is_dir():
            count = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.JPG"))) + len(list(folder.glob("*.png")))
            print(f"  {folder.name}: {count}")
            total += count

    print(f"\nTotal: {total} samples")


def main():
    parser = argparse.ArgumentParser(description="Fix nested PlantVillage folder structure")
    parser.add_argument('--data-dir', type=Path, default=Path("data"),
                        help='Path to data directory (default: data)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--count-only', action='store_true',
                        help='Only count samples, do not fix')
    args = parser.parse_args()

    if args.count_only:
        count_samples(args.data_dir)
        return

    print("=" * 60)
    print("PlantVillage Nested Folder Fixer")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN MODE - No changes will be made]\n")

    fixed = fix_dataset(args.data_dir, dry_run=args.dry_run)

    if not args.dry_run:
        print("\n" + "=" * 60)
        count_samples(args.data_dir)


if __name__ == "__main__":
    main()

