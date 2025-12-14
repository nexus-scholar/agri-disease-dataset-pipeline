"""PlantDoc dataset processing."""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, safe_rmtree
from .label_utils import normalize_label


SPLITS = ["train", "test"]


def _find_split_folder(root: Path, split_name: str, max_depth: int = 3) -> Optional[Path]:
    """Recursively find a train or test folder."""
    if not root.is_dir():
        return None

    # Check if this folder is the split folder
    if root.name.lower() == split_name.lower():
        return root

    # Check direct children first
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == split_name.lower():
            return child

    # Recurse into subfolders
    if max_depth > 0:
        for child in root.iterdir():
            if child.is_dir() and not child.name.startswith('.'):
                result = _find_split_folder(child, split_name, max_depth - 1)
                if result:
                    return result

    return None


def process_plantdoc(source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Process PlantDoc dataset.
    Structure: pd/train/Class_Name/images... and pd/test/Class_Name/images...
    Merges train and test into single dataset.
    """
    print("\n" + "=" * 60)
    print("PROCESSING PLANTDOC (pd) - Merging train/test")
    print("=" * 60)

    source_dir = Path(source_dir)
    extracted_root = source_dir
    output_dir = Path(output_dir)

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return []

    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data: list[dict[str, str]] = []
    stats = defaultdict(int)
    label_counters = defaultdict(int)

    for split in SPLITS:
        # Search recursively for the split folder
        split_dir = _find_split_folder(source_dir, split)
        if not split_dir:
            print(f"  WARNING: {split} folder not found")
            continue
        print(f"\n  Processing {split.upper()} (found at {split_dir.relative_to(source_dir)}):")
        print("  " + "-" * 50)
        class_folders = sorted(_iter_class_folders(split_dir))
        for class_folder in class_folders:
            original_name = class_folder.name
            label, crop, disease = normalize_label(original_name)
            images = sorted(f for f in class_folder.glob("*") if is_valid_image(f))
            if not images:
                continue
            class_out = output_dir / label
            class_out.mkdir(exist_ok=True)
            start_idx = label_counters[label]
            for idx, src in enumerate(images, start=start_idx + 1):
                ext = ".jpg" if src.suffix.lower() in {".jpg", ".jpeg"} else ".png"
                new_name = f"image-{idx:05d}{ext}"
                dst = class_out / new_name
                copy_file(src, dst)
                csv_data.append({
                    "filename": new_name,
                    "label": label,
                    "crop": crop,
                    "disease": disease,
                    "original_folder": original_name,
                    "original_split": split,
                    "path": f"PlantDoc_processed/{label}/{new_name}"
                })
                label_counters[label] += 1
            stats[label] += len(images)
            print(f"    {label:38} : {len(images):4} images ({split})")

    _write_metadata(output_dir, csv_data, stats)
    print("\n  " + "-" * 50)
    print(f"  TOTAL: {len(csv_data)} images, {len(stats)} classes")
    print(f"  CSV saved: {output_dir / 'labels.csv'}")

    safe_rmtree(extracted_root)
    return csv_data


def _iter_class_folders(source_dir: Path) -> Iterable[Path]:
    return [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]


def _write_metadata(output_dir: Path, csv_data, stats):
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "filename", "label", "crop", "disease", "original_folder", "original_split", "path"
        ])
        writer.writeheader()
        writer.writerows(csv_data)

    metadata = {
        "dataset": "PlantDoc",
        "processed_date": datetime.now().isoformat(),
        "total_images": len(csv_data),
        "num_classes": len(stats),
        "note": "train and test merged into single dataset",
        "classes": dict(stats)
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        import json
        json.dump(metadata, fh, indent=2)
