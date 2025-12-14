"""PlantVillage dataset processing."""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, safe_rmtree
from .label_utils import normalize_label


def _find_class_folders_recursive(root: Path, max_depth: int = 3) -> List[Path]:
    """
    Recursively find all class folders (folders containing images).

    A class folder is defined as a folder that:
    1. Contains image files directly
    2. Has a name that looks like a class label (contains underscore)
    """
    class_folders = []

    def scan(path: Path, depth: int):
        if depth > max_depth:
            return

        if not path.is_dir():
            return

        # Check if this folder contains images
        images = [f for f in path.iterdir() if f.is_file() and is_valid_image(f)]

        if images and '_' in path.name:
            # This is a class folder
            class_folders.append(path)
        else:
            # Recurse into subfolders
            for subfolder in path.iterdir():
                if subfolder.is_dir() and not subfolder.name.startswith('.'):
                    scan(subfolder, depth + 1)

    scan(root, 0)
    return class_folders


def process_plantvillage(source_dir: Path, output_dir: Path) -> list[dict[str, str]]:
    """Process PlantVillage dataset into normalized folder + CSV."""
    source_dir = Path(source_dir)
    extracted_root = source_dir
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("PROCESSING PLANTVILLAGE")
    print("=" * 60)

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return []

    # Recursively find all class folders regardless of nesting structure
    print(f"Scanning {source_dir} for class folders...")
    all_class_folders = _find_class_folders_recursive(source_dir)

    if not all_class_folders:
        print(f"ERROR: No class folders found in {source_dir}")
        return []

    print(f"Found {len(all_class_folders)} class folders")

    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data: list[dict[str, str]] = []
    stats = defaultdict(int)
    label_counters = defaultdict(int)

    # Group by normalized label to merge same classes from different sources
    class_folders_by_label = defaultdict(list)
    for folder in all_class_folders:
        label, _, _ = normalize_label(folder.name)
        class_folders_by_label[label].append(folder)

    print(f"Grouped into {len(class_folders_by_label)} unique classes")
    print("-" * 60)

    for label in sorted(class_folders_by_label.keys()):
        folders = class_folders_by_label[label]
        crop = label.split('_')[0] if '_' in label else label
        disease = '_'.join(label.split('_')[1:]) if '_' in label else 'unknown'

        # Collect all images from all folders for this label
        all_images = []
        for folder in folders:
            images = [f for f in folder.glob("*") if is_valid_image(f)]
            all_images.extend(images)

        if not all_images:
            print(f"  SKIP (no images): {label}")
            continue

        # Sort and deduplicate (by file content hash would be ideal, but use name for speed)
        all_images = sorted(set(all_images), key=lambda f: f.name)

        class_out = output_dir / label
        class_out.mkdir(exist_ok=True)

        start_idx = label_counters[label]
        for idx, src in enumerate(all_images, start=start_idx + 1):
            ext = ".jpg" if src.suffix.lower() in {".jpg", ".jpeg"} else ".png"
            new_name = f"image-{idx:05d}{ext}"
            dst = class_out / new_name
            copy_file(src, dst)
            csv_data.append({
                "filename": new_name,
                "label": label,
                "crop": crop,
                "disease": disease,
                "original_folder": folders[0].name,  # Use first folder name
                "path": f"PlantVillage_processed/{label}/{new_name}"
            })
            label_counters[label] += 1
        stats[label] += len(all_images)
        src_info = f" (from {len(folders)} folders)" if len(folders) > 1 else ""
        print(f"  {label:40} : {len(all_images):5} images{src_info}")

    _write_metadata(output_dir, csv_data, stats)
    print("-" * 60)
    print(f"TOTAL: {len(csv_data)} images, {len(stats)} classes")
    print(f"CSV saved: {output_dir / 'labels.csv'}")

    safe_rmtree(extracted_root)
    return csv_data


def _iter_class_folders(source_dir: Path) -> Iterable[Path]:
    return [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]


def _write_metadata(output_dir: Path, csv_data, stats):
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "filename", "label", "crop", "disease", "original_folder", "path"
        ])
        writer.writeheader()
        writer.writerows(csv_data)

    metadata = {
        "dataset": "PlantVillage",
        "processed_date": datetime.now().isoformat(),
        "total_images": len(csv_data),
        "num_classes": len(stats),
        "classes": dict(stats)
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        import json
        json.dump(metadata, fh, indent=2)
