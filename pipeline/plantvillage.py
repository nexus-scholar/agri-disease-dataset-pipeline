"""PlantVillage dataset processing."""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, safe_rmtree
from .label_utils import normalize_label


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

    nested = source_dir / "PlantVillage"
    if nested.exists() and nested.is_dir():
        nested_dirs = [d for d in nested.iterdir() if d.is_dir()]
        if nested_dirs and any(d.name not in {"train", "test"} for d in nested_dirs):
            print(f"Found nested PlantVillage folder, using: {nested}")
            source_dir = nested

    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data: list[dict[str, str]] = []
    stats = defaultdict(int)
    label_counters = defaultdict(int)

    class_folders = sorted(_iter_class_folders(source_dir))
    print(f"\nFound {len(class_folders)} class folders")
    print("-" * 60)

    for class_folder in class_folders:
        original_name = class_folder.name
        label, crop, disease = normalize_label(original_name)
        images = sorted(f for f in class_folder.glob("*") if is_valid_image(f))
        if not images:
            print(f"  SKIP (no images): {original_name}")
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
                "path": f"PlantVillage_processed/{label}/{new_name}"
            })
            label_counters[label] += 1
        stats[label] += len(images)
        print(f"  {label:40} : {len(images):5} images")

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
