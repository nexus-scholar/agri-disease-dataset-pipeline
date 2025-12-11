"""Tomato Leaf Multiclass dataset processing."""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .dataset_metadata import is_valid_image
from .fs_utils import copy_file, ensure_dir, safe_rmtree

DATASET_ROOT_NAME = "Tomato Leaf Dataset  A dataset for multiclass disease detection and classification"
RAW_FOLDER = "TomatoLeafMulticlass (Raw Data)"
ANNOTATED_FOLDER = "TomatoLeafMulticlass (Annotated)"
YOLO_SPLITS = ["train", "test"]


@dataclass
class TomatoLeafRecord:
    filename: str
    label: str
    bbox: tuple[float, float, float, float] | None
    source: str


def process_tomato_leaf(raw_dir: Path, processed_dir: Path) -> list[dict[str, str]]:
    """Process tomato leaf dataset into YOLO-friendly structure."""
    raw_dir = Path(raw_dir)
    extracted_root = raw_dir
    processed_dir = Path(processed_dir)

    print("\n" + "=" * 60)
    print("PROCESSING TOMATO LEAF DATASET")
    print("=" * 60)

    safe_rmtree(processed_dir)

    dataset_root = raw_dir
    if (raw_dir / DATASET_ROOT_NAME).exists():
        dataset_root = raw_dir / DATASET_ROOT_NAME

    raw_images_root = dataset_root / RAW_FOLDER / "images"
    annotated_root = dataset_root / ANNOTATED_FOLDER

    if not raw_images_root.exists():
        print(f"ERROR: Raw images folder not found: {raw_images_root}")
        return []

    safe_rmtree(processed_dir)
    ensure_dir(processed_dir)

    image_out_dir = processed_dir / "images"
    label_out_dir = processed_dir / "labels"
    ensure_dir(image_out_dir)
    ensure_dir(label_out_dir)

    records: List[dict[str, str]] = []
    stats = Counter()

    # Copy raw images
    print("Copying raw images...")
    for image_path in sorted(raw_images_root.glob("*.jpg")):
        dst = image_out_dir / image_path.name
        copy_file(image_path, dst)
        records.append({
            "filename": image_path.name,
            "label": "unknown",
            "crop": "tomato",
            "disease": "unknown",
            "source": "tomato_leaf_raw",
            "path": f"TomatoLeaf/images/{image_path.name}"
        })
        stats["raw_images"] += 1

    # Copy annotated YOLO structure if present
    for split in YOLO_SPLITS:
        split_images = annotated_root / split / "images"
        split_labels = annotated_root / split / "labels"
        if not split_images.exists():
            continue
        split_out_images = processed_dir / "annotated" / split / "images"
        split_out_labels = processed_dir / "annotated" / split / "labels"
        ensure_dir(split_out_images)
        ensure_dir(split_out_labels)
        print(f"Processing annotated split: {split}")
        for image_path in sorted(split_images.glob("*.jpg")):
            dst_img = split_out_images / image_path.name
            copy_file(image_path, dst_img)
            label_path = split_labels / (image_path.stem + ".txt")
            if label_path.exists():
                copy_file(label_path, split_out_labels / label_path.name)
            else:
                (split_out_labels / (image_path.stem + ".txt")).write_text("", encoding="utf-8")
            stats[f"annotated_{split}"] += 1
            records.append({
                "filename": image_path.name,
                "label": "tomato_leaf",
                "crop": "tomato",
                "disease": "multiclass",
                "source": f"tomato_leaf_{split}",
                "path": f"TomatoLeaf/annotated/{split}/images/{image_path.name}"
            })

    # Save metadata
    metadata = {
        "dataset": "TomatoLeafMulticlass",
        "processed": list(stats.items()),
        "total_records": len(records)
    }
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    safe_rmtree(extracted_root)
    return records
