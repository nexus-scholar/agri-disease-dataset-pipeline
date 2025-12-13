#!/usr/bin/env python3
"""Generate dataset summary tables (per dataset/crop/class).

Usage example:
    python scripts/generate_data_report.py \
        --processed-root data/processed/dataset \
        --exclude TomatoLeaf_processed \
        --min-count 100 \
        --output-prefix data/reports/dataset_report_no_tomatoleaf

The script scans each dataset directory under ``processed-root``. When a
``labels.csv`` exists it is parsed; otherwise image files are counted per
class folder. JSON + CSV outputs are emitted with the provided prefix.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.label_utils import canonicalize_label, extract_crop

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SOURCE_DATASETS = {"plantvillage_processed": "source"}
TARGET_DATASETS = {"plantdoc_processed": "target"}


@dataclass
class CropStats:
    counts: list[int]
    domain_counts: dict[str, int]

    def __init__(self):
        self.counts = []
        self.domain_counts = {"source": 0, "target": 0, "other": 0}

    def add_sample(self, count: int, domain: str):
        self.counts.append(count)
        if domain not in self.domain_counts:
            self.domain_counts[domain] = 0
        self.domain_counts[domain] += count

    def summary(self) -> dict:
        if not self.counts:
            mean = variance = 0.0
        else:
            mean = sum(self.counts) / len(self.counts)
            variance = (
                sum((c - mean) ** 2 for c in self.counts) / len(self.counts)
                if len(self.counts) > 1
                else 0.0
            )
        total = sum(self.domain_counts.values())
        source = self.domain_counts.get("source", 0)
        target = self.domain_counts.get("target", 0)
        source_share = source / total if total else 0.0
        target_share = target / total if total else 0.0
        return {
            "mean": mean,
            "variance": variance,
            "source_share": source_share,
            "target_share": target_share,
            "source_count": source,
            "target_count": target,
            "total_count": total,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize processed datasets.')
    parser.add_argument('--processed-root', default=os.path.join('data', 'processed', 'dataset'),
                        help='Root that contains dataset folders (default: data/processed/dataset).')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='Dataset folder names to skip (e.g., TomatoLeaf_processed).')
    parser.add_argument('--min-count', type=int, default=100,
                        help='Threshold for the "candidate_crops" list (default: 100).')
    parser.add_argument('--output-prefix', default=os.path.join('data', 'reports', 'dataset_report'),
                        help='Prefix for the output files (JSON + CSV).')
    return parser.parse_args()


def detect_label_field(fieldnames):
    if not fieldnames:
        return None
    preferred = ['label', 'class', 'class_label', 'label_name', 'category', 'target', 'disease']
    for cand in preferred:
        if cand in fieldnames:
            return cand
    return fieldnames[1] if len(fieldnames) > 1 else fieldnames[0]


def count_from_labels_csv(csv_path):
    classes = defaultdict(int)
    total = 0
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        label_field = detect_label_field(reader.fieldnames)
        if label_field is None:
            return classes, total
        for row in reader:
            label = canonicalize_label((row.get(label_field) or "").strip())
            if not label:
                continue
            classes[label] += 1
            total += 1
    return classes, total


def count_from_folders(dataset_path):
    classes = defaultdict(int)
    total = 0
    for entry in sorted(Path(dataset_path).iterdir()):
        if not entry.is_dir():
            continue
        cnt = 0
        for root, _, files in os.walk(entry):
            for fname in files:
                if Path(fname).suffix.lower() in IMAGE_EXTS:
                    cnt += 1
        if cnt:
            label = canonicalize_label(entry.name)
            classes[label] += cnt
            total += cnt
    return classes, total


def aggregate(processed_root, exclude):
    processed_root = Path(processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root not found: {processed_root}")
    datasets = {}
    total_images = 0
    crops = defaultdict(int)
    crop_stats: dict[str, CropStats] = defaultdict(CropStats)
    for entry in sorted(processed_root.iterdir()):
        if not entry.is_dir() or entry.name in exclude:
            continue
        info = {"labels_csv": False, "total_images": 0, "classes": {}}
        labels_csv = entry / "labels.csv"
        if labels_csv.exists():
            info["labels_csv"] = True
            classes, count = count_from_labels_csv(labels_csv)
        else:
            classes, count = count_from_folders(entry)
        info["classes"] = dict(sorted(classes.items()))
        info["total_images"] = count
        datasets[entry.name] = info
        total_images += count
        for cls, cnt in info["classes"].items():
            if not cls:
                continue
            crop = extract_crop(cls)
            domain = SOURCE_DATASETS.get(entry.name.lower(), TARGET_DATASETS.get(entry.name.lower(), "other"))
            crops[crop] += cnt
            crop_stats[crop].add_sample(cnt, domain)
    return datasets, total_images, dict(sorted(crops.items())), crop_stats


def build_split_recommendations(datasets, crop_stats, min_count):
    recommendations = {}
    for crop, stats in crop_stats.items():
        summary = stats.summary()
        if summary["total_count"] < min_count:
            continue
        # Very rough heuristic: suggested pool is min(target_count, source_count/4)
        pool = min(summary["target_count"], summary["source_count"] // 4)
        recommendations[crop] = {
            "total": summary["total_count"],
            "source_count": summary["source_count"],
            "target_count": summary["target_count"],
            "suggested_pool_size": pool,
            "notes": "Limited target data" if summary["target_count"] < min_count else "OK",
        }
    return recommendations


def write_outputs(prefix, datasets, crops, total_images, min_count, crop_stats):
    out_dir = Path(prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{prefix}.json")
    csv_path = Path(f"{prefix}.csv")
    per_crop_csv = Path(f"{prefix}_crops.csv")
    split_csv = Path(f"{prefix}_splits.csv")
    report = {
        "total_images": total_images,
        "datasets": datasets,
        "crops": crops,
        "candidate_crops": [crop for crop, cnt in crops.items() if cnt >= min_count],
        "per_crop_stats": {crop: stats.summary() for crop, stats in crop_stats.items()},
        "split_recommendations": build_split_recommendations(datasets, crop_stats, min_count),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "class_label", "crop", "count", "dataset_total", "labels_csv"])
        for dataset, info in datasets.items():
            for cls, cnt in info["classes"].items():
                crop = extract_crop(cls)
                writer.writerow([dataset, cls, crop, cnt, info["total_images"], info["labels_csv"]])
    with open(per_crop_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["crop", "mean", "variance", "source_share", "target_share", "source_count", "target_count", "total_count"])
        for crop, summary in report["per_crop_stats"].items():
            writer.writerow([
                crop,
                summary["mean"],
                summary["variance"],
                summary["source_share"],
                summary["target_share"],
                summary["source_count"],
                summary["target_count"],
                summary["total_count"],
            ])
    with open(split_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["crop", "total", "source_count", "target_count", "suggested_pool_size", "notes"])
        for crop, rec in report["split_recommendations"].items():
            writer.writerow([
                crop,
                rec["total"],
                rec["source_count"],
                rec["target_count"],
                rec["suggested_pool_size"],
                rec["notes"],
            ])
    return json_path, csv_path, per_crop_csv, split_csv, report


def main():
    args = parse_args()
    datasets, total_images, crops, crop_stats = aggregate(args.processed_root, set(args.exclude or []))
    paths = write_outputs(args.output_prefix, datasets, crops, total_images, args.min_count, crop_stats)
    json_path, csv_path, per_crop_csv, split_csv, report = paths
    print(f"Wrote JSON report to {json_path}")
    print(f"Wrote class CSV to {csv_path}")
    print(f"Wrote per-crop CSV to {per_crop_csv}")
    print(f"Wrote split recommendations CSV to {split_csv}")
    print("Candidate crops (>= {} images): {}".format(args.min_count, ", ".join(report["candidate_crops"]) or "None"))


if __name__ == '__main__':
    main()

