#!/usr/bin/env python3
"""
Complete Reproducible Dataset Processing Pipeline

Processes PlantVillage and PlantDoc datasets:
- Unzips datasets from data/raw/dataset/*.zip to data/processed/dataset/
- Merges PlantDoc train/test into single dataset
- Renames images to simple sequential format: image-00001.jpg
- Extracts crop and disease from folder names
- Creates CSV with filename, label (full class name), crop, disease
- Organizes output by class label (crop_disease format)
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import importlib

from pipeline.config import load_default_config, parse_args, DatasetSource
from pipeline.logging_utils import configure_logging
from pipeline.zip_utils import unzip_dataset
from pipeline.download_utils import download_datasets


def _import_processor(processor_path: str):
    module_name, func_name = processor_path.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def create_combined_csv(pv_data, pd_data, tl_data, output_path):
    """Create combined CSV with all data."""
    print("\n" + "=" * 60)
    print("CREATING COMBINED DATASET CSV")
    print("=" * 60)

    combined = []

    for row in pv_data:
        combined.append({
            'filename': row['filename'],
            'label': row['label'],
            'crop': row['crop'],
            'disease': row['disease'],
            'source': 'plantvillage',
            'path': row['path']
        })

    for row in pd_data:
        combined.append({
            'filename': row['filename'],
            'label': row['label'],
            'crop': row['crop'],
            'disease': row['disease'],
            'source': 'plantdoc',
            'path': row['path']
        })

    for row in tl_data:
        combined.append({
            'filename': row['filename'],
            'label': row['label'],
            'crop': row['crop'],
            'disease': row['disease'],
            'source': 'tomatoleaf',
            'path': row['path']
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'label', 'crop', 'disease', 'source', 'path'])
        writer.writeheader()
        writer.writerows(combined)

    pv_count = sum(1 for r in combined if r['source'] == 'plantvillage')
    pd_count = sum(1 for r in combined if r['source'] == 'plantdoc')
    tl_count = sum(1 for r in combined if r['source'] == 'tomatoleaf')

    print(f"\nPlantVillage: {pv_count} images")
    print(f"PlantDoc:     {pd_count} images")
    print(f"TomatoLeaf:   {tl_count} images")
    print(f"TOTAL:        {len(combined)} images")
    print(f"\nSaved: {output_path}")

    return combined


def print_summary(pv_data, pd_data, tl_data):
    """Print final summary with class overlap analysis."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    pv_labels = set(r['label'] for r in pv_data)
    pd_labels = set(r['label'] for r in pd_data)
    tl_labels = set(r['label'] for r in tl_data)

    pv_tomato = set(l for l in pv_labels if l.startswith('tomato'))
    pd_tomato = set(l for l in pd_labels if l.startswith('tomato'))
    tl_tomato = set(l for l in tl_labels if l.startswith('tomato'))
    tomato_overlap = pv_tomato & pd_tomato & tl_tomato if tl_data else pv_tomato & pd_tomato

    print(f"\nPlantVillage:")
    print(f"  Total images: {len(pv_data)}")
    print(f"  Total classes: {len(pv_labels)}")
    print(f"  Tomato classes: {len(pv_tomato)}")

    print(f"\nPlantDoc:")
    print(f"  Total images: {len(pd_data)}")
    print(f"  Total classes: {len(pd_labels)}")
    print(f"  Tomato classes: {len(pd_tomato)}")

    if tl_data:
        print(f"\nTomatoLeaf:")
        print(f"  Total images: {len(tl_data)}")
        print(f"  Total classes: {len(tl_labels)}")
        print(f"  Tomato classes: {len(tl_tomato)}")

    print(f"\nTomato Class Overlap (for domain adaptation):")
    print(f"  Common tomato classes: {len(tomato_overlap)}")
    if tomato_overlap:
        for label in sorted(tomato_overlap):
            pv_count = sum(1 for r in pv_data if r['label'] == label)
            pd_count = sum(1 for r in pd_data if r['label'] == label)
            tl_count = sum(1 for r in tl_data if r['label'] == label)
            print(f"    {label}: PV={pv_count}, PD={pd_count}, TL={tl_count}")

    print("\n" + "=" * 60)
    print("DATASET PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nOutput structure:")
    print("  data/processed/dataset/PlantVillage/")
    print("    └── (extracted raw data)")
    print("  data/processed/dataset/PlantVillage_processed/")
    print("    └── {class_label}/image-00001.jpg, image-00002.jpg, ...")
    print("    └── labels.csv")
    print("    └── metadata.json")
    print("  data/processed/dataset/PlantDoc/")
    print("    └── (extracted raw data)")
    print("  data/processed/dataset/PlantDoc_processed/")
    print("    └── {class_label}/image-00001.jpg, image-00002.jpg, ...")
    print("    └── labels.csv")
    print("    └── metadata.json")
    print("  data/processed/dataset/TomatoLeaf/")
    print("    └── (extracted raw data)")
    print("  data/processed/dataset/TomatoLeaf_processed/")
    print("    └── {class_label}/image-00001.jpg, image-00002.jpg, ...")
    print("    └── labels.csv")
    print("    └── metadata.json")
    print("  data/processed/dataset/combined_dataset.csv")
    print("\nTo reproduce: python process_datasets.py")


def main():
    config = parse_args()
    logger = configure_logging(config.paths.log_file, config.log_level)

    print("\n" + "=" * 60)
    print("REPRODUCIBLE DATASET PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Project root: {config.paths.project_root}")
    print(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if config.download:
        print("\n" + "=" * 60)
        print("STEP 0: DOWNLOADING DATASETS")
        print("=" * 60)
        download_datasets(config.datasets, config)
        if config.download_only:
            print("Download-only flag set; exiting after downloads.")
            return

    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTING DATASETS FROM ZIP FILES")
    print("=" * 60)

    processed_data = {}
    for dataset_name in config.datasets:
        source: DatasetSource | None = config.dataset_sources.get(dataset_name)
        if not source:
            print(f"WARNING: Dataset '{dataset_name}' not defined in datasets.json")
            continue
        zip_path = source.zip_path
        raw_dir = source.raw_dir
        print(f"\n--- {dataset_name.upper()} ---")
        extracted = unzip_dataset(zip_path, raw_dir, dataset_name)
        if not extracted:
            print(f"  Skipping {dataset_name} (extraction failed)")
            continue
        processor = _import_processor(source.processor_path)
        data = processor(extracted, source.processed_dir)
        processed_data[dataset_name] = data

    if not processed_data:
        print("\nERROR: No datasets were processed successfully.")
        return

    combined_path = config.paths.combined_csv
    pv_data = processed_data.get("plantvillage", [])
    pd_data = processed_data.get("plantdoc", [])
    tl_data = processed_data.get("tomatoleaf", [])
    create_combined_csv(pv_data, pd_data, tl_data, combined_path)
    print_summary(pv_data, pd_data, tl_data)


if __name__ == "__main__":
    main()
