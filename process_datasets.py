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

from pipeline.config import load_default_config, parse_args
from pipeline.logging_utils import configure_logging
from pipeline.plantdoc import process_plantdoc
from pipeline.plantvillage import process_plantvillage
from pipeline.tomato_leaf import process_tomato_leaf
from pipeline.zip_utils import unzip_dataset
from pipeline.download_utils import download_datasets


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

    pv_extracted = unzip_dataset(config.paths.plantvillage_zip, config.paths.plantvillage_raw, "PlantVillage") if "plantvillage" in config.datasets else None
    pd_extracted = unzip_dataset(config.paths.plantdoc_zip, config.paths.plantdoc_raw, "PlantDoc") if "plantdoc" in config.datasets else None
    tl_extracted = unzip_dataset(config.paths.tomato_leaf_zip, config.paths.tomato_leaf_raw, "TomatoLeaf") if "tomatoleaf" in config.datasets else None

    if not any([pv_extracted, pd_extracted, tl_extracted]):
        print("\nERROR: No datasets were extracted. Please check zip files exist at:")
        print(f"  - {config.paths.plantvillage_zip}")
        print(f"  - {config.paths.plantdoc_zip}")
        print(f"  - {config.paths.tomato_leaf_zip}")
        return

    print("\n" + "=" * 60)
    print("STEP 2: PROCESSING DATASETS")
    print("=" * 60)

    pv_data = process_plantvillage(pv_extracted, config.paths.plantvillage_processed) if pv_extracted else []
    pd_data = process_plantdoc(pd_extracted, config.paths.plantdoc_processed) if pd_extracted else []
    tl_data = process_tomato_leaf(tl_extracted, config.paths.tomato_leaf_processed) if tl_extracted else []

    combined_path = config.paths.combined_csv
    create_combined_csv(pv_data, pd_data, tl_data, combined_path)
    print_summary(pv_data, pd_data, tl_data)


if __name__ == "__main__":
    main()
