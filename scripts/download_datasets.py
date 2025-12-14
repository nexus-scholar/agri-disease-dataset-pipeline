#!/usr/bin/env python3
"""
Download PlantVillage and PlantDoc datasets directly.

This script downloads the datasets from their original sources:
- PlantVillage: From Kaggle (requires kaggle API) or direct mirror
- PlantDoc: From GitHub repository

Usage:
    # Download both datasets
    python scripts/download_datasets.py

    # Download only PlantVillage
    python scripts/download_datasets.py --plantvillage

    # Download only PlantDoc
    python scripts/download_datasets.py --plantdoc

    # Specify output directory
    python scripts/download_datasets.py --output data/raw/dataset
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


# Direct download URLs (mirrors/alternatives)
PLANTDOC_GITHUB_URL = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"

# Kaggle dataset identifiers
PLANTVILLAGE_KAGGLE = "emmarex/plantdisease"  # PlantVillage on Kaggle
PLANTVILLAGE_KAGGLE_ALT = "abdallahalidev/plantvillage-dataset"  # Alternative


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress indication."""
    print(f"{desc}: {url}")
    print(f"  -> {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try with urllib (works without extra dependencies)
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=show_progress)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_with_wget(url: str, dest: Path):
    """Download using wget (if available)."""
    try:
        subprocess.run(
            ["wget", "-O", str(dest), url],
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_plantdoc(output_dir: Path) -> bool:
    """Download PlantDoc dataset from GitHub."""
    print("\n" + "=" * 60)
    print("DOWNLOADING PLANTDOC")
    print("=" * 60)

    zip_path = output_dir / "PlantDoc.zip"

    # Download from GitHub
    success = download_file(
        PLANTDOC_GITHUB_URL,
        zip_path,
        "Downloading PlantDoc from GitHub"
    )

    if success and zip_path.exists():
        print(f"✓ PlantDoc downloaded: {zip_path}")
        print(f"  Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    else:
        print("✗ Failed to download PlantDoc")
        return False


def download_plantvillage_kaggle(output_dir: Path) -> bool:
    """Download PlantVillage from Kaggle (requires kaggle CLI)."""
    print("\n" + "=" * 60)
    print("DOWNLOADING PLANTVILLAGE (Kaggle)")
    print("=" * 60)

    # Check if kaggle is installed
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Kaggle CLI not found. Install with: pip install kaggle")
        print("Then configure with your API key from https://www.kaggle.com/account")
        return False

    # Download from Kaggle
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from Kaggle: {PLANTVILLAGE_KAGGLE}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", PLANTVILLAGE_KAGGLE, "-p", str(output_dir)],
            check=True
        )

        # Find the downloaded zip
        zip_files = list(output_dir.glob("*.zip"))
        if zip_files:
            # Rename to PlantVillage.zip
            src = zip_files[0]
            dest = output_dir / "PlantVillage.zip"
            if src != dest:
                shutil.move(str(src), str(dest))
            print(f"✓ PlantVillage downloaded: {dest}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed: {e}")

        # Try alternative dataset
        print(f"\nTrying alternative: {PLANTVILLAGE_KAGGLE_ALT}")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", PLANTVILLAGE_KAGGLE_ALT, "-p", str(output_dir)],
                check=True
            )
            zip_files = list(output_dir.glob("*.zip"))
            if zip_files:
                src = zip_files[0]
                dest = output_dir / "PlantVillage.zip"
                if src != dest:
                    shutil.move(str(src), str(dest))
                print(f"✓ PlantVillage downloaded: {dest}")
                return True
        except subprocess.CalledProcessError:
            pass

    return False


def download_plantvillage_gdown(output_dir: Path) -> bool:
    """Download PlantVillage using gdown (Google Drive alternative)."""
    print("\n" + "=" * 60)
    print("DOWNLOADING PLANTVILLAGE (gdown)")
    print("=" * 60)

    # Known Google Drive file IDs for PlantVillage
    # These are common shares - may need updating
    GDRIVE_IDS = [
        "0B_voCy5O5sXMTFByemhpZllYREU",  # Common PlantVillage share
    ]

    try:
        import gdown
    except ImportError:
        print("gdown not installed. Install with: pip install gdown")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "PlantVillage.zip"

    for file_id in GDRIVE_IDS:
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Trying: {url}")
            gdown.download(url, str(dest), quiet=False)
            if dest.exists() and dest.stat().st_size > 1000000:  # > 1MB
                print(f"✓ Downloaded: {dest}")
                return True
        except Exception as e:
            print(f"  Failed: {e}")

    return False


def setup_kaggle_colab():
    """Setup Kaggle credentials on Colab."""
    print("""
To use Kaggle on Colab, run these commands:

1. Upload your kaggle.json:
   from google.colab import files
   files.upload()  # Upload kaggle.json from your Kaggle account

2. Setup credentials:
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json

3. Then run this script again.

Get your kaggle.json from: https://www.kaggle.com/account -> Create New API Token
""")


def print_manual_instructions():
    """Print manual download instructions."""
    print("""
=" * 60
MANUAL DOWNLOAD INSTRUCTIONS
=" * 60

If automatic download fails, you can download manually:

1. PLANTVILLAGE (Kaggle):
   - Go to: https://www.kaggle.com/datasets/emmarex/plantdisease
   - Or: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
   - Click "Download" (requires Kaggle account)
   - Save as: data/raw/dataset/PlantVillage.zip

2. PLANTDOC (GitHub):
   - Go to: https://github.com/pratikkayal/PlantDoc-Dataset
   - Click "Code" -> "Download ZIP"
   - Save as: data/raw/dataset/PlantDoc.zip

3. Then run the processing pipeline:
   python process_datasets.py
""")


def main():
    parser = argparse.ArgumentParser(description="Download plant disease datasets")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/raw/dataset"),
                        help="Output directory for downloaded files")
    parser.add_argument("--plantvillage", action="store_true",
                        help="Download only PlantVillage")
    parser.add_argument("--plantdoc", action="store_true",
                        help="Download only PlantDoc")
    parser.add_argument("--kaggle-setup", action="store_true",
                        help="Show Kaggle setup instructions for Colab")
    args = parser.parse_args()

    if args.kaggle_setup:
        setup_kaggle_colab()
        return

    # If neither specified, download both
    if not args.plantvillage and not args.plantdoc:
        args.plantvillage = True
        args.plantdoc = True

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")

    results = {}

    # Download PlantDoc (easiest - direct from GitHub)
    if args.plantdoc:
        results['PlantDoc'] = download_plantdoc(output_dir)

    # Download PlantVillage (try multiple methods)
    if args.plantvillage:
        # Method 1: Kaggle CLI
        success = download_plantvillage_kaggle(output_dir)

        if not success:
            # Method 2: gdown (Google Drive)
            success = download_plantvillage_gdown(output_dir)

        results['PlantVillage'] = success

        if not success:
            print("\n⚠ PlantVillage download failed.")
            print_manual_instructions()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for dataset, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {dataset}: {status}")

    # Check what we have
    print("\nFiles in output directory:")
    for f in output_dir.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    # Next steps
    if all(results.values()):
        print("\n✓ All downloads complete!")
        print("\nNext step: Run the processing pipeline:")
        print("  python process_datasets.py")
    else:
        print("\n⚠ Some downloads failed. See instructions above.")


if __name__ == "__main__":
    main()

