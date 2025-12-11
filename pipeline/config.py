"""Central configuration for the dataset processing pipeline."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSource:
    name: str
    zip_path: Path
    url: str | None = None


@dataclass(frozen=True)
class Paths:
    """Holds canonical project paths relative to the repo root."""

    project_root: Path

    @property
    def raw_dataset_dir(self) -> Path:
        return self.project_root / "data" / "raw" / "dataset"

    @property
    def processed_dataset_dir(self) -> Path:
        return self.project_root / "data" / "processed" / "dataset"

    @property
    def plantvillage_zip(self) -> Path:
        return self.raw_dataset_dir / "plantvillage.zip"

    @property
    def plantdoc_zip(self) -> Path:
        return self.raw_dataset_dir / "plantdoc.zip"

    @property
    def tomato_leaf_zip(self) -> Path:
        return self.raw_dataset_dir / "Tomato Leaf Dataset  A dataset for multiclass disease detection and classification.zip"

    @property
    def plantvillage_raw(self) -> Path:
        return self.processed_dataset_dir / "PlantVillage"

    @property
    def plantdoc_raw(self) -> Path:
        return self.processed_dataset_dir / "PlantDoc"

    @property
    def tomato_leaf_raw(self) -> Path:
        return self.processed_dataset_dir / "TomatoLeaf"

    @property
    def plantvillage_processed(self) -> Path:
        return self.processed_dataset_dir / "PlantVillage_processed"

    @property
    def plantdoc_processed(self) -> Path:
        return self.processed_dataset_dir / "PlantDoc_processed"

    @property
    def tomato_leaf_processed(self) -> Path:
        return self.processed_dataset_dir / "TomatoLeaf_processed"

    @property
    def combined_csv(self) -> Path:
        return self.processed_dataset_dir / "combined_dataset.csv"

    @property
    def log_file(self) -> Path:
        return self.project_root / "processing.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregates user-configurable knobs for the pipeline run."""

    paths: Paths
    datasets: tuple[str, ...] = ("plantvillage", "plantdoc", "tomatoleaf")
    overwrite_existing: bool = True
    log_level: str = "INFO"
    download: bool = False
    download_only: bool = False
    dataset_urls: dict[str, str | None] = None


DEFAULT_DATASET_URLS = {
    "plantvillage": os.getenv("PLANTVILLAGE_URL", "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"),
    "plantdoc": os.getenv("PLANTDOC_URL", "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"),
    "tomatoleaf": os.getenv("TOMATOLEAF_URL", "https://data.mendeley.com/public-files/datasets/bpfd9cns5g/2/download"),
}


def load_default_config(project_root: Path | None = None, datasets: tuple[str, ...] | None = None,
                        download: bool = False, download_only: bool = False,
                        dataset_urls: dict[str, str | None] | None = None,
                        log_level: str = "INFO") -> PipelineConfig:
    """Create a default configuration for the repository."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    selected = datasets or ("plantvillage", "plantdoc", "tomatoleaf")
    urls = DEFAULT_DATASET_URLS.copy()
    if dataset_urls:
        urls.update(dataset_urls)
    return PipelineConfig(paths=Paths(project_root=project_root),
                          datasets=selected,
                          log_level=log_level,
                          download=download,
                          download_only=download_only,
                          dataset_urls=urls)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Process agricultural datasets")
    parser.add_argument("--datasets", nargs="*", default=["plantvillage", "plantdoc", "tomatoleaf"],
                        help="Datasets to process: plantvillage, plantdoc, tomatoleaf")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--download", action="store_true",
                        help="Download missing archives for the selected datasets before processing")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download archives; skip extraction and processing")
    parser.add_argument("--dataset-url", action="append", default=[],
                        help="Override dataset download URL, e.g. plantdoc=https://example.com/plantdoc.zip")
    args = parser.parse_args()

    dataset_url_overrides: dict[str, str | None] = {}
    for override in args.dataset_url:
        if "=" not in override:
            parser.error(f"Invalid --dataset-url value: {override}. Use name=url format.")
        name, url = override.split("=", 1)
        dataset_url_overrides[name.strip().lower()] = url.strip()

    download_flag = args.download or args.download_only
    cfg = load_default_config(datasets=tuple(args.datasets),
                              download=download_flag,
                              download_only=args.download_only,
                              dataset_urls=dataset_url_overrides,
                              log_level=args.log_level)
    return cfg
