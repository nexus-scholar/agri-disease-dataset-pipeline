"""Utilities for downloading dataset archives from public URLs."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable

import requests
from rich.progress import Progress, BarColumn, DownloadColumn, TextColumn, TimeRemainingColumn

CHUNK_SIZE = 1024 * 1024  # 1 MB


class DownloadError(RuntimeError):
    pass


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, overwrite: bool = False):
    dest = Path(dest)
    ensure_parent(dest)
    if dest.exists() and not overwrite:
        print(f"  Skipping download (exists): {dest}")
        return dest

    print(f"  Downloading {url}\n            -> {dest}")
    with requests.get(url, stream=True, timeout=60) as response:
        if response.status_code != 200:
            raise DownloadError(f"Failed to download {url}: HTTP {response.status_code}")
        total = int(response.headers.get("Content-Length", 0))
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
        )
        task_id = progress.add_task("download", total=total, description=f"  -> {dest.name}")
        with progress:
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        progress.update(task_id, advance=len(chunk))
    return dest


def download_datasets(dataset_names: Iterable[str], config) -> list[Path]:
    downloaded = []
    for name in dataset_names:
        url = config.dataset_urls.get(name)
        if not url:
            print(f"No download URL configured for {name}. Use --dataset-url or env var to provide one.")
            continue
        dest = {
            "plantvillage": config.paths.plantvillage_zip,
            "plantdoc": config.paths.plantdoc_zip,
            "tomatoleaf": config.paths.tomato_leaf_zip,
        }.get(name)
        if not dest:
            continue
        downloaded.append(download_file(url, dest, overwrite=config.overwrite_existing))
    return downloaded

