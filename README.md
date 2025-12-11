# agri-disease-dataset-pipeline

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20WSL-success.svg" alt="Platform">
  <img src="https://img.shields.io/badge/tests-pytest-lightgrey.svg" alt="Tests">
  <img src="https://img.shields.io/badge/license-Refer%20dataset%20authors-orange.svg" alt="License">
</p>

> **Reproducible long-path-safe pipeline for extracting, normalizing, and merging PlantVillage, PlantDoc, and Tomato Leaf datasets into a single ML-ready manifest.**

## Highlights

- 🔁 **End-to-end automation** – unzip, normalize, label, and aggregate datasets in one command.
- 🪟 **Windows-first resilience** – mitigates `MAX_PATH` issues via extended-path copying and rename manifests.
- 📊 **Consistent outputs** – sequential filenames, CSV labels, metadata JSON, and a unified `combined_dataset.csv`.
- 🧪 **Tested utilities** – label normalization covered by `pytest`; processors structured for extension.

## Repository Layout

```
├── data/
│   ├── raw/dataset/            # place plantvillage.zip, plantdoc.zip, tomato leaf zip
│   └── processed/dataset/      # generated outputs (cleaned on commit)
├── pipeline/                   # core modules (config, fs utils, processors)
├── tests/                      # pytest unit tests
├── docs/                       # architecture & dataset notes (+ diagrams/screenshots)
├── process_datasets.py         # CLI entry point
├── requirements.txt            # runtime deps
└── README.md
```

## Quick Links

| Topic | Resource |
|-------|----------|
| Architecture & diagrams | [`docs/pipeline_overview.md`](docs/pipeline_overview.md) |
| Dataset-specific nuances | [`docs/datasets.md`](docs/datasets.md) |
| Issue tracker | GitHub Issues (enable upon publishing) |

## Requirements

- Windows 10/11 with long-paths enabled (or WSL/Ubuntu)
- Python 3.10+
- ~10 GB free disk space for extraction

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Preparation

Copy the raw archives into `data/raw/dataset/` before running the pipeline:

- `plantvillage.zip`
- `plantdoc.zip`
- `Tomato Leaf Dataset  A dataset for multiclass disease detection and classification.zip`

Processed outputs are not tracked; they will be re-created in `data/processed/dataset/` on demand.

## Run the Pipeline

```bash
python process_datasets.py --datasets plantvillage plantdoc tomatoleaf --log-level INFO
```

Flags:

- `--datasets` – subset to run (`plantvillage`, `plantdoc`, `tomatoleaf`). Defaults to all.
- `--log-level` – logging verbosity (`INFO`, `DEBUG`, ...).

The script clears previous processed folders, extracts long-path-safe copies, and regenerates CSV/metadata files. `combined_dataset.csv` always reflects the datasets processed in the current run.

## Outputs

```
data/processed/dataset/
├── PlantVillage_processed/
│   ├── {class_label}/image-00001.jpg
│   ├── labels.csv
│   └── metadata.json
├── PlantDoc_processed/
│   ├── {class_label}/image-00001.jpg
│   ├── labels.csv
│   └── metadata.json
├── TomatoLeaf_processed/
│   ├── images/
│   ├── annotated/{train,test}/images
│   ├── labels/
│   └── metadata.json
└── combined_dataset.csv
```

Each extraction also emits `{Dataset}_renamed_files.json` if any filenames were truncated to satisfy Windows path constraints.

## Testing

```bash
.\.venv\Scripts\python.exe -m pytest tests
```

## Troubleshooting

| Symptom | Remedy |
|---------|--------|
| `WinError 3` / path too long | Ensure `LongPathsAware=1` or run under WSL; review rename manifest. |
| Missing dataset folders | Confirm zip presence under `data/raw/dataset/`. |
| Antivirus slows extraction | Temporarily exclude the repo path or run on SSD. |
| Need original filenames | Cross-reference `{Dataset}_renamed_files.json` for mappings. |

## Contributing

1. Fork & branch (`git checkout -b feature/...`).
2. Add/adjust processors or utilities under `pipeline/`.
3. Update docs/tests as needed.
4. Run `pytest` + sample `process_datasets.py` invocation.
5. Submit a PR with a concise summary and validation details.

## License / Citation

Refer to the original dataset licenses (PlantVillage, PlantDoc, Tomato Leaf). This pipeline is distributed under the repository’s default license; cite the dataset authors in downstream research.
