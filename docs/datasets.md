# Dataset Notes

## PlantVillage

- Structure: `PlantVillage/{ClassName}/image.jpg` with potential nested `PlantVillage/PlantVillage/...` directories.
- Processor (`pipeline/plantvillage.py`) normalizes class names via `label_utils.normalize_label`, renames images to `image-00001.jpg`, and writes:
  - `PlantVillage_processed/{label}/image-XXXXX.jpg`
  - `PlantVillage_processed/labels.csv`
  - `PlantVillage_processed/metadata.json`
- Classes are mapped into `crop_disease` format; duplicate folder names are merged automatically.

## PlantDoc

- Structure: `PlantDoc/train/{Class}/img.jpg` and `PlantDoc/test/{Class}/img.jpg`.
- Processor merges both splits, retaining `original_split` metadata in the CSV.
- Some filenames exceed Windows path limits; `zip_utils.unzip_dataset` logs truncated names in `PlantDoc_renamed_files.json`.
- After processing, outputs reside under `PlantDoc_processed/{label}/` plus CSV/metadata.

## Tomato Leaf Dataset

- Zip root: `Tomato Leaf Dataset  A dataset for multiclass disease detection and classification`.
- Contains both raw images and YOLO annotations:
  - `TomatoLeafMulticlass (Raw Data)/images/*.jpg`
  - `TomatoLeafMulticlass (Annotated)/(train|test)/(images|labels)`
- Processor copies raw images to `TomatoLeaf_processed/images/` and YOLO splits to `TomatoLeaf_processed/annotated/{split}/`.
- Labels remain YOLO-formatted `.txt` files; metadata includes counts for raw vs. annotated splits.
- A rename manifest `TomatoLeaf_renamed_files.json` is produced when long filenames are truncated.

## Combined CSV

- `data/processed/dataset/combined_dataset.csv` stacks rows from all processed datasets.
- Schema:
  - `filename`
  - `label`
  - `crop`
  - `disease`
  - `source` (`plantvillage`, `plantdoc`, `tomatoleaf`)
  - `path` (relative path into processed folders)
  - Additional per-dataset fields (e.g., `original_folder`, `original_split`) are preserved where available.

## Tips

- Always inspect `{dataset}_renamed_files.json` after extraction if you rely on original filenames.
- To add a new dataset, implement `pipeline/<dataset>.py` with a `process_*` function returning rows shaped like the combined CSV, then register it in `process_datasets.py`.

