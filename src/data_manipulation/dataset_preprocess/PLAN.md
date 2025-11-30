Preprocessing script plan (images/labels pruning)

- Goal: add a dataset cleaner that removes images lacking labels and images whose YOLO bounding boxes exceed 2% of image area.
- Inputs: dataset root with `images/` and `labels/` subfolders; optional split subdirectories (`train`, `val`, `test`) should be supported.
- Steps to implement:
  - Parse CLI args for dataset path, optional split, and box area threshold (default 0.02).
  - Walk image files, locate matching YOLO label files, and load image dimensions via Pillow.
  - Skip images with no label file or empty labels; mark for deletion.
  - For each bounding box, compute absolute area; if any exceeds threshold of image area, mark that image and its label for deletion.
  - Delete marked image and label files; print summary counts.
- Safety: dry-run flag to report would-be deletions; require confirmation unless `--yes` is passed.
- Output: console summary of kept vs. removed files; consider writing an optional log CSV of removed files.
