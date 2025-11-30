"""Copy a cleaned YOLO dataset, skipping unlabeled images and those with oversized boxes.

Example:
    # Edit CONFIG below, then run:
    python src/data_manipulation/dataset_preprocess/clean_dataset.py
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm

# ================== CONFIG ==================
# Set the source dataset root (expects images/ and labels/ children).
DATASET_DIR = Path("/lab/projects/fire_smoke_awr/data/detection/datasets/pyro-sdis/deduplicated/phash10")

# Destination for the cleaned copy; existing files are left untouched/overwritten as copied.
OUTPUT_DIR = Path("/lab/projects/fire_smoke_awr/data/detection/processed/pyro-sdis/cleaned")

# Optional split subfolder (e.g., "train", "val", "test"); leave empty to process top-level images/labels.
SPLIT = ""

# Any box with area greater than this fraction of the image area triggers removal (0.02 = 2%).
BOX_AREA_THRESHOLD = 0.02

# When True, only report actions; no files are copied.
DRY_RUN = False
# ============================================


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class RemovalCounts:
    removed_missing_label: int = 0
    removed_large_box: int = 0
    kept: int = 0

    @property
    def total_removed(self) -> int:
        return self.removed_missing_label + self.removed_large_box


def resolve_dirs(dataset_dir: Path, output_dir: Path, split: str) -> tuple[Path, Path, Path, Path]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    out_images_dir = output_dir / "images"
    out_labels_dir = output_dir / "labels"
    if split:
        images_dir = images_dir / split
        labels_dir = labels_dir / split
        out_images_dir = out_images_dir / split
        out_labels_dir = out_labels_dir / split
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir, out_images_dir, out_labels_dir


def iter_image_files(images_dir: Path) -> Iterable[Path]:
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_label_lines(label_path: Path) -> list[list[float]]:
    lines: list[list[float]] = []
    if not label_path.exists():
        return lines
    with label_path.open("r") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                lines.append(list(map(float, parts[:5])))
            except ValueError:
                continue
    return lines


def box_area_fraction(width: float, height: float, img_w: int, img_h: int) -> float:
    if width <= 1.0 and height <= 1.0:
        return width * height
    return (width * height) / (img_w * img_h)


def should_remove_for_boxes(label_lines: list[list[float]], img_w: int, img_h: int, threshold: float) -> bool:
    for _, _, _, width, height in label_lines:
        area_frac = box_area_fraction(width, height, img_w, img_h)
        if area_frac > threshold:
            return True
    return False


def copy_pair(src_image: Path, src_label: Path, dst_image: Path, dst_label: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image, dst_image)
    shutil.copy2(src_label, dst_label)


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    split: str,
    threshold: float,
    dry_run: bool,
) -> RemovalCounts:
    images_dir, labels_dir, out_images_dir, out_labels_dir = resolve_dirs(dataset_dir, output_dir, split)
    counts = RemovalCounts()

    for image_path in tqdm(list(iter_image_files(images_dir)), desc="Scanning images"):
        rel = image_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")
        label_lines = load_label_lines(label_path)

        if not label_lines:
            counts.removed_missing_label += 1
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        if should_remove_for_boxes(label_lines, img_w, img_h, threshold):
            counts.removed_large_box += 1
            continue

        dst_image = out_images_dir / rel
        dst_label = out_labels_dir / rel.with_suffix(".txt")
        copy_pair(image_path, label_path, dst_image, dst_label, dry_run)
        counts.kept += 1

    return counts


def print_summary(counts: RemovalCounts, dry_run: bool, output_dir: Path) -> None:
    mode = "DRY RUN (no files copied)" if dry_run else "Copied cleaned dataset"
    print(f"\n{mode} to {output_dir}")
    print(f"Skipped (missing labels): {counts.removed_missing_label}")
    print(f"Skipped (oversized boxes): {counts.removed_large_box}")
    print(f"Total skipped: {counts.total_removed}")
    print(f"Kept & copied: {counts.kept}")


def main() -> int:
    counts = process_dataset(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        split=SPLIT,
        threshold=BOX_AREA_THRESHOLD,
        dry_run=DRY_RUN,
    )
    print_summary(counts, dry_run=DRY_RUN, output_dir=OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
