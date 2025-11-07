"""Filter a dataset down to single-box smoke images with tiny footprints.

This script ingests a YOLO-style dataset and keeps only the samples where:
    1. The label file contains exactly one bounding box.
    2. The class id equals `CONFIG["target_class"]` (default: smoke = 1).
    3. The normalized box area is smaller than `CONFIG["max_area_percent"]`.
    4. The image resolution meets or exceeds `CONFIG["min_resolution"]` (>=1080p by default).

Example:
    python filter_small_smoke_only.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Pillow is required to inspect image resolution. Install it via `pip install pillow`."
    ) from exc


CONFIG = {
    # Source dataset (expects `images/` and `labels/` inside this folder).
    "source_root": "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/ABCDE_all",
    # Destination dataset root to receive filtered samples.
    "output_root": "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/ABCDE_all/early_smoke",
    "image_dirname": "images",
    "label_dirname": "labels",
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp"],
    "target_class": 1,
    "max_area_percent": 0.05,
    "min_resolution": (1080, 1080),  # width, height
}


def find_image_path(stem: str, images_dir: Path, extensions: list[str]) -> Path | None:
    for ext in extensions:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def label_meets_criteria(label_path: Path, cfg: dict) -> bool:
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        return False

    parts = lines[0].split()
    if len(parts) < 5:
        return False

    try:
        class_id = int(float(parts[0]))
        width = float(parts[3])
        height = float(parts[4])
    except ValueError:
        return False

    if class_id != cfg["target_class"]:
        return False

    area_percent = width * height * 100
    return area_percent < cfg["max_area_percent"]


def filter_dataset(cfg: dict) -> None:
    source_root = Path(cfg["source_root"])
    output_root = Path(cfg["output_root"])

    src_images = source_root / cfg["image_dirname"]
    src_labels = source_root / cfg["label_dirname"]

    dst_images = output_root / cfg["image_dirname"]
    dst_labels = output_root / cfg["label_dirname"]
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    kept = 0
    scanned = 0
    missing_images = 0
    too_small_resolution = 0

    for label_path in sorted(src_labels.glob("*.txt")):
        scanned += 1
        if not label_meets_criteria(label_path, cfg):
            continue

        stem = label_path.stem
        image_path = find_image_path(stem, src_images, cfg["image_extensions"])
        if image_path is None:
            missing_images += 1
            continue

        with Image.open(image_path) as img:
            width, height = img.size
        min_width, min_height = cfg["min_resolution"]
        if width < min_width or height < min_height:
            too_small_resolution += 1
            continue

        shutil.copy2(image_path, dst_images / image_path.name)
        shutil.copy2(label_path, dst_labels / label_path.name)
        kept += 1

    print(f"Scanned {scanned} label files.")
    print(f"Kept {kept} samples that met the criteria.")
    if missing_images:
        print(f"Skipped {missing_images} matches because the image file was missing.")
    if too_small_resolution:
        print(f"Skipped {too_small_resolution} samples due to resolution < {cfg['min_resolution']}.")


if __name__ == "__main__":
    filter_dataset(CONFIG)
