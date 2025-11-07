"""
Copy deduplicated image names (and their YOLO annotations) to a clean subset.

Example:
    python src/data_manipulation/deduplication/phash/get_images_from_csv.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

# ================ CONFIG ================
CONFIG = {
    # Directory that still contains the full dataset (expects images/ + labels/ subfolders)
    "PARENT_DIR": "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/E/original",
    # Destination directory that will receive the kept subset
    "OUTPUT_DIR": "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/E/deduplicated/dedup_phash10",
    # CSV listing the kept filenames (one row per image)
    "CSV_PATH": "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/src/data_manipulation/deduplication/phash/csvs/E_phash_10.csv",
    "IMAGE_COLUMN_NAME": "Kept Image Name",
    # Annotation extension to copy alongside images
    "ANNO_EXT": ".txt",
}
# ========================================


def load_kept_images(csv_path: Path, column: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")
    return df[column].dropna().astype(str).tolist()


def copy_subset(image_names: Iterable[str]) -> int:
    parent_dir = Path(CONFIG["PARENT_DIR"]).expanduser()
    output_dir = Path(CONFIG["OUTPUT_DIR"]).expanduser()
    anno_ext = CONFIG["ANNO_EXT"]

    image_dir = parent_dir / "images"
    anno_dir = parent_dir / "labels"

    out_image_dir = output_dir / "images"
    out_anno_dir = output_dir / "labels"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_anno_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_name in tqdm(image_names, desc="Copying images"):
        src_img = image_dir / img_name
        dst_img = out_image_dir / img_name

        if not src_img.exists():
            print(f"Warning: missing image {src_img}")
            continue

        shutil.copy2(src_img, dst_img)
        copied += 1

        base_name = Path(img_name).stem
        anno_name = f"{base_name}{anno_ext}"
        src_anno = anno_dir / anno_name
        dst_anno = out_anno_dir / anno_name

        if src_anno.exists():
            shutil.copy2(src_anno, dst_anno)
        else:
            print(f"Warning: missing annotation for {img_name}")

    return copied


def main() -> None:
    csv_path = Path(CONFIG["CSV_PATH"]).expanduser()
    print("Running with CONFIG:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    kept_images = load_kept_images(csv_path, CONFIG["IMAGE_COLUMN_NAME"])
    print(f"Found {len(kept_images)} filenames in CSV.")

    copied = copy_subset(kept_images)
    print(f"Copied {copied} images (with annotations where available) to {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
