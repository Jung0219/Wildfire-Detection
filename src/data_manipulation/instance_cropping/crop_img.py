"""
Crop fixed-size windows centered on each YOLO bounding box.

Example:
    python -m src.data_manipulation.instance_cropping.fixed_window_crop
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
from tqdm import tqdm

CONFIG = {
    "IMAGES_DIR": "data/detection/training/AD_phash3_early_smoke/original/images/train",
    "LABELS_DIR": "/lab/projects/fire_smoke_awr/outputs/yolo/detection/AD_phash3_early_smoke/900_AdamW/es_train/labels/fp_00",
    "OUTPUT_DIR": "/lab/projects/fire_smoke_awr/data/classification/datasets/AD_phash3_early_smoke/background",
    "OUTPUT_PREFIX": "fp_",  # optional prefix added to every saved crop filename
    "WINDOW_SIZE": (224, 224),  # (height, width)
    "ALLOWED_EXTS": (".jpg", ".jpeg", ".png"),
}


@dataclass
class YoloBox:
    """Container for YOLO-format bounding box."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float


def load_boxes(label_path: Path) -> List[YoloBox]:
    """Parse a YOLO label file into a list of boxes."""
    if not label_path.exists():
        return []
    boxes: List[YoloBox] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append(YoloBox(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


def compute_window(
    box: YoloBox, img_w: int, img_h: int, win_h: int, win_w: int
) -> Tuple[int, int, int, int]:
    """Return pixel bounds (x1, y1, x2, y2) for a fixed window centered on the box."""
    cx_px = box.cx * img_w
    cy_px = box.cy * img_h
    width = min(win_w, img_w)
    height = min(win_h, img_h)

    half_w = width / 2.0
    half_h = height / 2.0

    x1 = int(round(cx_px - half_w))
    y1 = int(round(cy_px - half_h))

    x1 = max(0, min(x1, img_w - width))
    y1 = max(0, min(y1, img_h - height))

    x2 = int(x1 + width)
    y2 = int(y1 + height)
    return x1, y1, x2, y2


def find_image(stem: str, image_dir: Path, extensions: Iterable[str]) -> Path | None:
    """Return the first image that matches the stem and allowed extensions."""
    for ext in extensions:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def crop_instances() -> None:
    """Main entry point for generating crops."""
    images_dir = Path(CONFIG["IMAGES_DIR"])
    labels_dir = Path(CONFIG["LABELS_DIR"])
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_prefix = CONFIG.get("OUTPUT_PREFIX", "")
    win_h, win_w = CONFIG["WINDOW_SIZE"]
    extensions = CONFIG["ALLOWED_EXTS"]

    for path in (images_dir, labels_dir):
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_images = 0
    total_boxes = 0
    saved_crops = 0
    missing_images = 0

    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in tqdm(label_files, desc="Cropping boxes"):
        boxes = load_boxes(label_path)
        if not boxes:
            continue

        image_path = find_image(label_path.stem, images_dir, extensions)
        if image_path is None:
            missing_images += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        processed_images += 1
        total_boxes += len(boxes)

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = compute_window(box, img_w, img_h, win_h, win_w)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_name = f"{output_prefix}{label_path.stem}_cls{box.cls}_{idx}.jpg"
            cv2.imwrite(str(output_dir / out_name), crop)
            saved_crops += 1

    print("\n=== Cropping Summary ===")
    print(f"Label files processed: {len(label_files)}")
    print(f"Images loaded: {processed_images}")
    print(f"Total boxes seen: {total_boxes}")
    print(f"Crops saved: {saved_crops}")
    print(f"Missing images: {missing_images}")


def main() -> None:
    """CLI hook."""
    crop_instances()


if __name__ == "__main__":
    main()
