"""
Generate fixed-size crops centered on each YOLO GT box and emit paired labels.

Example:
    python -m src.data_manipulation.instance_cropping.centered_crop_224
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
from tqdm.auto import tqdm

CONFIG = {
    # Preferred: set PARENT_DIR and the script will look under images/test and labels/test.
    "PARENT_DIR": Path("/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/phash10/original"),
    # If PARENT_DIR is None, set IMAGES_DIR and LABELS_DIR explicitly.
    "IMAGES_DIR": None,
    "LABELS_DIR": None,
    # Output base; crops go to <OUTPUT_BASE_DIR>/images, labels to <OUTPUT_BASE_DIR>/labels.
    "OUTPUT_BASE_DIR": Path("/lab/projects/fire_smoke_awr/data/detection/processed/secondary_detector/crops_224"),
    "WINDOW_SIZE": (224, 224),  # (height, width)
    "ALLOWED_EXTS": (".jpg", ".jpeg", ".png"),
    "OUTPUT_PREFIX": "crop_",  # optional prefix for saved files
}


@dataclass
class YoloBox:
    """YOLO-format bounding box."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def to_corners(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Return absolute pixel corners (x1, y1, x2, y2) in the source image."""
        x_center = self.cx * img_w
        y_center = self.cy * img_h
        half_w = (self.w * img_w) / 2.0
        half_h = (self.h * img_h) / 2.0
        x1 = x_center - half_w
        y1 = y_center - half_h
        x2 = x_center + half_w
        y2 = y_center + half_h
        return x1, y1, x2, y2


def resolve_paths() -> Tuple[Path, Path]:
    """Resolve image/label dirs based on PARENT_DIR or explicit paths."""
    parent = CONFIG.get("PARENT_DIR")
    if parent:
        parent_path = Path(parent)
        images_dir = parent_path / "images" / "train"
        labels_dir = parent_path / "labels" / "train"
    else:
        images_dir = Path(CONFIG["IMAGES_DIR"]) if CONFIG["IMAGES_DIR"] else None
        labels_dir = Path(CONFIG["LABELS_DIR"]) if CONFIG["LABELS_DIR"] else None

    if images_dir is None or labels_dir is None:
        raise ValueError("Provide PARENT_DIR or explicit IMAGES_DIR and LABELS_DIR in CONFIG.")
    return images_dir, labels_dir


def load_boxes(label_path: Path) -> List[YoloBox]:
    """Parse a YOLO label file into boxes (ignores confidences if present)."""
    if not label_path.exists():
        return []

    boxes: List[YoloBox] = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append(YoloBox(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


def find_image(stem: str, image_dir: Path, extensions: Iterable[str]) -> Path | None:
    """Return the first matching image path for a given stem."""
    for ext in extensions:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def compute_window(
    box: YoloBox, img_w: int, img_h: int, win_h: int, win_w: int
) -> Tuple[int, int, int, int, int, int]:
    """Compute crop window and cached window size."""
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
    return x1, y1, x2, y2, width, height


def relabel_box_for_crop(
    box: YoloBox, win_x1: int, win_y1: int, win_w: int, win_h: int, img_w: int, img_h: int
) -> Tuple[float, float, float, float] | None:
    """Shift box into crop space and return normalized YOLO coords; None if degenerate."""
    bx1, by1, bx2, by2 = box.to_corners(img_w, img_h)

    # Shift into crop coordinates.
    bx1_shift = bx1 - win_x1
    by1_shift = by1 - win_y1
    bx2_shift = bx2 - win_x1
    by2_shift = by2 - win_y1

    # Clamp to crop bounds.
    bx1_clamp = max(0.0, min(bx1_shift, win_w))
    by1_clamp = max(0.0, min(by1_shift, win_h))
    bx2_clamp = max(0.0, min(bx2_shift, win_w))
    by2_clamp = max(0.0, min(by2_shift, win_h))

    new_w = bx2_clamp - bx1_clamp
    new_h = by2_clamp - by1_clamp
    if new_w <= 0 or new_h <= 0:
        return None

    new_cx = (bx1_clamp + bx2_clamp) / 2.0 / win_w
    new_cy = (by1_clamp + by2_clamp) / 2.0 / win_h
    norm_w = new_w / win_w
    norm_h = new_h / win_h
    return new_cx, new_cy, norm_w, norm_h


def ensure_output_dirs(base_dir: Path) -> tuple[Path, Path]:
    """Create and return output image/label directories."""
    images_out = base_dir / "images"
    labels_out = base_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    return images_out, labels_out


def crop_boxes() -> None:
    """Main entry point for cropping."""
    images_dir, labels_dir = resolve_paths()
    output_base = Path(CONFIG["OUTPUT_BASE_DIR"])
    win_h, win_w = CONFIG["WINDOW_SIZE"]
    extensions = CONFIG["ALLOWED_EXTS"]
    prefix = CONFIG.get("OUTPUT_PREFIX", "")

    images_out, labels_out = ensure_output_dirs(output_base)

    print("CONFIG:")
    print(f"  IMAGES_DIR      = {images_dir}")
    print(f"  LABELS_DIR      = {labels_dir}")
    print(f"  OUTPUT_IMAGES   = {images_out}")
    print(f"  OUTPUT_LABELS   = {labels_out}")
    print(f"  WINDOW_SIZE     = {CONFIG['WINDOW_SIZE']}")
    print(f"  ALLOWED_EXTS    = {extensions}")
    print(f"  OUTPUT_PREFIX   = {prefix}")
    print()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    label_files = sorted(labels_dir.glob("*.txt"))
    processed_images = 0
    total_boxes = 0
    crops_saved = 0
    missing_images = 0
    degenerate_boxes = 0

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
            win_x1, win_y1, win_x2, win_y2, curr_w, curr_h = compute_window(
                box, img_w, img_h, win_h, win_w
            )
            crop = image[win_y1:win_y2, win_x1:win_x2]
            if crop.size == 0:
                degenerate_boxes += 1
                continue

            relabeled = relabel_box_for_crop(box, win_x1, win_y1, curr_w, curr_h, img_w, img_h)
            if relabeled is None:
                degenerate_boxes += 1
                continue

            out_stem = f"{prefix}{label_path.stem}_cls{box.cls}_{idx}"
            out_image_path = images_out / f"{out_stem}.jpg"
            out_label_path = labels_out / f"{out_stem}.txt"

            cv2.imwrite(str(out_image_path), crop)
            new_cx, new_cy, new_w, new_h = relabeled
            out_label_path.write_text(f"{box.cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")
            crops_saved += 1

    print("\n=== Cropping Summary ===")
    print(f"Label files processed: {len(label_files)}")
    print(f"Images loaded: {processed_images}")
    print(f"Total boxes seen: {total_boxes}")
    print(f"Crops saved: {crops_saved}")
    print(f"Missing images: {missing_images}")
    print(f"Skipped/degenerate boxes: {degenerate_boxes}")


def main() -> None:
    """CLI entry point."""
    crop_boxes()


if __name__ == "__main__":
    main()
