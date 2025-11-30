"""
Debug version of secondary_detector_crop_fps: applies the same filtering logic
but also saves annotated originals with GT boxes, FP boxes, and crop windows.

Example:
    python -m src.data_manipulation.instance_cropping.secondary_detector_crop_fps_debug
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
from tqdm.auto import tqdm

CONFIG = {
    "IMAGES_DIR": Path("/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/phash10/original/images/train"),
    "LABELS_DIR": Path("/lab/projects/fire_smoke_awr/outputs/yolo/detection/pyro-sdis/phash10/900/es_train/composites"),
    # Ground-truth labels; crops whose window overlaps any GT box are skipped.
    "GT_LABELS_DIR": Path("/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/phash10/original/labels/train"),
    "OUTPUT_BASE_DIR": Path("/lab/projects/fire_smoke_awr/data/detection/processed/secondary_detector/crops_224_debug"),
    "WINDOW_SIZE": (224, 224),  # (height, width)
    "ALLOWED_EXTS": (".jpg", ".jpeg", ".png"),
    "OUTPUT_PREFIX": "crop_",  # optional prefix for saved files
    # Confidence filter (inclusive). Set both to None to keep all.
    "CONF_MIN": None,
    "CONF_MAX": None,
    # Debug overlay directory; defaults to <OUTPUT_BASE_DIR>/debug if None.
    "DEBUG_DIR": Path("/lab/projects/fire_smoke_awr/src/data_manipulation/instance_cropping/crops_224_debug/debug"),
}


@dataclass
class YoloBox:
    """YOLO-format bounding box."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float | None = None

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


def resolve_paths() -> Tuple[Path, Path, Path]:
    """Resolve required paths from CONFIG."""
    images_dir = Path(CONFIG["IMAGES_DIR"]) if CONFIG["IMAGES_DIR"] else None
    labels_dir = Path(CONFIG["LABELS_DIR"]) if CONFIG["LABELS_DIR"] else None
    gt_labels_dir = Path(CONFIG["GT_LABELS_DIR"]) if CONFIG["GT_LABELS_DIR"] else None
    if images_dir is None or labels_dir is None or gt_labels_dir is None:
        raise ValueError("Set IMAGES_DIR, LABELS_DIR, and GT_LABELS_DIR in CONFIG.")
    return images_dir, labels_dir, gt_labels_dir


def load_boxes(label_path: Path) -> List[YoloBox]:
    """Parse a YOLO label file into boxes (ignores/confidence as optional sixth column)."""
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
        conf = float(parts[5]) if len(parts) >= 6 else None
        boxes.append(YoloBox(cls=cls, cx=cx, cy=cy, w=w, h=h, conf=conf))
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
    """Compute crop window centered on the box and clamp to image bounds."""
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


def window_overlaps_gt(win_x1: int, win_y1: int, win_x2: int, win_y2: int, gt_boxes: List[YoloBox], img_w: int, img_h: int) -> bool:
    """Return True if crop window has positive intersection with any GT box."""
    for gt_box in gt_boxes:
        gx1, gy1, gx2, gy2 = gt_box.to_corners(img_w, img_h)
        inter_x1 = max(win_x1, gx1)
        inter_y1 = max(win_y1, gy1)
        inter_x2 = min(win_x2, gx2)
        inter_y2 = min(win_y2, gy2)
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            return True
    return False


def draw_box(img, box, color, thickness=2, label: str | None = None):
    """Draw rectangle with optional label."""
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
    if label:
        cv2.putText(
            img,
            label,
            (int(box[0]), int(box[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )


def ensure_output_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    """Create and return output dirs for crops, labels, and debug overlays."""
    images_out = base_dir / "images"
    labels_out = base_dir / "labels"
    debug_dir = Path(CONFIG["DEBUG_DIR"]) if CONFIG["DEBUG_DIR"] else base_dir / "debug"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    return images_out, labels_out, debug_dir


def crop_boxes() -> None:
    """Main entry point for cropping and debugging overlays."""
    images_dir, labels_dir, gt_labels_dir = resolve_paths()
    output_base = Path(CONFIG["OUTPUT_BASE_DIR"])
    win_h, win_w = CONFIG["WINDOW_SIZE"]
    extensions = CONFIG["ALLOWED_EXTS"]
    prefix = CONFIG.get("OUTPUT_PREFIX", "")
    min_conf = CONFIG["CONF_MIN"]
    max_conf = CONFIG["CONF_MAX"]

    images_out, labels_out, debug_dir = ensure_output_dirs(output_base)

    print("CONFIG:")
    print(f"  IMAGES_DIR      = {images_dir}")
    print(f"  LABELS_DIR      = {labels_dir}")
    print(f"  GT_LABELS_DIR   = {gt_labels_dir}")
    print(f"  OUTPUT_IMAGES   = {images_out}")
    print(f"  OUTPUT_LABELS   = {labels_out}")
    print(f"  DEBUG_DIR       = {debug_dir}")
    print(f"  WINDOW_SIZE     = {CONFIG['WINDOW_SIZE']}")
    print(f"  ALLOWED_EXTS    = {extensions}")
    print(f"  OUTPUT_PREFIX   = {prefix}")
    print(f"  CONF_MIN/MAX    = {min_conf}, {max_conf}")
    print()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    if not gt_labels_dir.exists():
        raise FileNotFoundError(f"GT labels dir not found: {gt_labels_dir}")

    label_files = sorted(labels_dir.glob("*.txt"))
    processed_images = 0
    total_boxes = 0
    crops_saved = 0
    missing_images = 0
    degenerate_boxes = 0
    skipped_gt_overlap = 0
    skipped_conf = 0

    for label_path in tqdm(label_files, desc="Cropping boxes (debug)"):
        boxes = load_boxes(label_path)
        if not boxes:
            continue

        gt_label_path = gt_labels_dir / label_path.name
        gt_boxes = load_boxes(gt_label_path)

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
            if min_conf is not None or max_conf is not None:
                if box.conf is None:
                    skipped_conf += 1
                    continue
                if min_conf is not None and box.conf < min_conf:
                    skipped_conf += 1
                    continue
                if max_conf is not None and box.conf > max_conf:
                    skipped_conf += 1
                    continue

            win_x1, win_y1, win_x2, win_y2, curr_w, curr_h = compute_window(
                box, img_w, img_h, win_h, win_w
            )

            if window_overlaps_gt(win_x1, win_y1, win_x2, win_y2, gt_boxes, img_w, img_h):
                skipped_gt_overlap += 1
                continue

            crop = image[win_y1:win_y2, win_x1:win_y2]
            if crop.size == 0:
                degenerate_boxes += 1
                continue

            out_stem = f"{prefix}{label_path.stem}_cls{box.cls}_{idx}"
            out_image_path = images_out / f"{out_stem}.jpg"
            out_label_path = labels_out / f"{out_stem}.txt"
            out_debug_path = debug_dir / f"{out_stem}.jpg"

            # Save crop and empty label (false positive mining).
            cv2.imwrite(str(out_image_path), crop)
            out_label_path.write_text("")
            crops_saved += 1

            # Build annotated original and save.
            annotated = image.copy()
            fp_x1, fp_y1, fp_x2, fp_y2 = box.to_corners(img_w, img_h)
            # Draw GT boxes (green).
            for gt in gt_boxes:
                gx1, gy1, gx2, gy2 = gt.to_corners(img_w, img_h)
                draw_box(annotated, (gx1, gy1, gx2, gy2), color=(0, 255, 0), label="GT")
            # Draw FP box (red).
            fp_label = f"FP cls{box.cls}"
            if box.conf is not None:
                fp_label += f" conf{box.conf:.2f}"
            draw_box(annotated, (fp_x1, fp_y1, fp_x2, fp_y2), color=(0, 0, 255), label=fp_label)
            # Draw crop window (blue).
            draw_box(annotated, (win_x1, win_y1, win_x2, win_y2), color=(255, 0, 0), label="crop")

            cv2.imwrite(str(out_debug_path), annotated)

    print("\n=== Cropping Summary (debug) ===")
    print(f"Label files processed: {len(label_files)}")
    print(f"Images loaded: {processed_images}")
    print(f"Total boxes seen: {total_boxes}")
    print(f"Crops saved: {crops_saved}")
    print(f"Missing images: {missing_images}")
    print(f"Skipped/degenerate boxes: {degenerate_boxes}")
    print(f"Skipped due to GT overlap: {skipped_gt_overlap}")
    print(f"Skipped due to confidence filter: {skipped_conf}")


def main() -> None:
    """CLI entry point."""
    crop_boxes()


if __name__ == "__main__":
    main()
