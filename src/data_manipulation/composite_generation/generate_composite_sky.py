"""Generate composite images and YOLO labels using skyline-based composites.

Example:
    python src/util/etc/generate_composite.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# ====================== CONFIG ======================
CONFIG = {
    # Parent directories with YOLO-style splits (images/ + labels/)
    "INPUT_PARENT": "/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/original",
    "OUTPUT_PARENT": "/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/composite_skyline/900",
    # When DATASET_SPLITS is populated, IMAGE_ROOT/LABEL_ROOT define the shared directories.
    "DATASET_SPLITS": ["train", "val"],
    "IMAGE_ROOT": "images",
    "LABEL_ROOT": "labels",
    # Legacy single-dir override (set both to None when using DATASET_SPLITS)
    "IMAGE_SUBDIR": None,
    "LABEL_SUBDIR": None,
    "IMAGE_EXTENSIONS": [".jpg", ".jpeg", ".png"],
    # Canvas and intermediate sizing
    "CANVAS_SIZE": 640,
    "INTERMEDIATE_SIZE": 900,
    # Skyline composite tuning
    "ANCHOR_Y_FRAC": 0.25,
    # Optional limit if you only want the first N images (None = all)
    "MAX_SAMPLES": None,
}
# ====================================================


@dataclass
class YoloBox:
    """YOLO box in normalized xywh format."""

    cls: int
    xc: float
    yc: float
    w: float
    h: float

    @classmethod
    def from_line(cls, line: str) -> "YoloBox | None":
        parts = line.strip().split()
        if len(parts) < 5:
            return None
        return cls(
            cls=int(float(parts[0])),
            xc=float(parts[1]),
            yc=float(parts[2]),
            w=float(parts[3]),
            h=float(parts[4]),
        )

    def to_line(self) -> str:
        return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def load_boxes(label_path: Path) -> List[YoloBox]:
    """Load YOLO labels from a txt file."""
    if not label_path.exists():
        return []
    with label_path.open() as f:
        return [box for line in f if (box := YoloBox.from_line(line))]


def save_boxes(boxes: Sequence[YoloBox], label_path: Path) -> None:
    """Write YOLO labels to a txt file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        for box in boxes:
            f.write(box.to_line() + "\n")


def list_image_files(image_dir: Path, extensions: Sequence[str]) -> List[Path]:
    """List image files in a directory matching allowed extensions."""
    allowed = {ext.lower() for ext in extensions}
    files = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    ]
    return sorted(files)


def resolve_split_configs(config: dict) -> List[dict]:
    splits = config.get("DATASET_SPLITS")
    if splits:
        image_root = Path(config.get("IMAGE_ROOT", "images"))
        label_root = Path(config.get("LABEL_ROOT", "labels"))
        resolved = []
        for split in splits:
            resolved.append(
                {
                    "name": str(split),
                    "image_subdir": image_root / split,
                    "label_subdir": label_root / split,
                }
            )
        return resolved

    image_subdir = config.get("IMAGE_SUBDIR")
    label_subdir = config.get("LABEL_SUBDIR")
    if not image_subdir or not label_subdir:
        raise ValueError(
            "Either provide DATASET_SPLITS with IMAGE_ROOT/LABEL_ROOT or set IMAGE_SUBDIR/LABEL_SUBDIR."
        )
    return [
        {
            "name": config.get("SPLIT_NAME", Path(image_subdir).name),
            "image_subdir": Path(image_subdir),
            "label_subdir": Path(label_subdir),
        }
    ]


def to_xyxy_pixels(box: YoloBox, width: int, height: int) -> Tuple[float, float, float, float]:
    """Convert YOLO normalized box to pixel xyxy."""
    x1 = (box.xc - box.w / 2) * width
    y1 = (box.yc - box.h / 2) * height
    x2 = (box.xc + box.w / 2) * width
    y2 = (box.yc + box.h / 2) * height
    return x1, y1, x2, y2


def within_bounds(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> bool:
    """Check that a box is fully within image bounds."""
    if x2 <= x1 or y2 <= y1:
        return False
    return 0 <= x1 and 0 <= y1 and x2 <= width and y2 <= height


def to_yolo_box(
    cls_id: int, x1: float, y1: float, x2: float, y2: float, canvas_size: int
) -> YoloBox:
    """Convert pixel xyxy into YOLO normalized box for a square canvas."""
    xc = ((x1 + x2) / 2) / canvas_size
    yc = ((y1 + y2) / 2) / canvas_size
    w = (x2 - x1) / canvas_size
    h = (y2 - y1) / canvas_size
    return YoloBox(cls=cls_id, xc=xc, yc=yc, w=w, h=h)

def detect_skyline_y(img_bgr, cb_min=120, cb_max=255, cr_min=0, cr_max=130, sky_ratio_thresh=5.0):
    """Detect a skyline row; returns -1 if not found."""

    H, W = img_bgr.shape[:2]
    scale = 0.25
    resized_w = max(1, int(round(W * scale)))
    resized_h = max(1, int(round(H * scale)))
    resized_bgr = cv2.resize(img_bgr, (resized_w, resized_h))
    ycrcb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    y_thresh = float(Y.astype(np.float32).mean())

    def sky_mask(bgr):
        ycrcb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y_, Cr_, Cb_ = cv2.split(ycrcb_)
        return ((Y_ >= y_thresh) & (Cb_ >= cb_min) & (Cb_ <= cb_max) & (Cr_ >= cr_min) & (Cr_ <= cr_max)).astype(np.uint8)

    m_full = sky_mask(resized_bgr)
    counts = m_full.sum(axis=1) / float(resized_w)
    d = np.diff(counts)
    idx = int(np.argmin(d))
    y_candidate = int(np.clip(idx + 1, 0, resized_h - 1))
    above = int(m_full[:y_candidate, :].sum())
    below = int(m_full[y_candidate:, :].sum())
    ratio = (above + 1e-9) / (below + 1e-9)
    if ratio < sky_ratio_thresh:
        return -1
    scale_y = resized_h / float(H)
    y_scaled = int(round(y_candidate / scale_y))
    return int(np.clip(y_scaled, 0, H - 1))


def generate_composite_640x640(original_image, object_center_norm, intermediate_size, anchor_x_frac=0.5, anchor_y_frac=0.25):
    """Create a 640x640 composite for wide images; returns (composite, meta)."""

    TARGET_SIZE = 640
    orig_h, orig_w = original_image.shape[:2]

    scale_inter = intermediate_size / (orig_w if orig_w >= orig_h else orig_h)
    res_inter_w, res_inter_h = int(orig_w * scale_inter), int(orig_h * scale_inter)
    image_inter = cv2.resize(original_image, (res_inter_w, res_inter_h))

    scale_to_640 = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    resized_w, resized_h = int(orig_w * scale_to_640), int(orig_h * scale_to_640)
    resized_bottom = cv2.resize(original_image, (resized_w, resized_h))

    if resized_h == TARGET_SIZE and resized_w == TARGET_SIZE:
        return resized_bottom, {
            "div_y": TARGET_SIZE,
            "crop_x1": 0,
            "crop_y1": 0,
            "scale_inter": scale_inter,
            "scale_to_640": scale_to_640,
            "resized_w": resized_w,
            "resized_h": resized_h,
            "pad_top_left": 0,
            "pad_bottom_left": 0,
        }

    crop_h = TARGET_SIZE - resized_h
    crop_w = resized_w
    obj_x = int(np.clip(object_center_norm[0], 0, 1) * res_inter_w)
    obj_y = int(np.clip(object_center_norm[1], 0, 1) * res_inter_h)
    anchor_x = int(round(anchor_x_frac * crop_w))
    anchor_y = int(round(anchor_y_frac * crop_h))
    crop_x1 = max(0, obj_x - anchor_x)
    crop_y1 = max(0, obj_y - anchor_y)
    crop_x2 = min(crop_x1 + crop_w, res_inter_w)
    crop_y2 = min(crop_y1 + crop_h, res_inter_h)
    if crop_x2 - crop_x1 < crop_w:
        crop_x1 = max(0, crop_x2 - crop_w)
    if crop_y2 - crop_y1 < crop_h:
        crop_y1 = max(0, crop_y2 - crop_h)

    cropped_top = image_inter[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped_top.size == 0:
        cropped_top = np.zeros((max(1, crop_h), max(1, crop_w), 3), dtype=np.uint8)

    resized_crop = cv2.resize(cropped_top, (crop_w, crop_h))

    pad_left_top = (TARGET_SIZE - crop_w) // 2
    pad_left_bottom = (TARGET_SIZE - resized_w) // 2
    top_band = cv2.copyMakeBorder(
        resized_crop,
        0,
        0,
        pad_left_top,
        TARGET_SIZE - crop_w - pad_left_top,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    bottom_band = cv2.copyMakeBorder(
        resized_bottom,
        0,
        0,
        pad_left_bottom,
        TARGET_SIZE - resized_w - pad_left_bottom,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    composite = np.vstack([top_band, bottom_band])

    meta = {
        "div_y": crop_h,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "scale_inter": scale_inter,
        "scale_to_640": scale_to_640,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "pad_top_left": pad_left_top,
        "pad_bottom_left": pad_left_bottom,
    }
    return composite, meta


def pad_or_downscale_to_640(img, target_size=640, color=(114, 114, 114)):
    """Pad or downscale an image to a square canvas."""

    h, w = img.shape[:2]

    if h > target_size or w > target_size:
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        h, w = new_h, new_w
    else:
        scale = 1.0

    canvas = np.full((target_size, target_size, 3), color, dtype=img.dtype)
    y_off = (target_size - h) // 2
    x_off = (target_size - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = img

    return canvas, (x_off, y_off, scale)


def prepare_image_for_detection(image, intermediate_size: int, anchor_y_frac: float, canvas_size: int):
    """Select padding vs composite strategy and return (composite, meta)."""

    img_h, img_w = image.shape[:2]
    if img_h >= img_w or img_h < canvas_size or img_w < canvas_size:
        composite, (x_off, y_off, scale) = pad_or_downscale_to_640(image, canvas_size)
        meta = {"div_y": 0, "scale_to_640": scale, "x_off": x_off, "y_off": y_off}
    else:
        y_border = detect_skyline_y(image, 120, 255, 0, 130, 5.0)
        obj_center = (0.5, float(y_border) / img_h) if y_border >= 0 else (0.5, 0.5)
        composite, meta = generate_composite_640x640(
            image, obj_center, intermediate_size, anchor_y_frac=anchor_y_frac
        )
    return composite, meta


def map_boxes_to_composite(
    boxes: Sequence[YoloBox], meta: dict, orig_w: int, orig_h: int, canvas_size: int
) -> Tuple[List[YoloBox], int, int, int]:
    """Map YOLO boxes from original image to composite coordinates."""
    mapped: List[YoloBox] = []
    dropped_total = 0
    top_total = 0
    top_dropped = 0
    for box in boxes:
        mapped_this_box = False
        top_included = False
        x1, y1, x2, y2 = to_xyxy_pixels(box, orig_w, orig_h)
        if meta.get("div_y", 0) == 0 and "x_off" in meta:
            scale = meta["scale_to_640"]
            x1_c = x1 * scale + meta["x_off"]
            x2_c = x2 * scale + meta["x_off"]
            y1_c = y1 * scale + meta["y_off"]
            y2_c = y2 * scale + meta["y_off"]
            if within_bounds(x1_c, y1_c, x2_c, y2_c, canvas_size, canvas_size):
                mapped.append(to_yolo_box(box.cls, x1_c, y1_c, x2_c, y2_c, canvas_size))
                mapped_this_box = True
            if not mapped_this_box:
                dropped_total += 1
            continue

        # Bottom band (scaled full image)
        scale = meta["scale_to_640"]
        x1_b = x1 * scale + meta["pad_bottom_left"]
        x2_b = x2 * scale + meta["pad_bottom_left"]
        y1_b = y1 * scale + (canvas_size - meta["resized_h"])
        y2_b = y2 * scale + (canvas_size - meta["resized_h"])
        if within_bounds(x1_b, y1_b, x2_b, y2_b, canvas_size, canvas_size):
            mapped.append(to_yolo_box(box.cls, x1_b, y1_b, x2_b, y2_b, canvas_size))
            mapped_this_box = True

        # Top band (cropped from intermediate)
        crop_h = int(meta["div_y"])
        if crop_h <= 0:
            if not mapped_this_box:
                dropped_total += 1
            continue
        top_total += 1
        scale_inter = meta["scale_inter"]
        x1_t = x1 * scale_inter - meta["crop_x1"] + meta["pad_top_left"]
        x2_t = x2 * scale_inter - meta["crop_x1"] + meta["pad_top_left"]
        y1_t = y1 * scale_inter - meta["crop_y1"]
        y2_t = y2 * scale_inter - meta["crop_y1"]
        if within_bounds(x1_t, y1_t, x2_t, y2_t, canvas_size, crop_h):
            mapped.append(to_yolo_box(box.cls, x1_t, y1_t, x2_t, y2_t, canvas_size))
            mapped_this_box = True
            top_included = True

        if not mapped_this_box:
            dropped_total += 1
        if not top_included:
            top_dropped += 1
    return mapped, dropped_total, top_total, top_dropped


def process_split(
    input_parent: Path,
    output_parent: Path,
    split_config: dict,
    config: dict,
) -> dict:
    """Process a split and write composites + labels."""
    split_name = split_config.get("name", "split")
    image_dir = input_parent / split_config["image_subdir"]
    label_dir = input_parent / split_config["label_subdir"]
    out_img_dir = output_parent / split_config["image_subdir"]
    out_label_dir = output_parent / split_config["label_subdir"]

    if not image_dir.exists():
        raise FileNotFoundError(f"[{split_name}] Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"[{split_name}] Label directory not found: {label_dir}")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(image_dir, config["IMAGE_EXTENSIONS"])
    if not image_files:
        print(f"[{split_name}] No images found; skipping.")
        return {
            "split": split_name,
            "processed_images": 0,
            "missing_labels": 0,
            "total_boxes": 0,
            "dropped_boxes": 0,
            "top_total_boxes": 0,
            "top_dropped_boxes": 0,
        }

    max_samples = config.get("MAX_SAMPLES")
    if max_samples is not None:
        image_files = image_files[: int(max_samples)]

    processed = 0
    missing_labels = 0
    total_boxes = 0
    dropped_boxes = 0
    top_total_boxes = 0
    top_dropped_boxes = 0
    desc = f"Processing {split_name} images"

    for img_path in tqdm(image_files, desc=desc):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        boxes = load_boxes(label_path)
        total_boxes += len(boxes)
        composite, meta = prepare_image_for_detection(
            image,
            intermediate_size=int(config["INTERMEDIATE_SIZE"]),
            anchor_y_frac=float(config["ANCHOR_Y_FRAC"]),
            canvas_size=int(config["CANVAS_SIZE"]),
        )
        mapped_boxes, dropped, top_total, top_dropped = map_boxes_to_composite(
            boxes, meta, image.shape[1], image.shape[0], int(config["CANVAS_SIZE"])
        )
        dropped_boxes += dropped
        top_total_boxes += top_total
        top_dropped_boxes += top_dropped

        out_img_path = out_img_dir / img_path.name
        out_label_path = out_label_dir / f"{img_path.stem}.txt"
        cv2.imwrite(str(out_img_path), composite)
        save_boxes(mapped_boxes, out_label_path)
        processed += 1

    print(f"\n[{split_name}] Summary")
    print(f"Images processed: {processed}")
    print(f"Images skipped (missing label): {missing_labels}")
    print(f"GT boxes total: {total_boxes}")
    drop_pct = (100.0 * dropped_boxes / total_boxes) if total_boxes else 0.0
    print(f"GT boxes dropped (out of bounds): {dropped_boxes} ({drop_pct:.2f}%)")
    print(f"GT boxes considered for top band: {top_total_boxes}")
    top_drop_pct = (100.0 * top_dropped_boxes / top_total_boxes) if top_total_boxes else 0.0
    print(f"GT boxes dropped from top band: {top_dropped_boxes} ({top_drop_pct:.2f}%)")
    print(f"Output images dir: {out_img_dir}")
    print(f"Output labels dir: {out_label_dir}")
    return {
        "split": split_name,
        "processed_images": processed,
        "missing_labels": missing_labels,
        "total_boxes": total_boxes,
        "dropped_boxes": dropped_boxes,
        "top_total_boxes": top_total_boxes,
        "top_dropped_boxes": top_dropped_boxes,
    }


def main() -> None:
    """CLI entrypoint."""
    print("Running composite generation with CONFIG:")
    print(json.dumps(CONFIG, indent=2))

    input_parent = Path(CONFIG["INPUT_PARENT"]).expanduser()
    output_parent = Path(CONFIG["OUTPUT_PARENT"]).expanduser()
    split_configs = resolve_split_configs(CONFIG)

    summaries = []
    for split_cfg in split_configs:
        summaries.append(process_split(input_parent, output_parent, split_cfg, CONFIG))

    if len(summaries) > 1:
        total_processed = sum(s["processed_images"] for s in summaries)
        total_missing = sum(s["missing_labels"] for s in summaries)
        total_boxes = sum(s["total_boxes"] for s in summaries)
        total_dropped = sum(s["dropped_boxes"] for s in summaries)
        drop_pct = (100.0 * total_dropped / total_boxes) if total_boxes else 0.0
        top_total_boxes = sum(s["top_total_boxes"] for s in summaries)
        top_dropped_boxes = sum(s["top_dropped_boxes"] for s in summaries)
        top_drop_pct = (
            (100.0 * top_dropped_boxes / top_total_boxes) if top_total_boxes else 0.0
        )
        print("\n=== Aggregate Summary ===")
        print(f"Images processed: {total_processed}")
        print(f"Images skipped (missing label): {total_missing}")
        print(f"GT boxes total: {total_boxes}")
        print(f"GT boxes dropped (out of bounds): {total_dropped} ({drop_pct:.2f}%)")
        print(f"GT boxes considered for top band: {top_total_boxes}")
        print(
            f"GT boxes dropped from top band: {top_dropped_boxes} ({top_drop_pct:.2f}%)"
        )
        print(f"Outputs stored under: {output_parent}")


if __name__ == "__main__":
    main()
