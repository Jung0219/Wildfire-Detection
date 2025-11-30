"""
Generate square composite images (plus YOLO labels) across train/val splits,
producing multiple horizontal-crop variations per image for augmentation review.

Example:
    python src/data_manipulation/composite_generation/generate_composite.py
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
    "INPUT_PARENT": "/lab/projects/fire_smoke_awr/data/detection/training/AD_phash3_early_smoke/original",
    "OUTPUT_PARENT": "/lab/projects/fire_smoke_awr/data/detection/training/AD_phash3_early_smoke/composite/1100",
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
    "INTERMEDIATE_SIZE": 1100,
    # Choose which bounding box to influence the crop window (0 = first box)
    "PRIMARY_BOX_INDEX": 0,
    # Where should that box land horizontally within the crop? (0=left edge, 1=right edge)
    "HORIZONTAL_FOCI": [0.25, 0.50, 0.75],
    # Optional limit if you only want the first N images (None = all)
    "MAX_SAMPLES": None,
}
# ====================================================


@dataclass
class YoloBox:
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
    if not label_path.exists():
        return []
    with label_path.open() as f:
        return [box for line in f if (box := YoloBox.from_line(line))]
    """
    boxes = []
        for line in f:
            box = YoloBox.from_line(line)
            if box:
                boxes.append(box)
        return boxes
    """


def save_boxes(boxes: Sequence[YoloBox], label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        for box in boxes:
            f.write(box.to_line() + "\n")


def draw_boxes(image: np.ndarray, boxes: Sequence[YoloBox], out_path: Path) -> None:
    vis = image.copy()
    colors = [
        (255, 128, 64),
        (64, 192, 255),
        (128, 255, 128),
        (255, 64, 192),
        (255, 255, 0),
    ]

    for idx, box in enumerate(boxes):
        xc = box.xc * vis.shape[1]
        yc = box.yc * vis.shape[0]
        w = box.w * vis.shape[1]
        h = box.h * vis.shape[0]
        x1 = int(round(xc - w / 2))
        y1 = int(round(yc - h / 2))
        x2 = int(round(xc + w / 2))
        y2 = int(round(yc + h / 2))
        color = colors[idx % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{box.cls}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def to_pixels(box: YoloBox, width: int, height: int) -> Tuple[float, float, float, float]:
    return (
        box.xc * width,
        box.yc * height,
        box.w * width,
        box.h * height,
    )


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def list_image_files(image_dir: Path, extensions: Sequence[str]) -> List[Path]:
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


def build_bottom_band(
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    canvas: np.ndarray,
    canvas_size: int,
) -> Tuple[np.ndarray, List[YoloBox], float, int, int]:
    # create bottom band, place the resized images
    orig_h, orig_w = image.shape[:2]
    longest = max(orig_w, orig_h)
    scale = canvas_size / longest
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    resized = cv2.resize(image, (new_w, new_h))

    x_off = (canvas_size - new_w) // 2
    y_off = canvas_size - new_h
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    # adjust yolo labels
    transformed: List[YoloBox] = []
    for box in tqdm(boxes, desc="Transforming bottom boxes", leave=False):
        xc_px, yc_px, w_px, h_px = to_pixels(box, orig_w, orig_h)
        xc_canvas = (xc_px * scale + x_off) / canvas_size
        yc_canvas = (yc_px * scale + y_off) / canvas_size
        w_canvas = (w_px * scale) / canvas_size
        h_canvas = (h_px * scale) / canvas_size
        transformed.append(
            YoloBox(cls=box.cls, xc=xc_canvas, yc=yc_canvas, w=w_canvas, h=h_canvas)
        )

    top_height = y_off
    return canvas, transformed, scale, x_off, top_height


def pad_to_size(image: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    pad_x = max(0, target_w - w)
    pad_y = max(0, target_h - h)
    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, left, top


def build_top_band(
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    canvas: np.ndarray,
    canvas_size: int,
    top_height: int,
    intermediate_size: int,
    primary_box_idx: int,
    horizontal_focus: float,
) -> Tuple[np.ndarray, List[YoloBox]]:
    if top_height <= 0:
        return canvas, []

    orig_h, orig_w = image.shape[:2]
    longest = max(orig_w, orig_h)
    scale_inter = intermediate_size / longest
    inter_w, inter_h = int(round(orig_w * scale_inter)), int(round(orig_h * scale_inter))
    intermediate = cv2.resize(image, (inter_w, inter_h))

    padded, pad_left, pad_top = pad_to_size(intermediate, canvas_size, top_height)
    padded_h, padded_w = padded.shape[:2]

    target_idx = clamp(primary_box_idx, 0, max(0, len(boxes) - 1))
    if boxes:
        box = boxes[int(target_idx)]
        xc_px, yc_px, _, _ = to_pixels(box, orig_w, orig_h)
        obj_x = xc_px * scale_inter + pad_left
        obj_y = yc_px * scale_inter + pad_top
    else:
        obj_x, obj_y = padded_w / 2.0, padded_h / 2.0

    crop_w = canvas_size
    crop_h = top_height
    focus = clamp(horizontal_focus, 0.0, 1.0)
    crop_x1 = int(round(obj_x - focus * crop_w))
    crop_y1 = int(round(obj_y - crop_h / 2))
    crop_x1 = int(clamp(crop_x1, 0, max(0, padded_w - crop_w)))
    crop_y1 = int(clamp(crop_y1, 0, max(0, padded_h - crop_h)))
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    cropped = padded[crop_y1:crop_y2, crop_x1:crop_x2]

    canvas[0:top_height, 0:canvas_size] = cropped

    transformed: List[YoloBox] = []
    for box in tqdm(boxes, desc="Transforming top boxes", leave=False):
        xc_px, yc_px, w_px, h_px = to_pixels(box, orig_w, orig_h)
        xc_scaled = xc_px * scale_inter + pad_left
        yc_scaled = yc_px * scale_inter + pad_top
        x1 = xc_scaled - (w_px * scale_inter) / 2
        y1 = yc_scaled - (h_px * scale_inter) / 2
        x2 = xc_scaled + (w_px * scale_inter) / 2
        y2 = yc_scaled + (h_px * scale_inter) / 2

        x1_crop = max(0, x1 - crop_x1)
        x2_crop = min(crop_w, x2 - crop_x1)
        y1_crop = max(0, y1 - crop_y1)
        y2_crop = min(crop_h, y2 - crop_y1)

        if x2_crop <= x1_crop or y2_crop <= y1_crop:
            continue

        xc_canvas = ((x1_crop + x2_crop) / 2) / canvas_size
        yc_canvas = ((y1_crop + y2_crop) / 2) / canvas_size
        w_canvas = (x2_crop - x1_crop) / canvas_size
        h_canvas = (y2_crop - y1_crop) / canvas_size

        transformed.append(
            YoloBox(cls=box.cls, xc=xc_canvas, yc=yc_canvas, w=w_canvas, h=h_canvas)
        )

    return canvas, transformed


def process_sample(
    image_path: Path,
    label_path: Path,
    out_image_dir: Path,
    out_label_dir: Path,
    config: dict,
) -> List[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    boxes = load_boxes(label_path)

    canvas_size = int(config["CANVAS_SIZE"])
    intermediate_size = int(config["INTERMEDIATE_SIZE"])
    primary_idx = int(config["PRIMARY_BOX_INDEX"])
    horizontal_foci = config.get("HORIZONTAL_FOCI", [0.5])

    base_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=image.dtype)
    base_canvas, bottom_boxes, _, _, top_height = build_bottom_band(
        image, boxes, base_canvas, canvas_size
    )

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for focus in horizontal_foci:
        variant_canvas = base_canvas.copy()
        variant_canvas, top_boxes = build_top_band(
            image,
            boxes,
            variant_canvas,
            canvas_size,
            top_height,
            intermediate_size,
            primary_idx,
            focus,
        )

        suffix = f"focus{int(round(focus * 100)):02d}"
        variant_image_path = out_image_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
        variant_label_path = out_label_dir / f"{image_path.stem}_{suffix}.txt"
        combined_boxes = top_boxes + bottom_boxes
        cv2.imwrite(str(variant_image_path), variant_canvas)
        save_boxes(combined_boxes, variant_label_path)

        results.append(
            {
                "focus": focus,
                "image": variant_image_path,
                "label": variant_label_path,
                "top_height": top_height,
            }
        )
    return results


def process_split(
    input_parent: Path,
    output_parent: Path,
    split_config: dict,
    config: dict,
) -> dict:
    split_name = split_config.get("name", "split")
    image_dir = input_parent / split_config["image_subdir"]
    label_dir = input_parent / split_config["label_subdir"]
    out_image_dir = output_parent / split_config["image_subdir"]
    out_label_dir = output_parent / split_config["label_subdir"]

    if not image_dir.exists():
        raise FileNotFoundError(f"[{split_name}] Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"[{split_name}] Label directory not found: {label_dir}")

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(image_dir, config["IMAGE_EXTENSIONS"])
    if not image_files:
        print(f"[{split_name}] No images found matching extensions; skipping.")
        return {
            "split": split_name,
            "processed_images": 0,
            "missing_labels": 0,
            "composites": 0,
        }

    max_samples = config.get("MAX_SAMPLES")
    if max_samples is not None:
        image_files = image_files[: int(max_samples)]

    total_composites = 0
    missing_labels = 0
    desc = f"Processing {split_name} images"

    for img_path in tqdm(image_files, desc=desc):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[{split_name}] Warning: missing label for {img_path.name}, skipping.")
            missing_labels += 1
            continue

        results = process_sample(img_path, label_path, out_image_dir, out_label_dir, config)
        total_composites += len(results)

    processed_images = len(image_files) - missing_labels
    print(f"\n[{split_name}] Summary")
    print(f"Images processed: {processed_images}")
    print(f"Images skipped (missing label): {missing_labels}")
    print(f"Composites generated: {total_composites}")
    print(f"Output images dir: {out_image_dir}")
    print(f"Output labels dir: {out_label_dir}")
    return {
        "split": split_name,
        "processed_images": processed_images,
        "missing_labels": missing_labels,
        "composites": total_composites,
    }


def main() -> None:
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
        total_composites = sum(s["composites"] for s in summaries)
        print("\n=== Aggregate Summary ===")
        print(f"Images processed: {total_processed}")
        print(f"Images skipped (missing label): {total_missing}")
        print(f"Composites generated: {total_composites}")
        print(f"Outputs stored under: {output_parent}")


if __name__ == "__main__":
    main()
