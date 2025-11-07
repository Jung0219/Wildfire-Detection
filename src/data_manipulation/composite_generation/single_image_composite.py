"""
Generate a square composite image (and updated YOLO labels) for a single sample.

Example:
    python src/data_manipulation/composite_generation/single_image_composite.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ====================== CONFIG ======================
CONFIG = {
    # Paths for the single sample you want to process
    "IMAGE_PATH": "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/A/deduplicated/dedup_phash10/images/bothFireAndSmoke_UAV000000.jpg",
    "LABEL_PATH": "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/A/deduplicated/dedup_phash10/labels/bothFireAndSmoke_UAV000000.txt",
    # Where to save the composite image + labels
    "OUT_IMAGE_PATH": "/lab/projects/fire_smoke_awr/outputs/composite_generation/sample_composite.jpg",
    "OUT_LABEL_PATH": "/lab/projects/fire_smoke_awr/outputs/composite_generation/sample_composite.txt",
    # Canvas and intermediate sizing
    "CANVAS_SIZE": 640,
    "INTERMEDIATE_SIZE": 1024,
    # Choose which bounding box to center in the crop window (0 = first box)
    "PRIMARY_BOX_INDEX": 0,
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


def build_bottom_band(
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    canvas: np.ndarray,
    canvas_size: int,
) -> Tuple[np.ndarray, List[YoloBox], float, int, int]:
    orig_h, orig_w = image.shape[:2]
    longest = max(orig_w, orig_h)
    scale = canvas_size / longest
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    resized = cv2.resize(image, (new_w, new_h))

    x_off = (canvas_size - new_w) // 2
    y_off = canvas_size - new_h
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

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
    crop_x1 = int(round(obj_x - crop_w / 2))
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
        xc_crop = xc_scaled - crop_x1
        yc_crop = yc_scaled - crop_y1
        if not (0 <= xc_crop <= crop_w and 0 <= yc_crop <= crop_h):
            continue

        w_scaled = w_px * scale_inter
        h_scaled = h_px * scale_inter
        w_canvas = w_scaled / canvas_size
        h_canvas = h_scaled / canvas_size
        xc_canvas = xc_crop / canvas_size
        yc_canvas = yc_crop / canvas_size
        transformed.append(
            YoloBox(cls=box.cls, xc=xc_canvas, yc=yc_canvas, w=w_canvas, h=h_canvas)
        )

    return canvas, transformed


def main() -> None:
    print("Running composite generation with CONFIG:")
    print(json.dumps(CONFIG, indent=2))

    image_path = Path(CONFIG["IMAGE_PATH"]).expanduser()
    label_path = Path(CONFIG["LABEL_PATH"]).expanduser()
    out_image_path = Path(CONFIG["OUT_IMAGE_PATH"]).expanduser()
    out_label_path = Path(CONFIG["OUT_LABEL_PATH"]).expanduser()
    canvas_size = int(CONFIG["CANVAS_SIZE"])
    intermediate_size = int(CONFIG["INTERMEDIATE_SIZE"])
    primary_idx = int(CONFIG["PRIMARY_BOX_INDEX"])

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    boxes = load_boxes(label_path)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=image.dtype)
    canvas, bottom_boxes, _, _, top_height = build_bottom_band(image, boxes, canvas, canvas_size)
    canvas, top_boxes = build_top_band(
        image,
        boxes,
        canvas,
        canvas_size,
        top_height,
        intermediate_size,
        primary_idx,
    )

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image_path), canvas)

    combined_boxes = top_boxes + bottom_boxes
    save_boxes(combined_boxes, out_label_path)
    debug_path = out_image_path.with_name(out_image_path.stem + "_boxes.jpg")
    draw_boxes(canvas, combined_boxes, debug_path)

    print(f"Composite saved to: {out_image_path}")
    print(f"Wrote {len(combined_boxes)} labels to: {out_label_path}")
    print(f"Box visualization saved to: {debug_path}")
    print(f"Top band height: {top_height}px; bottom height: {canvas_size - top_height}px")


if __name__ == "__main__":
    main()
