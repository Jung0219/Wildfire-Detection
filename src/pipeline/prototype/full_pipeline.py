"""
Run the multiresolution detector and filter each detection with an image classifier.

Example:
    python -m src.pipeline.MR+classifier.mr+classifier
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure local src/ is on sys.path before importing project modules.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from src.pipeline.multiresolution.multiresolution import (
    apply_nms,
    detect_skyline_y,
    generate_composite_640x640,
    pad_or_downscale_to_640,
    yolo_to_original,
)
from src.pipeline.two_stage.classifiers import EVAClassifier, YOLOClassifier

# ================= CONFIG =================
DEFAULT_ROOT = Path(os.getenv("FS_ROOT_DIR", REPO_ROOT))

CONFIG = {
    # Data + weights
    "GT_DIR": Path(
        os.getenv(
            "MR_GT_DIR",
            DEFAULT_ROOT / "data/detection/training/AD_phash3_early_smoke/original",
        )
    ),  # data
    "YOLO_MODEL": Path(
        os.getenv(
            "MR_DET_WEIGHTS",
            DEFAULT_ROOT
            / "outputs/yolo/detection/AD_phash3_early_smoke/900_AdamW/train/weights/best.pt",
        )
    ),  # detector
    "CLASSIFIER": os.getenv("MR_CLASSIFIER", "eva"),  # "eva" or "yolo"
    "CLASSIFIER_WEIGHTS": Path(
        os.getenv(
            "MR_CLS_WEIGHTS",
            DEFAULT_ROOT
            / "/lab/projects/fire_smoke_awr/outputs/eva02/AD_phash3_early_smoke/train/weights/best_loss.pt",
        )
    ),
    "OUTPUT_LABEL_DIR": Path(
        os.getenv(
            "MR_OUTPUT_DIR",
            DEFAULT_ROOT
            / "outputs/yolo/detection/AD_phash3_early_smoke/900_AdamW/test/full_pipeline",
        )
    ),
    # Thresholds + resizing
    "INTERMEDIATE_SIZE": 900,
    "NMS_IOU_THRESH": 0.45,
    "CONF_LOW": 0.1,
    "CONF_HIGH": 0.4,
    "CLASSIFIER_CROP_SIZE": (224, 224),  # (height, width) crop centered on detection for classifier
    "ANCHOR_Y_FRAC": 0.6,
    # Debug/visualization
    "DEBUG_FIRST_N": None,  # when set, limit to first N images
    "SAVE_DEBUG_VIS": False,
    "DEBUG_DIR": Path(
        os.getenv(
            "MR_DEBUG_DIR",
            DEFAULT_ROOT / "outputs/yolo/detection/AD_phash3_early_smoke/900_AdamW_lr0001/debug_vis",
        )
    ),
}
# ==========================================

CLASS_NAMES = ["fire", "smoke"]


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_classifier(name: str, weights: str):
    """Factory for classifier wrappers."""
    if name == "eva":
        return EVAClassifier(weights_path=weights, device="cuda")
    if name == "yolo":
        return YOLOClassifier(weights_path=weights)
    raise ValueError(f"Unsupported classifier type: {name}")


def compute_centered_window(
    xc: float, yc: float, crop_size: Tuple[int, int], img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Return pixel window for a fixed-size crop centered on a normalized box center."""
    crop_h, crop_w = crop_size
    width = min(crop_w, img_w)
    height = min(crop_h, img_h)
    cx_px = xc * img_w
    cy_px = yc * img_h

    x1 = int(round(cx_px - width / 2))
    y1 = int(round(cy_px - height / 2))
    x1 = max(0, min(x1, img_w - width))
    y1 = max(0, min(y1, img_h - height))
    x2 = int(x1 + width)
    y2 = int(y1 + height)
    return x1, y1, x2, y2


def filter_with_classifier(
    detections: List[List[float]],
    image,
    classifier,
    conf_low: float,
    conf_high: float,
    crop_size: Tuple[int, int],
    crop_save_dir: Optional[Path] = None,
    img_base: str = "",
) -> List[List[float]]:
    """Gate detections based on confidence and classifier output."""
    if not detections:
        return []

    img_h, img_w = image.shape[:2]
    kept: List[List[float]] = []
    for det_idx, (cls_id, xc, yc, w, h, conf) in enumerate(detections):
        if conf < conf_low:
            continue

        if conf >= conf_high:
            kept.append([cls_id, xc, yc, w, h, conf])
            continue

        x1, y1, x2, y2 = compute_centered_window(xc, yc, crop_size, img_w, img_h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pred_label = classifier.predict(crop)
        if crop_save_dir:
            ensure_dir(crop_save_dir / img_base)
            out_path = (
                crop_save_dir
                / (img_base + f"det{det_idx}_conf{conf:.2f}_{pred_label.lower()}.jpg")
            )
            cv2.imwrite(str(out_path), crop)

        if pred_label.lower() == "background":
            continue
        kept.append([cls_id, xc, yc, w, h, conf])
    return kept


def draw_detections(
    image,
    detections: List[List[float]],
    class_names: Optional[List[str]] = None,
    color=(0, 255, 0),
):
    """Draw normalized detections on a copy of the image."""
    canvas = image.copy()
    img_h, img_w = canvas.shape[:2]
    for cls_id, xc, yc, w, h, conf in detections:
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = (
            class_names[int(cls_id)]
            if class_names and int(cls_id) < len(class_names)
            else str(int(cls_id))
        )
        cv2.putText(
            canvas,
            f"{label}:{conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_pipeline() -> None:
    gt_dir = Path(CONFIG["GT_DIR"])
    image_dir = gt_dir / "images" / "test"

    output_dir = Path(CONFIG["OUTPUT_LABEL_DIR"])
    filtered_dir = output_dir / f"{CONFIG['CLASSIFIER']}_{CONFIG['CONF_LOW']}_{CONFIG['CONF_HIGH']}"

    debug_dir = Path(CONFIG["DEBUG_DIR"])
    debug_composites = debug_dir / "composites"
    debug_crops = debug_dir / "classifier_crops"
    debug_boxes = debug_dir / "original_boxes"

    ensure_dir(filtered_dir)
    if CONFIG["SAVE_DEBUG_VIS"]:
        ensure_dir(debug_composites)
        ensure_dir(debug_crops)
        ensure_dir(debug_boxes)

    model = YOLO(CONFIG["YOLO_MODEL"])
    classifier = init_classifier(
        CONFIG["CLASSIFIER"], str(CONFIG["CLASSIFIER_WEIGHTS"])
    )

    image_files = sorted(
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if CONFIG.get("DEBUG_FIRST_N") is not None:
        image_files = image_files[: int(CONFIG["DEBUG_FIRST_N"])]

    total_dets = 0
    passed_classifier = 0
    skipped_low_conf = 0

    for idx, img_name in enumerate(tqdm(image_files, desc="Running MR+classifier")):
        img_path = image_dir / img_name
        base = img_path.stem
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        img_h, img_w = original.shape[:2]

        # Build composite using same logic as multiresolution pipeline
        if img_h >= img_w or img_h < 640 or img_w < 640:
            composite, (x_off, y_off, scale) = pad_or_downscale_to_640(original, 640)
            meta = {"div_y": 0, "scale_to_640": scale, "x_off": x_off, "y_off": y_off}
        else:
            y_border = detect_skyline_y(original, 120, 255, 0, 130, 5.0)
            obj_center = (0.5, float(y_border) / img_h) if y_border >= 0 else (0.5, 0.5)
            composite, meta = generate_composite_640x640(
                original,
                obj_center,
                CONFIG["INTERMEDIATE_SIZE"],
                anchor_y_frac=CONFIG["ANCHOR_Y_FRAC"],
            )

        if CONFIG["SAVE_DEBUG_VIS"]:
            cv2.imwrite(str(debug_composites / f"{idx+1:03d}_{base}.jpg"), composite)

        yolo_res = model.predict(composite, imgsz=640, conf=0.001, verbose=False)[0]
        dets_top, dets_bottom = [], []
        for box, conf, cls_id in zip(
            yolo_res.boxes.xywhn.cpu().numpy(),
            yolo_res.boxes.conf.cpu().numpy(),
            yolo_res.boxes.cls.cpu().numpy(),
        ):
            is_bottom = (box[1] * 640) >= meta.get("div_y", 0)
            mapped = yolo_to_original(box, meta, conf, cls_id, img_w, img_h, is_bottom)
            (dets_bottom if is_bottom else dets_top).append(mapped)

        merged = dets_bottom + dets_top
        total_dets += len(merged)

        if CONFIG["NMS_IOU_THRESH"] and CONFIG["NMS_IOU_THRESH"] > 0:
            final_dets = apply_nms(merged, CONFIG["NMS_IOU_THRESH"], img_w, img_h)
        else:
            final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in merged]

        filtered = filter_with_classifier(
            final_dets,
            original,
            classifier,
            CONFIG["CONF_LOW"],
            CONFIG["CONF_HIGH"],
            CONFIG["CLASSIFIER_CROP_SIZE"],
            crop_save_dir=debug_crops if CONFIG["SAVE_DEBUG_VIS"] else None,
            img_base=f"{idx+1:03d}_{base}",
        )
        passed_classifier += len(filtered)
        skipped_low_conf += sum(1 for det in final_dets if det[5] < CONFIG["CONF_LOW"])

        if CONFIG["SAVE_DEBUG_VIS"]:
            boxed = draw_detections(original, filtered, CLASS_NAMES)
            cv2.imwrite(str(debug_boxes / f"{idx+1:03d}_{base}.jpg"), boxed)

        out_txt = filtered_dir / f"{base}.txt"
        with open(out_txt, "w") as f:
            for cls_id, xc, yc, w, h, conf in filtered:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

    print("\n=== MR + Classifier Summary ===")
    print(f"Images processed: {len(image_files)}")
    print(f"Detections before filtering: {total_dets}")
    print(f"Skipped for low confidence (<{CONFIG['CONF_LOW']}): {skipped_low_conf}")
    print(f"Detections written after classifier: {passed_classifier}")
    print(f"Output labels saved to: {filtered_dir}")


if __name__ == "__main__":
    run_pipeline()
