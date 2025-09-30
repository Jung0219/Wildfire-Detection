#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage YOLO + classifier pipeline with confidence fusion strategies.
Selective mode: apply classifier only to detections with conf < CONF_THRESH.
"""

import os
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# ========== CONFIG ==========
IMAGES_DIR   = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire"
PRED_DIR     = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/new_model/early_fire/composites_wbf"
SAVE_DIR     = "wbf"
CLS_MODEL    = "/lab/projects/fire_smoke_awr/outputs/yolo/classification/fp_mined/train/weights/best.pt"
DEVICE       = 0
IMGSZ        = 224
CONF_THRESH  = None   # apply classifier only if det_conf < CONF_THRESH
FUSION_METHOD = "cls_only"   # choose from: cls_only, det_mul_cls, temp_cls, weighted_avg, logreg, temp_logreg
# =============================
SAVE_DIR = Path(PRED_DIR) / SAVE_DIR / FUSION_METHOD
IMAGES_DIR = Path(IMAGES_DIR) / "images/test"
# ---------- Helpers ----------

def yolo_to_pixels(x, y, w, h, img_w, img_h):
    """Convert normalized YOLO xywh to pixel coordinates (x1,y1,x2,y2)."""
    cx, cy = x * img_w, y * img_h
    pw, ph = w * img_w, h * img_h
    x1 = max(int(cx - pw / 2), 0)
    y1 = max(int(cy - ph / 2), 0)
    x2 = min(int(cx + pw / 2), img_w - 1)
    y2 = min(int(cy + ph / 2), img_h - 1)
    return x1, y1, x2, y2


def crop_image(image, bbox):
    """Crop image using (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def classify_crop(classifier, crop, imgsz=224, device=0):
    """Run classifier on crop and return (class id, confidence)."""
    if crop.size == 0:
        return None, None
    results = classifier.predict(crop, imgsz=imgsz, device=device, verbose=False)
    top1 = int(results[0].probs.top1)
    conf = float(results[0].probs.data[top1])  # probability of predicted class
    return top1, conf

def refine_with_classifier(classifier, crop, imgsz=224, device=0):
    """
    Run classifier on crop. 
    - If output is background (class 0), return (None, None) to discard box.
    - If fire (1) or smoke (2), return mapped YOLO class + confidence.
    """
    if crop.size == 0:
        return None, None

    cls_out, cls_conf = classify_crop(classifier, crop, imgsz, device)
    if cls_out is None:
        return None, None

    final_cls = map_class(cls_out)
    if final_cls is None:  # background
        return None, None

    return final_cls, cls_conf


def map_class(cls_out):
    """Map classifier output to final YOLO class. Return None if dropped."""
    if cls_out == 0:   # background
        return None
    elif cls_out == 1: # fire
        return 0
    elif cls_out == 2: # smoke
        return 1
    return None

def get_box_size_640(x, y, bw, bh, img_w, img_h, target=640):
    """
    Compute box size (sqrt of area) after letterboxing to 640.
    YOLO box format: normalized (x, y, w, h).
    """
    # Scale ratio (preserving aspect ratio)
    scale = min(target / img_w, target / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)

    # Scale box to resized dimensions
    bw_scaled = bw * new_w
    bh_scaled = bh * new_h

    # Compute size = sqrt(area)
    return (bw_scaled * bh_scaled) ** 0.5


# ================= Confidence Fusion Functions =================
def conf_brute_cls(det_conf, cls_conf):
    return cls_conf

def conf_brute_det_mul_cls(det_conf, cls_conf):
    return det_conf * cls_conf

def conf_temp_scaled_cls(det_conf, cls_conf, T=1.2543):
    p = max(min(cls_conf, 1 - 1e-6), 1e-6)  # clamp
    odds = (p / (1 - p)) ** (1 / T)
    return odds / (1 + odds)

def conf_weighted_avg(det_conf, cls_conf, a=0.9074):
    return a * det_conf + (1 - a) * cls_conf

def conf_logistic_reg(det_conf, cls_conf,
                      w1=5.9254198448270206,
                      w2=2.323336641431169,
                      b=-4.459283716646686):
    z = w1 * det_conf + w2 * cls_conf + b
    return 1 / (1 + pow(2.718281828, -z))

def conf_temp_scaled_logreg(det_conf, cls_conf,
                            T=1.2543,
                            w1=5.851216496433655,
                            w2=2.4262028129301187,
                            b=-4.4983072507311705):
    p = max(min(cls_conf, 1 - 1e-6), 1e-6)
    odds = (p / (1 - p)) ** (1 / T)
    cls_cal = odds / (1 + odds)
    z = w1 * det_conf + w2 * cls_cal + b
    return 1 / (1 + pow(2.718281828, -z))
# ----------------------------------------------------------------

FUSION_FUNCS = {
    "cls_only": conf_brute_cls,
    "det_mul_cls": conf_brute_det_mul_cls,
    "temp_cls": conf_temp_scaled_cls,
    "weighted_avg": conf_weighted_avg,
    "logreg": conf_logistic_reg,
    "temp_logreg": conf_temp_scaled_logreg,
}

# =============================

def main():
    # Load classifier
    classifier = YOLO(CLS_MODEL)
    print(f"[INFO] Loaded classifier with {len(classifier.names)} classes: {classifier.names}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    fusion_func = FUSION_FUNCS[FUSION_METHOD]

    # Process each image
    for img_path in tqdm(list(Path(IMAGES_DIR).glob("*.jpg")), desc="Processing"):
        stem = img_path.stem
        txt_path = Path(PRED_DIR) / f"{stem}.txt"
        if not txt_path.exists():
            continue

        image = cv2.imread(str(img_path))
        img_h, img_w = image.shape[:2]

        new_lines = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) not in (5, 6):
                    continue
                cls, x, y, bw, bh, det_conf = parts

                cls = int(float(cls))
                x, y, bw, bh, det_conf = map(float, (x, y, bw, bh, det_conf))

                box_size = get_box_size_640(x, y, bw, bh, img_w, img_h, target=640)

                final_cls = None
                final_conf = det_conf

                if box_size < 100:   # only refine small objects
                    bbox = yolo_to_pixels(x, y, bw, bh, img_w, img_h)
                    crop = crop_image(image, bbox)
                    cls_out, cls_conf = classify_crop(classifier, crop, IMGSZ, DEVICE)
                    final_cls = map_class(cls_out)

                    if final_cls is None:
                        continue

                    final_conf = fusion_func(det_conf, cls_conf)
                else:
                    final_cls = cls

                new_lines.append(
                    f"{final_cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {final_conf:.2f}\n"
                )

        if new_lines:
            save_path = Path(SAVE_DIR) / f"{stem}.txt"
            with open(save_path, "w") as f:
                f.writelines(new_lines)


if __name__ == "__main__":
    main()
    print("saved to ", SAVE_DIR)
