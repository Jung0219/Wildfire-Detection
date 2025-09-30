#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze object sizes from YOLO prediction files with letterbox scaling.
Computes box side lengths and areas in 640-scale pixels,
where the longer side of the original image is scaled to imgsz
and the shorter side is letterboxed.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# === CONFIGURATION ===
pred_dir = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_original_single_objects/composites/tp_below_0.163"   # YOLO txt files
img_dir  = "/lab/projects/fire_smoke_awr/data/detection/test_sets/A/original/single_objects/images/test"  # original images
imgsz = 640
save_plot = pred_dir + "/plot/object_size_distribution.png"
os.makedirs(os.path.dirname(save_plot), exist_ok=True)


def get_letterbox_dims(img_path, imgsz=640):
    """Compute resized width/height after letterbox scaling to imgsz."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = imgsz / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return new_w, new_h


def load_pred_boxes(file_path, img_path, imgsz=640):
    """
    Load YOLO predictions from a file, compute box sizes in letterboxed 640-scale.
    Format: cls cx cy w h conf (normalized by original image size).
    Returns list of (cls, w_px, h_px, area_px, side_px, conf).
    """
    dims = get_letterbox_dims(img_path, imgsz)
    if dims is None:
        return []
    new_w, new_h = dims

    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            cls, cx, cy, w, h, conf = parts
            w, h = float(w), float(h)

            w_px = w * new_w
            h_px = h * new_h
            area_px = w_px * h_px
            side_px = np.sqrt(area_px)
            boxes.append((int(cls), w_px, h_px, area_px, side_px, float(conf)))
    return boxes


# === Collect all objects ===
records = []
txt_files = glob(os.path.join(pred_dir, "*.txt"))

for txt_file in tqdm(txt_files):
    fname = os.path.splitext(os.path.basename(txt_file))[0]

    # look for matching image (try common extensions)
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        cand = os.path.join(img_dir, fname + ext)
        if os.path.exists(cand):
            img_path = cand
            break
    if img_path is None:
        continue

    preds = load_pred_boxes(txt_file, img_path, imgsz=imgsz)
    for (cls, w_px, h_px, area_px, side_px, conf) in preds:
        records.append({
            "image_id": fname,
            "class": cls,
            "w_px": w_px,
            "h_px": h_px,
            "area_px": area_px,
            "side_px": side_px,
            "conf": conf
        })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} objects from {len(df['image_id'].unique())} images.")

# === Summary statistics ===
print("\n=== Object Size Summary (letterboxed to 640) ===")
print(df[["w_px","h_px","area_px","side_px"]].describe())

# === Histogram plots ===
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.hist(df["w_px"], bins=40, color="blue", alpha=0.7)
plt.xlabel("Box width (px)")
plt.ylabel("Count")

plt.subplot(2,2,2)
plt.hist(df["h_px"], bins=40, color="green", alpha=0.7)
plt.xlabel("Box height (px)")
plt.ylabel("Count")

plt.subplot(2,2,3)
plt.hist(df["area_px"], bins=40, color="orange", alpha=0.7)
plt.xlabel("Box area (px²)")
plt.ylabel("Count")

plt.subplot(2,2,4)
plt.hist(df["side_px"], bins=40, color="purple", alpha=0.7)
plt.xlabel("Box side length sqrt(area) (px)")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig(save_plot, dpi=300)
plt.close()
print(f"\nSaved histogram plot to {save_plot}")
