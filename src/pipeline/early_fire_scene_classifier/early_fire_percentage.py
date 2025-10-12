#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==================== CONFIG ====================
IMAGE_DIR = "/lab/projects/fire_smoke_awr/data/detection/test_sets/ABCDE_noEF_10%"
Y_MIN = 80          # Minimum Y value (luminance)
Y_MAX = 255  
CB_MIN = 0         # Minimum Cb value for boundary
CB_MAX = 140        # Maximum Cb value for boundary
CR_MIN = 150         # Minimum Cr value for boundary
CR_MAX = 255         # Maximum Cr value for boundary
RATIO_TH = 0.05   # early fire threshold
# ================================================

def use_composite(img_bgr,
                  cb_min=CB_MIN, cb_max=CB_MAX,
                  cr_min=CR_MIN, cr_max=CR_MAX,
                  y_min=Y_MIN, y_max=Y_MAX,
                  ratio_thresh=RATIO_TH):
    """
    Use composite only if fire pixel ratio < ratio_thresh
    => possible early fire scene
    """
    H, W = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    m_full = ((Y >= y_min) & (Y <= y_max) &
              (Cb >= cb_min) & (Cb <= cb_max) &
              (Cr >= cr_min) & (Cr <= cr_max)).astype(np.uint8)

    ratio = m_full.sum() / (H * W)
    return ratio < ratio_thresh

def main():
    img_paths = [p for p in Path(IMAGE_DIR).rglob("*")  
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]

    if not img_paths:
        print("No images found in", IMAGE_DIR)
        return

    total = len(img_paths)
    qualified = 0

    for p in tqdm(img_paths, desc="Scanning images"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        if use_composite(img):
            qualified += 1

    perc = (qualified / total) * 100
    print(f"\nTotal images: {total}")
    print(f"Qualified (early fire): {qualified}")
    print(f"Percentage: {perc:.2f}%")

if __name__ == "__main__":
    main()
