#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run 3-class fire/smoke classifier (EVA02) on cropped YOLO detections.

- Class 0 = background
- Class 1 = fire
- Class 2 = smoke
- At inference: 
    - classifier confidence = P(fire) + P(smoke)
    - temp_conf = classifier confidence after temperature scaling
    - logreg_conf = calibrated probability from logistic regression on (det_conf, cls_conf)
- Saves CSV with:
  [image, det_conf, cls_conf, temp_conf, logreg_conf, label, logit_bg, logit_fire, logit_smoke]
"""
import sys
sys.path.append("/lab/projects/fire_smoke_awr")  # project root

import os
import csv
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image

from src.models.eva02.eva02_model import EVA02Classifier

# ========== CONFIG ==========
LABEL_DIR     = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/A_phash10_single_objects_pred_lt_100_merged_labels"
IMAGE_DIR     = "/lab/projects/fire_smoke_awr/data/detection/test_sets/A/deduplicated/phash10/single_objects/images/test"
CSV_PATH      = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/A_phash10_single_objects_pred_lt_100.csv"
MODEL_WEIGHTS = "/lab/projects/fire_smoke_awr/outputs/eva02/BCDE_val_set_letterbox/train/weights/best.pt"
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_SIZE      = 224
NUM_CLASSES   = 3   # background=0, fire=1, smoke=2
T             = 1.2542  # learned temperature
# Logistic regression calibration weights
W1, W2, B     = 5.9254198448270206, 2.323336641431169, -4.459283716646686
# ============================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logreg_conf_fn(det_conf, cls_conf):
    """Calibrated confidence using logistic regression weights."""
    return sigmoid(W1 * det_conf + W2 * cls_conf + B)

# load classifier
classifier = EVA02Classifier(
    model_name="eva02_base_patch16_clip_224",
    num_classes=NUM_CLASSES,
    pretrained=False,           # load trained weights
    transform="letterbox",      # or "centerpad"
    img_size=IMG_SIZE
).to(DEVICE)

classifier.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
classifier.eval()

transform = classifier.get_transform()

rows = []
label_files = glob(os.path.join(LABEL_DIR, "*.txt"))

for lf in tqdm(label_files, desc="Processing label files"):
    img_name = os.path.splitext(os.path.basename(lf))[0] + ".jpg"
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    with open(lf, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f" Cropping + classifying {img_name}", leave=False):
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls, x, y, w, h, det_conf = parts
        cls = int(cls)  # relabeled ground-truth: 1=TP, 0=FP
        x, y, w, h, det_conf = map(float, [x, y, w, h, det_conf])

        # convert YOLO format -> pixel box
        h_img, w_img = img.shape[:2]
        x_c, y_c = x * w_img, y * h_img
        bw, bh = w * w_img, h * h_img
        x1, y1 = int(x_c - bw/2), int(y_c - bh/2)
        x2, y2 = int(x_c + bw/2), int(y_c + bh/2)

        crop = img[max(0,y1):min(h_img,y2), max(0,x1):min(w_img,x2)]
        if crop.size == 0:
            continue

        # convert crop -> PIL for transform
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = classifier(crop_tensor)  # shape [1, 3]
            logit_vals = logits.cpu().numpy()[0]  # raw logits

            # uncalibrated probabilities
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            cls_conf = (probs[1] + probs[2]).clip(0, 1)

            # temperature-scaled probabilities
            logits_scaled = logits / T
            probs_scaled = F.softmax(logits_scaled, dim=1).cpu().numpy()[0]
            temp_conf = (probs_scaled[1] + probs_scaled[2]).clip(0, 1)

            # logistic regression calibrated confidence
            logreg_conf = logreg_conf_fn(det_conf, cls_conf)

        rows.append([
            img_name,
            det_conf,
            cls_conf,
            temp_conf,
            logreg_conf,
            cls,
            logit_vals[0], logit_vals[1], logit_vals[2]
        ])

# save to CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image", "det_conf", "cls_conf", "temp_conf", "logreg_conf",
        "label", "logit_bg", "logit_fire", "logit_smoke"
    ])
    writer.writerows(rows)

print(f"✅ Saved {len(rows)} entries with logits + temp_conf + logreg_conf to {CSV_PATH}")
