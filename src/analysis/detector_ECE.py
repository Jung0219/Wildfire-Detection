#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reliability analysis (ECE + reliability diagram) for YOLO predictions.
Matches predictions to GT, collects (confidence, TP/FP),
and evaluates calibration quality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict

# ================= CONFIG =================
PARENT_DIR     = "/lab/projects/fire_smoke_awr/data/detection/test_sets/A/original/single_objects_lt_80"
PRED_LABEL_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_original_single_objects_lt_80/composites"

IOU_THRESH     = 0.5
CLASS_AWARE    = False   # if True, match only same-class
IMG_EXTS       = [".jpg", ".png", ".jpeg"]

IMG_DIR      = os.path.join(PARENT_DIR, "images/test")
GT_LABEL_DIR = os.path.join(PARENT_DIR, "labels/test")
save_path    = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_original_single_objects_lt_80/composites/plots/reliability_diagram.png"

# ==========================================

def load_yolo_labels(path, is_pred=False):
    """Load YOLO-format labels."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if is_pred:
                if len(parts) != 6:  # cls cx cy w h conf
                    continue
                cls, cx, cy, w, h, conf = parts
                boxes.append((int(cls), float(cx), float(cy),
                              float(w), float(h), float(conf)))
            else:
                if len(parts) != 5:  # cls cx cy w h
                    continue
                cls, cx, cy, w, h = parts
                boxes.append((int(cls), float(cx), float(cy),
                              float(w), float(h)))
    return boxes

def iou(box1, box2):
    """IoU between two YOLO-format boxes (normalized)."""
    _, x1, y1, w1, h1 = box1[:5]
    _, x2, y2, w2, h2 = box2[:5]

    xa1, ya1 = x1 - w1/2, y1 - h1/2
    xa2, ya2 = x1 + w1/2, y1 + h1/2
    xb1, yb1 = x2 - w2/2, y2 - h2/2
    xb2, yb2 = x2 + w2/2, y2 + h2/2

    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

# === Step 1: Gather all predictions and GTs ===
scores, labels = [], []

gt_files = {os.path.splitext(os.path.basename(f))[0]: f
            for f in glob(os.path.join(GT_LABEL_DIR, "*.txt"))}

pred_files = {os.path.splitext(os.path.basename(f))[0]: f
              for f in glob(os.path.join(PRED_LABEL_DIR, "*.txt"))}

for img_id, gt_file in gt_files.items():
    gts = load_yolo_labels(gt_file, is_pred=False)
    preds = load_yolo_labels(pred_files.get(img_id, ""), is_pred=True)

    used_gt = [False] * len(gts)
    preds = sorted(preds, key=lambda b: -b[5])  # sort by conf

    for pb in preds:
        best_i, best_iou = -1, 0.0
        for i, gb in enumerate(gts):
            if used_gt[i]:
                continue
            if CLASS_AWARE and pb[0] != gb[0]:
                continue
            iou_val = iou(pb, gb)
            if iou_val >= IOU_THRESH and iou_val > best_iou:
                best_i, best_iou = i, iou_val
        if best_i >= 0:
            labels.append(1)          # TP
            used_gt[best_i] = True
        else:
            labels.append(0)          # FP
        scores.append(pb[5])

scores = np.array(scores)
labels = np.array(labels)

print(f"Collected {len(scores)} predictions, {labels.sum()} TP, {(labels==0).sum()} FP")

# === Step 2: Reliability diagram + ECE ===
def reliability_diagram(scores, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    bin_acc, bin_conf, bin_count = [], [], []

    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i+1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = scores[mask].mean()
        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_count.append(mask.sum())

    # Expected Calibration Error (ECE)
    total = len(scores)
    ece = sum(c/total * abs(a - m) for a,m,c in zip(bin_acc, bin_conf, bin_count))

    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],'k--',label="Perfectly calibrated")
    plt.plot(bin_conf, bin_acc, marker='o', label="Model")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.3f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)

    return ece

ece = reliability_diagram(scores, labels, n_bins=10)
print(f"ECE = {ece:.4f}")
