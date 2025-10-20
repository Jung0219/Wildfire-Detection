import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from collections import defaultdict

# === CONFIG ===
GT_PARENT = "/lab/projects/fire_smoke_awr/data/detection/test_sets/A/deduplicated/phash10/single_objects"  # contains images/test and labels/test
PRED_DIR  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/composites"  # YOLO txt files with conf
SAVE_FIG  = "composites_confidence_vs_metrics.png"

IOU_THRESH = 0.5
SAVE_FIG   = os.path.join(os.path.dirname(PRED_DIR), SAVE_FIG)
MAX_DETS   = 100  # COCO uses top-100 per image

# ---------------- Helper Functions ----------------
def load_yolo_labels(path, is_pred=False):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if is_pred:
                if len(parts) != 6: 
                    continue
                cls, x, y, w, h, conf = parts
                boxes.append((int(cls), float(x), float(y), float(w), float(h), float(conf)))
            else:
                if len(parts) != 5: 
                    continue
                cls, x, y, w, h = parts
                boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes

def iou(box1, box2):
    _, x1, y1, w1, h1 = box1[:5]
    _, x2, y2, w2, h2 = box2[:5]

    xa1, ya1 = x1 - w1/2, y1 - h1/2
    xa2, ya2 = x1 + w1/2, y1 + h1/2
    xb1, yb1 = x2 - w2/2, y2 - h2/2
    xb2, yb2 = x2 + w2/2, y2 + h2/2

    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def compute_coco_ap(all_scores, all_matches, num_gt):
    # Sort predictions by confidence (descending)
    sorted_indices = np.argsort(-np.array(all_scores))
    matches = np.array(all_matches)[sorted_indices]

    tp = np.cumsum(matches == 1)
    fp = np.cumsum(matches == 0)

    recalls = tp / num_gt
    precisions = tp / (tp + fp + 1e-6)

    # Precision monotonic decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    prec_at_recalls = []
    for r in recall_points:
        inds = np.where(recalls >= r)[0]
        p = np.max(precisions[inds]) if inds.size > 0 else 0
        prec_at_recalls.append(p)

    ap = np.mean(prec_at_recalls)
    return ap, precisions, recalls

# ---------------- Main Eval ----------------
gt_dir = os.path.join(GT_PARENT, "labels", "test")
image_files = sorted(glob(os.path.join(GT_PARENT, "images", "test", "*.jpg")))

# Store predictions and GTs per class
preds_by_class = defaultdict(list)
gts_by_class   = defaultdict(list)

print("Collecting predictions...")
for img_path in tqdm(image_files):
    fname = os.path.splitext(os.path.basename(img_path))[0]
    gt_boxes   = load_yolo_labels(os.path.join(gt_dir, fname + ".txt"), is_pred=False)
    pred_boxes = load_yolo_labels(os.path.join(PRED_DIR, fname + ".txt"), is_pred=True)

    # Register GTs
    for gb in gt_boxes:
        gts_by_class[gb[0]].append((fname, gb))

    # Register predictions (sorted & truncated to max 100)
    pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)[:MAX_DETS]
    for pb in pred_boxes:
        preds_by_class[pb[0]].append((fname, pb))

# ---------------- Per-class AP ----------------
aps = []
for cls, preds in preds_by_class.items():
    preds = sorted(preds, key=lambda x: x[1][5], reverse=True)  # sort by conf

    # Prepare GT match flags
    gt_flags = {i: False for i in range(len(gts_by_class[cls]))}
    gt_images = [g[0] for g in gts_by_class[cls]]
    gt_boxes  = [g[1] for g in gts_by_class[cls]]
    num_gt    = len(gt_boxes)

    all_scores, all_matches = [], []

    for img_id, pb in preds:
        best_iou, best_idx = 0, -1
        for i, (gb_img, gb) in enumerate(zip(gt_images, gt_boxes)):
            if gb_img != img_id:  # must be same image
                continue
            if gt_flags[i]:  # already matched
                continue
            iou_val = iou(pb, gb)
            if iou_val >= IOU_THRESH and iou_val > best_iou:
                best_iou, best_idx = iou_val, i
        if best_idx >= 0:
            all_matches.append(1)  # TP
            gt_flags[best_idx] = True
        else:
            all_matches.append(0)  # FP
        all_scores.append(pb[5])

    if num_gt > 0:
        ap, _, _ = compute_coco_ap(all_scores, all_matches, num_gt)
        aps.append(ap)
        print(f"Class {cls}: AP@{IOU_THRESH:.2f} = {ap:.4f}")

# ---------------- Final mAP ----------------
mAP = np.mean(aps) if aps else 0
print(f"COCO-style mAP@{IOU_THRESH:.2f}: {mAP:.4f}")