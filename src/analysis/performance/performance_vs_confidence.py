import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from collections import defaultdict

# === CONFIG ===
GT_PARENT = "/lab/projects/fire_smoke_awr/data/detection/test_sets/A/deduplicated/phash10/single_objects"
PRED_DIR  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/composites"
SAVE_FIG  = os.path.join(os.path.dirname(PRED_DIR), "composites_metrics.png")

IOU_THRESH = 0.5
MAX_DETS   = 100
# =================================================

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
    if num_gt == 0:
        return 0, [], []

    sorted_indices = np.argsort(-np.array(all_scores))
    matches = np.array(all_matches)[sorted_indices]

    tp = np.cumsum(matches == 1)
    fp = np.cumsum(matches == 0)

    recalls = tp / num_gt
    precisions = tp / (tp + fp + 1e-6)

    # Precision monotone decreasing
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

# Group by class
gts_by_class   = defaultdict(lambda: defaultdict(list))  # gts_by_class[class][image] = list of boxes
preds_by_class = defaultdict(lambda: defaultdict(list))

print("Collecting predictions...")
for img_path in tqdm(image_files):
    fname = os.path.splitext(os.path.basename(img_path))[0]
    gt_boxes   = load_yolo_labels(os.path.join(gt_dir, fname + ".txt"), is_pred=False)
    pred_boxes = load_yolo_labels(os.path.join(PRED_DIR, fname + ".txt"), is_pred=True)

    for gb in gt_boxes:
        gts_by_class[gb[0]][fname].append(gb)

    pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)[:MAX_DETS]
    for pb in pred_boxes:
        preds_by_class[pb[0]][fname].append(pb)

# ---------------- Per-class Eval ----------------
metrics_per_class = {}
aps = {}

for cls in sorted(gts_by_class.keys() | preds_by_class.keys()):
    flat_preds = []
    for img_id, boxes in preds_by_class[cls].items():
        flat_preds.extend((img_id, pb) for pb in boxes)
    flat_preds = sorted(flat_preds, key=lambda x: x[1][5], reverse=True)

    num_gt = sum(len(boxes) for boxes in gts_by_class[cls].values())
    gt_flags = {img_id: [False]*len(gts_by_class[cls][img_id]) for img_id in gts_by_class[cls]}

    all_scores, all_matches = [], []

    for img_id, pb in flat_preds:
        best_iou, best_idx = 0, -1
        for i, gb in enumerate(gts_by_class[cls].get(img_id, [])):
            if gt_flags[img_id][i]:
                continue
            iou_val = iou(pb, gb)
            if iou_val >= IOU_THRESH and iou_val > best_iou:
                best_iou, best_idx = iou_val, i
        if best_idx >= 0:
            all_matches.append(1)
            gt_flags[img_id][best_idx] = True
        else:
            all_matches.append(0)
        all_scores.append(pb[5])

    if num_gt > 0:
        ap, _, _ = compute_coco_ap(all_scores, all_matches, num_gt)
    else:
        ap = 0

    aps[cls] = ap

    # Confidence sweep
    if len(all_scores) > 0:
        sorted_idx = np.argsort(-np.array(all_scores))
        scores_sorted = np.array(all_scores)[sorted_idx]
        matches_sorted = np.array(all_matches)[sorted_idx]

        tp = np.cumsum(matches_sorted == 1)
        fp = np.cumsum(matches_sorted == 0)
        precisions = tp / (tp + fp + 1e-6)
        recalls    = tp / (num_gt + 1e-6)
        f1_scores  = 2 * precisions * recalls / (precisions + recalls + 1e-6)

        metrics_per_class[cls] = {
            "scores": scores_sorted,
            "prec": precisions,
            "rec": recalls,
            "f1": f1_scores
        }
    else:
        metrics_per_class[cls] = {"scores": [], "prec": [], "rec": [], "f1": []}

# ---------------- Plotting ----------------
colors = ["blue", "green", "red", "orange", "purple"]

plt.figure(figsize=(14, 10))

# 1. Confidence vs Precision
plt.subplot(2, 2, 1)
for i, (cls, m) in enumerate(metrics_per_class.items()):
    plt.plot(m["scores"], m["prec"], label=f"Class {cls}", color=colors[i % len(colors)])
if metrics_per_class:
    avg_prec = np.mean([m["prec"] for m in metrics_per_class.values() if len(m["prec"])], axis=0)
    avg_scores = list(metrics_per_class.values())[0]["scores"]
    plt.plot(avg_scores, avg_prec, label="Average", color="black", linestyle="--")
plt.xlabel("Confidence Threshold")
plt.ylabel("Precision")
plt.title("Confidence vs Precision")
plt.legend()
plt.grid(True)

# 2. Confidence vs Recall
plt.subplot(2, 2, 2)
for i, (cls, m) in enumerate(metrics_per_class.items()):
    plt.plot(m["scores"], m["rec"], label=f"Class {cls}", color=colors[i % len(colors)])
if metrics_per_class:
    avg_rec = np.mean([m["rec"] for m in metrics_per_class.values() if len(m["rec"])], axis=0)
    avg_scores = list(metrics_per_class.values())[0]["scores"]
    plt.plot(avg_scores, avg_rec, label="Average", color="black", linestyle="--")
plt.xlabel("Confidence Threshold")
plt.ylabel("Recall")
plt.title("Confidence vs Recall")
plt.legend()
plt.grid(True)

# 3. Confidence vs F1
plt.subplot(2, 2, 3)
for i, (cls, m) in enumerate(metrics_per_class.items()):
    plt.plot(m["scores"], m["f1"], label=f"Class {cls}", color=colors[i % len(colors)])
if metrics_per_class:
    avg_f1 = np.mean([m["f1"] for m in metrics_per_class.values() if len(m["f1"])], axis=0)
    avg_scores = list(metrics_per_class.values())[0]["scores"]
    plt.plot(avg_scores, avg_f1, label="Average", color="black", linestyle="--")
plt.xlabel("Confidence Threshold")
plt.ylabel("F1")
plt.title("Confidence vs F1")
plt.legend()
plt.grid(True)

# 4. Precision-Recall curve
plt.subplot(2, 2, 4)
for i, (cls, m) in enumerate(metrics_per_class.items()):
    plt.plot(m["rec"], m["prec"], label=f"Class {cls} (AP={aps[cls]:.3f})", color=colors[i % len(colors)])
if metrics_per_class:
    avg_rec = np.mean([m["rec"] for m in metrics_per_class.values() if len(m["rec"])], axis=0)
    avg_prec = np.mean([m["prec"] for m in metrics_per_class.values() if len(m["prec"])], axis=0)
    plt.plot(avg_rec, avg_prec, label=f"Average (mAP={np.mean(list(aps.values())):.3f})", color="black", linestyle="--")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=300)
plt.show()

print("Per-class AP:", aps)
print(f"Final mAP@{IOU_THRESH:.2f} = {np.mean(list(aps.values())):.4f}")
