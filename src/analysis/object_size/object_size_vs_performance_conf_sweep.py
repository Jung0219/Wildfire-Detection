import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# === CONFIGURATION ===
parent_dir = "/lab/projects/fire_smoke_awr/data/detection/test_sets/A/deduplicated/phash10/single_objects"  # contains images/test and labels/test

pred_dirs = {
    "Original": {
        "dir": "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/labels",
        "conf": 0
    },
    "Composites": {
        "dir": "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/composites",
        "conf": 0
    }
}

save_path        = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/composites/object_size_vs_performance_fixed_bins.png"
smoothing_sigma  = 0.3
iou_thresh       = 0.5

# Bin config (side length in pixels)
min_size_px  = 0
max_size_px  = 640
num_bins     = 30

# === Directories ===
img_dir = os.path.join(parent_dir, "images", "test")
gt_dir  = os.path.join(parent_dir, "labels", "test")

# === Helper Functions ===
IMGSZ_FOR_SIZE = 640  # used only to convert normalized area to pixel side-length

def side_len_px(w, h, imgsz=IMGSZ_FOR_SIZE):
    return np.sqrt(w * h) * imgsz

def load_yolo_boxes(path, is_pred=False, conf_thresh=None):
    """
    GT: expected 5 fields:   cls cx cy w h
    Pred: expected 6 fields: cls cx cy w h conf
    """
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if is_pred:
                if len(parts) == 6:
                    cls, x, y, w, h, conf = parts
                else:
                    # strictly require confidences for preds in this loader
                    continue
                cls = int(cls)
                x, y, w, h = map(float, (x, y, w, h))
                conf = float(conf)
                if conf_thresh is None or conf >= conf_thresh:
                    boxes.append((cls, x, y, w, h, conf))
            else:
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = parts
                boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes

def iou(box1, box2):
    """
    IoU in normalized coords (YOLO format).
    box with conf is allowed as box1; we use indices 1:5 for coords.
    """
    x1, y1, w1, h1 = box1[1:5]
    x2, y2, w2, h2 = box2[1:5]
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

def compute_pr_ap(scores, matches, num_gt):
    """101-point interpolated AP; matches is 1 for TP, 0 for FP (scores-sorted)."""
    recall_points = np.linspace(0, 1, 101)
    if num_gt == 0 or len(scores) == 0:
        return 0.0, np.array([]), np.array([]), recall_points, np.zeros_like(recall_points)

    order = np.argsort(-np.asarray(scores))
    m = (np.asarray(matches)[order] == 1).astype(np.float32)

    tp = np.cumsum(m)
    fp = np.cumsum(1 - m)
    recalls = tp / max(num_gt, 1)
    precisions = tp / np.maximum(tp + fp, 1e-9)

    # make precision non-increasing as recall grows
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    prec_interp = []
    for r in recall_points:
        inds = np.where(recalls >= r)[0]
        prec_interp.append(np.max(precisions[inds]) if inds.size > 0 else 0.0)
    ap = float(np.mean(prec_interp))
    return ap, precisions, recalls, recall_points, np.asarray(prec_interp)

def sweep_bin_metrics(bin_objects, pred_dir, low, high, iou_thresh=0.5):
    """
    Confidence-swept PR for ONE size bin (MACRO aggregation):
      - GTs: only those whose size falls inside the bin [low, high)
      - Preds: all predictions from the same images, matched to in-bin GTs
      - Macro aggregate: compute AP/Prec/Rec per class, then average
    Returns: P_macro, R_macro, F1_macro, AP_macro
    """
    from collections import defaultdict

    gts_by_class_img = defaultdict(lambda: defaultdict(list))
    preds_by_class_img = defaultdict(lambda: defaultdict(list))

    img_ids = [fname for fname, _ in bin_objects]
    img_id_set = set(img_ids)

    gt_files = {os.path.splitext(os.path.basename(f))[0]: f
                for f in glob(os.path.join(gt_dir, "*.txt"))}

    def load_preds(path):
        boxes = []
        if not os.path.exists(path):
            return boxes
        with open(path, "r") as f:
            for line in f:
                p = line.strip().split()
                cls, x, y, w, h, conf = p
                boxes.append((int(cls), float(x), float(y), float(w), float(h), float(conf)))
        return boxes

    # load and partition
    for img_id in img_id_set:
        gt_path = gt_files.get(img_id)
        gts = load_yolo_boxes(gt_path, is_pred=False) if gt_path else []

        pred_path = os.path.join(pred_dir, img_id + ".txt")
        preds = load_preds(pred_path)

        gts_in_bin = []
        for gb in gts:
            _, _, _, w, h = gb
            if low <= side_len_px(w, h) < high:
                gts_in_bin.append(gb)

        for gb in gts_in_bin:
            gts_by_class_img[gb[0]][img_id].append(gb)
        for pb in preds:
            preds_by_class_img[pb[0]][img_id].append(pb)

    classes = sorted(set(list(gts_by_class_img.keys()) + list(preds_by_class_img.keys())))
    per_class_metrics = []

    for cls in classes:
        all_scores = []
        all_matches = []
        num_gt = sum(len(v) for v in gts_by_class_img[cls].values())
        if num_gt == 0:
            continue  # skip class with no GT in this bin

        # greedy matching
        flat_preds = []
        for img_id, pboxes in preds_by_class_img[cls].items():
            flat_preds.extend((img_id, pb) for pb in pboxes)
        flat_preds.sort(key=lambda t: t[1][5], reverse=True)

        gt_used = {img_id: [False] * len(gts_by_class_img[cls][img_id]) for img_id in gts_by_class_img[cls]}
        for img_id, pb in flat_preds:
            best_i, best = -1, 0.0
            for i, gb in enumerate(gts_by_class_img[cls].get(img_id, [])):
                if gt_used[img_id][i]:
                    continue
                iou_val = iou(pb, gb)
                if iou_val >= iou_thresh and iou_val > best:
                    best, best_i = iou_val, i
            if best_i >= 0:
                all_matches.append(1)
                gt_used[img_id][best_i] = True
            else:
                all_matches.append(0)
            all_scores.append(pb[5])

        ap, prec, rec, _, _ = compute_pr_ap(all_scores, all_matches, num_gt)
        if prec.size > 0:
            f1 = (2 * prec * rec) / np.maximum(prec + rec, 1e-9)
            k = int(np.argmax(f1))
            P_opt, R_opt, F1_opt = float(prec[k]), float(rec[k]), float(f1[k])
        else:
            P_opt = R_opt = F1_opt = 0.0

        per_class_metrics.append((P_opt, R_opt, F1_opt, ap))

    if len(per_class_metrics) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # macro average acros s classes
    arr = np.array(per_class_metrics)
    return arr[:,0].mean(), arr[:,1].mean(), arr[:,2].mean(), arr[:,3].mean()

# === Step 1: Collect object sizes (side length in pixels) ===
print("Collecting object sizes...")
objects = []  # (fname, side_len_px)
for gt_file in tqdm(glob(os.path.join(gt_dir, "*.txt"))):
    fname = os.path.splitext(os.path.basename(gt_file))[0]
    gt_boxes = load_yolo_boxes(gt_file, is_pred=False)
    for _, _, _, w, h in gt_boxes:
        side_len = side_len_px(w, h, imgsz=IMGSZ_FOR_SIZE)
        if min_size_px <= side_len <= max_size_px:
            objects.append((fname, side_len))
objects = np.array(objects, dtype=object)

# === Step 2: Bin edges ===
bin_edges = np.linspace(min_size_px, max_size_px, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# === Step 3: Evaluate per bin (confidence-swept) ===
results = {name: {"P": [], "R": [], "F1": [], "mAP": []} for name in pred_dirs.keys()}
counts = np.zeros(num_bins, dtype=int)

print("Evaluating bins (confidence sweep per bin)...")
for b in range(num_bins):
    low, high = bin_edges[b], bin_edges[b + 1]
    bin_objects = [(f, s) for f, s in objects if low <= s < high]
    counts[b] = len(bin_objects)

    for pred_name, cfg in pred_dirs.items():
        pred_dir = cfg["dir"]
        P_opt, R_opt, F1_opt, AP_101 = sweep_bin_metrics(
            bin_objects, pred_dir, low, high, iou_thresh=iou_thresh
        )
        results[pred_name]["P"].append(P_opt)
        results[pred_name]["R"].append(R_opt)
        results[pred_name]["F1"].append(F1_opt)
        results[pred_name]["mAP"].append(AP_101)

print(counts)

# ====== Step 4: Plotting =====
metrics = ["P", "R", "F1", "mAP"]
titles  = ["Precision", "Recall", "F1 Score", "mAP@0.5"]
colors  = plt.cm.tab10.colors

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (m, t) in enumerate(zip(metrics, titles)):
    ax1 = axes[i]
    for j, (pred_name, res) in enumerate(results.items()):
        y_vals = np.array(res[m])
        if smoothing_sigma > 0:
            y_vals = gaussian_filter1d(y_vals, sigma=smoothing_sigma)
        ax1.plot(
            bin_centers, y_vals,
            label=pred_name,
            color=colors[j % len(colors)],
            linewidth=2
        )
    ax1.set_title(t)
    ax1.set_xlabel("Object Size (px side length)")
    ax1.set_ylabel(t)
    ax1.set_ylim(0, 1)
    ax1.grid(True)

    # overlay object counts
    ax2 = ax1.twinx()
    ax2.bar(
        bin_centers, counts,
        width=(bin_edges[1] - bin_edges[0]) * 0.8,
        alpha=0.2, color="gray"
    )
    ax2.set_ylabel("Object Count", color="gray")

# add one legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(pred_dirs), fontsize=10)

fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\nSaved comparison plot with bins to: {save_path}")
