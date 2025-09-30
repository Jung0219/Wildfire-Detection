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

save_path    = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/composites/object_size_vs_performance_fixed_bins.png"
smoothing_sigma = 0.3
iou_thresh   = 0.5

# Bin config (side length in pixels)
min_size_px  = 0
max_size_px  = 640
num_bins     = 30

# === Directories ===
img_dir = os.path.join(parent_dir, "images", "test")
gt_dir  = os.path.join(parent_dir, "labels", "test")

# === Helper Functions ===
def load_yolo_boxes(path, is_pred=False, conf_thresh=None):
    boxes = []
    if not os.path.exists(path): return boxes
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if is_pred and len(parts) != 6: continue
            if not is_pred and len(parts) != 5: continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if is_pred else None
            if not is_pred or conf >= conf_thresh:
                boxes.append((cls, x, y, w, h, conf))
    return boxes

def iou(box1, box2):
    x1, y1, w1, h1 = box1[1:5]
    x2, y2, w2, h2 = box2[1:5]
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

def evaluate(gt_boxes, pred_boxes):
    matched = set()
    TP, FP = 0, 0
    for pb in pred_boxes:
        found = False
        for i, gb in enumerate(gt_boxes):
            if gb[0] == pb[0] and i not in matched and iou(pb, gb) >= iou_thresh:
                TP += 1
                matched.add(i)
                found = True
                break
        if not found: FP += 1
    FN = len(gt_boxes) - TP
    return TP, FP, FN

# === Step 1: Collect object sizes (side length in pixels) ===
print("Collecting object sizes...")
objects = []  # (fname, side_len_px)
for gt_file in tqdm(glob(os.path.join(gt_dir, "*.txt"))):
    fname = os.path.splitext(os.path.basename(gt_file))[0]
    gt_boxes = load_yolo_boxes(gt_file, is_pred=False)
    for _, _, _, w, h, _ in gt_boxes:
        side_len = np.sqrt(w * h) * 640  # YOLO normalized → px
        if min_size_px <= side_len <= max_size_px:
            objects.append((fname, side_len))
objects = np.array(objects, dtype=object)

# === Step 2: Bin edges ===
bin_edges = np.linspace(min_size_px, max_size_px, num_bins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# === Step 3: Evaluate per bin ===
results = {name: {"P": [], "R": [], "F1": [], "mAP": []}
           for name in pred_dirs.keys()}
counts = np.zeros(num_bins, dtype=int)

print("Evaluating bins...")
for b in range(num_bins):
    low, high = bin_edges[b], bin_edges[b+1]
    bin_objects = [(f,s) for f,s in objects if low <= s < high]
    counts[b] = len(bin_objects)

    for pred_name, cfg in pred_dirs.items():
        pred_dir = cfg["dir"]
        conf_thresh = cfg["conf"]
        TP = FP = FN = 0
        for fname, _ in bin_objects:
            gt_path   = os.path.join(gt_dir,  fname + ".txt")
            pred_path = os.path.join(pred_dir,fname + ".txt")
            gt   = load_yolo_boxes(gt_path, is_pred=False)
            pred = load_yolo_boxes(pred_path, is_pred=True, conf_thresh=conf_thresh)
            tp, fp, fn = evaluate(gt, pred)
            TP += tp; FP += fp; FN += fn
        P  = TP/(TP+FP) if (TP+FP)>0 else 0
        R  = TP/(TP+FN) if (TP+FN)>0 else 0
        F1 = 2*P*R/(P+R) if (P+R)>0 else 0
        AP = P
        results[pred_name]["P"].append(P)
        results[pred_name]["R"].append(R)
        results[pred_name]["F1"].append(F1)
        results[pred_name]["mAP"].append(AP)

print(counts)

# ======Step 4: Plotting=====
metrics = ["P","R","F1","mAP"]
titles  = ["Precision","Recall","F1 Score","mAP@0.5"]
colors  = plt.cm.tab10.colors

fig, axes = plt.subplots(2,2,figsize=(12,8))
axes = axes.flatten()

for i,(m,t) in enumerate(zip(metrics,titles)):
    ax1 = axes[i]
    for j,(pred_name,res) in enumerate(results.items()):
        y_vals = np.array(res[m])
        if smoothing_sigma > 0:
            y_vals = gaussian_filter1d(y_vals, sigma=smoothing_sigma)
        ax1.plot(bin_centers, y_vals,
                 label=pred_name,             # label line by name
                 color=colors[j%len(colors)],
                 linewidth=2)
    ax1.set_title(t)
    ax1.set_xlabel("Object Size (px side length)")
    ax1.set_ylabel(t)
    ax1.set_ylim(0,1)
    ax1.grid(True)

    # overlay object counts
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0])*0.8,
            alpha=0.2, color="gray")
    ax2.set_ylabel("Object Count", color="gray")

# add one legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(pred_dirs), fontsize=10)

fig.tight_layout(rect=[0,0.05,1,1])  # leave space at bottom for legend
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\n✅ Saved comparison plot with bins to: {save_path}")
