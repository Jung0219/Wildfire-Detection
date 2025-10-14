import os
import numpy as np

# ================= CONFIG =================
GT_DIR     = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire/labels/test"  # ground-truth labels
PRED_DIR   = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire/test_set/composites_orig_top" # predictions
STEP       = 0.05   # confidence bin step
START      = 0.0    # min confidence bound
END        = 1    # max confidence bound
# ==========================================
IOU_THRESH = 0.5    # IoU threshold for TP/FP
# ==========================================
print("prediction: ", os.path.basename(PRED_DIR))
def parse_gt_line(line):
    """GT format: class x y w h"""
    cls, x, y, w, h = line.strip().split()
    return int(cls), float(x), float(y), float(w), float(h)

def parse_pred_line(line):
    """Pred format: class x y w h conf"""
    cls, x, y, w, h, conf = line.strip().split()
    return int(cls), float(x), float(y), float(w), float(h), float(conf)

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def evaluate_bins(gt_dir, pred_dir, step, start, end, iou_thresh):
    edges = np.arange(start, end + step, step)
    edges[-1] = end
    bin_ranges = [(edges[i], edges[i+1]) for i in range(len(edges) - 1)]

    results = {}
    total_gt = 0
    for fname in os.listdir(gt_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(gt_dir, fname)) as f:
                total_gt += sum(1 for l in f if l.strip())

    for low, high in bin_ranges:
        TP, FP = 0, 0

        for fname in os.listdir(pred_dir):
            if not fname.endswith(".txt"):
                continue

            pred_file = os.path.join(pred_dir, fname)
            gt_file   = os.path.join(gt_dir, fname)
            if not os.path.exists(gt_file):
                continue

            with open(pred_file) as f:
                preds = [parse_pred_line(l) for l in f if l.strip()]
            with open(gt_file) as f:
                gts = [parse_gt_line(l) for l in f if l.strip()]

            preds_bin = [(c, (x, y, w, h)) for c, x, y, w, h, conf in preds
                         if low <= conf < high or (conf == end and high == end)]

            matched = set()
            for cls, box in preds_bin:
                best_iou, best_idx = 0.0, -1
                for i, (gcls, gx, gy, gw, gh) in enumerate(gts):
                    if gcls != cls or i in matched:
                        continue
                    iou_val = iou(box, (gx, gy, gw, gh))
                    if iou_val > best_iou:
                        best_iou, best_idx = iou_val, i
                if best_iou >= iou_thresh:
                    TP += 1
                    matched.add(best_idx)
                else:
                    FP += 1

        tp_ratio = TP / total_gt if total_gt > 0 else 0.0
        fp_ratio = FP / total_gt if total_gt > 0 else 0.0
        results[f"{low:.2f}-{high:.2f}"] = {
            "TP": TP,
            "FP": FP,
            "TP/GT": tp_ratio,
            "FP/GT": fp_ratio,
            "GT": total_gt
        }

    return results, total_gt

if __name__ == "__main__":
    results, total_gt = evaluate_bins(GT_DIR, PRED_DIR, STEP, START, END, IOU_THRESH)
    print(f"=== Ratios by Confidence Bin ===\nTotal GT count = {total_gt}\n")
    for bin_range, stats in results.items():
        print(
            f"{bin_range}: "
            f"TP={stats['TP']} FP={stats['FP']} "
            f"TP/GT={stats['TP/GT']:.3f} FP/GT={stats['FP/GT']:.3f} "
            f"(GT={stats['GT']})"
        )
