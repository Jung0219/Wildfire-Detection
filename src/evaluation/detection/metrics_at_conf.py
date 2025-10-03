import os
import glob
from collections import defaultdict

# ========== CONFIG ==========
GT_DIR     = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/dev/labels/test"
PRED_DIR   = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites/two_stage/yolo_0.3_0.525"
CONF_THRESH = 0.3
# ============================

def yolo_to_xywhn(line):
    parts = line.strip().split()
    if len(parts) == 5:  # GT
        cls, x, y, w, h = parts
        return int(cls), float(x), float(y), float(w), float(h), None
    elif len(parts) == 6:  # prediction
        cls, x, y, w, h, conf = parts
        return int(cls), float(x), float(y), float(w), float(h), float(conf)
    else:
        return None

def xywhn_to_xyxy(box):
    x, y, w, h = box
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return [x1, y1, x2, y2]

def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def evaluate(gt_dir, pred_dir, conf_thresh=0.3, iou_thresh=0.5):
    # per-class counts
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)

    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))

    for gt_file in gt_files:
        base = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, base)

        # load GT
        gts = []
        with open(gt_file) as f:
            for line in f:
                cls, x, y, w, h, _ = yolo_to_xywhn(line)
                gts.append({"cls": cls, "box": xywhn_to_xyxy((x,y,w,h)), "used": False})

        # load preds
        preds = []
        if os.path.exists(pred_file):
            with open(pred_file) as f:
                for line in f:
                    cls, x, y, w, h, conf = yolo_to_xywhn(line)
                    if conf >= conf_thresh:
                        preds.append({"cls": cls, "box": xywhn_to_xyxy((x,y,w,h)), "conf": conf})

        # match preds to GT
        preds = sorted(preds, key=lambda x: -x["conf"])
        for pred in preds:
            best_iou, best_gt = 0, None
            for gt in gts:
                if gt["cls"] != pred["cls"] or gt["used"]:
                    continue
                iou_val = iou(pred["box"], gt["box"])
                if iou_val > best_iou:
                    best_iou, best_gt = iou_val, gt
            if best_iou >= iou_thresh:
                TP[pred["cls"]] += 1
                best_gt["used"] = True
            else:
                FP[pred["cls"]] += 1

        # leftover GTs are FN
        for g in gts:
            if not g["used"]:
                FN[g["cls"]] += 1

    # compute metrics per class
    results = {}
    for cls in set(list(TP.keys()) + list(FP.keys()) + list(FN.keys())):
        tp, fp, fn = TP[cls], FP[cls], FN[cls]
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
        results[cls] = (prec, rec, f1)

    # macro average
    n_classes = len(results)
    macro_prec = sum(r[0] for r in results.values()) / n_classes
    macro_rec  = sum(r[1] for r in results.values()) / n_classes
    macro_f1   = sum(r[2] for r in results.values()) / n_classes

    return results, (macro_prec, macro_rec, macro_f1)

if __name__ == "__main__":
    per_class, macro = evaluate(GT_DIR, PRED_DIR, CONF_THRESH)
    for cls, (p, r, f1) in per_class.items():
        print(f"Class {cls}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    print(f"\nMacro Avg: Precision={macro[0]:.3f}, Recall={macro[1]:.3f}, F1={macro[2]:.3f}")
