import os
from glob import glob
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
# Set your folders and options here.
GT_DIR: str = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire"     # contains images/test and labels/test
PRED_DIR: str = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/test_set/target_crop_fixed_window"  # e.g., "/path/to/pred_labels" (YOLO txt)
IOU_THRESH: float = 0.5
MAX_DETS: Optional[int] = 100  # e.g., 100 to cap per-image detections
SAVE_JSON: Optional[str] = None # e.g., "/path/to/results.json"
SAVE_CSV: Optional[str] = None  # e.g., "/path/to/per_class.csv"
PLOTS_DIR: Optional[str] = os.path.join(PRED_DIR, "plots")  # e.g., "/path/to/pred_labels/plots"
# ==========================================

# YOLO box types
# GT box:    (cls, cx, cy, w, h)
# Pred box:  (cls, cx, cy, w, h, conf)
GT_DIR = GT_DIR + "/labels/test"

# Helpers
def load_yolo_labels(path: str, is_pred: bool = False) -> List[Tuple]:
    boxes: List[Tuple] = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if is_pred:
                if len(parts) == 6:
                    cls, x, y, w, h, conf = parts
                    boxes.append((int(float(cls)), float(x), float(y), float(w), float(h), float(conf)))
                elif len(parts) == 5:
                    raise ValueError(f"Prediction box line with 5 fields (no confidence) found in {path}, but is_pred=True") 
                else:
                    continue
            else:
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = parts
                boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def iou_norm(box1: Tuple, box2: Tuple) -> float:
    """IoU in normalized coordinates (YOLO format)."""
    x1, y1, w1, h1 = box1[1:5]
    x2, y2, w2, h2 = box2[1:5]
    xa1, ya1 = x1 - w1 / 2.0, y1 - h1 / 2.0
    xa2, ya2 = x1 + w1 / 2.0, y1 + h1 / 2.0
    xb1, yb1 = x2 - w2 / 2.0, y2 - h2 / 2.0
    xb2, yb2 = x2 + w2 / 2.0, y2 + h2 / 2.0

    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_pr_ap(scores: List[float], matches: List[int], num_gt: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute AP with 101-point interpolation.
    Returns: (AP, precision, recall, recall_grid, precision_interp)
    precision/recall are along score-sorted thresholds; precision_interp is at recall_grid.
    """
    recall_points = np.linspace(0, 1, 101)
    if num_gt == 0 or len(scores) == 0:
        # Return empty threshold curves but zero precision over the recall grid
        return 0.0, np.array([], dtype=float), np.array([], dtype=float), recall_points, np.zeros_like(recall_points)

    order = np.argsort(-np.asarray(scores))
    m = (np.asarray(matches)[order] == 1).astype(np.float32)

    tp = np.cumsum(m)
    fp = np.cumsum(1 - m)
    recalls = tp / max(num_gt, 1)
    precisions = tp / np.maximum(tp + fp, 1e-9)

    # Make precision monotonically non-increasing as recall grows
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    prec_interp = []
    for r in recall_points:
        inds = np.where(recalls >= r)[0]
        prec_interp.append(np.max(precisions[inds]) if inds.size > 0 else 0.0)
    ap = float(np.mean(prec_interp))
    return ap, precisions, recalls, recall_points, np.asarray(prec_interp)


# Main Calculation
def evaluate_directories( # change it so it takes in objects instead of paths. 
    gt_dir: str,
    pred_dir: str,
    iou_thresh: float = 0.5,
    max_dets_per_image: Optional[int] = None,
) -> Dict:
    """
    Evaluate predictions folder against ground-truth folder in YOLO format.

    Returns dict with per-class AP, overall mAP, and summary counts.
    """
    # Index files by basename
    gt_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(gt_dir, "*.txt"))}
    pred_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(pred_dir, "*.txt"))}
    image_ids = sorted(set(gt_files.keys()) | set(pred_files.keys()))

    # Group data by class
    gts_by_class: Dict[int, Dict[str, List[Tuple]]] = defaultdict(lambda: defaultdict(list))
    preds_by_class: Dict[int, Dict[str, List[Tuple]]] = defaultdict(lambda: defaultdict(list))

    for img_id in image_ids:
        gt_path = gt_files.get(img_id)
        pred_path = pred_files.get(img_id)

        gt_boxes = load_yolo_labels(gt_path, is_pred=False) if gt_path else []
        pred_boxes = load_yolo_labels(pred_path, is_pred=True) if pred_path else []

        for gb in gt_boxes:
            gts_by_class[gb[0]][img_id].append(gb)

        if max_dets_per_image is not None and len(pred_boxes) > 0:
            pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)[:max_dets_per_image]

        for pb in pred_boxes:
            preds_by_class[pb[0]][img_id].append(pb)

    # Evaluate per-class
    per_class = {}
    ap_values = {}
    # Store per-class interpolated PR for later macro averaging
    pr_interp_by_class: Dict[int, Dict[str, np.ndarray]] = {}
    total_tp = total_fp = total_fn = 0

    classes = sorted(set(list(gts_by_class.keys()) + list(preds_by_class.keys())))
    for cls in classes:
        # Flatten predictions for this class across images, sort by score
        flat_preds: List[Tuple[str, Tuple]] = []
        for img_id, boxes in preds_by_class[cls].items():
            flat_preds.extend((img_id, pb) for pb in boxes)
        flat_preds.sort(key=lambda t: t[1][5], reverse=True)

        # Mark matches
        num_gt = sum(len(v) for v in gts_by_class[cls].values())
        gt_used = {img_id: [False] * len(gts_by_class[cls][img_id]) for img_id in gts_by_class[cls]}

        scores: List[float] = []
        matches: List[int] = []  # 1 for TP, 0 for FP

        for img_id, pb in flat_preds:
            best_iou = 0.0
            best_idx = -1
            gt_list = gts_by_class[cls].get(img_id, [])
            for i, gb in enumerate(gt_list):
                if gt_used.get(img_id, [])[i]:
                    continue
                iou_val = iou_norm(pb, gb)
                if iou_val >= iou_thresh and iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_idx >= 0:
                matches.append(1)
                gt_used[img_id][best_idx] = True
            else:
                matches.append(0)
            scores.append(pb[5])

        # Counts
        tp = int(np.sum(matches))
        fp = int(len(matches) - tp)
        fn = int(num_gt - tp)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        ap, prec, rec, rgrid, pinterp = compute_pr_ap(scores, matches, num_gt)
        ap_values[cls] = ap
        pr_interp_by_class[cls] = {"recall": rgrid, "precision": pinterp}

        # Best F1 along the sorted threshold sweep
        if prec.size > 0:
            f1 = (2 * prec * rec) / np.maximum(prec + rec, 1e-9)
            best_idx = int(np.argmax(f1))
            # scores aligned with prec/rec are the scores sorted in descending order
            order = np.argsort(-np.asarray(scores))
            scores_sorted = np.asarray(scores)[order]
            best = {
                "precision": float(prec[best_idx]),
                "recall": float(rec[best_idx]),
                "f1": float(f1[best_idx]),
                "score_threshold": float(scores_sorted[best_idx]) if scores else 0.0,
            }
        else:
            best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "score_threshold": 0.0}

        # Curves for plotting
        if prec.size > 0:
            order = np.argsort(-np.asarray(scores))
            scores_sorted = np.asarray(scores)[order]
            f1_curve = (2 * prec * rec) / np.maximum(prec + rec, 1e-9)
        else:
            scores_sorted = np.array([], dtype=float)
            f1_curve = np.array([], dtype=float)

        per_class[cls] = {
            "ap": float(ap),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "best_f1": best,
            "curve": {
                "scores": scores_sorted.tolist(),
                "precision": prec.tolist(),
                "recall": rec.tolist(),
                "f1": f1_curve.tolist(),
                "interp": {
                    "recall": rgrid.tolist(),
                    "precision": pinterp.tolist(),
                },
            },
        }


    mAP = float(np.mean(list(ap_values.values()))) if ap_values else 0.0

    summary = {
        "mAP@{:.2f}".format(iou_thresh): mAP,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "num_images": len(image_ids),
        "num_classes": len(classes),
    }

    # Macro-averaged overall curves
    # 1) PR macro over recall grid
    valid_classes = [c for c in classes if c in pr_interp_by_class]
    if valid_classes:
        rgrid = pr_interp_by_class[valid_classes[0]]["recall"]
        stack_prec = np.stack([pr_interp_by_class[c]["precision"] for c in valid_classes], axis=0)
        macro_prec = np.mean(stack_prec, axis=0)
    else:
        rgrid = np.linspace(0, 1, 101)
        macro_prec = np.zeros_like(rgrid)

    # 2) Confidence-vs-metrics macro by aligning per-rank index
    non_empty = [per_class[c]["curve"] for c in classes if len(per_class[c]["curve"]["scores"]) > 0]
    if non_empty:
        min_len = min(len(c["scores"]) for c in non_empty)
        if min_len > 0:
            scores_mat = np.stack([np.asarray(c["scores"])[:min_len] for c in non_empty], axis=0)
            prec_mat = np.stack([np.asarray(c["precision"])[:min_len] for c in non_empty], axis=0)
            rec_mat = np.stack([np.asarray(c["recall"])[:min_len] for c in non_empty], axis=0)
            f1_mat = np.stack([np.asarray(c["f1"])[:min_len] for c in non_empty], axis=0)
            o_scores_sorted = np.mean(scores_mat, axis=0)
            o_prec = np.mean(prec_mat, axis=0)
            o_rec = np.mean(rec_mat, axis=0)
            o_f1 = np.mean(f1_mat, axis=0)
        else:
            o_scores_sorted = np.array([], dtype=float)
            o_prec = np.array([], dtype=float)
            o_rec = np.array([], dtype=float)
            o_f1 = np.array([], dtype=float)
    else:
        o_scores_sorted = np.array([], dtype=float)
        o_prec = np.array([], dtype=float)
        o_rec = np.array([], dtype=float)
        o_f1 = np.array([], dtype=float)

    return {
        "iou_thresh": iou_thresh,
        "classes": classes,
        "per_class": per_class,
        "summary": summary,
        "overall_curve": {
            "scores": o_scores_sorted.tolist(),
            "precision": o_prec.tolist(),
            "recall": o_rec.tolist(),
            "f1": o_f1.tolist(),
            "ap": float(mAP),
            "pr_interp": {
                "recall": rgrid.tolist(),
                "precision": macro_prec.tolist(),
            },
        },
    }


def main():
    if not GT_DIR or not PRED_DIR:
        raise SystemExit("Please set GT_DIR and PRED_DIR at the top of the file.")

    res = evaluate_directories(
        GT_DIR,
        PRED_DIR,
        iou_thresh=IOU_THRESH,
        max_dets_per_image=MAX_DETS,
    )

    # Print concise summary
    print("Evaluation Summary:")
    for k, v in res["summary"].items():
        print(f"- {k}: {v}")

    print("\nPer-class metrics:")
    for cls in res["classes"]:
        m = res["per_class"][cls]
        best = m["best_f1"]
        print(
            f"class {cls}: AP={m['ap']:.4f} TP={m['tp']} FP={m['fp']} FN={m['fn']} "
            f"| bestF1={best['f1']:.4f} @ thr={best['score_threshold']:.3f} (P={best['precision']:.3f}, R={best['recall']:.3f})"
        )

    # Optional saves
    if SAVE_JSON:
        import json
        os.makedirs(os.path.dirname(SAVE_JSON), exist_ok=True)
        with open(SAVE_JSON, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved JSON: {SAVE_JSON}")

    if SAVE_CSV:
        os.makedirs(os.path.dirname(SAVE_CSV), exist_ok=True)
        with open(SAVE_CSV, "w") as f:
            f.write("class,AP,TP,FP,FN,bestF1,thr,prec,rec\n")
            for cls in res["classes"]:
                m = res["per_class"][cls]
                b = m["best_f1"]
                f.write(
                    f"{cls},{m['ap']:.6f},{m['tp']},{m['fp']},{m['fn']},{b['f1']:.6f},{b['score_threshold']:.6f},{b['precision']:.6f},{b['recall']:.6f}\n"
                )
        print(f"Saved CSV: {SAVE_CSV}")

    # Plot confidence vs precision/recall/F1 and PR curve
    # Plot confidence vs precision/recall/F1 and PR curve
    out_dir = PLOTS_DIR or os.path.join(PRED_DIR, "metrics")
    os.makedirs(out_dir, exist_ok=True)

     # ...existing code...
    
    plt.figure(figsize=(14, 10)) 

    # 1) Confidence vs Precision
    plt.subplot(2, 2, 1)
    for cls in res["classes"]:
        c = res["per_class"][cls]["curve"]
        if len(c["scores"]) == 0:
            continue
        s = np.asarray(c["scores"]); p = np.asarray(c["precision"])
        imax = int(np.argmax(p)) if p.size else None
        label = f"class {cls}"
        if p.size:
            label += f" (maxP={p[imax]:.3f}@{s[imax]:.3f})"
        plt.plot(s, p, label=label)
    oc = res["overall_curve"]
    if len(oc["scores"]) > 0:
        s = np.asarray(oc["scores"]); p = np.asarray(oc["precision"])
        imax = int(np.argmax(p)) if p.size else None
        olabel = "overall"
        if p.size:
            olabel += f" (maxP={p[imax]:.3f}@{s[imax]:.3f})"
        plt.plot(s, p, label=olabel, color="black", linestyle="--")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Precision")
    plt.title("Confidence vs Precision")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 2) Confidence vs Recall
    plt.subplot(2, 2, 2)
    for cls in res["classes"]:
        c = res["per_class"][cls]["curve"]
        if len(c["scores"]) == 0:
            continue
        s = np.asarray(c["scores"]); r = np.asarray(c["recall"])
        imax = int(np.argmax(r)) if r.size else None
        label = f"class {cls}"
        if r.size:
            label += f" (maxR={r[imax]:.3f}@{s[imax]:.3f})"
        plt.plot(s, r, label=label)
    if len(oc["scores"]) > 0:
        s = np.asarray(oc["scores"]); r = np.asarray(oc["recall"])
        imax = int(np.argmax(r)) if r.size else None
        olabel = "overall"
        if r.size:
            olabel += f" (maxR={r[imax]:.3f}@{s[imax]:.3f})"
        plt.plot(s, r, label=olabel, color="black", linestyle="--")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Recall")
    plt.title("Confidence vs Recall")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 3) Confidence vs F1
    plt.subplot(2, 2, 3)
    for cls in res["classes"]:
        c = res["per_class"][cls]["curve"]
        if len(c["scores"]) == 0:
            continue
        s = np.asarray(c["scores"]); f1 = np.asarray(c["f1"])
        imax = int(np.argmax(f1)) if f1.size else None
        label = f"class {cls}"
        if f1.size:
            label += f" (maxF1={f1[imax]:.3f}@{s[imax]:.3f})"
        plt.plot(s, f1, label=label)
    if len(oc["scores"]) > 0:
        s = np.asarray(oc["scores"]); f1 = np.asarray(oc["f1"])
        imax = int(np.argmax(f1)) if f1.size else None
        olabel = "overall"
        if f1.size:
            olabel += f" (maxF1={f1[imax]:.3f}@{s[imax]:.3f})"
        plt.plot(s, f1, label=olabel, color="black", linestyle="--")
    plt.xlabel("Confidence threshold")
    plt.ylabel("F1")
    plt.title("Confidence vs F1")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 4) Precision–Recall curve (legend shows AP and best-F1@thr)
    plt.subplot(2, 2, 4)
    for cls in res["classes"]:
        c = res["per_class"][cls]["curve"]
        if len(c["scores"]) == 0:
            continue
        s = np.asarray(c["scores"])
        p = np.asarray(c["precision"]); r = np.asarray(c["recall"])
        f1 = (2 * p * r) / np.maximum(p + r, 1e-9) if p.size else np.array([])
        imax = int(np.argmax(f1)) if f1.size else None
        ap = res["per_class"][cls]["ap"]
        label = f"class {cls} (AP={ap:.3f})"
        if f1.size:
            label += f", bestF1={f1[imax]:.3f}@{s[imax]:.3f}"
        plt.plot(r, p, label=label)
    if "pr_interp" in oc and len(oc["pr_interp"]["recall"]) > 0:
        # Overall PR (interpolated) line; add AP and best-F1@thr from overall curves if present
        r_i = np.asarray(oc["pr_interp"]["recall"])
        p_i = np.asarray(oc["pr_interp"]["precision"])
        olabel = f"overall (mAP={oc['ap']:.3f})"
        if len(oc["scores"]) > 0:
            s = np.asarray(oc["scores"])
            p = np.asarray(oc["precision"]); r = np.asarray(oc["recall"])
            f1 = (2 * p * r) / np.maximum(p + r, 1e-9) if p.size else np.array([])
            if f1.size:
                imax = int(np.argmax(f1))
                olabel += f", bestF1={f1[imax]:.3f}@{s[imax]:.3f}"
        plt.plot(r_i, p_i, label=olabel, color="black", linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plot_path = os.path.join(out_dir, "confidence_precision_recall_f1_pr.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    try:
        plt.close()
    except Exception:
        pass
    print(f"Saved plots: {plot_path}")

# ...existing code...



if __name__ == "__main__":
    main()
