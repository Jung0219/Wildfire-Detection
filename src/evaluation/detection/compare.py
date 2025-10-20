import os
from glob import glob
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
GT_DIR: str        = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire"     # contains images/test and labels/test
PRED_BASELINE: str = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_training_test_set/labels"  # e.g., "/path/to/pred_labels"
PRED_NEW: str      = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_training_test_set/target_cropping/target_crop_dynamic_window_800"
IOU_THRESH: float = 0.5
MAX_DETS: Optional[int] = 100
CONF_GRID_STEPS: int = 201
OUT_DIR: str = os.path.join(PRED_NEW, "plots_baseline_comparison")
# ==========================================
GT_DIR = GT_DIR + "/labels/test"

def load_yolo_labels(path: str, is_pred: bool = False) -> List[Tuple]:
    boxes: List[Tuple] = []
    if not path or not os.path.exists(path):
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
            else:
                if len(parts) == 5:
                    cls, x, y, w, h = parts
                    boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def iou_norm(box1: Tuple, box2: Tuple) -> float:
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


def compute_pr(scores: List[float], matches: List[int], num_gt: int):
    if num_gt == 0 or len(scores) == 0:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(-np.asarray(scores))
    m = (np.asarray(matches)[order] == 1).astype(np.float32)
    tp = np.cumsum(m)
    fp = np.cumsum(1 - m)
    recalls = tp / max(num_gt, 1)
    precisions = tp / np.maximum(tp + fp, 1e-9)
    # enforce monotonic precision
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    scores_sorted = np.asarray(scores)[order]
    return scores_sorted, precisions, recalls


def compute_ap_coco(scores: List[float], matches: List[int], num_gt: int) -> float:
    """COCO 101-point interpolation AP@0.5."""
    recall_points = np.linspace(0, 1, 101)
    if num_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    m = (np.asarray(matches)[order] == 1).astype(np.float32)
    tp = np.cumsum(m)
    fp = np.cumsum(1 - m)
    recalls = tp / max(num_gt, 1)
    precisions = tp / np.maximum(tp + fp, 1e-9)
    # enforce monotonic precision
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap_vals = []
    for rp in recall_points:
        inds = np.where(recalls >= rp)[0]
        p_val = np.max(precisions[inds]) if inds.size > 0 else 0.0
        ap_vals.append(p_val)
    return float(np.mean(ap_vals))


def eval_per_class_curves(gt_dir: str, pred_dir: str, iou_thresh: float, max_dets: Optional[int]):
    gt_files   = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(gt_dir, "*.txt"))}
    pred_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(pred_dir, "*.txt"))}
    ids = sorted(set(gt_files) | set(pred_files))
    gts_by_class  = defaultdict(lambda: defaultdict(list))
    preds_by_class= defaultdict(lambda: defaultdict(list))

    for img_id in ids:
        g = load_yolo_labels(gt_files.get(img_id), is_pred=False) if gt_files.get(img_id) else []
        p = load_yolo_labels(pred_files.get(img_id), is_pred=True) if pred_files.get(img_id) else []
        if max_dets is not None and p:
            p = sorted(p, key=lambda x: x[5], reverse=True)[:max_dets]
        for gb in g: gts_by_class[gb[0]][img_id].append(gb)
        for pb in p: preds_by_class[pb[0]][img_id].append(pb)

    per_class = {}
    ap_values = {}
    classes = sorted(set(list(gts_by_class.keys()) + list(preds_by_class.keys())))
    for cls in classes:
        flat_preds = [(img_id, pb) for img_id, boxes in preds_by_class[cls].items() for pb in boxes]
        flat_preds.sort(key=lambda t: t[1][5], reverse=True)
        num_gt = sum(len(v) for v in gts_by_class[cls].values())
        gt_used = {img_id: [False]*len(gts_by_class[cls][img_id]) for img_id in gts_by_class[cls]}
        scores, matches = [], []
        for img_id, pb in flat_preds:
            best_iou, best_idx = 0.0, -1
            for i, gb in enumerate(gts_by_class[cls].get(img_id, [])):
                if gt_used[img_id][i]: continue
                iou = iou_norm(pb, gb)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0:
                matches.append(1); gt_used[img_id][best_idx] = True
            else:
                matches.append(0)
            scores.append(pb[5])
        s, p, r = compute_pr(scores, matches, num_gt)
        ap = compute_ap_coco(scores, matches, num_gt)
        ap_values[cls] = ap
        per_class[cls] = {"scores": s, "precision": p, "recall": r}
    return per_class, ap_values


def macro_on_conf_grid(per_class_curves: Dict[int, Dict[str, np.ndarray]], steps: int = 201):
    conf_grid = np.linspace(0.0, 1.0, steps)
    all_prec, all_rec, all_f1 = [], [], []
    for c, cur in per_class_curves.items():
        s = cur["scores"]; p = cur["precision"]; r = cur["recall"]
        if s.size == 0: continue
        s_asc, p_asc, r_asc = s[::-1], p[::-1], r[::-1]
        p_i = np.interp(conf_grid, s_asc, p_asc, left=p_asc[0], right=p_asc[-1])
        r_i = np.interp(conf_grid, s_asc, r_asc, left=r_asc[0], right=r_asc[-1])
        f1_i = (2 * p_i * r_i) / np.maximum(p_i + r_i, 1e-9)
        all_prec.append(p_i); all_rec.append(r_i); all_f1.append(f1_i)
    if not all_rec:
        return conf_grid, np.zeros_like(conf_grid), np.zeros_like(conf_grid), np.zeros_like(conf_grid)
    return conf_grid, np.mean(all_prec, axis=0), np.mean(all_rec, axis=0), np.mean(all_f1, axis=0)


def compare_methods(gt_dir: str, pred_old: str, pred_new: str, iou: float, max_dets: Optional[int], steps: int):
    curves_old, ap_old_values = eval_per_class_curves(gt_dir, pred_old, iou, max_dets)
    curves_new, ap_new_values = eval_per_class_curves(gt_dir, pred_new, iou, max_dets)

    conf, P_old, R_old, F1_old = macro_on_conf_grid(curves_old, steps)
    _,    P_new, R_new, F1_new = macro_on_conf_grid(curves_new, steps)

    dR, dP, dF1 = R_new - R_old, P_new - P_old, F1_new - F1_old

    mAP_old = float(np.mean(list(ap_old_values.values()))) if ap_old_values else 0.0
    mAP_new = float(np.mean(list(ap_new_values.values()))) if ap_new_values else 0.0
    delta_mAP = mAP_new - mAP_old

    return {
        "conf": conf,
        "old": {"P": P_old, "R": R_old, "F1": F1_old, "mAP": mAP_old},
        "new": {"P": P_new, "R": R_new, "F1": F1_new, "mAP": mAP_new},
        "delta": {"dP": dP, "dR": dR, "dF1": dF1, "dmAP": delta_mAP},
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    res = compare_methods(GT_DIR, PRED_BASELINE, PRED_NEW, IOU_THRESH, MAX_DETS, CONF_GRID_STEPS)

    conf = res["conf"]
    R_old, R_new = res["old"]["R"], res["new"]["R"]
    P_old, P_new = res["old"]["P"], res["new"]["P"]
    F1_old, F1_new = res["old"]["F1"], res["new"]["F1"]

    dR, dP, dF1 = res["delta"]["dR"], res["delta"]["dP"], res["delta"]["dF1"]
    mAP_old, mAP_new, delta_mAP = res["old"]["mAP"], res["new"]["mAP"], res["delta"]["dmAP"]

    print("\n=== Evaluation Summary (COCO 101-point) ===")
    print(f"Baseline mAP: {mAP_old:.4f}")
    print(f"New mAP     : {mAP_new:.4f}")
    print(f"ΔmAP        : {delta_mAP:+.4f}")
    print("\n--- Delta Metrics (macro-averaged across confidence grid) ---")
    print(f"Mean ΔRecall   : {np.mean(dR):+.4f}")
    print(f"Mean ΔPrecision: {np.mean(dP):+.4f}")
    print(f"Mean ΔF1       : {np.mean(dF1):+.4f}")

    # === Plotting ===
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    axes[0,0].plot(conf, R_old, '--', label="baseline")
    axes[0,0].plot(conf, R_new, label="new")
    axes[0,0].set(title="Recall", xlabel="Conf", ylabel="Recall", xlim=(0,1), ylim=(0,1))
    axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(conf, P_old, '--', label="baseline")
    axes[0,1].plot(conf, P_new, label="new")
    axes[0,1].set(title="Precision", xlabel="Conf", ylabel="Precision", xlim=(0,1), ylim=(0,1))
    axes[0,1].legend(); axes[0,1].grid(True)

    axes[0,2].plot(conf, F1_old, '--', label="baseline")
    axes[0,2].plot(conf, F1_new, label="new")
    axes[0,2].set(title="F1", xlabel="Conf", ylabel="F1", xlim=(0,1), ylim=(0,1))
    axes[0,2].legend(); axes[0,2].grid(True)

    axes[0,3].plot(R_old, P_old, '--', label=f"baseline (mAP={mAP_old:.3f})")
    axes[0,3].plot(R_new, P_new, label=f"new (mAP={mAP_new:.3f})")
    axes[0,3].set(title="Macro PR Curve", xlabel="Recall", ylabel="Precision", xlim=(0,1), ylim=(0,1))
    axes[0,3].legend(); axes[0,3].grid(True)

    axes[1,0].plot(conf, dR, label="ΔRecall"); axes[1,0].axhline(0,color="black")
    axes[1,0].set(title="Recall Gain", xlabel="Conf", ylabel="ΔRecall", xlim=(0,1))
    axes[1,0].grid(True); axes[1,0].legend()

    axes[1,1].plot(conf, dP, label="ΔPrecision"); axes[1,1].axhline(0,color="black")
    axes[1,1].set(title="Precision Gain", xlabel="Conf", ylabel="ΔPrecision", xlim=(0,1))
    axes[1,1].grid(True); axes[1,1].legend()

    axes[1,2].plot(conf, dF1, label="ΔF1"); axes[1,2].axhline(0,color="black")
    axes[1,2].set(title="F1 Gain", xlabel="Conf", ylabel="ΔF1", xlim=(0,1))
    axes[1,2].grid(True); axes[1,2].legend()

    axes[1,3].axis("off")
    axes[1,3].text(0.1, 0.55,
                   f"Baseline mAP: {mAP_old:.4f}\n"
                   f"New mAP: {mAP_new:.4f}\n"
                   f"ΔmAP: {delta_mAP:+.4f}\n"
                   f"Mean ΔRecall: {np.mean(dR):+.4f}\n"
                   f"Mean ΔPrecision: {np.mean(dP):+.4f}\n"
                   f"Mean ΔF1: {np.mean(dF1):+.4f}\n",
                   fontsize=14, bbox=dict(facecolor="white", alpha=0.7))

    plt.tight_layout()
    # Make plot filename adaptive based on baseline and new prediction directory names
    baseline_name = os.path.basename(os.path.normpath(PRED_BASELINE))
    new_name = os.path.basename(os.path.normpath(PRED_NEW))
    plot_filename = f"all_metrics_grid_{baseline_name}_vs_{new_name}.png"
    plt.savefig(os.path.join(OUT_DIR, plot_filename), dpi=300)
    plt.close()
    print(f"\nSaved combined plot to {os.path.join(OUT_DIR, plot_filename)}")
