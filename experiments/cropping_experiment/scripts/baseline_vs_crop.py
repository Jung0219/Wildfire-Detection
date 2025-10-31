import os, glob, math, shutil, numpy as np, pandas as pd
from collections import defaultdict
import cv2

# ============================================================
# CONFIGURATION
# ============================================================
LABELS_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/early_smoke_A/labels/test"
IMAGES_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/early_smoke_A/images/test"
PRED_BASE_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_smoke_A/test_set/labels"
PRED_CROP_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_smoke_A/test_set/target_crop_dynamic_window_800"
SAVE_CSV = "/lab/projects/fire_smoke_awr/experiments/cropping_experiment/results/per_image_delta_yolo.csv"
WORSE_DIR = "/lab/projects/fire_smoke_awr/experiments/cropping_experiment/results/worse_cases"
IOU_THR = 0.5
CONF_THR = 0.25
CLASS_AWARE = True
# ============================================================


def load_yolo_dir(dir_path, images_dir):
    """Load YOLO-format directory into dict[image_name] -> list of detections."""
    preds = {}
    for txt_path in glob.glob(os.path.join(dir_path, "*.txt")):
        name = os.path.splitext(os.path.basename(txt_path))[0]
        img_path = os.path.join(images_dir, f"{name}.jpg")
        if not os.path.exists(img_path):
            continue
        h, w = get_image_hw(img_path)
        boxes = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh, conf = map(float, parts[1:6])
                if conf < CONF_THR:
                    continue
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append({"bbox": [x1, y1, x2, y2], "cls": cls, "score": conf})
        preds[name] = boxes
    return preds


def load_yolo_gt(labels_dir, images_dir):
    """Load YOLO ground-truth annotations."""
    gts = {}
    for label_path in glob.glob(os.path.join(labels_dir, "*.txt")):
        name = os.path.splitext(os.path.basename(label_path))[0]
        img_path = os.path.join(images_dir, f"{name}.jpg")
        if not os.path.exists(img_path):
            continue
        h, w = get_image_hw(img_path)
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append({"bbox": [x1, y1, x2, y2], "cls": cls})
        gts[name] = boxes
    return gts


def get_image_hw(path):
    im = cv2.imread(path)
    return im.shape[:2]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def match_image(preds, gts, iou_thr=0.5, class_aware=True):
    used_gt = set()
    tp, fp = 0, 0
    for p in preds:
        best_iou, best_idx = 0.0, -1
        for j, gt in enumerate(gts):
            if j in used_gt:
                continue
            if class_aware and (p["cls"] != gt["cls"]):
                continue
            iou = iou_xyxy(p["bbox"], gt["bbox"])
            if iou >= iou_thr and iou > best_iou:
                best_iou, best_idx = iou, j
        if best_idx >= 0:
            tp += 1
            used_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gts) - len(used_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return dict(tp=tp, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1)


def main():
    print("[1/4] Loading ground truth and YOLO prediction directories...")
    gts = load_yolo_gt(LABELS_DIR, IMAGES_DIR)
    preds_base = load_yolo_dir(PRED_BASE_DIR, IMAGES_DIR)
    preds_crop = load_yolo_dir(PRED_CROP_DIR, IMAGES_DIR)

    rows = []
    print("[2/4] Computing per-image F1...")
    for name in sorted(gts.keys()):
        gt = gts[name]
        pb = preds_base.get(name, [])
        pc = preds_crop.get(name, [])
        mb = match_image(pb, gt, IOU_THR, CLASS_AWARE)
        mc = match_image(pc, gt, IOU_THR, CLASS_AWARE)
        rows.append({
            "image_name": name,
            "gt_count": len(gt),
            "base_f1": mb["f1"],
            "crop_f1": mc["f1"],
            "delta_f1": mc["f1"] - mb["f1"],
            "base_tp": mb["tp"], "crop_tp": mc["tp"],
            "base_fp": mb["fp"], "crop_fp": mc["fp"],
            "base_fn": mb["fn"], "crop_fn": mc["fn"]
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SAVE_CSV), exist_ok=True)
    df.to_csv(SAVE_CSV, index=False)
    print(f"[3/4] Saved metrics to: {SAVE_CSV}")

    # Identify worse cases
    worse_df = df[df["delta_f1"] < 0]
    print(f"[4/4] Found {len(worse_df)} images where cropping worsened F1")

    os.makedirs(WORSE_DIR, exist_ok=True)
    for _, row in worse_df.iterrows():
        src_img = os.path.join(IMAGES_DIR, f"{row['image_name']}.jpg")
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(WORSE_DIR, f"{row['image_name']}.jpg"))

    print(f"[✓] Copied worse images to: {WORSE_DIR}")
    print("\nTop 10 worst cases:")
    print(worse_df.sort_values('delta_f1').head(10)[["image_name", "base_f1", "crop_f1", "delta_f1"]])


if __name__ == "__main__":
    main()
