import os, shutil
from pathlib import Path
import numpy as np

# ===================================================
# CONFIGURATION
# ===================================================

GT_DIR      = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire_test_clean"
PRED_BASE_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/test_set_clean/manual_resize_original_inference/labels"
PRED_CROP_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/test_set_clean/manual_resize_crop_inference/labels"
SAVE_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/early_fire_pad_aug/test_set_clean/manual_resize_crop_inference/baseline_comparison"

IOU_THR = 0.5
CONF_DROP_THR = 0.1

# ===================================================
IMAGE_DIR = os.path.join(GT_DIR, "images/test")
LABEL_DIR = os.path.join(GT_DIR, "labels/test")

# ===================================================
categories = ["False_Negatives", "False_Positives", "TruePos_LowerConf"]
for cat in categories:
    for sub in ["images/test", "labels/test", "baseline", "cropped"]:
        Path(SAVE_DIR, cat, sub).mkdir(parents=True, exist_ok=True)

def read_yolo_file(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return [list(map(float, line.strip().split())) for line in f if line.strip()]

def iou(box1, box2):
    _, x1, y1, w1, h1 = box1[:5]
    _, x2, y2, w2, h2 = box2[:5]
    b1 = [x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2]
    b2 = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]
    inter_x1, inter_y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    inter_x2, inter_y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter + 1e-9)

def match_and_count_fp(gt_boxes, preds):
    if not gt_boxes or not preds:
        return set(), len(preds)

    iou_pairs = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(preds):
            if int(g[0]) != int(p[0]):  # class mismatch
                continue
            iou_v = iou(g, p)
            if iou_v >= IOU_THR:
                iou_pairs.append((iou_v, gi, pi))

    iou_pairs.sort(reverse=True, key=lambda x: x[0])
    matched_gt, matched_preds = set(), set()
    for _, gi, pi in iou_pairs:
        if gi not in matched_gt and pi not in matched_preds:
            matched_gt.add(gi)
            matched_preds.add(pi)

    fp_count = len(preds) - len(matched_preds)
    return matched_gt, fp_count

# ===================================================
# MAIN LOOP
# ===================================================
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue
    stem = Path(img_name).stem

    img_path = Path(IMAGE_DIR) / img_name
    gt_path = Path(LABEL_DIR) / f"{stem}.txt"
    base_path = Path(PRED_BASE_DIR) / f"{stem}.txt"
    crop_path = Path(PRED_CROP_DIR) / f"{stem}.txt"

    gt = read_yolo_file(gt_path)
    base = read_yolo_file(base_path)
    crop = read_yolo_file(crop_path)

    base_matched, base_fp = match_and_count_fp(gt, base)
    crop_matched, crop_fp = match_and_count_fp(gt, crop)

    # ---- NEW: FN just means cropped missed any GT box
    fn_increase = len(crop_matched) < len(gt)
    fp_increase = crop_fp > base_fp
    
    # Confidence drop (if conf values exist)
    conf_base = np.median([p[5] for p in base if len(p) > 5]) if any(len(p) > 5 for p in base) else None
    conf_crop = np.median([p[5] for p in crop if len(p) > 5]) if any(len(p) > 5 for p in crop) else None
    lower_conf = conf_base is not None and conf_crop is not None and conf_crop < conf_base - CONF_DROP_THR

    if fn_increase:
        cat = "False_Negatives"
    elif fp_increase:
        cat = "False_Positives"
    elif lower_conf:
        cat = "TruePos_LowerConf"
    else:
        continue

    shutil.copy(img_path, Path(SAVE_DIR, cat, "images/test", img_name))
    if gt_path.exists():
        shutil.copy(gt_path, Path(SAVE_DIR, cat, "labels/test", f"{stem}.txt"))
    if base_path.exists():
        shutil.copy(base_path, Path(SAVE_DIR, cat, "baseline", f"{stem}.txt"))
    if crop_path.exists():
        shutil.copy(crop_path, Path(SAVE_DIR, cat, "cropped", f"{stem}.txt"))
