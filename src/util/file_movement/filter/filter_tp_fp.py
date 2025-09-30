import os
import glob
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
GT_DIR = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/dev/labels/test"         # ground truth labels (.txt)
PRED_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites/conf_lt_0.3"     # predicted labels (.txt)
TP_OUT_DIR = os.path.join(PRED_DIR, "tp")
FP_OUT_DIR = os.path.join(PRED_DIR, "fp")
TP_IOU_THRESHOLD = 0.5                # IoU cutoff for TP
FP_IOU_THRESHOLD = 0.5                # IoU cutoff for FP
CLASS_AWARE = False                   # True = class-aware, False = class-agnostic
# =========================

os.makedirs(TP_OUT_DIR, exist_ok=True)
os.makedirs(FP_OUT_DIR, exist_ok=True)


def compute_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def load_labels(path):
    boxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            rest = parts[5:]  # keep confidence or extras if present
            boxes.append((cls, [x, y, w, h], rest, line.strip()))
    return boxes


def evaluate(gt_dir, pred_dir, tp_thr, fp_thr, tp_out, fp_out, class_aware=True):
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))

    total_tp, total_fp = 0, 0

    for gt_file in tqdm(gt_files, desc="Processing files"):
        file_name = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, file_name)

        tp_lines, fp_lines = [], []

        if not os.path.exists(pred_file):
            # if prediction file missing, skip but create empties
            open(os.path.join(tp_out, file_name), "w").close()
            open(os.path.join(fp_out, file_name), "w").close()
            continue

        gt_boxes = load_labels(gt_file)
        pred_boxes = load_labels(pred_file)

        # matched = set()
        for cls, pbox, rest, raw_line in pred_boxes:
            best_iou = 0
            best_idx = -1

            for i, (gt_cls, gt_box, _, _) in enumerate(gt_boxes):
                if class_aware and cls != gt_cls:
                    continue
                # if i in matched:
                #    continue
                iou = compute_iou(pbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                if best_iou >= tp_thr:
                    tp_lines.append(raw_line + "\n")
                    # matched.add(best_idx)
                    total_tp += 1
                elif best_iou <= fp_thr:
                    fp_lines.append(raw_line + "\n")
                    total_fp += 1
            else:
                fp_lines.append(raw_line + "\n")
                total_fp += 1

        # Write TP/FP results
        with open(os.path.join(tp_out, file_name), "w") as f:
            f.writelines(tp_lines)
        with open(os.path.join(fp_out, file_name), "w") as f:
            f.writelines(fp_lines)

    return total_tp, total_fp


if __name__ == "__main__":
    tp_count, fp_count = evaluate(
        GT_DIR, PRED_DIR,
        TP_IOU_THRESHOLD, FP_IOU_THRESHOLD,
        TP_OUT_DIR, FP_OUT_DIR,
        class_aware=CLASS_AWARE
    )
    print(f"Done. TP files in {TP_OUT_DIR}, FP files in {FP_OUT_DIR}")
    print(f"Total TPs: {tp_count}")
    print(f"Total FPs: {fp_count}")
