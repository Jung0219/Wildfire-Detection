import os

# ================= CONFIG =================
GT_DIR     = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/dev/labels/test"
PRED_DIR   = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites"
OUT_DIR    = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites/tps"
CONF_CUT   = 1.0   # only keep TPs <= this confidence
IOU_THRESH = 0.5   # IoU threshold for TP matching
MODE       = "strict"  # "strict" = classwise, one GT match
                     # "loose"  = class-agnostic, multiple TPs per GT
# ==========================================


def parse_gt_line(line):
    cls, x, y, w, h = line.strip().split()
    return int(cls), float(x), float(y), float(w), float(h)


def parse_pred_line(line):
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


def filter_tp(gt_dir, pred_dir, out_dir, conf_cut, iou_thresh, mode):
    os.makedirs(out_dir, exist_ok=True)

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

        preds_out = []
        matched = set()

        for cls, x, y, w, h, conf in preds:
            best_iou, best_idx = 0.0, -1
            for i, (gcls, gx, gy, gw, gh) in enumerate(gts):
                # strict: enforce class match and unique GT usage
                if mode == "strict" and (gcls != cls or i in matched):
                    continue
                # loose: ignore class and allow multiple matches
                iou_val = iou((x, y, w, h), (gx, gy, gw, gh))
                if iou_val > best_iou:
                    best_iou, best_idx = iou_val, i

            if best_iou >= iou_thresh:
                if conf <= conf_cut:
                    preds_out.append(f"{cls} {x} {y} {w} {h} {conf}\n")
                if mode == "strict":
                    matched.add(best_idx)  # block reuse only in strict mode

        out_file = os.path.join(out_dir, fname)
        with open(out_file, "w") as f:
            f.writelines(preds_out)


if __name__ == "__main__":
    filter_tp(GT_DIR, PRED_DIR, OUT_DIR, CONF_CUT, IOU_THRESH, MODE)
    print(f"Filtered TPs written to {OUT_DIR}")
