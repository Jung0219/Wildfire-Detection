import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= CONFIG =================
IMG_DIR     = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/dev/images/test"
PRED_DIR    = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites/conf_lt_0.3/fp"   # YOLO-format .txt files (one per image)
IOU_THRESH  = 0.25
OUT_DIR     = PRED_DIR + "/analysis_plots"
os.makedirs(OUT_DIR, exist_ok=True)
# ==========================================


def parse_pred_line(line):
    """YOLO pred format: cls x y w h conf (normalized)."""
    parts = line.strip().split()
    return {
        "cls": int(parts[0]),
        "x": float(parts[1]),
        "y": float(parts[2]),
        "w": float(parts[3]),
        "h": float(parts[4]),
        "conf": float(parts[5]) if len(parts) > 5 else 1.0
    }


def iou(box1, box2):
    # boxes are [x_center, y_center, w, h], normalized [0,1]
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


def extract_hue_saturation(img, box):
    H, W = img.shape[:2]
    xc, yc, w, h = box
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0, 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:, :, 0].mean()
    s_mean = hsv[:, :, 1].mean()
    return h_mean, s_mean


def yolo_box_on_letterbox(box, H, W, target=640):
    """
    Convert normalized YOLO box [xc, yc, w, h] into absolute
    width, height after letterbox resize to target x target.
    """
    r = min(target / W, target / H)
    new_w, new_h = int(round(W * r)), int(round(H * r))

    pad_w = (target - new_w) / 2
    pad_h = (target - new_h) / 2

    xc, yc, bw, bh = box
    bw_abs = bw * W * r
    bh_abs = bh * H * r
    xc_abs = xc * W * r + pad_w
    yc_abs = yc * H * r + pad_h

    return bw_abs, bh_abs


def main():
    all_traits = []

    pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith(".txt")]
    for pred_file in tqdm(pred_files, desc="Processing prediction files"):
        img_file = os.path.splitext(pred_file)[0] + ".jpg"  # adjust if .png
        img_path = os.path.join(IMG_DIR, img_file)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        with open(os.path.join(PRED_DIR, pred_file), "r") as f:
            preds = [parse_pred_line(line) for line in f if line.strip()]

        for i, p in enumerate(preds):
            # area in original pixel units
            area = p["w"] * W * p["h"] * H

            # size in letterboxed 640x640
            bw_640, bh_640 = yolo_box_on_letterbox([p["x"], p["y"], p["w"], p["h"]], H, W, target=640)
            size_640 = np.sqrt(bw_640 * bh_640)

            # hue/saturation
            hue, sat = extract_hue_saturation(img, [p["x"], p["y"], p["w"], p["h"]])

            # overlaps
            overlaps = 0
            for j, q in enumerate(preds):
                if i == j:
                    continue
                if iou([p["x"], p["y"], p["w"], p["h"]],
                       [q["x"], q["y"], q["w"], q["h"]]) >= IOU_THRESH:
                    overlaps += 1

            all_traits.append({
                "image": img_file,
                "class": p["cls"],
                "area": area,
                "size_640": size_640,
                "hue": hue,
                "saturation": sat,
                "overlaps": overlaps
            })

    df = pd.DataFrame(all_traits)

    # violin plots in one figure
    # violin plots (without area) on one figure with fixed y-axis
    traits = ["size_640", "hue", "saturation", "overlaps"]
    ylims = {
        "size_640": (0, 200),
        "hue": (0, 150),
        "saturation": (0, 160),
        "overlaps": (0, 10)
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, trait in zip(axes.flatten(), traits):
        sns.violinplot(y=df[trait], inner="box", ax=ax)
        ax.set_ylim(ylims[trait])
        ax.set_title(f"{trait.capitalize()} distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "all_traits_violin.png"))
    plt.close()


    print(f"Done. Saved violin plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
