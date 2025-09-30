import os
import csv
import cv2
import torch
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

# ========== CONFIG ==========
LABEL_DIR     = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/merged_labels"    # relabeled tp/fp labels
IMAGE_DIR     = "/lab/projects/fire_smoke_awr/data/detection/test_sets/A/deduplicated/phash10/images/test"
CSV_PATH      = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/output.csv"
MODEL_PARENT  = "/lab/projects/fire_smoke_awr/outputs/yolo/classification/BCDE_val_ground_truth_tps_ternary"
DEVICE        = 0  # or "cpu"
# ============================

# load classifier
classifier = YOLO(MODEL_PARENT + "/train/weights/best.pt")

# Get classifier class names
rows = []
label_files = glob(os.path.join(LABEL_DIR, "*.txt"))

for lf in tqdm(label_files, desc="Processing label files"):
    img_name = os.path.splitext(os.path.basename(lf))[0] + ".jpg"
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    with open(lf, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f" Cropping + classifying {img_name}", leave=False):
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls, x, y, w, h, det_conf = parts
        cls = int(cls)  # 1=TP, 0=FP
        x, y, w, h, det_conf = map(float, [x, y, w, h, det_conf])

        # convert YOLO format -> pixel box
        h_img, w_img = img.shape[:2]
        x_c, y_c = x * w_img, y * h_img
        bw, bh = w * w_img, h * h_img
        x1, y1 = int(x_c - bw/2), int(y_c - bh/2)
        x2, y2 = int(x_c + bw/2), int(y_c + bh/2)

        crop = img[max(0,y1):min(h_img,y2), max(0,x1):min(w_img,x2)]
        if crop.size == 0:
            continue

        # run classifier
        results = classifier(crop, device=DEVICE, verbose=False)
        if results[0].probs is None:
            continue

        # directly get probabilities as numpy
        probs = results[0].probs.data.cpu().numpy()   # shape = (3,)

        # TP confidence = fire + smoke
        cls_conf = (probs[1] + probs[2]).clip(0, 1)

        rows.append([img_name, det_conf, cls_conf, cls])


# save to CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "det_conf", "cls_conf", "label"])
    writer.writerows(rows)

print(f"Saved {len(rows)} entries to {CSV_PATH}")
