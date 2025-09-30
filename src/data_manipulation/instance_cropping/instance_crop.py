import os
import cv2
from glob import glob
from tqdm import tqdm

# =============== CONFIG ===============
PARENT_DIR     = "/lab/projects/fire_smoke_awr/data/detection/training/ABCDE_noEF"      
PRED_LABEL_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/inference_train_data/composites/fp"
OUTPUT_DIR     = "/lab/projects/fire_smoke_awr/data/classification/datasets/ABCDE_noEF_train/detector_tp+fp"
# ======================================

# known classes (adjust as needed)
CLASS_MAP = {
    0: "background",
    1: "background"
}

IMG_DIR  = os.path.join(PARENT_DIR, "images/train")
IMG_EXTS = [".jpg", ".png", ".jpeg"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def yolo_to_xyxy(box, img_w, img_h):
    """Convert YOLO [cls cx cy w h] to absolute [x1, y1, x2, y2, cls]."""
    cls, cx, cy, w, h = box
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return [max(0, x1), max(0, y1), min(img_w-1, x2), min(img_h-1, y2), int(cls)]

# collect prediction files
pred_files = glob(os.path.join(PRED_LABEL_DIR, "*.txt"))

for pred_file in tqdm(pred_files, desc="Cropping instances"):
    base = os.path.splitext(os.path.basename(pred_file))[0]

    # find matching image file
    img_file = None
    for ext in IMG_EXTS:
        cand = os.path.join(IMG_DIR, base + ext)
        if os.path.exists(cand):
            img_file = cand
            break
    if img_file is None:
        continue

    img = cv2.imread(img_file)
    if img is None:
        continue
    H, W = img.shape[:2]

    with open(pred_file) as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        box = list(map(float, parts[:5]))  # cls cx cy w h
        x1, y1, x2, y2, cls = yolo_to_xyxy(box, W, H)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        class_name = CLASS_MAP.get(cls, f"class{cls}")
        filename   = f"{base}_{class_name}_{idx}.jpg"

        # save under OUTPUT_DIR/<class_name>/
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        cv2.imwrite(os.path.join(class_dir, filename), crop)

print(f"\n[INFO] Finished. Crops saved under {OUTPUT_DIR}/<class_name>/")
