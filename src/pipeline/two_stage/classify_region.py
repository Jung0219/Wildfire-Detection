# pipeline.py
import os
import glob
import cv2
from tqdm import tqdm

from src.pipeline.two_stage.classifiers import YOLOClassifier, EVAClassifier
# python -m src.pipeline.two_stage.classify_region
# ================= CONFIG =================
IMG_DIR     = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/dev/images/test"
LABEL_DIR   = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites"
CONF_LOW    = 0.3
CONF_HIGH   = 0.525
CLASSIFIER  = "eva"                  # "yolo" or "eva"
WEIGHTS     = "/lab/projects/fire_smoke_awr/outputs/eva02/fp_mined_v2/train/weights/best.pt"
OUT_DIR     = LABEL_DIR + f"/two_stage/{CLASSIFIER}_{CONF_LOW}_{CONF_HIGH}"
# ==========================================

def yolo_to_xyxy(line, img_w, img_h):
    cls, x, y, w, h, conf = line.strip().split()
    cls, x, y, w, h, conf = int(cls), float(x), float(y), float(w), float(h), float(conf)
    x1 = int((x - w/2) * img_w)
    y1 = int((y - h/2) * img_h)
    x2 = int((x + w/2) * img_w)
    y2 = int((y + h/2) * img_h)
    return cls, x1, y1, x2, y2, conf

def xyxy_to_yolo(cls, x1, y1, x2, y2, conf, img_w, img_h):
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}\n"

def process_image(image_path, label_path, out_dir, classifier, conf_low, conf_high):
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    if not os.path.exists(label_path):
        return

    with open(label_path) as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        cls, x, y, w, h, conf = int(parts[0]), float(parts[1]), float(parts[2]), \
                                float(parts[3]), float(parts[4]), float(parts[5])

        x1 = int((x - w/2) * img_w)
        y1 = int((y - h/2) * img_h)
        x2 = int((x + w/2) * img_w)
        y2 = int((y + h/2) * img_h)

        if conf < conf_low:
            continue
        elif conf_low <= conf < conf_high:
            crop = img[max(0,y1):min(img_h,y2), max(0,x1):min(img_w,x2)]
            if crop.size == 0:
                continue
            pred_label = classifier.predict(crop)
            if pred_label == "background":
                continue
        # conf >= conf_high → always keep

        out_lines.append(xyxy_to_yolo(cls, x1, y1, x2, y2, conf, img_w, img_h))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(label_path))
    with open(out_path, "w") as f:
        f.writelines(out_lines)

def main():
    if CLASSIFIER == "yolo":
        clf = YOLOClassifier(WEIGHTS)
    elif CLASSIFIER == "eva":
        clf = EVAClassifier(WEIGHTS, device="cuda")
    else:
        raise ValueError("Unsupported classifier type")

    image_files = glob.glob(os.path.join(IMG_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(IMG_DIR, "*.png"))

    for img_path in tqdm(image_files, desc="Processing images"):
        base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, base)
        process_image(img_path, label_path, OUT_DIR, clf, CONF_LOW, CONF_HIGH)

if __name__ == "__main__":
    main()
