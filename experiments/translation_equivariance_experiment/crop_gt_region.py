import cv2
import numpy as np
import os
from pathlib import Path

# ---------------- CONFIG ----------------
CROP_SIZE = 320
IMG_PATH = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/all/images/test/smoke_UAV003066.jpg"
LABEL_PATH = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/all/labels/test/smoke_UAV003066.txt"
OUTPUT_PATH = "/lab/projects/fire_smoke_awr/src/data_manipulation/translation_equivariance/" + os.path.basename(IMG_PATH) + f"_{CROP_SIZE}cropped.png"
# ----------------------------------------


def crop_centered_on_object(img_path, label_path, out_path, crop_size=320):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h, w = img.shape[:2]
    label_file = Path(label_path)
    if not label_file.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    with open(label_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise ValueError("No objects found in label file")

    # Use first object in label
    _, x, y, _, _ = map(float, lines[0].strip().split())

    # Convert YOLO normalized center (x, y) to pixel coordinates
    cx = int(x * w)
    cy = int(y * h)

    # Crop boundaries
    half = crop_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)

    # Adjust if near border
    if x2 - x1 < crop_size:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size:
        y1 = max(0, y2 - crop_size)

    crop = img[y1:y2, x1:x2]

    # Save output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    print(f"✅ Saved cropped image: {out_path}")


if __name__ == "__main__":
    crop_centered_on_object(IMG_PATH, LABEL_PATH, OUTPUT_PATH, CROP_SIZE)
