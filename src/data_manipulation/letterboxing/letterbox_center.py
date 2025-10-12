import os
import cv2
import numpy as np
from pathlib import Path

# --------------------- CONFIGURATION ---------------------
PARENT_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire"  # contains images/ and labels/
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire_letterbox"  # output root for images/ and labels/
IMG_SIZE = 640
STRIDE = 32
PADDING_COLOR = (114, 114, 114)
# ---------------------------------------------------------

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        new_unpad = new_shape
        dw, dh = 0, 0
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, dw, dh

def adjust_yolo_labels(label_path, out_label_path, r, dw, dh, orig_w, orig_h, img_size=640):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)

        x *= orig_w
        y *= orig_h
        w *= orig_w
        h *= orig_h

        x = x * r + dw
        y = y * r + dh
        w *= r
        h *= r

        x /= img_size
        y /= img_size
        w /= img_size
        h /= img_size

        new_line = f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        new_lines.append(new_line)

    with open(out_label_path, 'w') as f:
        f.write("\n".join(new_lines))

def process_dataset_with_labels(parent_dir, output_dir, img_size=640, stride=32, padding_color=(114, 114, 114)):
    input_img_root = Path(parent_dir) / "images/train"
    input_lbl_root = Path(parent_dir) / "labels/train"

    image_files = list(input_img_root.rglob("*.*"))
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    count = 0
    for img_path in image_files:
        if img_path.suffix.lower() not in valid_exts:
            continue
        relative_path = img_path.relative_to(input_img_root)
        label_path = input_lbl_root / relative_path.with_suffix(".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]
        img_lb, r, dw, dh = letterbox(img, new_shape=img_size, color=padding_color, auto=False, stride=stride)

        out_img_path = Path(output_dir) / "images/train" / relative_path
        out_lbl_path = Path(output_dir) / "labels/train" / relative_path.with_suffix(".txt")

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_img_path), img_lb)

        if label_path.exists():
            adjust_yolo_labels(label_path, out_lbl_path, r, dw, dh, orig_w, orig_h, img_size)

        count += 1

    print(f"Processed {count} images with labels into: {output_dir}")

if __name__ == "__main__":
    process_dataset_with_labels(PARENT_DIR, OUTPUT_DIR, IMG_SIZE, STRIDE, PADDING_COLOR)
