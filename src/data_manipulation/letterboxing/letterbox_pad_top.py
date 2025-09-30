import os
import cv2
import numpy as np
from pathlib import Path

# --------------------- CONFIGURATION ---------------------
PARENT_DIR = "/lab/projects/fire_smoke_awr/data/test_sets/A/deduplicated/phash10/single_objects"  # contains images/ and labels/
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/data/test_sets/A/deduplicated/phash10/single_objects/letterboxed_pad_top"
IMG_SIZE = 640
STRIDE = 32
PADDING_COLOR = (114, 114, 114)
# ---------------------------------------------------------

def letterbox_top(img, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[1] / shape[1], new_shape[0] / shape[0])  # scale based on height
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # width, height
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding

    # Apply vertical padding only on top, horizontal padding centered
    top = dh
    bottom = 0
    left = dw // 2
    right = dw - left

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, dw, dh  # dw is total width padding, dh is total top padding

def adjust_yolo_labels(label_path, out_label_path, r, dw, dh, orig_w, orig_h, img_size=640):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)

        # Convert from normalized to pixel
        x *= orig_w
        y *= orig_h
        w *= orig_w
        h *= orig_h

        # Apply resize scaling and padding
        x = x * r + dw / 2  # horizontal padding is centered
        y = y * r + dh      # vertical padding only on top
        w *= r
        h *= r

        # Convert back to normalized coordinates
        x /= img_size
        y /= img_size
        w /= img_size
        h /= img_size

        new_line = f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        new_lines.append(new_line)

    with open(out_label_path, 'w') as f:
        f.write("\n".join(new_lines))

def process_dataset_with_labels(parent_dir, output_dir, img_size=640, padding_color=(114, 114, 114)):
    input_img_root = Path(parent_dir) / "images"
    input_lbl_root = Path(parent_dir) / "labels"

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
        img_lb, r, dw, dh = letterbox_top(img, new_shape=(img_size, img_size), color=padding_color, scaleup=True)

        out_img_path = Path(output_dir) / "images" / relative_path
        out_lbl_path = Path(output_dir) / "labels" / relative_path.with_suffix(".txt")

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_img_path), img_lb)

        if label_path.exists():
            adjust_yolo_labels(label_path, out_lbl_path, r, dw, dh, orig_w, orig_h, img_size)

        count += 1

    print(f"Processed {count} images with labels into: {output_dir}")

if __name__ == "__main__":
    process_dataset_with_labels(PARENT_DIR, OUTPUT_DIR, IMG_SIZE, PADDING_COLOR)
