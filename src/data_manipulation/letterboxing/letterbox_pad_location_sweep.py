import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
PARENT_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/ABCDE_noEF_pad_aug"
OUTPUT_DIR = PARENT_DIR # + "/padding_augmented"

IMAGES_SUBDIR = "images/train"
LABELS_SUBDIR = "labels/train"
IMG_SIZE = 640
PADDING_COLOR = (114, 114, 114)

PAD_RATIO_W = 0.5   # keep horizontal centered
PAD_RATIO_H_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # sweep vertically
# ----------------------------------------


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              scaleup=True, stride=32, pad_ratio_w=0.5, pad_ratio_h=0.5):
    """Resize and pad image to a square shape with uneven padding distribution."""
    shape = img.shape[:2]  # (height, width)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute resize ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # New resized (unpadded) shape
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # Distribute padding
    left = int(round(dw * (1 - pad_ratio_w)))
    right = int(round(dw * pad_ratio_w))
    top = int(round(dh * (1 - pad_ratio_h)))
    bottom = int(round(dh * pad_ratio_h))

    # Resize + pad
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, left, top


def adjust_yolo_labels(label_path, out_label_path, r, left, top, orig_w, orig_h, img_size=640):
    """Adjust YOLO label coordinates based on resize and padding."""
    if not label_path.exists():
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)

        # Denormalize (to pixels)
        x *= orig_w
        y *= orig_h
        w *= orig_w
        h *= orig_h

        # Apply resize and padding
        x = x * r + left
        y = y * r + top
        w *= r
        h *= r

        # Normalize to new image size
        x /= img_size
        y /= img_size
        w /= img_size
        h /= img_size

        new_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_label_path, "w") as f:
        f.write("\n".join(new_lines))


def process_with_multiple_pads(parent_dir, output_dir, img_size=640,
                               pad_ratio_w=0.5, pad_ratio_h_values=None, padding_color=(114, 114, 114)):
    """Process each image and create multiple padded versions with different vertical ratios."""
    if pad_ratio_h_values is None:
        pad_ratio_h_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    img_root = Path(parent_dir) / IMAGES_SUBDIR
    lbl_root = Path(parent_dir) / LABELS_SUBDIR
    out_img_root = Path(output_dir) / IMAGES_SUBDIR
    out_lbl_root = Path(output_dir) / LABELS_SUBDIR

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = [p for p in img_root.rglob("*") if p.suffix.lower() in valid_exts]

    for img_path in tqdm(image_files, desc="Processing images", unit="img"):
        label_path = lbl_root / img_path.relative_to(img_root).with_suffix(".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"[WARN] Skipped unreadable image: {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]

        # Create padded versions
        for pad_h in pad_ratio_h_values:
            img_lb, r, left, top = letterbox(
                img,
                new_shape=img_size,
                color=padding_color,
                pad_ratio_w=pad_ratio_w,
                pad_ratio_h=pad_h,
            )

            suffix = f"_padH{pad_h:.2f}".replace(".", "")
            out_img_version = out_img_root / img_path.relative_to(img_root).with_name(img_path.stem + suffix + img_path.suffix)
            out_lbl_version = out_lbl_root / label_path.relative_to(lbl_root).with_name(label_path.stem + suffix + ".txt")

            cv2.imwrite(str(out_img_version), img_lb)
            adjust_yolo_labels(label_path, out_lbl_version, r, left, top, orig_w, orig_h, img_size)

    print(f"✅ Finished creating 5 padded versions (plus original) for {len(image_files)} images into: {output_dir}")


if __name__ == "__main__":
    process_with_multiple_pads(
        PARENT_DIR,
        OUTPUT_DIR,
        img_size=IMG_SIZE,
        pad_ratio_w=PAD_RATIO_W,
        pad_ratio_h_values=PAD_RATIO_H_VALUES,
        padding_color=PADDING_COLOR
    )
