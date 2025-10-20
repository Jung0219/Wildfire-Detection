import cv2
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
IMG_PATH   = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/all/images/test/smoke_UAV003830.jpg"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/data_manipulation/letterboxing/samples/images"

IMG_SIZE = 640
PADDING_COLOR = (114, 114, 114)
PAD_RATIO_W = 0.5
PAD_RATIO_H_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
# ----------------------------------------


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              scaleup=True, pad_ratio_w=0.5, pad_ratio_h=0.5):
    """Resize and pad image to a square shape with uneven padding distribution."""
    shape = img.shape[:2]  # (height, width)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Resize ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    left = int(round(dw * (1 - pad_ratio_w)))
    right = int(round(dw * pad_ratio_w))
    top = int(round(dh * (1 - pad_ratio_h)))
    bottom = int(round(dh * pad_ratio_h))

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img


def resize_preserve_aspect(img, size):
    """Resize image keeping aspect ratio, no padding."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


if __name__ == "__main__":
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(IMG_PATH).stem
    ext = Path(IMG_PATH).suffix

    # (1) Save baseline resized version (aspect ratio preserved)
    resized_img = resize_preserve_aspect(img, IMG_SIZE)
    out_path = out_dir / f"{base_name}_resized{ext}"
    cv2.imwrite(str(out_path), resized_img)
    print(f"✅ Saved: {out_path}")

    # (2) Save padded versions (different vertical pad ratios)
    for pad_h in PAD_RATIO_H_VALUES:
        padded = letterbox(
            img,
            new_shape=IMG_SIZE,
            color=PADDING_COLOR,
            pad_ratio_w=PAD_RATIO_W,
            pad_ratio_h=pad_h,
        )
        suffix = f"_padH{pad_h:.2f}".replace(".", "")
        out_path = out_dir / f"{base_name}{suffix}{ext}"
        cv2.imwrite(str(out_path), padded)
        print(f"✅ Saved: {out_path}")

    print("✅ Done — generated 6 total versions (1 resized + 5 padded).")
