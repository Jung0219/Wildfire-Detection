#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
INPUT_DIR  = "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/all/images/test"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/pipeline/early_fire_scene_classifier/fire_overlay"

# Define Y, Cb, Cr thresholds (inclusive)
Y_MIN = 80          # Minimum Y value (luminance)
Y_MAX = 255         # Maximum Y value
CB_MIN = 0          # Minimum Cb value
CB_MAX = 140        # Maximum Cb value
CR_MIN = 150        # Minimum Cr value
CR_MAX = 255        # Maximum Cr value

MASK_COLOR = (0, 255, 0)   # overlay color
ALPHA = 0.4                # mask transparency
MASK_RATIO_THRESH = 0.05    # threshold for splitting into folders 5%
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ==========================================

def list_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS]

def bgr_to_ycbcr(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)  # note: OpenCV order is Y, Cr, Cb
    return Y, Cb, Cr

def make_mask(img, y_min, y_max, cb_min, cb_max, cr_min, cr_max):
    Y, Cb, Cr = bgr_to_ycbcr(img)
    mask = (
        (Y >= y_min) & (Y <= y_max) &
        (Cb >= cb_min) & (Cb <= cb_max) &
        (Cr >= cr_min) & (Cr <= cr_max)
    )
    return mask.astype(np.uint8)

def overlay_mask(img, mask, color=(0,0,255), alpha=0.6):
    """Apply colored overlay on masked regions."""
    vis = img.copy()
    if mask.any():
        overlay = np.full_like(vis, color, dtype=np.uint8)
        vis = np.where(mask[...,None].astype(bool),
                       (alpha*overlay + (1-alpha)*vis).astype(np.uint8),
                       vis)
    return vis

def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)

    # Create subdirectories
    regular_dir = out_dir / "regular"
    multires_dir = out_dir / "multiresolution"
    regular_dir.mkdir(parents=True, exist_ok=True)
    multires_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list_images(in_dir)
    if not img_paths:
        raise SystemExit("No images found in input folder.")

    for path in tqdm(img_paths, desc="Processing images", unit="img"):
        img = cv2.imread(str(path))
        if img is None:
            continue

        mask = make_mask(img, Y_MIN, Y_MAX, CB_MIN, CB_MAX, CR_MIN, CR_MAX)
        vis  = overlay_mask(img, mask, color=MASK_COLOR, alpha=ALPHA)

        # Compute mask ratio
        ratio = mask.mean()  # since mask is 0/1, mean = fraction of masked pixels

        # Decide output folder
        if ratio > MASK_RATIO_THRESH:
            save_dir = regular_dir
        else:
            save_dir = multires_dir

        # Side-by-side concat
        combined = np.hstack([img, vis])
        out_path = save_dir / f"{path.stem}_sidebyside.png"
        cv2.imwrite(str(out_path), combined)

    print(f"\n✅ Done. Saved {len(img_paths)} results into:")
    print(f"   Regular (> {MASK_RATIO_THRESH:.2f} mask ratio): {regular_dir}")
    print(f"   Multiresolution (<= {MASK_RATIO_THRESH:.2f} mask ratio): {multires_dir}")

if __name__ == "__main__":
    main()
