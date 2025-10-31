import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= CONFIG =================
GT_DIR     = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/data_manipulation/crop_generation/crops"

CROP_RATIO = 0.9   # proportion of original image width used for crop
ANCHOR_FROM_GT = True  # use GT box center
# ==========================================

# Input directories
IMAGE_DIR = os.path.join(GT_DIR, "images/test")
LABEL_DIR = os.path.join(GT_DIR, "labels/test")

# Output directories
IMG_OUTDIR = os.path.join(OUTPUT_DIR, "images/test")
LBL_OUTDIR = os.path.join(OUTPUT_DIR, "labels/test")
os.makedirs(IMG_OUTDIR, exist_ok=True)
os.makedirs(LBL_OUTDIR, exist_ok=True)

# ==============================================================
def load_gt_center(label_path, use_mean=False):
    """Reads YOLO label file and returns (xc, yc) of first or mean box."""
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return None

    boxes = [tuple(map(float, line.split()[1:3])) for line in lines if len(line.split()) >= 5]
    if not boxes:
        return None

    if use_mean:
        arr = np.array(boxes)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    else:
        return boxes[0]


def generate_crop_scaled_from_aspect(original_image, object_center_norm, crop_ratio):
    """
    Generate crop window centered on GT using 'padding-difference' logic.
      - crop_width  = crop_ratio * original width
      - crop_height = crop_ratio * (orig_w - orig_h)
        (i.e., removing the vertical padding that would exist when letterboxed)
    """
    orig_h, orig_w = original_image.shape[:2]

    # --- Step 1: Compute target window size ---
    win_w = int(crop_ratio * orig_w)
    win_h = int(crop_ratio * (orig_w - orig_h))
    win_h = max(1, min(win_h, orig_h))  # clamp to valid range

    # --- Step 2: Center crop around GT ---
    cx = int(np.clip(object_center_norm[0], 0, 1) * orig_w)
    cy = int(np.clip(object_center_norm[1], 0, 1) * orig_h)

    x1 = int(cx - win_w / 2)
    y1 = int(cy - win_h / 2)
    x2 = x1 + win_w
    y2 = y1 + win_h

    # --- Step 3: Clamp within image bounds ---
    if x1 < 0:
        x2 += -x1; x1 = 0
    if y1 < 0:
        y2 += -y1; y1 = 0
    if x2 > orig_w:
        x1 -= (x2 - orig_w); x2 = orig_w
    if y2 > orig_h:
        y1 -= (y2 - orig_h); y2 = orig_h

    # --- Step 4: Crop directly from original image ---
    cropped = original_image[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = np.zeros((win_h, win_w, 3), dtype=np.uint8)

    meta = {
        "crop_x1": x1, "crop_y1": y1,
        "crop_w": x2 - x1, "crop_h": y2 - y1,
        "crop_ratio": crop_ratio,
        "orig_w": orig_w, "orig_h": orig_h,
        "derived_padding": orig_w - orig_h,
    }
    return cropped, meta


def adjust_label_for_crop(label_path, meta, save_path):
    """Adjust YOLO labels for the cropped region and save new label file."""
    if not os.path.exists(label_path):
        return

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return

    crop_x1, crop_y1 = meta["crop_x1"], meta["crop_y1"]
    crop_w, crop_h = meta["crop_w"], meta["crop_h"]
    orig_w, orig_h = meta["orig_w"], meta["orig_h"]

    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, xc, yc, w, h = parts[:5]
        xc, yc, w, h = map(float, (xc, yc, w, h))

        # Convert normalized to pixel coordinates
        xc_pix = xc * orig_w
        yc_pix = yc * orig_h
        w_pix  = w * orig_w
        h_pix  = h * orig_h

        # Shift to crop coordinates
        xc_pix -= crop_x1
        yc_pix -= crop_y1

        # Skip boxes fully outside crop
        if (xc_pix + w_pix/2 < 0 or xc_pix - w_pix/2 > crop_w or
            yc_pix + h_pix/2 < 0 or yc_pix - h_pix/2 > crop_h):
            continue

        # Clip boxes to crop
        x1 = max(0, xc_pix - w_pix / 2)
        y1 = max(0, yc_pix - h_pix / 2)
        x2 = min(crop_w, xc_pix + w_pix / 2)
        y2 = min(crop_h, yc_pix + h_pix / 2)

        new_xc = (x1 + x2) / 2 / crop_w
        new_yc = (y1 + y2) / 2 / crop_h
        new_w  = (x2 - x1) / crop_w
        new_h  = (y2 - y1) / crop_h

        if new_w > 0 and new_h > 0:
            new_lines.append(f"{cls} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

    if new_lines:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("\n".join(new_lines))

# ==============================================================
# MAIN LOOP
# ==============================================================
for img_name in tqdm(sorted(os.listdir(IMAGE_DIR))):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    gt_center = load_gt_center(label_path)
    if gt_center is None:
        continue

    cropped_img, meta = generate_crop_scaled_from_aspect(image, gt_center, CROP_RATIO)

    base_name = os.path.splitext(img_name)[0]
    new_img_name = f"{base_name}_crop_{int(CROP_RATIO*100)}.jpg"
    new_lbl_name = f"{base_name}_crop_{int(CROP_RATIO*100)}.txt"

    save_img_path = os.path.join(IMG_OUTDIR, new_img_name)
    save_lbl_path = os.path.join(LBL_OUTDIR, new_lbl_name)

    cv2.imwrite(save_img_path, cropped_img)
    adjust_label_for_crop(label_path, meta, save_lbl_path)

print(f"\n✅ Cropped images and labels saved to: {OUTPUT_DIR}")
