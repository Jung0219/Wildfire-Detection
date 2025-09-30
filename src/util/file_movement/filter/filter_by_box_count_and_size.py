import os
import shutil
import math
from PIL import Image

# ==============================
# CONFIGURATION
# ==============================
CONFIG = {
    # Input dataset parent with images/ and labels/
    "input_parent": "/lab/projects/fire_smoke_awr/data/detection/training/ABCDE_noEF",

    # Output directory for filtered dataset
    "output_parent": "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/fp_mining",

    # Target image size after letterboxing
    "target_size": 640,

    # Required number of boxes per image
    "n_boxes": 1,

    # Max allowed box size (sqrt(w*h), after letterbox scaling)
    "max_box_size": 130,

    # File extensions considered as images
    "img_exts": [".jpg", ".jpeg", ".png"],
}
# ==============================


def letterbox_scale(img_w, img_h, target):
    """Return scaling factor and padding for letterbox resize to (target, target)."""
    scale = min(target / img_w, target / img_h)
    new_w, new_h = int(round(img_w * scale)), int(round(img_h * scale))
    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2
    return scale, pad_w, pad_h


def filter_dataset(cfg):
    img_dir = os.path.join(cfg["input_parent"], "images/train")
    lbl_dir = os.path.join(cfg["input_parent"], "labels/train")

    out_img_dir = os.path.join(cfg["output_parent"], "images/test")
    out_lbl_dir = os.path.join(cfg["output_parent"], "labels/test")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    kept = 0
    total = 0

    for file in os.listdir(img_dir):
        if not any(file.lower().endswith(ext) for ext in cfg["img_exts"]):
            continue

        total += 1
        img_path = os.path.join(img_dir, file)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(file)[0] + ".txt")

        if not os.path.exists(lbl_path):
            continue

        # Load image size
        with Image.open(img_path) as im:
            w, h = im.size

        scale, pad_w, pad_h = letterbox_scale(w, h, cfg["target_size"])

        box_sizes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, x, y, bw, bh = parts
                x, y, bw, bh = map(float, (x, y, bw, bh))

                # Convert YOLO normalized bbox to pixel size
                bw_px = bw * w
                bh_px = bh * h

                # Scale after letterboxing
                bw_scaled = bw_px * scale
                bh_scaled = bh_px * scale

                # Compute box size = sqrt(w*h)
                box_size = math.sqrt(bw_scaled * bh_scaled)
                box_sizes.append(box_size)

        # Filter: correct count and all box sizes below threshold
        if len(box_sizes) == cfg["n_boxes"] and all(s < cfg["max_box_size"] for s in box_sizes):
            shutil.copy2(img_path, os.path.join(out_img_dir, file))
            shutil.copy2(lbl_path, os.path.join(out_lbl_dir, os.path.basename(lbl_path)))
            kept += 1

    print(f"Kept {kept}/{total} images.")


if __name__ == "__main__":
    filter_dataset(CONFIG)
