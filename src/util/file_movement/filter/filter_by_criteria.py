import os
import shutil
import math
from tqdm import tqdm
from PIL import Image, ImageStat

# ==============================
# CONFIGURATION
# ==============================
CONFIG = {
    "input_parent": "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/early_fire_A_only",
    "output_parent": "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/early_smoke_A",
    "target_size": 640,
    "n_boxes": 1,
    "max_box_size": 80,
    "min_brightness": 80,
    "target_class": 1,  # <-- NEW: keep only this class
    "img_exts": [".jpg", ".jpeg", ".png"],
}
# ==============================


def letterbox_scale(img_w, img_h, target):
    scale = min(target / img_w, target / img_h)
    new_w, new_h = int(round(img_w * scale)), int(round(img_h * scale))
    pad_w = (target - new_w) // 2
    pad_h = (target - new_h) // 2
    return scale, pad_w, pad_h


def compute_brightness(image_path):
    with Image.open(image_path) as im:
        gray = im.convert("L")
        stat = ImageStat.Stat(gray)
        return stat.mean[0]


def filter_dataset(cfg):
    img_dir = os.path.join(cfg["input_parent"], "images/test")
    lbl_dir = os.path.join(cfg["input_parent"], "labels/test")

    out_img_dir = os.path.join(cfg["output_parent"], "images/test")
    out_lbl_dir = os.path.join(cfg["output_parent"], "labels/test")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    kept = 0
    total = 0

    for file in tqdm(os.listdir(img_dir), desc="Processing images"):
        if not any(file.lower().endswith(ext) for ext in cfg["img_exts"]):
            continue

        total += 1
        img_path = os.path.join(img_dir, file)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(file)[0] + ".txt")

        if not os.path.exists(lbl_path):
            continue

        # Brightness filter
        brightness = compute_brightness(img_path)
        if brightness < cfg["min_brightness"]:
            continue

        with Image.open(img_path) as im:
            w, h = im.size
        scale, pad_w, pad_h = letterbox_scale(w, h, cfg["target_size"])

        box_sizes = []
        class_ids = []

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, x, y, bw, bh = parts
                cls = int(cls)
                x, y, bw, bh = map(float, (x, y, bw, bh))

                bw_px = bw * w
                bh_px = bh * h
                bw_scaled = bw_px * scale
                bh_scaled = bh_px * scale
                box_size = math.sqrt(bw_scaled * bh_scaled)

                box_sizes.append(box_size)
                class_ids.append(cls)

        # Filter: correct count, class match, small box
        if (
            len(box_sizes) == cfg["n_boxes"]
            and all(s < cfg["max_box_size"] for s in box_sizes)
            and all(c == cfg["target_class"] for c in class_ids)
        ):
            shutil.copy2(img_path, os.path.join(out_img_dir, file))
            shutil.copy2(lbl_path, os.path.join(out_lbl_dir, os.path.basename(lbl_path)))
            kept += 1

    print(f"Kept {kept}/{total} images.")


if __name__ == "__main__":
    filter_dataset(CONFIG)
