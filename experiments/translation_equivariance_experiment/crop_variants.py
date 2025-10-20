import cv2
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
IMG_PATH = "/lab/projects/fire_smoke_awr/src/data_manipulation/translation_equivariance_experiment/smoke_UAV003066.jpg_160cropped.png"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/data_manipulation/translation_equivariance_experiment/experiment3/images"
CANVAS_SIZE = 640
FILL_COLOR = (114, 114, 114)
# ----------------------------------------


def place_on_canvas(img_path, out_dir, canvas_size=640, fill_color=(114, 114, 114)):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h, w = img.shape[:2]
    base_name = Path(img_path).stem
    ext = Path(img_path).suffix

    # Compute positions for 9 placements
    offsets = [
        (0, 0),  # top-left
        ((canvas_size - w)//2, 0),  # top-center
        (canvas_size - w, 0),  # top-right
        (0, (canvas_size - h)//2),  # middle-left
        ((canvas_size - w)//2, (canvas_size - h)//2),  # center
        (canvas_size - w, (canvas_size - h)//2),  # middle-right
        (0, canvas_size - h),  # bottom-left
        ((canvas_size - w)//2, canvas_size - h),  # bottom-center
        (canvas_size - w, canvas_size - h)  # bottom-right
    ]

    names = [
        "top_left", "top_mid", "top_right",
        "mid_left", "center", "mid_right",
        "bot_left", "bot_mid", "bot_right"
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (x, y) in zip(names, offsets):
        canvas = np.full((canvas_size, canvas_size, 3), fill_color, dtype=np.uint8)
        canvas[y:y+h, x:x+w] = img
        out_path = out_dir / f"{base_name}_{name}{ext}"
        cv2.imwrite(str(out_path), canvas)
        print(f"✅ Saved: {out_path}")

    print("✅ Finished generating 9 placements.")


if __name__ == "__main__":
    place_on_canvas(IMG_PATH, OUTPUT_DIR, CANVAS_SIZE, FILL_COLOR)
