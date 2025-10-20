import cv2
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
IMG_PATH = "/lab/projects/fire_smoke_awr/src/data_manipulation/translation_equivariance_experiment/smoke_UAV003066.jpg_cropped.png"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/data_manipulation/translation_equivariance_experiment/experiment5/images"

CANVAS_SIZE = 640
FILL_COLOR = (114, 114, 114)

RESIZE_BEFORE_PLACING = True   # whether to resize the image first
RESIZE_DIM = (80, 80)          # (width, height) if resizing is enabled
# ----------------------------------------


def place_on_canvas(img_path, out_dir, canvas_size=640, fill_color=(114, 114, 114),
                    resize_before=False, resize_dim=(320, 320)):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    if resize_before:
        img = cv2.resize(img, resize_dim)
        print(f"Resized image to {resize_dim}")

    h, w = img.shape[:2]
    base_name = Path(img_path).stem
    ext = Path(img_path).suffix

    # Divide canvas into 3x3 grid cells
    cell_size = canvas_size // 3

    # Compute centers of each grid cell
    centers = [(int((2 * i + 1) * cell_size / 2), int((2 * j + 1) * cell_size / 2))
               for j in range(3) for i in range(3)]

    names = [
        "top_left", "top_mid", "top_right",
        "mid_left", "center", "mid_right",
        "bot_left", "bot_mid", "bot_right"
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (cx, cy) in zip(names, centers):
        canvas = np.full((canvas_size, canvas_size, 3), fill_color, dtype=np.uint8)

        # Top-left corner of the image so that its center is at (cx, cy)
        x1 = int(cx - w // 2)
        y1 = int(cy - h // 2)
        x2 = x1 + w
        y2 = y1 + h

        # Boundary checks (clip if near edges)
        x1_clip, y1_clip = max(x1, 0), max(y1, 0)
        x2_clip, y2_clip = min(x2, canvas_size), min(y2, canvas_size)

        img_x1 = x1_clip - x1
        img_y1 = y1_clip - y1
        img_x2 = w - (x2 - x2_clip)
        img_y2 = h - (y2 - y2_clip)

        # Place cropped region of image on canvas
        canvas[y1_clip:y2_clip, x1_clip:x2_clip] = img[img_y1:img_y2, img_x1:img_x2]

        out_path = out_dir / f"{base_name}_{name}{ext}"
        cv2.imwrite(str(out_path), canvas)
        print(f"✅ Saved: {out_path}")

    print("✅ Finished generating 9 centered placements in 3x3 grid.")


if __name__ == "__main__":
    place_on_canvas(
        IMG_PATH,
        OUTPUT_DIR,
        CANVAS_SIZE,
        FILL_COLOR,
        RESIZE_BEFORE_PLACING,
        RESIZE_DIM
    )
