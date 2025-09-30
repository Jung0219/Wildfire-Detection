import cv2
import numpy as np
import os

# ==== CONFIGURATION ====
IMAGE_PATH = "/lab/projects/fire_smoke_awr/data/test_sets/A+B+C+D+E/deduplicated/phash10/small_smoke/images/test/AoF00002.jpg"  # Set your image path here
X_COORD = 154
Y_COORD = 28
# ========================

def get_pixel_values(image_path, x, y):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    h, w, _ = image.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"Coordinates ({x}, {y}) out of bounds for image size ({w}, {h})")

    # BGR from OpenCV
    b, g, r = image[y, x]

    # Convert to YCrCb and get Y
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_val = ycrcb[y, x, 0]

    print(f"Pixel at ({x}, {y}):")
    print(f"  Y (luminance): {y_val}")
    print(f"  R: {r}")
    print(f"  G: {g}")
    print(f"  B: {b}")

if __name__ == "__main__":
    get_pixel_values(IMAGE_PATH, X_COORD, Y_COORD)
