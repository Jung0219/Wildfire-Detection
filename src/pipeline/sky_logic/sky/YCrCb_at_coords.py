#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import numpy as np

# ======================= CONFIG =======================
IMAGE_PATH   = "/lab/projects/fire_smoke_awr/src/util/experiment/sky/sky_masked/bothFireAndSmoke_UAV000073_side.png"  # <-- set this
PRINT_BGR    = True                     # also print BGR at the pixel
ONE_BASED_IN = False                     # set True if you type 1-based coords
PROMPT       = True                      # show a small prompt while waiting
# ======================================================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def to_ycbcr_arrays(img_bgr):
    # OpenCV conversion yields YCrCb; reorder to Y, Cb, Cr
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    return Y, Cb, Cr

def print_point(x, y, Y, Cb, Cr, img_bgr=None, print_bgr=False, one_based=False):
    H, W = Y.shape
    x_disp, y_disp = x, y
    if one_based:
        x -= 1
        y -= 1
    if not (0 <= x < W and 0 <= y < H):
        sys.stdout.write(f"(x={x_disp}, y={y_disp}) -> OUT OF BOUNDS for image {W}x{H}\n")
        sys.stdout.flush()
        return
    yv, cbv, crv = int(Y[y, x]), int(Cb[y, x]), int(Cr[y, x])
    msg = f"(x={x_disp}, y={y_disp}) -> Y={yv}, Cb={cbv}, Cr={crv}"
    if print_bgr and img_bgr is not None:
        b, g, r = img_bgr[y, x]
        msg += f" | BGR=({int(b)}, {int(g)}, {int(r)})"
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def parse_xy(s):
    s = s.strip()
    if not s:
        raise ValueError("empty line")
    if "," in s:
        a, b = s.split(",", 1)
        return int(a.strip()), int(b.strip())
    parts = s.split()
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("expected 'x y' or 'x,y'")

def main():
    img_bgr = load_image(IMAGE_PATH)
    Y, Cb, Cr = to_ycbcr_arrays(img_bgr)
    H, W = Y.shape

    sys.stdout.write(
        f"Loaded {Path(IMAGE_PATH).name}: size={W}x{H} (WxH). "
        f"Input coords are {'1-based' if ONE_BASED_IN else '0-based'}.\n"
        "Enter pixel coords as 'x y' or 'x,y'. Type 'q' to quit.\n"
    )
    sys.stdout.flush()

    while True:
        if PROMPT:
            sys.stdout.write("> ")
            sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:  # EOF
            break
        s = line.strip()
        if not s:
            continue
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            x, y = parse_xy(s)
            print_point(x, y, Y, Cb, Cr, img_bgr, print_bgr=PRINT_BGR, one_based=ONE_BASED_IN)
        except Exception as e:
            sys.stdout.write(f"Parse error: {e}. Try 'x y' or 'x,y' (or 'q' to quit).\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
