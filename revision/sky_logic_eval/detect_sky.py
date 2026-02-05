"""Sky presence detector using skyline estimation.

Example:
    python /data/lab/projects/fire_smoke_awr/revision/sky_logic_eval/detect_sky.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# CONFIG (editable)
IMAGE_DIR = "/lab/projects/fire_smoke_awr/data/detection/processed/early_smoke/images"
OUTPUT_PATH = "/lab/projects/fire_smoke_awr/revision/sky_logic_eval/es_all/ours.txt"
CB_MIN = 120
CB_MAX = 255
CR_MIN = 0
CR_MAX = 130
SKY_RATIO_THRESH = 5.0
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def detect_skyline_y(
    img_bgr: np.ndarray,
    cb_min: int = CB_MIN,
    cb_max: int = CB_MAX,
    cr_min: int = CR_MIN,
    cr_max: int = CR_MAX,
    sky_ratio_thresh: float = SKY_RATIO_THRESH,
) -> int:
    """Detect a skyline row; returns -1 if not found."""

    H, W = img_bgr.shape[:2]
    scale = 0.25
    resized_w = max(1, int(round(W * scale)))
    resized_h = max(1, int(round(H * scale)))
    resized_bgr = cv2.resize(img_bgr, (resized_w, resized_h))
    ycrcb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    y_thresh = float(Y.astype(np.float32).mean())

    def sky_mask(bgr):
        ycrcb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y_, Cr_, Cb_ = cv2.split(ycrcb_)
        return ((Y_ >= y_thresh) & (Cb_ >= cb_min) & (Cb_ <= cb_max) & (Cr_ >= cr_min) & (Cr_ <= cr_max)).astype(np.uint8)

    m_full = sky_mask(resized_bgr)
    counts = m_full.sum(axis=1) / float(resized_w)
    d = np.diff(counts)
    idx = int(np.argmin(d))
    y_candidate = int(np.clip(idx + 1, 0, resized_h - 1))
    above = int(m_full[:y_candidate, :].sum())
    below = int(m_full[y_candidate:, :].sum())
    ratio = (above + 1e-9) / (below + 1e-9)
    if ratio < sky_ratio_thresh:
        return -1
    scale_y = resized_h / float(H)
    y_scaled = int(round(y_candidate / scale_y))
    return int(np.clip(y_scaled, 0, H - 1))


def sky_present(
    img_bgr: np.ndarray,
    cb_min: int = CB_MIN,
    cb_max: int = CB_MAX,
    cr_min: int = CR_MIN,
    cr_max: int = CR_MAX,
    sky_ratio_thresh: float = SKY_RATIO_THRESH,
) -> bool:
    """Return True if sky is detected using skyline estimation."""
    return detect_skyline_y(
        img_bgr,
        cb_min=cb_min,
        cb_max=cb_max,
        cr_min=cr_min,
        cr_max=cr_max,
        sky_ratio_thresh=sky_ratio_thresh,
    ) != -1


def _iter_images(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def main() -> int:
    image_dir = Path(IMAGE_DIR)
    if not image_dir.is_dir():
        raise SystemExit(f"ERROR: not a directory: {image_dir}")

    images = _iter_images(image_dir)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sky_count = 0
    no_sky_count = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for img_path in tqdm(images, desc="Detecting"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                skipped += 1
                continue
            has_sky = 1 if sky_present(img_bgr) else 0
            if has_sky:
                sky_count += 1
            else:
                no_sky_count += 1
            f.write(f"{img_path.name} {has_sky}\n")

    total = len(images)
    sky_pct = (sky_count / total * 100.0) if total else 0.0
    no_sky_pct = (no_sky_count / total * 100.0) if total else 0.0
    print(f"Wrote {total} labels to {output_path}")
    print(f"Sky present: {sky_count} ({sky_pct:.1f}%)")
    print(f"No sky: {no_sky_count} ({no_sky_pct:.1f}%)")
    if skipped:
        print(f"Skipped unreadable: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
