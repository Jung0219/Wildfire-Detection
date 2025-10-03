#!/usr/bin/env python3
import os
from pathlib import Path
import random
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== CONFIG =====================
IMAGES_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/ABCDE_noEF/images/val"
LABELS_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/ABCDE_noEF/labels/val"
OUT_DIR    = "/lab/projects/fire_smoke_awr/src/pipeline/early_fire_scene_classifier/analysis"

FIRE_CLASS_IDS = {0}         # <-- set to your 'fire' class id(s); can be multiple
BOX_SHRINK = 0.6             # shrink factor (0.6 keeps 60% of width/height, centered)
MAX_PIXELS_PER_IMAGE = 20000 # subsample per image for speed/memory; None for all
RANDOM_SEED = 1337

# Optional: keep only moderately bright pixels to avoid dark smoke/ash bias
Y_MIN = None   # e.g., 64 or 80; set None to disable
Y_MAX = None   # e.g., 255; set None to disable

# Percentiles to report
PCTS = [1, 5, 25, 50, 75, 95, 99]

# File extensions
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Plots
MAKE_PLOTS = True
HEXBINS = 120  # resolution for 2D hexbin
# ==================================================

def yolo_to_xyxy(line, w, h):
    """Parse one YOLO line -> (class_id, x1, y1, x2, y2) in pixel coords."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = (cx - bw/2.0) * w
    y1 = (cy - bh/2.0) * h
    x2 = (cx + bw/2.0) * w
    y2 = (cy + bh/2.0) * h
    return cls, x1, y1, x2, y2

def shrink_box(x1, y1, x2, y2, factor):
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = (x2 - x1) * 0.5 * factor
    half_h = (y2 - y1) * 0.5 * factor
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h

def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w-1, int(round(x1))))
    y1 = max(0, min(h-1, int(round(y1))))
    x2 = max(0, min(w-1, int(round(x2))))
    y2 = max(0, min(h-1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def iter_image_label_pairs(images_dir, labels_dir):
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if Path(fn).suffix.lower() in EXTS:
                img_path = Path(root) / fn
                rel = img_path.relative_to(images_dir)
                lbl_path = Path(labels_dir) / rel.with_suffix(".txt")
                yield img_path, lbl_path

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    all_cb = []
    all_cr = []
    all_y  = []

    pairs = list(iter_image_label_pairs(IMAGES_DIR, LABELS_DIR))
    if not pairs:
        raise SystemExit("No image/label pairs found.")

    for img_path, lbl_path in tqdm(pairs, desc="Scanning dataset"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        if not lbl_path.exists():
            continue

        with open(lbl_path, "r") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]

        if not lines:
            continue

        # convert once
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)  # note: OpenCV order is Y, Cr, Cb

        # collect pixels from shrunken fire boxes
        pixels_cb_cr_y = []

        for ln in lines:
            parsed = yolo_to_xyxy(ln, W, H)
            if parsed is None:
                continue
            cls, x1, y1, x2, y2 = parsed
            if cls not in FIRE_CLASS_IDS:
                continue

            if BOX_SHRINK is not None and 0 < BOX_SHRINK < 1.0:
                x1, y1, x2, y2 = shrink_box(x1, y1, x2, y2, BOX_SHRINK)

            clipped = clip_box(x1, y1, x2, y2, W, H)
            if clipped is None:
                continue
            x1c, y1c, x2c, y2c = clipped

            # extract region
            Y_roi  = Y[y1c:y2c, x1c:x2c]
            Cr_roi = Cr[y1c:y2c, x1c:x2c]
            Cb_roi = Cb[y1c:y2c, x1c:x2c]

            # optional brightness gating
            if Y_MIN is not None or Y_MAX is not None:
                mask = np.ones_like(Y_roi, dtype=bool)
                if Y_MIN is not None:
                    mask &= (Y_roi >= Y_MIN)
                if Y_MAX is not None:
                    mask &= (Y_roi <= Y_MAX)
                if not mask.any():
                    continue
                Y_vals  = Y_roi[mask]
                Cr_vals = Cr_roi[mask]
                Cb_vals = Cb_roi[mask]
            else:
                Y_vals  = Y_roi.reshape(-1)
                Cr_vals = Cr_roi.reshape(-1)
                Cb_vals = Cb_roi.reshape(-1)

            if Y_vals.size == 0:
                continue

            # subsample to limit per-image contribution
            if MAX_PIXELS_PER_IMAGE is not None and Y_vals.size > MAX_PIXELS_PER_IMAGE:
                idx = np.random.choice(Y_vals.size, MAX_PIXELS_PER_IMAGE, replace=False)
                Y_vals, Cr_vals, Cb_vals = Y_vals[idx], Cr_vals[idx], Cb_vals[idx]

            pixels_cb_cr_y.append((Cb_vals, Cr_vals, Y_vals))

        if not pixels_cb_cr_y:
            continue

        # concat for this image
        Cb_img = np.concatenate([t[0] for t in pixels_cb_cr_y])
        Cr_img = np.concatenate([t[1] for t in pixels_cb_cr_y])
        Y_img  = np.concatenate([t[2] for t in pixels_cb_cr_y])

        all_cb.append(Cb_img.astype(np.float32))
        all_cr.append(Cr_img.astype(np.float32))
        all_y.append(Y_img.astype(np.float32))

    if not all_cb:
        raise SystemExit("No fire pixels collected. Check FIRE_CLASS_IDS and label paths.")

    cb = np.concatenate(all_cb)
    cr = np.concatenate(all_cr)
    y  = np.concatenate(all_y)

    # --------- Stats & suggested ranges ----------
    def pct(v, ps):
        return {f"p{p}": float(np.percentile(v, p)) for p in ps}

    cb_pct = pct(cb, PCTS)
    cr_pct = pct(cr, PCTS)
    y_pct  = pct(y,  PCTS)

    suggested = {
        "Cb_min": cb_pct["p5"], "Cb_max": cb_pct["p95"],
        "Cr_min": cr_pct["p5"], "Cr_max": cr_pct["p95"],
    }

    # save CSV summary
    csv_path = Path(OUT_DIR) / "fire_cbcr_y_percentiles.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel"] + [f"p{p}" for p in PCTS] + ["mean", "std"])
        writer.writerow(["Cb"] + [f"{cb_pct[f'p{p}']:.2f}" for p in PCTS] + [f"{cb.mean():.2f}", f"{cb.std():.2f}"])
        writer.writerow(["Cr"] + [f"{cr_pct[f'p{p}']:.2f}" for p in PCTS] + [f"{cr.mean():.2f}", f"{cr.std():.2f}"])
        writer.writerow(["Y"]  + [f"{y_pct[f'p{p}']:.2f}"  for p in PCTS] + [f"{y.mean():.2f}",  f"{y.std():.2f}"])

    # print concise report
    print("\n=== Fire Pixel YCbCr Stats (across sampled pixels) ===")
    print("Cb percentiles:", {k: round(v,2) for k,v in cb_pct.items()})
    print("Cr percentiles:", {k: round(v,2) for k,v in cr_pct.items()})
    print("Y  percentiles:", {k: round(v,2) for k,v in y_pct.items()})
    print(f"Suggested rectangular range (5–95%): "
          f"Cb ∈ [{suggested['Cb_min']:.0f}, {suggested['Cb_max']:.0f}], "
          f"Cr ∈ [{suggested['Cr_min']:.0f}, {suggested['Cr_max']:.0f}]")
    print(f"CSV saved to: {csv_path}")

    # --------- Plots ----------
    if MAKE_PLOTS:
        out_dir = Path(OUT_DIR)
        # Histograms
        plt.figure()
        plt.hist(cb, bins=256, alpha=0.7)
        plt.title("Cb histogram (fire pixels)")
        plt.xlabel("Cb"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "hist_cb.png"); plt.close()

        plt.figure()
        plt.hist(cr, bins=256, alpha=0.7)
        plt.title("Cr histogram (fire pixels)")
        plt.xlabel("Cr"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "hist_cr.png"); plt.close()

        plt.figure()
        plt.hist(y, bins=256, alpha=0.7)
        plt.title("Y histogram (fire pixels)")
        plt.xlabel("Y"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "hist_y.png"); plt.close()

        # 2D hexbin
        plt.figure()
        hb = plt.hexbin(cb, cr, gridsize=HEXBINS, bins='log')
        plt.xlabel("Cb"); plt.ylabel("Cr"); plt.title("Cb–Cr density (fire pixels)")
        plt.colorbar(hb, label='log(count)')
        plt.tight_layout()
        plt.savefig(out_dir / "hexbin_cb_cr.png"); plt.close()

        # Extra 2D relations: Y–Cr, Y–Cb
        plt.figure()
        hb = plt.hexbin(y, cr, gridsize=HEXBINS, bins='log')
        plt.xlabel("Y"); plt.ylabel("Cr"); plt.title("Y–Cr density (fire pixels)")
        plt.colorbar(hb, label='log(count)')
        plt.tight_layout()
        plt.savefig(out_dir / "hexbin_y_cr.png"); plt.close()

        plt.figure()
        hb = plt.hexbin(y, cb, gridsize=HEXBINS, bins='log')
        plt.xlabel("Y"); plt.ylabel("Cb"); plt.title("Y–Cb density (fire pixels)")
        plt.colorbar(hb, label='log(count)')
        plt.tight_layout()
        plt.savefig(out_dir / "hexbin_y_cb.png"); plt.close()

        # Optional: stratify by brightness buckets to see drift
        buckets = [(0,64), (64,128), (128,192), (192,256)]
        for lo, hi in buckets:
            m = (y >= lo) & (y < hi)
            if m.sum() < 1000:
                continue
            plt.figure()
            hb = plt.hexbin(cb[m], cr[m], gridsize=HEXBINS, bins='log')
            plt.xlabel("Cb"); plt.ylabel("Cr")
            plt.title(f"Cb–Cr density | Y in [{lo},{hi})")
            plt.colorbar(hb, label='log(count)')
            plt.tight_layout()
            plt.savefig(out_dir / f"hexbin_cb_cr_Y_{lo}_{hi}.png"); plt.close()

    # --------- Save raw samples (optional) ----------
    np.savez_compressed(Path(OUT_DIR) / "fire_cbcr_y_samples.npz", Cb=cb, Cr=cr, Y=y)

if __name__ == "__main__":
    main()
