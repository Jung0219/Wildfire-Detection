import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================== CONFIG (only what you asked to see) ========================
INPUT_DIR  = "/lab/projects/fire_smoke_awr/data/final/test_data/small_object_set/hand-filtered"
PARENT_OUT = "/lab/projects/fire_smoke_awr/src/util/experiment/sky/samples"

# Absolute channel ranges (YCbCr)
CB_MIN_ABS  = 120
CB_MAX_ABS  = 255
CR_MIN_ABS  = 0
CR_MAX_ABS  = 130

# Preproc for graph only (speed)
PREPROC_FOR_GRAPH   = True    # downscale+blur before computing the graph mask (we rescale to original size)
GRAPH_INPUT_SIZE    = 256     # target max side for the graph preproc (keeps aspect ratio)
GRAPH_BLUR_K        = 11      # odd
GRAPH_BLUR_SIGMA    = 1.5

# Optional smoothing of per-row counts
COUNT_SMOOTH        = False
COUNT_SMOOTH_SIGMA  = 2.0
COUNT_SMOOTH_K      = 11      # odd; set 0 to let OpenCV infer from sigma
# =====================================================================================

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(root, exts=EXTS):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]

def bgr_to_ycbcr(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    return Y, Cb, Cr

def range_sky_mask(img, y_min, cb_min, cb_max, cr_min, cr_max):
    """Binary mask using absolute Y, Cb, Cr ranges (no Mahalanobis, no means/std)."""
    Y, Cb, Cr = bgr_to_ycbcr(img)
    # cast Y to float to avoid overflow/underflow in comparisons when y_min is float
    mask = (Y.astype(np.float32) >= float(y_min)) & (Cb >= cb_min) & (Cb <= cb_max) & (Cr >= cr_min) & (Cr <= cr_max)
    return mask.astype(np.uint8)

def overlay_mask(img_bgr, mask, mask_color=(255, 255, 0), alpha=0.6):
    """Colorize masked pixels on top of original image."""
    vis = img_bgr.copy()
    if mask is not None and mask.any():
        m3 = mask.astype(bool)[..., None]
        m3 = np.repeat(m3, 3, axis=2)
        overlay = np.empty_like(vis); overlay[:] = mask_color
        blended = (alpha * overlay + (1 - alpha) * vis).astype(np.uint8)
        vis = np.where(m3, blended, vis)
    return vis

def compute_row_counts_norm(img, y_min, cb_min, cb_max, cr_min, cr_max):
    """
    Returns per-row sky-pixel counts normalized to [0, 1], where 1.0 == image width.
    Keeps height = original H. If PREPROC_FOR_GRAPH is True, compute on a smaller
    image for speed, then interpolate back to H while staying normalized.
    """
    H, W = img.shape[:2]

    if PREPROC_FOR_GRAPH:
        scale = GRAPH_INPUT_SIZE / max(H, W)
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        img_g = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if GRAPH_BLUR_K and GRAPH_BLUR_K % 2 == 1:
            img_g = cv2.GaussianBlur(img_g, (GRAPH_BLUR_K, GRAPH_BLUR_K), GRAPH_BLUR_SIGMA)

        mask_g = range_sky_mask(img_g, y_min, cb_min, cb_max, cr_min, cr_max)
        counts_g = mask_g.sum(axis=1).astype(np.float32)                    # len=new_h, in pixels of new_w
        counts_g_norm = counts_g / float(mask_g.shape[1])                   # -> [0,1] relative to new_w

        # upsample back to original H (still normalized)
        y_src = np.linspace(0, H - 1, num=counts_g_norm.shape[0], dtype=np.float32)
        y_dst = np.arange(H, dtype=np.float32)
        counts_norm = np.interp(y_dst, y_src, counts_g_norm).astype(np.float32)
    else:
        mask = range_sky_mask(img, y_min, cb_min, cb_max, cr_min, cr_max)
        counts = mask.sum(axis=1).astype(np.float32)                        # len=H, in pixels of W
        counts_norm = counts / float(W)                                     # -> [0,1]

    # optional vertical smoothing (operate in normalized space)
    if COUNT_SMOOTH:
        ksize = (COUNT_SMOOTH_K if (COUNT_SMOOTH_K and COUNT_SMOOTH_K % 2 == 1) else 0, 1)
        counts_norm = cv2.GaussianBlur(counts_norm.reshape(-1, 1), ksize, COUNT_SMOOTH_SIGMA).ravel()

    return np.clip(counts_norm, 0.0, 1.0)

def save_row_profile_plot_matplotlib(counts_norm, out_path, H, W, dpi=100, y_thresh=None):
    fig_w, fig_h = W / dpi, H / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.plot(counts_norm, np.arange(H), linewidth=1)
    ax.set_xlim(0.0, 1.0)          # normalized: 1.0 == full image width
    ax.set_ylim(H - 1, 0)          # invert Y (0 at top)
    ax.set_xlabel("Sky fraction per row (0–1)")
    ax.set_ylabel("Row")
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    txt = f"max={counts_norm.max():.2f}"
    if y_thresh is not None:
        txt += f" | meanY={y_thresh:.1f}"
    ax.text(0.65, 10, txt, fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.96)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":

    PARENT_OUT = Path(PARENT_OUT)
    OUTPUT_DIR = PARENT_OUT / "preview"
    GRAPH_DIR  = PARENT_OUT / "graphs"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(GRAPH_DIR).mkdir(parents=True, exist_ok=True)

    img_paths = list_images(INPUT_DIR)
    if not img_paths:
        raise SystemExit("No images found in input folder.")

    for path in tqdm(img_paths, desc="Processing images", unit="img"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # --- Adaptive Y threshold: per-image global mean Y ---
        Y_full, _, _ = bgr_to_ycbcr(img)
        y_thresh = float(Y_full.mean())

        # Full-res mask & overlay for the side-by-side image
        mask_full = range_sky_mask(
            img,
            y_min=y_thresh,  # adaptive
            cb_min=CB_MIN_ABS, cb_max=CB_MAX_ABS,
            cr_min=CR_MIN_ABS, cr_max=CR_MAX_ABS
        )
        vis = overlay_mask(img, mask_full)

        # --- Save ONLY original | masked side-by-side ---
        combined = np.hstack([img, vis])
        side_path = Path(OUTPUT_DIR) / f"{Path(path).stem}_orig_mask.png"
        cv2.imwrite(str(side_path), combined)

        # --- Save graph as a separate file, same basename as image ---
        counts = compute_row_counts_norm(
            img,
            y_min=y_thresh,  # adaptive
            cb_min=CB_MIN_ABS, cb_max=CB_MAX_ABS,
            cr_min=CR_MIN_ABS, cr_max=CR_MAX_ABS
        )
        graph_path = Path(GRAPH_DIR) / f"{Path(path).stem}.png"
        save_row_profile_plot_matplotlib(counts, str(graph_path), H=H, W=W, y_thresh=y_thresh)
