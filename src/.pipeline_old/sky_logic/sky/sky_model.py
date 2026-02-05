import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

# ======================== CONFIG =========================
DATASET_DIR = "/lab/projects/fire_smoke_awr/data/datasets/.HPWREN_FigLib/deduplicated/phash20/images"
EXTS        = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ROI geometry (normalized to H/W)
TOP_FRAC        = 0.20   # vertical band height (20% of H)
TOP_OFFSET_FRAC = 0.10   # start 10% below the top edge
SIDE_CROP_FRAC  = 0.05   # crop 5% from left and 5% from right (keep center 90%)

# Modeling
PERCENTILE  = 90         # Mahalanobis distance percentile for T
REG_EPS     = 1.0        # diagonal regularization for covariance
SEED        = 0
# MODEL_JSON  = "sky_model.json"

# Cap total pixels used for fitting (random subsample if exceeded)
MAX_PIXELS  = 500_000_000

# Pixel-level mask inside ROI
USE_MASK    = True       # apply pixel-level mask inside the ROI
Y_MIN       = 60         # min luminance for candidate pixels (tune)
SAT_MAX     = 90         # max chroma distance: |Cb-128| + |Cr-128| (tune)

# ROI previews
SAVE_ROI_PREVIEWS = False
ROI_PREVIEW_DIR   = "/lab/projects/fire_smoke_awr/src/util/experiment/sky/roi_preview"
MAX_ROI_PREVIEWS  = 100     # save at most this many previews
DRAW_BOX_THICK    = 2

# Plots
PLOT_DIR          = "/lab/projects/fire_smoke_awr/src/util/experiment/sky/plots"
PLOT_DPI          = 160
SCATTER_MAX_N     = 300_000
# =========================================================


def list_images(root, exts=EXTS):
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]


def bgr_to_ycbcr(bgr):
    """
    OpenCV returns YCrCb; split as Y, Cr, Cb then reorder to Y, Cb, Cr.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    return Y, Cb, Cr


def roi_bounds(h, w):
    y0 = int(round(h * TOP_OFFSET_FRAC))
    band_h = max(1, int(round(h * TOP_FRAC)))
    y1 = min(h, y0 + band_h)
    side_w = int(round(w * SIDE_CROP_FRAC))
    x0 = side_w
    x1 = max(x0 + 1, w - side_w)  # ensure at least 1 px width
    return y0, y1, x0, x1


def sample_top_band(img):
    """
    Return (Cb, Cr) pairs from ROI.
    If USE_MASK is True, keep only pixels passing (Y >= Y_MIN) and
    (|Cb-128| + |Cr-128| <= SAT_MAX). Otherwise, keep all ROI pixels.
    """
    H, W = img.shape[:2]
    y0, y1, x0, x1 = roi_bounds(H, W)

    # Guard against degenerate ROI
    if y1 - y0 <= 0 or x1 - x0 <= 0:
        return np.empty((0, 2), dtype=np.float32)

    Y, Cb, Cr = bgr_to_ycbcr(img)
    # Use int16 for safe abs arithmetic, then cast back to float32
    Yt  = Y [y0:y1, x0:x1].astype(np.int16)
    Cbt = Cb[y0:y1, x0:x1].astype(np.int16)
    Crt = Cr[y0:y1, x0:x1].astype(np.int16)

    if USE_MASK:
        chroma_dist = np.abs(Cbt - 128) + np.abs(Crt - 128)
        mask = (Yt >= Y_MIN) & (chroma_dist <= SAT_MAX)
        if not np.any(mask):
            return np.empty((0, 2), dtype=np.float32)
        Cb_sel = Cbt[mask].astype(np.float32)
        Cr_sel = Crt[mask].astype(np.float32)
        return np.stack([Cb_sel, Cr_sel], axis=1)

    # Original behavior: use all pixels in ROI
    return np.stack([Cbt.ravel().astype(np.float32),
                     Crt.ravel().astype(np.float32)], axis=1)


def make_roi_preview(img):
    """
    Create a side-by-side preview showing:
    [original with green ROI box | ROI-only (rest black)].
    """
    H, W = img.shape[:2]
    y0, y1, x0, x1 = roi_bounds(H, W)

    # If ROI is degenerate, just return original twice
    if y1 - y0 <= 0 or x1 - x0 <= 0:
        return np.hstack([img, img])

    # Left: original with ROI box
    left = img.copy()
    cv2.rectangle(left, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), DRAW_BOX_THICK)

    # Right: only ROI content, rest black
    right = np.zeros_like(img)
    right[y0:y1, x0:x1] = img[y0:y1, x0:x1]

    return np.hstack([left, right])


def fit_model(samples):
    """
    Fit mu, Sigma, and get T as the chosen percentile of Mahalanobis radii.
    """
    print("samples shape:", samples.shape, "dtype:", samples.dtype)

    # Explicit per-column mean
    mu_cb = samples[:, 0].mean()
    mu_cr = samples[:, 1].mean()
    mu = np.array([mu_cb, mu_cr], dtype=np.float32)

    # Center the data
    Xc = samples - mu

    # Covariance (unbiased, ddof=1)
    Sigma = (np.dot(Xc.T, Xc)) / max(1, len(samples) - 1)
    Sigma = Sigma.astype(np.float32)
    Sigma += np.eye(2, dtype=np.float32) * REG_EPS  # regularize

    # Mahalanobis cutoff
    Si = np.linalg.inv(Sigma)
    d2 = np.einsum('ij,jk,ik->i', Xc, Si, Xc)  # squared Mahalanobis
    d = np.sqrt(np.maximum(d2, 0.0))
    T = np.percentile(d, PERCENTILE)

    return mu, Sigma, float(T)


if __name__ == "__main__":
    np.random.seed(SEED)

    # Prepare output dirs
    if SAVE_ROI_PREVIEWS:
        Path(ROI_PREVIEW_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

    imgs = list_images(DATASET_DIR)
    if not imgs:
        raise SystemExit("No images found.")

    saved_previews = 0
    all_samples = []
    total_pixels = 0

    for path in tqdm(imgs, desc="Processing images", unit="img"):
        img = cv2.imread(path)
        if img is None:
            continue

        # Save an ROI side-by-side preview (first N images)
        if SAVE_ROI_PREVIEWS and saved_previews < MAX_ROI_PREVIEWS:
            preview = make_roi_preview(img)
            out_name = f"{Path(path).stem}_roi_side.png"
            cv2.imwrite(str(Path(ROI_PREVIEW_DIR) / out_name), preview)
            saved_previews += 1

        # Collect samples
        X = sample_top_band(img)
        if len(X) == 0:
            continue
        all_samples.append(X)
        total_pixels += X.shape[0]

    if not all_samples:
        raise SystemExit("No samples collected.")

    Xall = np.vstack(all_samples)

    # Global cap on pixels used for fitting
    original_count = len(Xall)
    if original_count > MAX_PIXELS:
        idx = np.random.choice(original_count, MAX_PIXELS, replace=False)
        Xall = Xall[idx]

    mu, Sigma, T = fit_model(Xall)

    # Report scalars
    var_cb = Xall[:, 0].var(ddof=1)  # variance of Cb
    var_cr = Xall[:, 1].var(ddof=1)  # variance of Cr
    cov_cb_cr = np.cov(Xall[:, 0], Xall[:, 1], ddof=1)[0, 1]  # covariance Cb,Cr

    print(f"\nProcessed {len(imgs)} images")
    print(f"Total ROI pixels collected: {total_pixels}")
    print(f"Used for model (after cap): {len(Xall)} (cap={MAX_PIXELS}, original={original_count})")
    print(f"mu (Cb, Cr) = {mu}")
    print(f"Sigma =\n{Sigma}")
    print(f"T (Mahalanobis cutoff @ {PERCENTILE}%) = {T}")
    print("================================================")
    print(f"mu (Cb, Cr) = {mu}, var(Cb)={var_cb:.4f}, var(Cr)={var_cr:.4f}, cov(Cb,Cr)={cov_cb_cr:.4f}")

    # Save model (commented out per your earlier preference)
    model_data = {
        "mu": mu.tolist(),
        "Sigma": Sigma.tolist(),
        "T": T,
        "PERCENTILE": PERCENTILE,
        "TOP_FRAC": TOP_FRAC,
        "TOP_OFFSET_FRAC": TOP_OFFSET_FRAC,
        "SIDE_CROP_FRAC": SIDE_CROP_FRAC,
        "REG_EPS": REG_EPS,
        "MAX_PIXELS": MAX_PIXELS,
        "DATASET_DIR": DATASET_DIR,
        "USE_MASK": USE_MASK,
        "Y_MIN": Y_MIN,
        "SAT_MAX": SAT_MAX
    }
    # with open(MODEL_JSON, "w") as f:
    #     json.dump(model_data, f, indent=4)
    # print(f"\nModel saved to {MODEL_JSON}")

    # ====== Plots: Cb–Cr density + scatter with ellipse ======
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    def mahalanobis_ellipse(mu_, Sigma_, T_, num=360):
        """Return x,y points of the Mahalanobis ellipse: (x-mu)^T Sigma^-1 (x-mu) = T^2."""
        vals, vecs = np.linalg.eigh(Sigma_)          # vals ascending, vecs columns
        axes = T_ * np.sqrt(np.maximum(vals, 0))     # radii along principal axes
        theta = np.linspace(0, 2*np.pi, num=num)
        circle = np.vstack([np.cos(theta), np.sin(theta)])  # 2 x num
        ellipse = (vecs @ np.diag(axes) @ circle).T + mu_
        return ellipse[:, 0], ellipse[:, 1]

    # 1) Full-data density (2D histogram)
    H, xedges, yedges = np.histogram2d(
        Xall[:, 0],  # Cb
        Xall[:, 1],  # Cr
        bins=256,
        range=[[0, 255], [0, 255]]
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(
        H.T,
        origin="lower",
        extent=[0, 255, 0, 255],
        aspect="equal",
        norm=LogNorm()
    )
    plt.xlabel("Cb")
    plt.ylabel("Cr")
    title_mask = "masked ROI" if USE_MASK else "ROI (all pixels)"
    plt.title(f"Cb–Cr density ({title_mask})")
    ex, ey = mahalanobis_ellipse(mu, Sigma, T)
    plt.plot(ex, ey, linewidth=1.25)
    plt.scatter([mu[0]], [mu[1]], marker="x")
    plt.tight_layout()
    density_path = Path(PLOT_DIR) / "cbcr_density.png"
    plt.savefig(str(density_path), dpi=PLOT_DPI)
    plt.close()

    # 2) Scatter of a random subsample (for the raw-points feel)
    rng = np.random.default_rng(0)
    N = min(len(Xall), SCATTER_MAX_N)
    idx = rng.choice(len(Xall), N, replace=False)
    sub = Xall[idx]

    plt.figure(figsize=(6, 6))
    plt.plot(sub[:, 0], sub[:, 1], ".", markersize=0.5, alpha=0.15)
    plt.xlabel("Cb")
    plt.ylabel("Cr")
    plt.title(f"Cb–Cr scatter ({title_mask}, n={N:,})")
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.gca().set_aspect("equal", adjustable="box")
    ex, ey = mahalanobis_ellipse(mu, Sigma, T)
    plt.plot(ex, ey, linewidth=1.25)
    plt.scatter([mu[0]], [mu[1]], marker="x")
    plt.tight_layout()
    scatter_path = Path(PLOT_DIR) / "cbcr_scatter.png"
    plt.savefig(str(scatter_path), dpi=PLOT_DPI)
    plt.close()

    print(f"\nSaved plots:\n - {density_path}\n - {scatter_path}")
    if SAVE_ROI_PREVIEWS:
        print(f"Saved ROI previews (up to {MAX_ROI_PREVIEWS}) to: {ROI_PREVIEW_DIR}")
