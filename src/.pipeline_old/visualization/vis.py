"""Run skyline detection on the first image in a dataset folder.

Example:
    python /data/lab/projects/fire_smoke_awr/src/.pipeline_old/sky_logic/sky/skyliine_single.py
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- CONFIG (edit as needed) ----
IMAGE_PATH = Path(
    "/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/phash10/original/images/test/force-06_cabanelle-244_2024-04-10T10-50-31.jpg"
)
OUTPUT_DIR = Path(
    "/lab/projects/fire_smoke_awr/src/.pipeline_old/visualization/example"
)
CB_MIN = 120
CB_MAX = 255
CR_MIN = 0
CR_MAX = 130
INTERMEDIATE_SIZE = 900
ANCHOR_X_FRAC = 0.5
ANCHOR_Y_FRAC = 0.25
# -------------------------------

def detect_skyline_y(
    img_path: str,
    cb_min: int = 120,
    cb_max: int = 255,
    cr_min: int = 0,
    cr_max: int = 130,
    output_dir: str | None = None,
) -> int:
    """
    Returns the estimated sky/ground border row (y) using:
      1) adaptive Y threshold (mean Y) for sky mask
      2) per-row sky fraction "graph" with downscale+blur for speed
      3) boundary at the most negative derivative
      4) 5× rule: sky pixels above / below >= 5 to accept; else returns -1
    """
    # ---- internal defaults ----
    PREPROC_FOR_GRAPH = True
    GRAPH_INPUT_SIZE = 256
    GRAPH_BLUR_K = 11
    GRAPH_BLUR_SIGMA = 1.5

    COUNT_SMOOTH = True
    COUNT_SMOOTH_K = 11
    COUNT_SMOOTH_SIGMA = 2.0

    DERIV_SMOOTH = True
    DERIV_SMOOTH_K = 11
    DERIV_SMOOTH_SIG = 2.0

    IGNORE_EDGE_FRAC = 0.02
    SKY_RATIO_THRESH = 5.0
    # ---------------------------

    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    H, W = img.shape[:2]
    quarter_w = max(1, W // 4)
    quarter_h = max(1, H // 4)
    img_small = cv2.resize(img, (quarter_w, quarter_h), interpolation=cv2.INTER_AREA)
    Hs, Ws = img_small.shape[:2]

    # YCbCr and adaptive Y threshold (mean Y)
    ycrcb = cv2.cvtColor(img_small, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y = Y.astype(np.float32)
    y_thresh = float(Y.mean())

    # Helper: sky mask under absolute ranges
    def range_sky_mask(bgr, y_min, cb_min_, cb_max_, cr_min_, cr_max_):
        ycrcb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y_, Cr_, Cb_ = cv2.split(ycrcb_)
        return (
            (Y_.astype(np.float32) >= y_min)
            & (Cb_ >= cb_min_)
            & (Cb_ <= cb_max_)
            & (Cr_ >= cr_min_)
            & (Cr_ <= cr_max_)
        ).astype(np.uint8)

    def compute_row_counts_norm(img, y_min, cb_min_, cb_max_, cr_min_, cr_max_):
        """
        Returns per-row sky-pixel counts normalized to [0, 1], where 1.0 == image width.
        Keeps height = original H. If PREPROC_FOR_GRAPH is True, compute on a smaller
        image for speed, then interpolate back to H while staying normalized.
        """
        H_, W_ = img.shape[:2]

        if PREPROC_FOR_GRAPH:
            scale = GRAPH_INPUT_SIZE / max(H_, W_)
            new_w = max(1, int(round(W_ * scale)))
            new_h = max(1, int(round(H_ * scale)))
            img_g = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if GRAPH_BLUR_K and GRAPH_BLUR_K % 2 == 1:
                img_g = cv2.GaussianBlur(img_g, (GRAPH_BLUR_K, GRAPH_BLUR_K), GRAPH_BLUR_SIGMA)

            mask_g = range_sky_mask(img_g, y_min, cb_min_, cb_max_, cr_min_, cr_max_)
            counts_g = mask_g.sum(axis=1).astype(np.float32)
            counts_g_norm = counts_g / float(mask_g.shape[1])

            y_src = np.linspace(0, H_ - 1, num=counts_g_norm.shape[0], dtype=np.float32)
            y_dst = np.arange(H_, dtype=np.float32)
            counts_norm = np.interp(y_dst, y_src, counts_g_norm).astype(np.float32)
        else:
            mask = range_sky_mask(img, y_min, cb_min_, cb_max_, cr_min_, cr_max_)
            counts = mask.sum(axis=1).astype(np.float32)
            counts_norm = counts / float(W_)

        if COUNT_SMOOTH:
            ksize = (COUNT_SMOOTH_K if (COUNT_SMOOTH_K and COUNT_SMOOTH_K % 2 == 1) else 0, 1)
            counts_norm = cv2.GaussianBlur(
                counts_norm.reshape(-1, 1), ksize, COUNT_SMOOTH_SIGMA
            ).ravel()

        return np.clip(counts_norm, 0.0, 1.0)

    def save_row_profile_plot_matplotlib(counts_norm, out_path, H_, W_, dpi=100, y_thresh=None):
        fig_w, fig_h = W_ / dpi, H_ / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        counts_plot = np.clip(counts_norm, 0.001, 0.999)
        y_vals = np.arange(H_)
        ax.fill_betweenx(
            y_vals,
            0.0,
            counts_plot,
            step="mid",
            color="tab:blue",
            alpha=0.6,
        )
        ax.plot(counts_plot, y_vals, linewidth=2.5, color="tab:blue")
        ax.set_xlim(0.0, 1.1)
        ax.set_ylim(H_ - 1, 0)
        ax.set_xlabel("Sky fraction for horizontal profile", fontsize=32)
        ax.set_ylabel("Vertical profile", fontsize=32)
        ax.tick_params(axis="both", labelsize=24)
        ax.grid(False)
        ax.axvline(0.0, color="black", linewidth=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.96)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

    # Build per-row sky fraction graph (normalized to [0,1])
    graph_small = compute_row_counts_norm(img_small, y_thresh, cb_min, cb_max, cr_min, cr_max)

    # Smooth before derivative (optional)
    if DERIV_SMOOTH:
        kx = (DERIV_SMOOTH_K if (DERIV_SMOOTH_K and DERIV_SMOOTH_K % 2 == 1) else 0, 1)
        graph_small = cv2.GaussianBlur(
            graph_small.reshape(-1, 1), kx, DERIV_SMOOTH_SIG
        ).ravel()

    # Find steepest negative drop
    d = np.diff(graph_small)  # s[y+1] - s[y]
    top_ignore = int(round(IGNORE_EDGE_FRAC * Hs))
    bot_ignore = int(round(IGNORE_EDGE_FRAC * Hs))
    lo = max(0, top_ignore)
    hi = max(lo + 1, (Hs - 1) - bot_ignore)
    idx = lo + int(np.argmin(d[lo:hi]))
    y_candidate = int(np.clip(idx + 1, 0, Hs - 1))
    
    print(y_candidate)

    # Validate with full-res mask and 5× rule
    m_small = range_sky_mask(img_small, y_thresh, cb_min, cb_max, cr_min, cr_max)
    above = int(m_small[:y_candidate, :].sum())
    below = int(m_small[y_candidate:, :].sum())
    ratio = (above + 1e-9) / (below + 1e-9)

    y_final_small = y_candidate if ratio >= SKY_RATIO_THRESH else -1
    if y_final_small >= 0:
        y_final = int(round((y_final_small / float(Hs)) * H))
        y_final = int(np.clip(y_final, 0, H - 1))
    else:
        y_final = -1

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = p.stem
        suffix = p.suffix if p.suffix else ".png"

        original_path = out_dir / f"skyline_original{suffix}"
        cv2.imwrite(str(original_path), img)

        label_pad = max(90, int(round(0.16 * W)))
        left_pad = max(180, int(round(0.16 * W)))
        line_pad = 40
        line_img = cv2.copyMakeBorder(
            img,
            0,
            0,
            left_pad,
            label_pad,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        caption_x = 10
        if y_final >= 0:
            cv2.line(
                line_img,
                (left_pad - line_pad, y_final),
                (left_pad + W + line_pad - 1, y_final),
                (0, 0, 255),
                4,
            )
            label_y = int(np.clip(y_final - 10, 30, H - 10))
            cv2.putText(
                line_img,
                "skyline",
                (caption_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 0, 255),
                3,
            )
        else:
            cv2.putText(
                line_img,
                "no_sky",
                (caption_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 0, 255),
                3,
            )
        skyline_path = out_dir / f"skyline_line{suffix}"
        cv2.imwrite(str(skyline_path), line_img)

        y_src = np.linspace(0, H - 1, num=graph_small.shape[0], dtype=np.float32)
        y_dst = np.arange(H, dtype=np.float32)
        graph = np.interp(y_dst, y_src, graph_small).astype(np.float32)

        graph_path = out_dir / f"skyline_graph{suffix}"
        save_row_profile_plot_matplotlib(graph, graph_path, H, W, dpi=100, y_thresh=y_thresh)

    return y_final


def generate_composite_640x640(
    original_image,
    object_center_norm,
    intermediate_size,
    anchor_x_frac=0.5,
    anchor_y_frac=0.25,
):
    """
    Generates a 640x640 composite image from a wider original image.

    The composite image is created by stacking two components vertically:
    1. A top band: A cropped region from an upscaled "intermediate" version
       of the original image. The crop is centered around a specified object
       center.
    2. A bottom band: The original image resized to a width of 640 pixels,
       maintaining its aspect ratio.

    This allows for a high-resolution view of a region of interest (top) while
    maintaining the overall context of the scene (bottom).

    Args:
        original_image (np.ndarray): The original input image.
        object_center_norm (tuple): Normalized (x, y) coordinates of the
                                    object/region of interest.
        intermediate_size (int): The size to which the largest dimension of the
                                 original image is scaled for the intermediate
                                 (top crop) version.
        anchor_x_frac (float): The horizontal anchor point within the crop window.
        anchor_y_frac (float): The vertical anchor point within the crop window.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The generated 640x640 composite image.
            - dict: A metadata dictionary with parameters used for the
                    transformation, required for mapping coordinates back to the
                    original image.
    """
    TARGET_SIZE = 640
    orig_h, orig_w = original_image.shape[:2]

    scale_inter = intermediate_size / (orig_w if orig_w >= orig_h else orig_h)
    res_inter_w, res_inter_h = int(orig_w * scale_inter), int(orig_h * scale_inter)
    image_inter = cv2.resize(original_image, (res_inter_w, res_inter_h))

    scale_to_640 = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    resized_w, resized_h = int(orig_w * scale_to_640), int(orig_h * scale_to_640)
    resized_bottom = cv2.resize(original_image, (resized_w, resized_h))

    if resized_h == TARGET_SIZE and resized_w == TARGET_SIZE:
        return resized_bottom, {
            "div_y": TARGET_SIZE,
            "crop_x1": 0,
            "crop_y1": 0,
            "scale_inter": scale_inter,
            "scale_to_640": scale_to_640,
            "resized_w": resized_w,
            "resized_h": resized_h,
            "pad_top_left": 0,
            "pad_bottom_left": 0,
        }

    crop_h = TARGET_SIZE - resized_h
    crop_w = resized_w
    obj_x = int(np.clip(object_center_norm[0], 0, 1) * res_inter_w)
    obj_y = int(np.clip(object_center_norm[1], 0, 1) * res_inter_h)
    anchor_x = int(round(anchor_x_frac * crop_w))
    anchor_y = int(round(anchor_y_frac * crop_h))
    crop_x1 = max(0, obj_x - anchor_x)
    crop_y1 = max(0, obj_y - anchor_y)
    crop_x2 = min(crop_x1 + crop_w, res_inter_w)
    crop_y2 = min(crop_y1 + crop_h, res_inter_h)
    if crop_x2 - crop_x1 < crop_w:
        crop_x1 = max(0, crop_x2 - crop_w)
    if crop_y2 - crop_y1 < crop_h:
        crop_y1 = max(0, crop_y2 - crop_h)

    cropped_top = image_inter[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped_top.size == 0:
        cropped_top = np.zeros((max(1, crop_h), max(1, crop_w), 3), dtype=np.uint8)

    resized_crop = cv2.resize(cropped_top, (crop_w, crop_h))

    pad_left_top = (TARGET_SIZE - crop_w) // 2
    pad_left_bottom = (TARGET_SIZE - resized_w) // 2
    top_band = cv2.copyMakeBorder(
        resized_crop,
        0,
        0,
        pad_left_top,
        TARGET_SIZE - crop_w - pad_left_top,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    bottom_band = cv2.copyMakeBorder(
        resized_bottom,
        0,
        0,
        pad_left_bottom,
        TARGET_SIZE - resized_w - pad_left_bottom,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    composite = np.vstack([top_band, bottom_band])

    meta = {
        "div_y": crop_h,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "scale_inter": scale_inter,
        "scale_to_640": scale_to_640,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "pad_top_left": pad_left_top,
        "pad_bottom_left": pad_left_bottom,
    }
    return composite, meta

def _ensure_image_path(path: Path) -> Path:
    """Validate and return the image path."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


if __name__ == "__main__":
    img_path = _ensure_image_path(IMAGE_PATH)
    y = detect_skyline_y(
        str(img_path),
        cb_min=CB_MIN,
        cb_max=CB_MAX,
        cr_min=CR_MIN,
        cr_max=CR_MAX,
        output_dir=str(OUTPUT_DIR),
    )
    print(f"{y} for {img_path.name}")  # -1 means "no sky" under the 5x rule

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_h, img_w = img.shape[:2]
    obj_center = (0.5, float(y) / img_h) if y >= 0 else (0.5, 0.5)
    composite, meta = generate_composite_640x640(
        img,
        obj_center,
        INTERMEDIATE_SIZE,
        anchor_x_frac=ANCHOR_X_FRAC,
        anchor_y_frac=ANCHOR_Y_FRAC,
    )

    scale_inter = meta["scale_inter"]
    res_inter_w = int(img_w * scale_inter)
    res_inter_h = int(img_h * scale_inter)
    image_inter = cv2.resize(img, (res_inter_w, res_inter_h))
    crop_w = meta["resized_w"]
    crop_h = meta["div_y"]
    crop_x1 = int(meta["crop_x1"])
    crop_y1 = int(meta["crop_y1"])
    crop_x2 = int(np.clip(crop_x1 + crop_w, 0, res_inter_w))
    crop_y2 = int(np.clip(crop_y1 + crop_h, 0, res_inter_h))

    roi_vis = image_inter.copy()
    cv2.rectangle(roi_vis, (crop_x1, crop_y1), (crop_x2 - 1, crop_y2 - 1), (0, 0, 255), 2)
