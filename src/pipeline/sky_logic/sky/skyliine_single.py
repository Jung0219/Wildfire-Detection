import cv2
import numpy as np
from pathlib import Path

def detect_skyline_y(img_path: str, cb_min: int = 120, cb_max: int = 255,
                     cr_min: int = 0, cr_max: int = 130) -> int:
    """
    Returns the estimated sky/ground border row (y) using:
      1) adaptive Y threshold (mean Y) for sky mask
      2) per-row sky fraction "graph" with downscale+blur for speed
      3) boundary at the most negative derivative
      4) 5× rule: sky pixels above / below >= 5 to accept; else returns -1
    """
    # ---- internal defaults ----
    PREPROC_FOR_GRAPH = True
    GRAPH_INPUT_SIZE  = 256
    GRAPH_BLUR_K      = 11
    GRAPH_BLUR_SIGMA  = 1.5

    DERIV_SMOOTH      = True
    DERIV_SMOOTH_K    = 11
    DERIV_SMOOTH_SIG  = 2.0

    IGNORE_EDGE_FRAC  = 0.02
    SKY_RATIO_THRESH  = 5.0
    # ---------------------------

    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    H, W = img.shape[:2]

    # YCbCr and adaptive Y threshold (mean Y)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y = Y.astype(np.float32)
    y_thresh = float(Y.mean())

    # Helper: sky mask under absolute ranges
    def sky_mask(bgr):
        ycrcb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y_, Cr_, Cb_ = cv2.split(ycrcb_)
        return ((Y_.astype(np.float32) >= y_thresh) &
                (Cb_ >= cb_min) & (Cb_ <= cb_max) &
                (Cr_ >= cr_min) & (Cr_ <= cr_max)).astype(np.uint8)

    # Build per-row sky fraction graph (normalized to [0,1])
    if PREPROC_FOR_GRAPH:
        scale = GRAPH_INPUT_SIZE / max(H, W)
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if GRAPH_BLUR_K % 2 == 1 and GRAPH_BLUR_K > 0:
            small = cv2.GaussianBlur(small, (GRAPH_BLUR_K, GRAPH_BLUR_K), GRAPH_BLUR_SIGMA)

        m_small = sky_mask(small)
        counts = m_small.sum(axis=1).astype(np.float32)                 # length new_h
        counts /= float(m_small.shape[1])                               # normalize by width

        y_src = np.linspace(0, H - 1, num=counts.shape[0], dtype=np.float32)
        y_dst = np.arange(H, dtype=np.float32)
        graph = np.interp(y_dst, y_src, counts).astype(np.float32)
    else:
        m_full = sky_mask(img)
        graph = m_full.sum(axis=1).astype(np.float32) / float(W)

    # Smooth before derivative (optional)
    if DERIV_SMOOTH:
        kx = (DERIV_SMOOTH_K if (DERIV_SMOOTH_K and DERIV_SMOOTH_K % 2 == 1) else 0, 1)
        graph = cv2.GaussianBlur(graph.reshape(-1, 1), kx, DERIV_SMOOTH_SIG).ravel()

    # Find steepest negative drop
    d = np.diff(graph)  # s[y+1] - s[y]
    top_ignore = int(round(IGNORE_EDGE_FRAC * H))
    bot_ignore = int(round(IGNORE_EDGE_FRAC * H))
    lo = max(0, top_ignore)
    hi = max(lo + 1, (H - 1) - bot_ignore)
    idx = lo + int(np.argmin(d[lo:hi]))
    y_candidate = int(np.clip(idx + 1, 0, H - 1))

    # Validate with full-res mask and 5× rule
    m_full = sky_mask(img)
    above = int(m_full[:y_candidate, :].sum())
    below = int(m_full[y_candidate:, :].sum())
    ratio = (above + 1e-9) / (below + 1e-9)

    return y_candidate if ratio >= SKY_RATIO_THRESH else -1

# Example:
# y = detect_skyline_y("/path/to/img.jpg", cb_min=120, cb_max=255, cr_min=0, cr_max=130)
# print("skyline y:", y)  # -1 means "no sky" under the 5× rule
