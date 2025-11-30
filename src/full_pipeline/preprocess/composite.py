"""Preprocessing utilities to build composites or padded images for detection.

Example:
    from src.pipeline.mr_modular.preprocess.composite import prepare_image_for_detection
    composite, meta = prepare_image_for_detection(img, intermediate_size=900, anchor_y_frac=0.6)
"""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def detect_skyline_y(img_bgr, cb_min=120, cb_max=255, cr_min=0, cr_max=130, sky_ratio_thresh=5.0):
    """Detect a skyline row; returns -1 if not found."""

    H, W = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    y_thresh = float(Y.astype(np.float32).mean())

    def sky_mask(bgr):
        ycrcb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y_, Cr_, Cb_ = cv2.split(ycrcb_)
        return ((Y_ >= y_thresh) & (Cb_ >= cb_min) & (Cb_ <= cb_max) & (Cr_ >= cr_min) & (Cr_ <= cr_max)).astype(np.uint8)

    m_full = sky_mask(img_bgr)
    counts = m_full.sum(axis=1) / float(W)
    d = np.diff(counts)
    idx = int(np.argmin(d))
    y_candidate = int(np.clip(idx + 1, 0, H - 1))
    above = int(m_full[:y_candidate, :].sum())
    below = int(m_full[y_candidate:, :].sum())
    ratio = (above + 1e-9) / (below + 1e-9)
    return y_candidate if ratio >= sky_ratio_thresh else -1


def generate_composite_640x640(original_image, object_center_norm, intermediate_size, anchor_x_frac=0.5, anchor_y_frac=0.25):
    """Create a 640x640 composite for wide images; returns (composite, meta)."""

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


def pad_or_downscale_to_640(img, target_size=640, color=(114, 114, 114)):
    """Pad or downscale an image to a square canvas."""

    h, w = img.shape[:2]

    if h > target_size or w > target_size:
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        h, w = new_h, new_w
    else:
        scale = 1.0

    canvas = np.full((target_size, target_size, 3), color, dtype=img.dtype)
    y_off = (target_size - h) // 2
    x_off = (target_size - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = img

    return canvas, (x_off, y_off, scale)


def prepare_image_for_detection(image, intermediate_size: int, anchor_y_frac: float):
    """Select padding vs composite strategy and return (composite, meta)."""

    img_h, img_w = image.shape[:2]
    if img_h >= img_w or img_h < 640 or img_w < 640:
        composite, (x_off, y_off, scale) = pad_or_downscale_to_640(image, 640)
        meta = {"div_y": 0, "scale_to_640": scale, "x_off": x_off, "y_off": y_off}
    else:
        y_border = detect_skyline_y(image, 120, 255, 0, 130, 5.0)
        obj_center = (0.5, float(y_border) / img_h) if y_border >= 0 else (0.5, 0.5)
        composite, meta = generate_composite_640x640(
            image, obj_center, intermediate_size, anchor_y_frac=anchor_y_frac
        )
    return composite, meta

