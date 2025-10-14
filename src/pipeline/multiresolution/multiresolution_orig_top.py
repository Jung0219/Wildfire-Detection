import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO   # pip install ultralytics
from torchvision.ops import nms
from tqdm import tqdm

# ================= CONFIG =================
GT_DIR      = "/lab/projects/fire_smoke_awr/data/detection/training/ABCDE_all_pad_aug"
PARENT_DIR  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_all_pad_aug/test_set"
YOLO_MODEL  = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_all_pad_aug/train/weights/best.pt"

## best values 
INTERMEDIATE_SIZE = 780
NMS_IOU_THRESH = 0.45
SAVE_IMG = True

# ==========================================
IMAGE_DIR = os.path.join(GT_DIR, "images/test")
OUTPUT_DIR = os.path.join(PARENT_DIR, "composites_orig_top")
composite_dir = os.path.join(OUTPUT_DIR, "composite_images")
if SAVE_IMG:
    os.makedirs(composite_dir, exist_ok=True)
# ==========================================


def detect_skyline_y(img_bgr, cb_min=120, cb_max=255, cr_min=0, cr_max=130, sky_ratio_thresh=5.0):
    """
    Heuristic 'skyline' predictor: looks for a sharp drop in sky-like pixels
    across rows to estimate a horizon y. Returns -1 if not confident.
    """
    H, W = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    y_thresh = float(Y.astype(np.float32).mean())

    def sky_mask(bgr):
        ycrcb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y_, Cr_, Cb_ = cv2.split(ycrcb_)
        return ((Y_ >= y_thresh) &
                (Cb_ >= cb_min) & (Cb_ <= cb_max) &
                (Cr_ >= cr_min) & (Cr_ <= cr_max)).astype(np.uint8)

    m_full = sky_mask(img_bgr)
    counts = m_full.sum(axis=1) / float(W)
    d = np.diff(counts)
    idx = int(np.argmin(d))
    y_candidate = int(np.clip(idx + 1, 0, H - 1))
    above = int(m_full[:y_candidate, :].sum())
    below = int(m_full[y_candidate:, :].sum())
    ratio = (above + 1e-9) / (below + 1e-9)
    return y_candidate if ratio >= sky_ratio_thresh else -1


def generate_composite_640x640(original_image, object_center_norm, intermediate_size,
                               anchor_x_frac=0.5, anchor_y_frac=0.25):
    """
    Build a 640x640 composite with:
      - TOP band: resized ORIGINAL image (centered horizontally)
      - BOTTOM band: CROPPED region (from intermediate upscaled image), sized to fill remaining height
    Returns (composite, meta) where meta holds everything needed to invert coordinates.
    """
    TARGET_SIZE = 640
    orig_h, orig_w = original_image.shape[:2]

    # --- Build intermediate (for crop stage) ---
    # scale_inter upscales based on the longer side
    scale_inter = intermediate_size / (orig_w if orig_w >= orig_h else orig_h)
    res_inter_w, res_inter_h = int(orig_w * scale_inter), int(orig_h * scale_inter)
    image_inter = cv2.resize(original_image, (res_inter_w, res_inter_h))

    # --- Build top band (original→fit in 640 box with uniform scale) ---
    scale_to_640 = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    resized_w, resized_h = int(orig_w * scale_to_640), int(orig_h * scale_to_640)
    resized_original = cv2.resize(original_image, (resized_w, resized_h))

    # If it perfectly fills 640x640, the bottom crop band is zero height
    if resized_h == TARGET_SIZE and resized_w == TARGET_SIZE:
        return resized_original, {
            # y-split between TOP (original) and BOTTOM (crop). Here, bottom band has 0 height.
            "div_y": TARGET_SIZE,
            # crop window offsets inside intermediate image (degenerate)
            "crop_x1": 0, "crop_y1": 0,
            # scales
            "scale_inter": scale_inter, "scale_to_640": scale_to_640,
            # sizes of the top (original) band before padding
            "resized_w": resized_w, "resized_h": resized_h,
            # horizontal paddings applied to each band inside the composite
            "pad_crop_left": 0, "pad_orig_left": 0
        }

    # --- Compute crop window that will fill the remaining vertical space ---
    crop_h = TARGET_SIZE - resized_h           # bottom band height
    crop_w = resized_w                         # bottom band width matches top band width
    # Object center in intermediate pixels
    obj_x = int(np.clip(object_center_norm[0], 0, 1) * res_inter_w)
    obj_y = int(np.clip(object_center_norm[1], 0, 1) * res_inter_h)
    # Window anchor offset (how far from window center to its corner)
    anchor_x = int(round(anchor_x_frac * crop_w))
    anchor_y = int(round(anchor_y_frac * crop_h))
    # Window top-left in intermediate coords
    crop_x1 = max(0, obj_x - anchor_x)
    crop_y1 = max(0, obj_y - anchor_y)
    crop_x2 = min(crop_x1 + crop_w, res_inter_w)
    crop_y2 = min(crop_y1 + crop_h, res_inter_h)
    # Adjust for boundary clipping to ensure exact size
    if crop_x2 - crop_x1 < crop_w: crop_x1 = max(0, crop_x2 - crop_w)
    if crop_y2 - crop_y1 < crop_h: crop_y1 = max(0, crop_y2 - crop_h)

    # --- Extract crop and normalize tiny rounding drift to exact (crop_w, crop_h) ---
    crop_bottom = image_inter[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop_bottom.size == 0:
        crop_bottom = np.zeros((max(1, crop_h), max(1, crop_w), 3), dtype=np.uint8)
    resized_crop = cv2.resize(crop_bottom, (crop_w, crop_h))  # safety 1–2 px correction

    # --- Horizontal centering paddings for both bands inside the 640 canvas ---
    pad_crop_left = (TARGET_SIZE - crop_w) // 2      # for bottom (crop) band
    pad_orig_left = (TARGET_SIZE - resized_w) // 2   # for top (original) band

    # --- Build bands with side paddings only (no vertical padding; vertical is by stacking) ---
    top_band = cv2.copyMakeBorder(resized_original, 0, 0,
                                  pad_orig_left, TARGET_SIZE - resized_w - pad_orig_left,
                                  cv2.BORDER_CONSTANT, value=0)  # ORIGINAL goes TOP
    bottom_band = cv2.copyMakeBorder(resized_crop, 0, 0,
                                     pad_crop_left, TARGET_SIZE - crop_w - pad_crop_left,
                                     cv2.BORDER_CONSTANT, value=0)  # CROP goes BOTTOM

    # --- Stack TOP(original) over BOTTOM(crop) ---
    composite = np.vstack([top_band, bottom_band])

    # meta carries everything needed to invert composite→original coordinates
    meta = {
        "div_y": resized_h,            # y-row where bottom band starts
        "crop_x1": crop_x1, "crop_y1": crop_y1,   # crop window location in intermediate image
        "scale_inter": scale_inter, "scale_to_640": scale_to_640,
        "resized_w": resized_w, "resized_h": resized_h,
        "pad_crop_left": pad_crop_left, "pad_orig_left": pad_orig_left
    }
    return composite, meta


def yolo_to_original(box_norm, meta, conf, cls_id, orig_w, orig_h, in_original_band):
    """
    Map a YOLO detection from composite (normalized) back to ORIGINAL image coordinates.
      - If in_original_band=True → detection lies in TOP band (original image after scale_to_640 + horizontal pad)
      - Else → detection lies in BOTTOM band (cropped region from intermediate image)
    Returns: [cls, xc_norm, yc_norm, w_norm, h_norm, conf, x1_px, y1_px, x2_px, y2_px] in original pixels.
    """
    # Convert normalized composite coords to composite pixels
    xc, yc, w, h = box_norm
    x1 = (xc - w/2) * 640
    y1 = (yc - h/2) * 640
    x2 = (xc + w/2) * 640
    y2 = (yc + h/2) * 640

    # Special case: tall/small image path which used pad_or_downscale_to_640 (single band)
    if meta.get("div_y", 0) == 0 and "x_off" in meta:
        # Undo center padding & uniform scaling applied by pad_or_downscale_to_640
        x1 -= meta["x_off"]; x2 -= meta["x_off"]
        y1 -= meta["y_off"]; y2 -= meta["y_off"]
        x1 /= meta["scale_to_640"]; x2 /= meta["scale_to_640"]
        y1 /= meta["scale_to_640"]; y2 /= meta["scale_to_640"]

    elif in_original_band:
        # === TOP BAND: ORIGINAL IMAGE ===
        # Remove the horizontal centering applied to the top band
        x1 -= meta["pad_orig_left"]; x2 -= meta["pad_orig_left"]
        # No vertical shift: top band starts at y=0 in the composite
        # Undo the uniform scaling from original → (resized_w, resized_h)
        x1 /= meta["scale_to_640"]; x2 /= meta["scale_to_640"]
        y1 /= meta["scale_to_640"]; y2 /= meta["scale_to_640"]

    else:
        # === BOTTOM BAND: CROPPED REGION (from INTERMEDIATE) ===
        # Remove the horizontal centering applied to the bottom band
        x1 -= meta["pad_crop_left"]; x2 -= meta["pad_crop_left"]

        # Shift vertically down by the height of the top (original) band
        # because the bottom band starts at y = meta["div_y"] in the composite.
        y1 -= meta["resized_h"]; y2 -= meta["resized_h"]

        # Map from crop band pixels → intermediate image pixels
        x1 += meta["crop_x1"]; x2 += meta["crop_x1"]
        y1 += meta["crop_y1"]; y2 += meta["crop_y1"]

        # Undo the intermediate upscaling to reach original pixels
        x1 /= meta["scale_inter"]; x2 /= meta["scale_inter"]
        y1 /= meta["scale_inter"]; y2 /= meta["scale_inter"]


    return [int(cls_id),
            ((x1 + x2) / 2) / orig_w,
            ((y1 + y2) / 2) / orig_h,
            (x2 - x1) / orig_w,
            (y2 - y1) / orig_h,
            float(conf), x1, y1, x2, y2]


def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def apply_nms(dets, iou_thresh, orig_w, orig_h):
    if not dets:
        return []

    boxes = torch.tensor([d[6:10] for d in dets], dtype=torch.float32)  # x1,y1,x2,y2 in original px
    scores = torch.tensor([d[5] for d in dets], dtype=torch.float32)
    classes = torch.tensor([d[0] for d in dets], dtype=torch.int64)

    keep = []
    for cls in torch.unique(classes):
        idxs = torch.where(classes == cls)[0]
        keep_idx = nms(boxes[idxs], scores[idxs], iou_thresh)
        keep.extend(idxs[keep_idx].tolist())

    final_dets = []
    for i in keep:
        cls_id, xc, yc, w, h, conf, *_ = dets[i]
        final_dets.append([cls_id, xc, yc, w, h, conf])

    return final_dets


def pad_or_downscale_to_640(img, target_size=640, color=(114, 114, 114)):
    """
    Fallback path for tall/small images: fit whole image into 640x640 by uniform scaling + center padding.
    Returns (canvas, (x_off, y_off, scale)).
    """
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
    canvas[y_off:y_off+h, x_off:x_off+w] = img

    return canvas, (x_off, y_off, scale)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_IMG:
        os.makedirs(composite_dir, exist_ok=True)

    model = YOLO(YOLO_MODEL)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        base, ext = os.path.splitext(img_name)

        out_txt = os.path.join(OUTPUT_DIR, base + ".txt")
        out_composite = os.path.join(composite_dir, base + "_composite.jpg")

        original = cv2.imread(img_path)
        H, W = original.shape[:2]

        # Heuristic: if tall or small, use simple pad/downscale (single band).
        if H >= W or H < 640 or W < 640:
            composite, (x_off, y_off, scale) = pad_or_downscale_to_640(original, 640)
            meta = {
                "div_y": 0,                 # single-band path
                "scale_to_640": scale,
                "x_off": x_off,
                "y_off": y_off
            }
        else:
            # Wide image → two-band composite (TOP=original, BOTTOM=crop)
            y_border = detect_skyline_y(original, 120, 255, 0, 130, 5.0)
            obj_center = (0.5, float(y_border) / H) if y_border >= 0 else (0.5, 0.5)
            composite, meta = generate_composite_640x640(original, obj_center, INTERMEDIATE_SIZE)

        if SAVE_IMG:
            cv2.imwrite(out_composite, composite)

        # Run detector on composite
        results = model.predict(composite, imgsz=640, conf=0.001, verbose=False)[0]

        dets_topband, dets_bottomband = [], []  # naming by *visual* position: top=original, bottom=crop

        for box, conf, cls_id in zip(results.boxes.xywhn.cpu().numpy(),
                                     results.boxes.conf.cpu().numpy(),
                                     results.boxes.cls.cpu().numpy()):
            # TOP band test: center y (in px) < div_y means it's in the original band
            in_original_band = (box[1] * 640 < meta.get("div_y", 0) or meta.get("div_y", 0) == 0)
            mapped = yolo_to_original(box, meta, conf, cls_id, W, H, in_original_band)
            if in_original_band:
                dets_topband.append(mapped)     # detections from original (top)
            else:
                dets_bottomband.append(mapped)  # detections from crop (bottom)

        # (Optional) filter either list here if desired; currently pass-through
        merged = dets_topband + dets_bottomband

        # Per-class NMS on original-image pixel boxes
        if NMS_IOU_THRESH and NMS_IOU_THRESH > 0:
            final_dets = apply_nms(merged, NMS_IOU_THRESH, W, H)
        else:
            final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in merged]

        # Save YOLO-format (normalized) + conf to txt
        with open(out_txt, "w") as f:
            for cls_id, xc, yc, w, h, conf in final_dets:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

    print(f"✅ Finished processing {len(image_files)} images. Results saved to {OUTPUT_DIR}")
