import cv2
import math
import numpy as np
import torch
from torchvision.ops import nms

# =======================
# Common IoU Helper
# =======================
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].

    Args:
        box1 (array-like): [x1, y1, x2, y2]
        box2 (array-like): [x1, y1, x2, y2]

    Returns:
        float: IoU value in [0,1].
    """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-9
    return inter / union


def detect_skyline_y(img_bgr, cb_min=120, cb_max=255, cr_min=0, cr_max=130, sky_ratio_thresh=5.0):
    """
    Detect the approximate horizon/skyline Y-coordinate in an image using YCrCb color filtering.

    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        cb_min, cb_max, cr_min, cr_max (int): Color thresholds in YCrCb space to isolate sky regions.
        sky_ratio_thresh (float): Minimum ratio of sky pixels above the candidate boundary
                                  relative to below it, to validate the skyline.

    Returns:
        int: y-coordinate of the skyline if detected, else -1.
    """
    H, W = img_bgr.shape[:2]
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    y_thresh = float(Y.astype(np.float32).mean())  # dynamic brightness threshold

    def sky_mask(bgr):
        """Binary mask of likely sky pixels based on thresholds."""
        ycrcb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y_, Cr_, Cb_ = cv2.split(ycrcb_)
        return ((Y_ >= y_thresh) &
                (Cb_ >= cb_min) & (Cb_ <= cb_max) &
                (Cr_ >= cr_min) & (Cr_ <= cr_max)).astype(np.uint8)

    m_full = sky_mask(img_bgr)
    counts = m_full.sum(axis=1) / float(W)  # ratio of sky pixels per row
    d = np.diff(counts)                     # find row with sharpest change
    idx = int(np.argmin(d))                 # candidate skyline
    y_candidate = int(np.clip(idx + 1, 0, H - 1))
    above = int(m_full[:y_candidate, :].sum())
    below = int(m_full[y_candidate:, :].sum())
    ratio = (above + 1e-9) / (below + 1e-9)

    return y_candidate if ratio >= sky_ratio_thresh else -1


def generate_composite_640x640(original_image, object_center_norm, intermediate_size,
                               anchor_x_frac=0.5, anchor_y_frac=0.25):
    """
    Build a composite 640x640 image consisting of:
      - Top band: cropped zoom-in region around an object of interest.
      - Bottom band: the entire image resized and letterboxed.

    Args:
        original_image (np.ndarray): Original input image (BGR).
        object_center_norm (tuple): Normalized (x,y) center of object of interest in [0,1].
        intermediate_size (int): Resizing dimension for intermediate zoom crop.
        anchor_x_frac, anchor_y_frac (float): Fractions controlling crop anchor relative to object center.

    Returns:
        composite (np.ndarray): 640x640 composite image.
        meta (dict): Metadata needed for mapping detections back to original coordinates.
    """
    TARGET_SIZE = 640
    orig_h, orig_w = original_image.shape[:2]

    # Resize to intermediate size (zoomed-in version)
    scale_inter = intermediate_size / (orig_w if orig_w >= orig_h else orig_h)
    res_inter_w, res_inter_h = int(orig_w * scale_inter), int(orig_h * scale_inter)
    image_inter = cv2.resize(original_image, (res_inter_w, res_inter_h))

    # Resize original for bottom band
    scale_to_640 = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    resized_w, resized_h = int(orig_w * scale_to_640), int(orig_h * scale_to_640)
    resized_bottom = cv2.resize(original_image, (resized_w, resized_h))

    # If already fits 640x640 → no need to composite
    if resized_h == TARGET_SIZE and resized_w == TARGET_SIZE:
        return resized_bottom, {
            "div_y": TARGET_SIZE, "crop_x1": 0, "crop_y1": 0,
            "scale_inter": scale_inter, "scale_to_640": scale_to_640,
            "resized_w": resized_w, "resized_h": resized_h,
            "pad_top_left": 0, "pad_bottom_left": 0
        }

    # Height of top crop = leftover after bottom band is placed
    crop_h = TARGET_SIZE - resized_h
    crop_w = resized_w

    # Object center in intermediate coordinates
    obj_x = int(np.clip(object_center_norm[0], 0, 1) * res_inter_w)
    obj_y = int(np.clip(object_center_norm[1], 0, 1) * res_inter_h)

    # Define crop anchor (offset from object center)
    anchor_x = int(round(anchor_x_frac * crop_w))
    anchor_y = int(round(anchor_y_frac * crop_h))

    # Crop boundaries
    crop_x1 = max(0, obj_x - anchor_x)
    crop_y1 = max(0, obj_y - anchor_y)
    crop_x2 = min(crop_x1 + crop_w, res_inter_w)
    crop_y2 = min(crop_y1 + crop_h, res_inter_h)
    if crop_x2 - crop_x1 < crop_w: crop_x1 = max(0, crop_x2 - crop_w)
    if crop_y2 - crop_y1 < crop_h: crop_y1 = max(0, crop_y2 - crop_h)

    cropped_top = image_inter[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped_top.size == 0:
        cropped_top = np.zeros((max(1, crop_h), max(1, crop_w), 3), dtype=np.uint8)

    resized_crop = cv2.resize(cropped_top, (crop_w, crop_h))

    # Pad top and bottom bands into 640x640
    pad_left_top = (TARGET_SIZE - crop_w) // 2
    pad_left_bottom = (TARGET_SIZE - resized_w) // 2
    top_band = cv2.copyMakeBorder(resized_crop, 0, 0,
                                  pad_left_top, TARGET_SIZE - crop_w - pad_left_top,
                                  cv2.BORDER_CONSTANT, value=0)
    bottom_band = cv2.copyMakeBorder(resized_bottom, 0, 0,
                                     pad_left_bottom, TARGET_SIZE - resized_w - pad_left_bottom,
                                     cv2.BORDER_CONSTANT, value=0)

    composite = np.vstack([top_band, bottom_band])

    meta = {
        "div_y": crop_h, "crop_x1": crop_x1, "crop_y1": crop_y1,
        "scale_inter": scale_inter, "scale_to_640": scale_to_640,
        "resized_w": resized_w, "resized_h": resized_h,
        "pad_top_left": pad_left_top, "pad_bottom_left": pad_left_bottom
    }
    return composite, meta


def yolo_to_original(box_norm, meta, conf, cls_id, orig_w, orig_h, is_bottom):
    """
    Map YOLO normalized detection (on composite) back to original image coordinates.

    Args:
        box_norm (list): [xc, yc, w, h] normalized in [0,1] from YOLO output.
        meta (dict): Metadata from composite creation.
        conf (float): Confidence score.
        cls_id (int): Predicted class id.
        orig_w, orig_h (int): Original image size.
        is_bottom (bool): Whether the detection came from bottom band.

    Returns:
        list: [cls, xc_norm, yc_norm, w_norm, h_norm, conf, x1, y1, x2, y2]
              with normalized coords relative to original image and absolute box corners.
    """
    # Convert to absolute composite coordinates
    xc, yc, w, h = box_norm
    x1 = (xc - w/2) * 640
    y1 = (yc - h/2) * 640
    x2 = (xc + w/2) * 640
    y2 = (yc + h/2) * 640

    # Different mapping depending on whether detection is in top or bottom band
    if meta.get("div_y", 0) == 0 and "x_off" in meta:
        # Tall image → scaled directly
        x1 -= meta["x_off"]; x2 -= meta["x_off"]
        y1 -= meta["y_off"]; y2 -= meta["y_off"]
        x1 /= meta["scale_to_640"]; x2 /= meta["scale_to_640"]
        y1 /= meta["scale_to_640"]; y2 /= meta["scale_to_640"]

    elif is_bottom:  # bottom band
        x1 -= meta["pad_bottom_left"]; x2 -= meta["pad_bottom_left"]
        y1 -= (640 - meta["resized_h"]); y2 -= (640 - meta["resized_h"])
        x1 /= meta["scale_to_640"]; x2 /= meta["scale_to_640"]
        y1 /= meta["scale_to_640"]; y2 /= meta["scale_to_640"]

    else:  # top band
        x1 -= meta["pad_top_left"]; x2 -= meta["pad_top_left"]
        x1 += meta["crop_x1"]; x2 += meta["crop_x1"]
        y1 += meta["crop_y1"]; y2 += meta["crop_y1"]
        x1 /= meta["scale_inter"]; x2 /= meta["scale_inter"]
        y1 /= meta["scale_inter"]; y2 /= meta["scale_inter"]

    return [int(cls_id),
            ((x1 + x2) / 2) / orig_w,
            ((y1 + y2) / 2) / orig_h,
            (x2 - x1) / orig_w,
            (y2 - y1) / orig_h,
            float(conf), x1, y1, x2, y2]


def apply_nms(dets, iou_thresh, orig_w, orig_h):
    """Standard hard NMS using torchvision.ops.nms."""
    if not dets:
        return []

    boxes = torch.tensor([d[6:10] for d in dets], dtype=torch.float32)
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
    Resize or pad an image into a square 640x640 canvas, preserving aspect ratio.

    Args:
        img (np.ndarray): Original image (BGR).
        target_size (int): Desired output dimension.
        color (tuple): Padding color.

    Returns:
        padded_img (np.ndarray): Resulting 640x640 image.
        (x_off, y_off, scale) (tuple): Offsets and scale factor used for mapping boxes back.
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


def soft_nms(dets, sigma=0.5, Nt=0.5, thresh=0.001, method=1):
    """
    Soft Non-Maximum Suppression.
    Reduces confidence of overlapping boxes instead of discarding them.
    """
    if not dets:
        return []

    dets = dets.copy()
    boxes = np.array([d[6:10] for d in dets], dtype=np.float32)
    scores = np.array([d[5] for d in dets], dtype=np.float32)
    N = len(dets)
    indices = np.arange(N)

    for i in range(N):
        max_pos = i + np.argmax(scores[i:])
        boxes[[i, max_pos]] = boxes[[max_pos, i]]
        scores[[i, max_pos]] = scores[[max_pos, i]]
        indices[[i, max_pos]] = indices[[max_pos, i]]

        box_i = boxes[i]
        pos = i + 1
        while pos < N:
            box_j = boxes[pos]
            ovr = compute_iou(box_i, box_j)

            if method == 1:  # linear
                if ovr > Nt:
                    scores[pos] *= (1 - ovr)
            elif method == 2:  # gaussian
                scores[pos] *= np.exp(-(ovr * ovr) / sigma)
            else:  # hard
                if ovr > Nt:
                    scores[pos] = 0

            if scores[pos] < thresh:
                boxes[pos] = boxes[N - 1]
                scores[pos] = scores[N - 1]
                indices[pos] = indices[N - 1]
                N -= 1
                pos -= 1
            pos += 1

    keep = []
    for idx in range(N):
        cls_id, xc, yc, w, h, _, x1, y1, x2, y2 = dets[indices[idx]]
        keep.append([cls_id, xc, yc, w, h, float(scores[idx])])
    return keep


def weighted_box_fusion(dets, iou_thresh=0.55, conf_type="avg"):
    """
    Weighted Box Fusion (WBF).
    Fuses overlapping boxes instead of suppressing them.
    """
    if not dets:
        return []

    boxes = []
    for d in dets:
        xc, yc, w, h = d[1:5]
        x1, y1 = xc - w/2, yc - h/2
        x2, y2 = xc + w/2, yc + h/2
        boxes.append([x1, y1, x2, y2])
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array([d[5] for d in dets], dtype=np.float32)
    labels = np.array([d[0] for d in dets], dtype=np.int32)

    used = [False] * len(dets)
    fused = []

    for i in np.argsort(-scores):
        if used[i]:
            continue
        same_cluster = [i]
        used[i] = True

        for j in np.argsort(-scores):
            if used[j] or labels[i] != labels[j]:
                continue
            if compute_iou(boxes[i], boxes[j]) >= iou_thresh:
                same_cluster.append(j)
                used[j] = True

        cluster_boxes = boxes[same_cluster]
        cluster_scores = scores[same_cluster]
        weights = cluster_scores / (cluster_scores.sum() + 1e-9)

        x1 = (cluster_boxes[:, 0] * weights).sum()
        y1 = (cluster_boxes[:, 1] * weights).sum()
        x2 = (cluster_boxes[:, 2] * weights).sum()
        y2 = (cluster_boxes[:, 3] * weights).sum()

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1

        if conf_type == "avg":
            conf = float(cluster_scores.mean())
        else:
            conf = float(cluster_scores.max())

        fused.append([int(labels[i]), xc, yc, w, h, conf])
    return fused    


def diou_nms(dets, diou_thresh):
    """
    Distance-IoU NMS (DIoU-NMS).
    Suppresses boxes based on both IoU and center distance.
    """
    if not dets:
        return []

    dets = sorted(dets, key=lambda x: x[5], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        remain = []

        for d in dets:
            iou = compute_iou(best[6:10], d[6:10])

            # Distance penalty
            x1, y1, x2, y2 = best[6:10]
            x1g, y1g, x2g, y2g = d[6:10]
            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
            cx2, cy2 = (x1g + x2g) / 2, (y1g + y2g) / 2
            rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

            enc_x1, enc_y1 = min(x1, x1g), min(y1, y1g)
            enc_x2, enc_y2 = max(x2, x2g), max(y2, y2g)
            c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-9

            diou_val = iou - rho2 / c2
            if diou_val <= diou_thresh:
                remain.append(d)

        dets = remain

    return [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in keep]


def ciou_nms(dets, ciou_thresh):
    """
    Complete-IoU NMS (CIoU-NMS).
    Suppresses boxes based on IoU, center distance, and aspect ratio consistency.

    Args:
        dets (list): detections in format
                     [cls, xc, yc, w, h, conf, x1, y1, x2, y2]
        ciou_thresh (float): threshold applied on CIoU score.

    Returns:
        list: filtered detections [cls, xc, yc, w, h, conf]
    """
    if not dets:
        return []

    # Sort detections by confidence
    dets = sorted(dets, key=lambda x: x[5], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        remain = []

        for d in dets:
            # --- IoU ---
            iou = compute_iou(best[6:10], d[6:10])

            # --- Distance penalty ---
            x1, y1, x2, y2 = best[6:10]
            x1g, y1g, x2g, y2g = d[6:10]
            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
            cx2, cy2 = (x1g + x2g) / 2, (y1g + y2g) / 2
            rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

            enc_x1, enc_y1 = min(x1, x1g), min(y1, y1g)
            enc_x2, enc_y2 = max(x2, x2g), max(y2, y2g)
            c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-9

            dist_penalty = rho2 / c2

            # --- Aspect ratio penalty ---
            w1, h1 = (x2 - x1), (y2 - y1)
            w2, h2 = (x2g - x1g), (y2g - y1g)
            v = (4 / math.pi**2) * (math.atan(w1 / h1) - math.atan(w2 / h2))**2
            alpha = v / (1 - iou + v + 1e-9)

            aspect_penalty = alpha * v

            # --- CIoU score ---
            ciou_val = iou - dist_penalty - aspect_penalty

            # Suppress if similarity score is above threshold
            if ciou_val <= ciou_thresh:
                remain.append(d)

        dets = remain

    return [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in keep]


def process_image(img_path, model, out_txt, out_composite,
                  intermediate_size, conf_thresh=None,
                  size_thresh=None, nms_iou_thresh=None,
                  save_img=False, postproc="nms"):
    """
    High-level wrapper for full composite detection pipeline:
      1. Build composite (zoomed + full image bands).
      2. Run YOLO detection.
      3. Map detections back to original coordinates.
      4. Apply filtering (confidence, size).
      5. Apply post-processing (NMS / Soft-NMS / DIoU-NMS / CIoU-NMS / WBF).
      6. Save results in YOLO .txt format.

    Args:
        img_path (str): Path to input image.
        model (YOLO): Ultralytics YOLO model instance.
        out_txt (str): Path to save detections (YOLO txt format).
        out_composite (str): Path to save composite visualization.
        intermediate_size (int): Size for intermediate zoom.
        conf_thresh (float): Confidence threshold.
        size_thresh (float): Max size (sqrt(w*h)) for detections.
        nms_iou_thresh (float): IoU threshold for NMS.
        save_img (bool): Whether to save composite image.
        postproc (str): One of ["nms", "softnms", "wbf", "none"].

    Returns:
        None (saves results to disk).
    """
    original = cv2.imread(img_path)
    H, W = original.shape[:2]

    # Step 1: build composite
    if H >= W or H < 640 or W < 640:
        composite, (x_off, y_off, scale) = pad_or_downscale_to_640(original, 640)
        meta = {"div_y": 0, "scale_to_640": scale, "x_off": x_off, "y_off": y_off}
    else:
        y_border = detect_skyline_y(original, 120, 255, 0, 130, 5.0)
        obj_center = (0.5, float(y_border) / H) if y_border >= 0 else (0.5, 0.5)
        composite, meta = generate_composite_640x640(original, obj_center, intermediate_size)

    if save_img:
        cv2.imwrite(out_composite, composite)

    # Step 2: run YOLO
    results = model.predict(composite, imgsz=640, conf=0.001, verbose=False)[0]
    dets_top, dets_bottom = [], []

    for box, conf, cls_id in zip(results.boxes.xywhn.cpu().numpy(),
                                 results.boxes.conf.cpu().numpy(),
                                 results.boxes.cls.cpu().numpy()):
        is_bottom = (box[1] * 640 >= meta["div_y"])
        mapped = yolo_to_original(box, meta, conf, cls_id, W, H, is_bottom)
        (dets_bottom if is_bottom else dets_top).append(mapped)

    # Step 3: filter top detections
    filtered_top = []
    for det in dets_top:
        cls_id, xc, yc, w, h, conf, x1, y1, x2, y2 = det
        size = np.sqrt((x2 - x1) * (y2 - y1))
        if conf_thresh is not None and conf < conf_thresh:
            continue
        if size_thresh is not None and size >= size_thresh:
            continue
        filtered_top.append(det)

    merged = dets_bottom + filtered_top

    # Step 4: apply post-processing
    if postproc == "nms" and nms_iou_thresh and nms_iou_thresh > 0:
        final_dets = apply_nms(merged, nms_iou_thresh, W, H)
    elif postproc == "softnms":
        final_dets = soft_nms(merged, sigma=0.5, Nt=nms_iou_thresh, method=1)
    elif postproc == "diounms":
        final_dets = diou_nms(merged, diou_thresh=nms_iou_thresh)
    elif postproc == "wbf":
        final_dets = weighted_box_fusion(merged, iou_thresh=nms_iou_thresh)
    elif postproc == "ciounms":
        final_dets = ciou_nms(merged, ciou_thresh=nms_iou_thresh)
    else:  # "none"
        final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in merged]

    # Step 5: save YOLO-format txt
    with open(out_txt, "w") as f:
        for cls_id, xc, yc, w, h, conf in final_dets:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")


