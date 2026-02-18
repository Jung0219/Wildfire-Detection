"""Two-stage gating with a secondary classifier."""

from __future__ import annotations

from typing import Dict, List, Tuple
import cv2


def extract_centered_crop(image, xc_norm: float, yc_norm: float, crop_size: int):
    """Create a fixed-size crop centered on normalized box center."""

    img_h, img_w = image.shape[:2]
    crop_h = crop_w = crop_size
    cx_px = xc_norm * img_w
    cy_px = yc_norm * img_h

    x1 = int(round(cx_px - crop_w / 2))
    y1 = int(round(cy_px - crop_h / 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(img_w, x2)
    y2_clamped = min(img_h, y2)

    crop = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    if crop.size == 0:
        return None

    if pad_left or pad_right or pad_top or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    return crop


def predict_label(classifier_model, crop) -> str:
    """Run the secondary classifier on a crop; return label or 'background'."""

    if crop is None or crop.size == 0:
        return "background"

    result = classifier_model.predict(crop)
    if isinstance(result, str):
        return result

    # Handle raw ultralytics models
    results = result[0] if hasattr(result, "__getitem__") else result

    if getattr(results, "probs", None) is not None:
        cls_id = int(results.probs.top1)
        names = results.names
        if isinstance(names, dict):
            return names.get(cls_id, "background")
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)

    boxes = getattr(results, "boxes", None)
    if boxes is None or boxes.data.numel() == 0:
        return "background"

    best_idx = int(boxes.conf.argmax()) if boxes.conf is not None else 0
    cls_id = int(boxes.cls[best_idx])
    names = results.names
    if isinstance(names, dict):
        return names.get(cls_id, "background")
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def verify_with_classifier(
    detections: List[List[float]],
    image,
    classifier_model,
    conf_low: float,
    conf_high: float,
    crop_size: int,
) -> Tuple[List[List[float]], Dict[str, int]]:
    """Apply confidence gating and classifier verification."""

    counts = {
        "dropped_low": 0,
        "auto_kept_high": 0,
        "sent_to_classifier": 0,
        "kept_after_classifier": 0,
    }
    if not detections:
        return [], counts

    kept: List[List[float]] = []
    for cls_id, xc, yc, w, h, conf in detections:
        if conf < conf_low:
            counts["dropped_low"] += 1
            continue

        if conf >= conf_high:
            kept.append([cls_id, xc, yc, w, h, conf])
            counts["auto_kept_high"] += 1
            continue

        crop = extract_centered_crop(image, xc, yc, crop_size)
        counts["sent_to_classifier"] += 1
        if crop is None or crop.size == 0:
            continue

        pred_label = predict_label(classifier_model, crop)
        if pred_label.lower() != "background":
            kept.append([cls_id, xc, yc, w, h, conf])
            counts["kept_after_classifier"] += 1

    return kept, counts
