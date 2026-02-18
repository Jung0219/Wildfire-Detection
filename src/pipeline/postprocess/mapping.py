"""Mapping utilities to convert detections back to original image coords."""

from __future__ import annotations

from typing import List


def yolo_to_original(box_norm, meta, conf, cls_id, orig_w, orig_h, is_bottom):
    """Map a detection from the composite/padded space back to original coords."""

    target_size = int(meta.get("target_size", 640))
    xc, yc, w, h = box_norm
    x1 = (xc - w / 2) * target_size
    y1 = (yc - h / 2) * target_size
    x2 = (xc + w / 2) * target_size
    y2 = (yc + h / 2) * target_size

    if meta.get("div_y", 0) == 0 and "x_off" in meta:
        x1 -= meta["x_off"]
        x2 -= meta["x_off"]
        y1 -= meta["y_off"]
        y2 -= meta["y_off"]
        x1 /= meta["scale_to_640"]
        x2 /= meta["scale_to_640"]
        y1 /= meta["scale_to_640"]
        y2 /= meta["scale_to_640"]
    elif is_bottom:
        x1 -= meta["pad_bottom_left"]
        x2 -= meta["pad_bottom_left"]
        y1 -= target_size - meta["resized_h"]
        y2 -= target_size - meta["resized_h"]
        x1 /= meta["scale_to_640"]
        x2 /= meta["scale_to_640"]
        y1 /= meta["scale_to_640"]
        y2 /= meta["scale_to_640"]
    else:
        x1 -= meta["pad_top_left"]
        x2 -= meta["pad_top_left"]
        x1 += meta["crop_x1"]
        x2 += meta["crop_x1"]
        y1 += meta["crop_y1"]
        y2 += meta["crop_y1"]
        x1 /= meta["scale_inter"]
        x2 /= meta["scale_inter"]
        y1 /= meta["scale_inter"]
        y2 /= meta["scale_inter"]

    return [
        int(cls_id),
        ((x1 + x2) / 2) / orig_w,
        ((y1 + y2) / 2) / orig_h,
        (x2 - x1) / orig_w,
        (y2 - y1) / orig_h,
        float(conf),
        x1,
        y1,
        x2,
        y2,
    ]


def map_detections(yolo_result, meta, orig_shape) -> List[List[float]]:
    """Map YOLO detections back to original coordinates."""

    H, W = orig_shape[:2]
    target_size = int(meta.get("target_size", 640))
    mapped = []
    for box, conf, cls_id in zip(
        yolo_result.boxes.xywhn.cpu().numpy(),
        yolo_result.boxes.conf.cpu().numpy(),
        yolo_result.boxes.cls.cpu().numpy(),
    ):
        is_bottom = (box[1] * target_size) >= meta.get("div_y", 0)
        mapped.append(yolo_to_original(box, meta, conf, cls_id, W, H, is_bottom))
    return mapped
