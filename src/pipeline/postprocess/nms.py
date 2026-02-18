"""Non-maximum suppression utilities."""

from __future__ import annotations

from typing import List
import torch
from torchvision.ops import nms


def apply_nms(dets: List[List[float]], iou_thresh: float) -> List[List[float]]:
    """Apply per-class NMS on detections (expects x1..y2 at positions 6:10)."""

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

