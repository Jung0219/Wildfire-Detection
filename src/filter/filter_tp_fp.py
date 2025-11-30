"""
Split YOLO predictions into true positives and false positives based on IoU.

Example:
    python -m src.filter.filter_tp_fp
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from collections import defaultdict

from tqdm import tqdm

CONFIG = {
    "GT_DIR": "/lab/projects/fire_smoke_awr/data/detection/training/AD_phash3_early_smoke/original/labels/test", # the labels folder
    "PRED_DIR": "/lab/projects/fire_smoke_awr/outputs/yolo/detection/AD_phash3_early_smoke/AdamW/800_AdamW/test/labels",
    "TP_OUT_DIR_NAME": "tp",
    "FP_OUT_DIR_NAME": "fp",
    "TP_IOU_THRESH": 0.5,
    "FP_IOU_THRESH": 0.5-1e-6,
    "CONF_MIN": 0.0,
    "CONF_MAX": 1.0,
}


@dataclass
class YoloBox:
    """YOLO-format bounding box with optional confidence."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float = 1.0
    raw_line: str = ""


@dataclass
class MatchResult:
    """Stores the result of matching a prediction box against GT."""

    pred_index: int
    status: Optional[str]
    iou: float


def load_yolo_boxes(file_path: Path) -> List[YoloBox]:
    """Parse YOLO txt file into a list of YoloBox entries."""
    if not file_path.exists():
        return []
    boxes: List[YoloBox] = []
    for line in file_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        boxes.append(
            YoloBox(
                cls=cls,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                conf=conf,
                raw_line=line.strip(),
            )
        )
    return boxes


def yolo_to_xyxy(box: YoloBox) -> tuple[float, float, float, float]:
    """Convert YOLO center/width/height to corner coordinates."""
    x1 = box.cx - box.w / 2.0
    y1 = box.cy - box.h / 2.0
    x2 = box.cx + box.w / 2.0
    y2 = box.cy + box.h / 2.0
    return x1, y1, x2, y2


def box_iou(box_a: YoloBox, box_b: YoloBox) -> float:
    """Compute IoU between two YOLO boxes in normalized coordinates."""
    ax1, ay1, ax2, ay2 = yolo_to_xyxy(box_a)
    bx1, by1, bx2, by2 = yolo_to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h

    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_predictions(
    gt_boxes: List[YoloBox],
    pred_boxes: List[YoloBox],
    tp_iou_thresh: float,
    fp_iou_thresh: float,
) -> Dict[int, MatchResult]:
    """Assign prediction boxes to GT boxes with IoU-based TP/FP labels."""
    match_map: Dict[int, MatchResult] = {}
    used_gt = [False] * len(gt_boxes)
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i].conf, reverse=True)

    for pred_idx in order:
        pred_box = pred_boxes[pred_idx]
        best_same_iou = 0.0
        best_gt_idx: Optional[int] = None
        best_any_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = box_iou(pred_box, gt_box)
            if iou > best_any_iou:
                best_any_iou = iou
            if used_gt[gt_idx] or gt_box.cls != pred_box.cls:
                continue
            if iou > best_same_iou:
                best_same_iou = iou
                best_gt_idx = gt_idx

        status: Optional[str] = None
        if best_gt_idx is not None and best_same_iou >= tp_iou_thresh:
            status = "tp"
            used_gt[best_gt_idx] = True
        elif best_any_iou <= fp_iou_thresh:
            status = "fp"

        reported_iou = best_same_iou if best_gt_idx is not None else best_any_iou
        match_map[pred_idx] = MatchResult(pred_index=pred_idx, status=status, iou=reported_iou)
    return match_map


def ensure_directory(path: Path) -> None:
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_lines(file_path: Path, lines: Iterable[str]) -> None:
    """Write YOLO lines to disk (empty file if no lines)."""
    content = "\n".join(lines)
    file_path.write_text(content)


def process_predictions() -> None:
    """Entry point for filtering predictions into TP/FP splits."""
    gt_dir = Path(CONFIG["GT_DIR"])
    pred_dir = Path(CONFIG["PRED_DIR"])
    tp_out_dirname = str(CONFIG["TP_OUT_DIR_NAME"])
    fp_out_dirname = str(CONFIG["FP_OUT_DIR_NAME"])
    tp_iou_thresh = float(CONFIG["TP_IOU_THRESH"])
    fp_iou_thresh = float(CONFIG["FP_IOU_THRESH"])
    conf_min = float(CONFIG["CONF_MIN"])
    conf_max = float(CONFIG["CONF_MAX"])
    tp_dir = pred_dir / tp_out_dirname
    fp_dir = pred_dir / fp_out_dirname

    if fp_iou_thresh >= tp_iou_thresh:
        raise ValueError("FP_IOU_THRESH must be lower than TP_IOU_THRESH.")
    if conf_min > conf_max:
        raise ValueError("CONF_MIN must be lower than or equal to CONF_MAX.")

    for path in (gt_dir, pred_dir):
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
    if not tp_out_dirname:
        raise ValueError("TP_OUT_DIR_NAME cannot be empty.")
    if not fp_out_dirname:
        raise ValueError("FP_OUT_DIR_NAME cannot be empty.")

    ensure_directory(tp_dir)
    ensure_directory(fp_dir)

    print("CONFIG:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print(f"  TP_DIR: {tp_dir}")
    print(f"  FP_DIR: {fp_dir}")

    total_preds = 0
    total_tp = 0
    total_fp = 0
    ignored_preds = 0
    filtered_by_conf = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0})

    pred_files = sorted(pred_dir.glob("*.txt"))
    for pred_file in tqdm(pred_files, desc="Filtering predictions"):
        preds_all = load_yolo_boxes(pred_file)
        preds = [box for box in preds_all if conf_min <= box.conf <= conf_max]
        filtered_by_conf += len(preds_all) - len(preds)

        gt_file = gt_dir / pred_file.name
        gt_boxes = load_yolo_boxes(gt_file)

        matches = match_predictions(gt_boxes, preds, tp_iou_thresh, fp_iou_thresh)

        tp_lines: List[str] = []
        fp_lines: List[str] = []
        for idx, box in enumerate(preds):
            result = matches.get(idx)
            if result and result.status == "tp":
                tp_lines.append(box.raw_line)
                total_tp += 1
                per_class[box.cls]["tp"] += 1
            elif result and result.status == "fp":
                fp_lines.append(box.raw_line)
                total_fp += 1
                per_class[box.cls]["fp"] += 1
            else:
                ignored_preds += 1

        total_preds += len(preds)

        tp_out = tp_dir / pred_file.name
        fp_out = fp_dir / pred_file.name
        tp_out.parent.mkdir(parents=True, exist_ok=True)
        fp_out.parent.mkdir(parents=True, exist_ok=True)
        write_lines(tp_out, tp_lines)
        write_lines(fp_out, fp_lines)

    print("\n=== Filtering Summary ===")
    print(f"Processed prediction files: {len(pred_files)}")
    print(f"Predictions (after conf filter): {total_preds}")
    print(f"True positives written: {total_tp}")
    print(f"False positives written: {total_fp}")
    print(f"Ignored predictions (IoU between thresholds): {ignored_preds}")
    print(f"Filtered out by confidence bounds: {filtered_by_conf}")

    if per_class:
        print("\nPer-class counts:")
        for cls_id in sorted(per_class.keys()):
            stats = per_class[cls_id]
            print(f"  class {cls_id}: TP={stats['tp']} | FP={stats['fp']}")


def main() -> None:
    """CLI entry point."""
    process_predictions()


if __name__ == "__main__":
    main()
