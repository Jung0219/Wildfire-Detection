"""
Compare YOLO prediction annotations against ground truth and report overlap rates.

Example:
    python -m src.util.etc.compare_overlap
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from tqdm.auto import tqdm

CONFIG = {
    # Directory containing YOLO GT label files (one .txt per image).
    "GT_DIR": Path("/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/phash10/original/labels/test"),
    # Directory containing YOLO prediction label files to evaluate.
    "PRED_DIR": Path("/lab/projects/fire_smoke_awr/outputs/yolo/detection/pyro-sdis/phash10/900/es_test/composites/fp_labels_01_03"),
}

MIN_IOU_FOR_MATCH = 0.0  # Any positive intersection counts as a match.


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_corners(self) -> tuple[float, float, float, float]:
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        x1 = self.x_center - half_w
        y1 = self.y_center - half_h
        x2 = self.x_center + half_w
        y2 = self.y_center + half_h
        return x1, y1, x2, y2


def read_yolo_file(path: Path) -> List[YoloBox]:
    if not path.exists():
        return []

    boxes: List[YoloBox] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 6:
            class_id, x_center, y_center, width, height, _conf = parts
        elif len(parts) == 5:
            class_id, x_center, y_center, width, height = parts
        else:
            raise ValueError(
                f"Invalid YOLO row in {path}: expected 5 or 6 columns, got {len(parts)} :: {line}"
            )
        boxes.append(
            YoloBox(
                class_id=int(class_id),
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
            )
        )
    return boxes


def iou(box_a: YoloBox, box_b: YoloBox) -> float:
    ax1, ay1, ax2, ay2 = box_a.to_corners()
    bx1, by1, bx2, by2 = box_b.to_corners()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def count_matches(pred_boxes: Iterable[YoloBox], gt_boxes: List[YoloBox]) -> int:
    gt_by_class = {}
    for gt in gt_boxes:
        gt_by_class.setdefault(gt.class_id, []).append(gt)

    matches = 0
    for pred in pred_boxes:
        candidates = gt_by_class.get(pred.class_id, [])
        if any(iou(pred, gt) > MIN_IOU_FOR_MATCH for gt in candidates):
            matches += 1
    return matches


def compare_overlaps(gt_dir: Path, pred_dir: Path) -> None:
    pred_files = sorted(pred_dir.rglob("*.txt"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction label files found under {pred_dir}")

    total_preds = 0
    matched_preds = 0

    print("CONFIG:")
    print(f"  GT_DIR   = {gt_dir}")
    print(f"  PRED_DIR = {pred_dir}")
    print(f"  MIN_IOU_FOR_MATCH = {MIN_IOU_FOR_MATCH}")
    print()

    for pred_file in tqdm(pred_files, desc="Comparing predictions"):
        relative_path = pred_file.relative_to(pred_dir)
        gt_file = gt_dir / relative_path

        pred_boxes = read_yolo_file(pred_file)
        gt_boxes = read_yolo_file(gt_file)

        matched = count_matches(pred_boxes, gt_boxes)
        total_preds += len(pred_boxes)
        matched_preds += matched

    if total_preds == 0:
        print("No predicted boxes found to compare.")
        return

    percent = (matched_preds / total_preds) * 100.0
    print(f"Total predicted boxes: {total_preds}")
    print(f"Predicted boxes with overlapping GT of same class: {matched_preds}")
    print(f"Percentage matched: {percent:.2f}%")


def main() -> None:
    compare_overlaps(CONFIG["GT_DIR"], CONFIG["PRED_DIR"])


if __name__ == "__main__":
    main()
