"""Evaluate classifier-gated detection refinement on a confidence band using YOLO classifier.

Example:
    python -m src.pipeline.two_stage.classifier_accuracy
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

from src.evaluation.detection.eval_metrics import evaluate_directories


# ============ CONFIG ============
PRED_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/pyro-sdis/phash10/900/es_test/composites"
IMG_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/phash10/original/images/test"
GT_LABELS_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/pyro-sdis/phash10/original/labels/test"

CLASSIFIER_WEIGHTS = "/lab/projects/fire_smoke_awr/outputs/yolo/classification/pyro_sdis/224x224/train/weights/best.pt"  # YOLO classifier weights
IMG_SIZE = 224
CLS_POSITIVE_LABEL = 1  # treat this classifier output as foreground

CONF_MIN = 0.1
CONF_MAX = 0.5
IOU_THRESH = 0.5
OUTPUT_ROOT = "/lab/projects/fire_smoke_awr/src/pipeline/two_stage/classifier_accuracy_verification"
DEVICE = "cuda"  # YOLO will fallback if unavailable
# =================================


@dataclass
class PredLine:
    cls: str
    x: float
    y: float
    w: float
    h: float
    conf: float
    raw: str


def load_pred_lines(path: Path) -> List[PredLine]:
    lines: List[PredLine] = []
    if not path.exists():
        return lines
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls, x, y, w, h, conf = parts[:6]
        lines.append(
            PredLine(
                cls=cls,
                x=float(x),
                y=float(y),
                w=float(w),
                h=float(h),
                conf=float(conf),
                raw=line.strip(),
            )
        )
    return lines


def find_image(img_root: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = img_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def crop_fixed_center(img: Image.Image, box: PredLine, size: int) -> Image.Image:
    """Crop a fixed-size window centered on the YOLO box."""
    w, h = img.size
    cx = box.x * w
    cy = box.y * h
    half = size / 2
    left = max(0, cx - half)
    top = max(0, cy - half)
    right = left + size
    bottom = top + size

    # Clamp if we run past edges
    if right > w:
        overflow = right - w
        left = max(0, left - overflow)
        right = w
    if bottom > h:
        overflow = bottom - h
        top = max(0, top - overflow)
        bottom = h
    return img.crop((int(left), int(top), int(right), int(bottom)))


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def yolo_to_xyxy(box: PredLine, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert YOLO normalized (cx, cy, w, h) to pixel xyxy."""
    cx, cy, bw, bh = box.x * img_w, box.y * img_h, box.w * img_w, box.h * img_h
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(float(img_w), cx + bw / 2.0)
    y2 = min(float(img_h), cy + bh / 2.0)
    return x1, y1, x2, y2


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[PredLine],
    color: str,
    label: str,
) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    w, h = image.size
    for b in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(b, w, h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text_pos = (x1, max(0, y1 - 10))
        draw.text(text_pos, label, fill=color, font=font)


def classify_batch(model: YOLO, crops: List[Image.Image]) -> List[int]:
    """Run YOLO classifier on a list of crops and return top1 class ids."""
    results = model.predict(crops, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
    preds: List[int] = []
    for res in results:
        if res.probs is not None and res.probs.top1 is not None:
            preds.append(int(res.probs.top1))
        else:
            preds.append(-1)
    return preds


def main() -> None:
    run_name = f"classifier_accuracy_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(OUTPUT_ROOT) / run_name
    refined_dir = out_root / "refined"
    band_before_dir = out_root / "band_before"
    band_after_dir = out_root / "band_after"
    vis_dir = out_root / "visualizations"
    out_root.mkdir(parents=True, exist_ok=True)

    print("CONFIG:")
    print(
        json.dumps(
            {
                "pred_dir": PRED_DIR,
                "img_dir": IMG_DIR,
                "gt_labels_dir": GT_LABELS_DIR,
                "classifier_weights": CLASSIFIER_WEIGHTS,
                "img_size": IMG_SIZE,
                "cls_positive_label": CLS_POSITIVE_LABEL,
                "conf_min": CONF_MIN,
                "conf_max": CONF_MAX,
                "iou_thresh": IOU_THRESH,
                "output_root": str(out_root),
                "device": DEVICE,
            },
            indent=2,
        )
    )

    model = YOLO(CLASSIFIER_WEIGHTS)

    pred_root = Path(PRED_DIR)
    img_root = Path(IMG_DIR)
    pred_files = sorted(pred_root.glob("*.txt"))

    total_candidates = total_kept = total_pass_through = total_preds = 0
    for pred_path in tqdm(pred_files, desc="Processing files"):
        stem = pred_path.stem
        img_path = find_image(img_root, stem)
        if img_path is None:
            print(f"[WARN] Missing image for {stem}, skipping.")
            continue

        preds = load_pred_lines(pred_path)
        if not preds:
            write_lines(refined_dir / pred_path.name, [])
            write_lines(band_before_dir / pred_path.name, [])
            write_lines(band_after_dir / pred_path.name, [])
            continue

        image = Image.open(img_path).convert("RGB")

        band_candidates: List[PredLine] = []
        pass_through: List[PredLine] = []
        for p in preds:
            if CONF_MIN <= p.conf <= CONF_MAX:
                band_candidates.append(p)
            else:
                pass_through.append(p)

        # classify candidates
        kept_candidates: List[PredLine] = []
        if band_candidates:
            crops = [crop_fixed_center(image, p, IMG_SIZE) for p in band_candidates]
            cls_preds = classify_batch(model, crops)
            for p, cls_pred in zip(band_candidates, cls_preds):
                if cls_pred == CLS_POSITIVE_LABEL:
                    kept_candidates.append(p)

        refined_lines = [p.raw for p in pass_through + kept_candidates]
        before_lines = [p.raw for p in band_candidates]
        after_lines = [p.raw for p in kept_candidates]

        write_lines(refined_dir / pred_path.name, refined_lines)
        write_lines(band_before_dir / pred_path.name, before_lines)
        write_lines(band_after_dir / pred_path.name, after_lines)

        # Visualization: show kept vs removed (band candidates that were dropped).
        vis_img = image.copy()
        draw_boxes(vis_img, kept_candidates + pass_through, color="green", label="kept")
        removed = [p for p in band_candidates if p not in kept_candidates]
        if removed:
            draw_boxes(vis_img, removed, color="red", label="removed")
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_img.save(vis_dir / f"{stem}.jpg")

        total_preds += len(preds)
        total_pass_through += len(pass_through)
        total_candidates += len(band_candidates)
        total_kept += len(kept_candidates)

    print(
        f"Processed {len(pred_files)} files | total boxes: {total_preds} | "
        f"band candidates: {total_candidates} | kept after classifier: {total_kept} | "
        f"pass-through: {total_pass_through}"
    )

    print("Evaluating band-only before/after...")
    metrics_before = evaluate_directories(GT_LABELS_DIR, str(band_before_dir), iou_thresh=IOU_THRESH)
    metrics_after = evaluate_directories(GT_LABELS_DIR, str(band_after_dir), iou_thresh=IOU_THRESH)

    before_summary = metrics_before.get("summary", {})
    after_summary = metrics_after.get("summary", {})
    before_tp = int(before_summary.get("total_tp", 0))
    before_fp = int(before_summary.get("total_fp", 0))
    before_fn = int(before_summary.get("total_fn", 0))
    after_tp = int(after_summary.get("total_tp", 0))
    after_fp = int(after_summary.get("total_fp", 0))
    after_fn = int(after_summary.get("total_fn", 0))

    summary = {
        "config": {
            "pred_dir": PRED_DIR,
            "img_dir": IMG_DIR,
            "gt_labels_dir": GT_LABELS_DIR,
            "classifier_weights": CLASSIFIER_WEIGHTS,
            "img_size": IMG_SIZE,
            "cls_positive_label": CLS_POSITIVE_LABEL,
            "conf_min": CONF_MIN,
            "conf_max": CONF_MAX,
            "iou_thresh": IOU_THRESH,
            "output_root": str(out_root),
            "device": DEVICE,
        },
        "counts": {
            "files": len(pred_files),
            "total_boxes": total_preds,
            "band_candidates": total_candidates,
            "kept_after_classifier": total_kept,
            "pass_through": total_pass_through,
        },
        "band_metrics_before": metrics_before,
        "band_metrics_after": metrics_after,
        "band_analysis": {
            "band_predictions_total": total_candidates,
            "band_predictions_kept": total_kept,
            "kept_fraction_of_band": total_kept / total_candidates if total_candidates else 0.0,
            "tp_before": before_tp,
            "fp_before": before_fp,
            "fn_before": before_fn,
            "tp_after": after_tp,
            "fp_after": after_fp,
            "fn_after": after_fn,
            "tp_retained_fraction": (after_tp / before_tp) if before_tp else 0.0,
            "saved_predictions_true_fraction": (after_tp / total_kept) if total_kept else 0.0,
            "description": "Band-only comparison: kept_fraction_of_band = kept / total band preds; tp_retained_fraction = band TP kept vs band TP before; saved_predictions_true_fraction = TP among kept preds.",
        },
    }

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved refined predictions to {refined_dir}")
    print(f"Saved band-only before/after to {band_before_dir} and {band_after_dir}")
    print(f"Summary written to {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
