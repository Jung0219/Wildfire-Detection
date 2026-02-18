"""Single-image API for the MR + classifier pipeline.

Example:
    from pathlib import Path
    import cv2
    from src.pipeline.mr_modular.api import run_mr_classifier_on_image
    from src.pipeline.mr_modular.config.config import MRClassifierConfig
    from src.pipeline.mr_modular.models.detector import load_detector
    from src.pipeline.mr_modular.models.classifier import load_classifier

    cfg = MRClassifierConfig()
    det = load_detector(cfg.detector_weights, device=cfg.device)
    cls = load_classifier(cfg.classifier_weights, device=cfg.device)
    image = cv2.imread("/path/to/image.jpg")
    result = run_mr_classifier_on_image(image, cfg, det, cls, base_name="image")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2

from src.full_pipeline.classifier_stage.gating import verify_with_classifier
from src.full_pipeline.config.config import MRClassifierConfig, ensure_dir
from src.full_pipeline.data.loader import base_name as basename_from_path
from src.full_pipeline.data.loader import load_image
from src.full_pipeline.io.save import save_image, write_labels
from src.full_pipeline.postprocess.mapping import map_detections
from src.full_pipeline.postprocess.nms import apply_nms
from src.full_pipeline.preprocess.composite import prepare_image_for_detection


@dataclass
class PipelineResult:
    """Structured output from a single run."""

    base_name: str
    detections: List[List[float]]
    counts: Dict[str, int]
    meta: Dict[str, Any]
    composite_path: Optional[Path] = None
    labels_path: Optional[Path] = None


def run_mr_classifier_on_image(
    image_or_path,
    config: MRClassifierConfig,
    detector,
    classifier,
    base_name: str | None = None,
    save_labels: bool = True,
) -> PipelineResult:
    """Run the MR + classifier pipeline on a single image."""

    if isinstance(image_or_path, (str, Path)):
        image = load_image(image_or_path)
        base = base_name or basename_from_path(image_or_path)
    else:
        image = image_or_path
        base = base_name or "image"

    composite, meta = prepare_image_for_detection(
        image, config.intermediate_size, config.anchor_y_frac
    )

    yolo_res = detector.predict(composite, imgsz=640, conf=0.001, verbose=False)[0]
    mapped = map_detections(yolo_res, meta, image.shape)
    if config.nms_iou_thresh and config.nms_iou_thresh > 0:
        final_dets = apply_nms(mapped, config.nms_iou_thresh)
    else:
        final_dets = [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in mapped]

    verified, counts = verify_with_classifier(
        final_dets,
        image,
        classifier,
        config.conf_low,
        config.conf_high,
        config.classifier_crop_size,
    )

    labels_path = None
    if save_labels:
        labels_path = config.output_dir / f"{base}.txt"
        ensure_dir(labels_path.parent)
        write_labels(labels_path, verified)

    composite_path = None
    if config.save_composites or config.save_debug:
        composite_path = (config.debug_dir if config.save_debug else config.output_dir / "composites") / (
            f"{base}_composite.jpg"
        )
        save_image(composite_path, composite)

    return PipelineResult(
        base_name=base,
        detections=verified,
        counts=counts,
        meta=meta,
        composite_path=composite_path,
        labels_path=labels_path,
    )

