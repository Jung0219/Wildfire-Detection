"""Configuration helpers for the modular MR + classifier pipeline.

Example:
    python -m src.pipeline.mr_modular.cli.main \
        --image-dir /path/to/images \
        --output-dir /tmp/labels \
        --detector-weights weights/best.pt \
        --classifier-weights weights/cls.pt
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(os.getenv("FS_ROOT_DIR", REPO_ROOT))


@dataclass
class MRClassifierConfig:
    """Settings for the MR + classifier pipeline.

    Provide these values explicitly (via constructor or YAML/CLI). No hard-coded
    defaults are set to avoid leaking experiment-specific paths.
    """

    image_dir: Path | None = None
    output_dir: Path | None = None
    detector_weights: Path | None = None
    classifier_weights: Path | None = None
    intermediate_size: int | None = None
    nms_iou_thresh: float | None = None
    conf_low: float | None = None
    conf_high: float | None = None
    classifier_crop_size: int | None = None
    anchor_y_frac: float | None = None
    save_debug: bool | None = None
    debug_dir: Path | None = None
    save_composites: bool | None = None
    device: str | None = None


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path
