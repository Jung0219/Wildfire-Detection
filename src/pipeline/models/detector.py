"""Detector loader for the MR pipeline.

Example:
    from src.pipeline.mr_modular.models.detector import load_detector
    model = load_detector("weights/best.pt", device="cuda")
"""

from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


def load_detector(weights: str | Path, device: str | None = None) -> YOLO:
    """Load a YOLO detector model."""

    model = YOLO(str(weights))
    if device:
        model.to(device)
    return model

