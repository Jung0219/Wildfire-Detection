"""IO helpers for saving labels and debug artifacts."""

from __future__ import annotations

from pathlib import Path
import cv2


def write_labels(path: Path, detections) -> None:
    """Write YOLO-formatted labels to a txt file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for cls_id, xc, yc, w, h, conf in detections:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")


def save_image(path: Path, image) -> None:
    """Save an image (BGR) to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

