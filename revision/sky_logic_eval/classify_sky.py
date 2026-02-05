"""Batch sky presence classification using image segmentation.

Example:
    python /data/lab/projects/fire_smoke_awr/revision/sky_logic_eval/classify_sky.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

# CONFIG (editable)
IMAGE_DIR = "/lab/projects/fire_smoke_awr/data/detection/processed/early_smoke/images"
OUTPUT_PATH = "/lab/projects/fire_smoke_awr/revision/sky_logic_eval/es_all/sky_gt.txt"
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
SKY_RATIO_THRESH = 0.2
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _iter_images(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def _sky_ratio(seg_output: list[dict]) -> float:
    sky_masks = [np.array(x["mask"]) > 0 for x in seg_output if x["label"].lower() == "sky"]
    if not sky_masks:
        return 0.0
    return float(np.logical_or.reduce(sky_masks).mean())


def main() -> int:
    image_dir = Path(IMAGE_DIR)
    if not image_dir.is_dir():
        raise SystemExit(f"ERROR: not a directory: {image_dir}")

    images = _iter_images(image_dir)
    seg = pipeline("image-segmentation", model=MODEL_ID)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        sky_count = 0
        no_sky_count = 0
        for img_path in tqdm(images, desc="Segmenting"):
            img = Image.open(img_path).convert("RGB")
            out = seg(img)
            ratio = _sky_ratio(out)
            has_sky = 1 if ratio >= SKY_RATIO_THRESH else 0
            if has_sky:
                sky_count += 1
            else:
                no_sky_count += 1
            f.write(f"{img_path.name} {has_sky}\n")

    total = len(images)
    sky_pct = (sky_count / total * 100.0) if total else 0.0
    no_sky_pct = (no_sky_count / total * 100.0) if total else 0.0
    print(f"Wrote {total} labels to {output_path}")
    print(f"Sky present: {sky_count} ({sky_pct:.1f}%)")
    print(f"No sky: {no_sky_count} ({no_sky_pct:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
