"""Run SegFormer on images and save sky masks + accumulated density.

Example:
    python src/.pipeline_old/sky_logic/YCbCr_map/segformer_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# ====================== CONFIG ======================
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
IMAGE_PATH = "/lab/projects/fire_smoke_awr/data/samples/early_smoke/images/early_smoke_sample_1.jpg"
IMAGE_DIR = "/lab/projects/fire_smoke_awr/data/detection/processed/AD_phash3_early_smoke/images"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
Y_TARGETS = [64, 128, 160, 192]
RUN_SINGLE_IMAGE = False
RUN_DATASET = True
OUTPUT_DIR = Path(__file__).resolve().parent
MASK_DIR = OUTPUT_DIR / "masks"
DENSITY_PATH = OUTPUT_DIR / "dataset_sky_density.npz"
# =====================================================


def load_model(model_id: str) -> tuple[str, AutoImageProcessor, AutoModelForSemanticSegmentation]:
    """Load SegFormer model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    return device, processor, model


def get_sky_mask(
    img: Image.Image,
    device: str,
    processor: AutoImageProcessor,
    model: AutoModelForSemanticSegmentation,
) -> np.ndarray:
    """Return a binary sky mask (0/1) for the input image."""
    h, w = img.size[1], img.size[0]
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_up = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits_up.argmax(dim=1)[0].cpu().numpy()

    id2label = model.config.id2label
    sky_ids = [k for k, v in id2label.items() if "sky" in v.lower()]
    if len(sky_ids) == 0:
        raise RuntimeError("No 'sky' class found in this model's id2label mapping.")
    sky_id = sky_ids[0]
    return (pred == sky_id).astype(np.uint8)


def list_images(image_dir: Path) -> list[Path]:
    """List image files under a directory."""
    allowed = {ext.lower() for ext in IMAGE_EXTENSIONS}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in allowed])


def save_mask(mask: np.ndarray, mask_path: Path) -> None:
    """Save a binary mask to disk."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, mask)


def accumulate_density_from_mask(
    img: Image.Image,
    mask: np.ndarray,
    densities: dict[int, np.ndarray],
    y_targets: list[int],
) -> None:
    """Accumulate Cb/Cr densities for sky pixels from one image + mask."""
    ycbcr = np.array(img.convert("YCbCr"))
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    sky = mask == 1
    for y_target in y_targets:
        select = sky & (y == y_target)
        if not np.any(select):
            continue
        cb_vals = cb[select].astype(np.int32)
        cr_vals = cr[select].astype(np.int32)
        idx = cr_vals * 256 + cb_vals
        counts = np.bincount(idx, minlength=256 * 256).reshape(256, 256)
        densities[y_target] += counts


def save_density(densities: dict[int, np.ndarray], out_path: Path) -> None:
    """Save accumulated densities to a compressed .npz file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"y{y_target}": density for y_target, density in densities.items()}
    np.savez_compressed(out_path, **payload)


def mask_path_for_image(image_path: Path, mask_dir: Path) -> Path:
    """Return the mask path for an image."""
    return mask_dir / f"{image_path.stem}_sky_mask.npy"


def run_single_image(
    image_path: Path,
    device: str,
    processor: AutoImageProcessor,
    model: AutoModelForSemanticSegmentation,
) -> None:
    """Generate a sky mask for a single image."""
    img = Image.open(image_path).convert("RGB")
    mask = get_sky_mask(img, device, processor, model)
    out_path = mask_path_for_image(image_path, MASK_DIR)
    save_mask(mask, out_path)
    print(f"Saved mask: {out_path}")


def run_dataset(
    image_dir: Path,
    device: str,
    processor: AutoImageProcessor,
    model: AutoModelForSemanticSegmentation,
) -> None:
    """Generate masks for a dataset and save accumulated density."""
    image_paths = list_images(image_dir)
    densities = {y: np.zeros((256, 256), dtype=np.float32) for y in Y_TARGETS}

    for img_path in tqdm(image_paths, desc="Generating masks"):
        img = Image.open(img_path).convert("RGB")
        mask = get_sky_mask(img, device, processor, model)
        out_path = mask_path_for_image(img_path, MASK_DIR)
        save_mask(mask, out_path)
        accumulate_density_from_mask(img, mask, densities, Y_TARGETS)

    save_density(densities, DENSITY_PATH)
    print(f"Saved density: {DENSITY_PATH}")
    print(f"Masks saved under: {MASK_DIR}")


def main() -> None:
    device, processor, model = load_model(MODEL_ID)

    if RUN_SINGLE_IMAGE:
        run_single_image(Path(IMAGE_PATH), device, processor, model)

    if RUN_DATASET:
        run_dataset(Path(IMAGE_DIR), device, processor, model)


if __name__ == "__main__":
    main()
