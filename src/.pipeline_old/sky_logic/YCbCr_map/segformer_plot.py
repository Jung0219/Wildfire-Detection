"""Plot YCbCr planes from saved sky masks and accumulated density.

Example:
    python src/.pipeline_old/sky_logic/YCbCr_map/segformer_plot.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====================== CONFIG ======================
IMAGE_DIR = "/lab/projects/fire_smoke_awr/data/detection/processed/AD_phash3_early_smoke/images"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
Y_TARGETS = [64, 128, 160, 192]
OUTPUT_DIR = Path(__file__).resolve().parent
MASK_DIR = OUTPUT_DIR / "masks"
DENSITY_PATH = OUTPUT_DIR / "dataset_sky_density.npz"
# =====================================================


def list_images(image_dir: Path) -> list[Path]:
    """List image files under a directory."""
    allowed = {ext.lower() for ext in IMAGE_EXTENSIONS}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in allowed])


def mask_path_for_image(image_path: Path, mask_dir: Path) -> Path:
    """Return the mask path for an image."""
    return mask_dir / f"{image_path.stem}_sky_mask.npy"


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a saved mask from disk."""
    return np.load(mask_path)


def ycrcb_base_plane(y_target: int) -> np.ndarray:
    """Create the base YCrCb plane (converted to RGB) for a fixed Y."""
    cb_grid = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    cr_grid = np.tile(np.arange(256, dtype=np.uint8).reshape(-1, 1), (1, 256))
    y_plane = np.full((256, 256), y_target, dtype=np.uint8)
    ycrcb_plane = np.dstack([y_plane, cr_grid, cb_grid])
    return cv2.cvtColor(ycrcb_plane, cv2.COLOR_YCrCb2RGB)


def add_bounds_labels(ax: plt.Axes, cb_min: int, cb_max: int, cr_min: int, cr_max: int) -> None:
    """Annotate Cb/Cr bounds just outside the axes frame."""
    ax.annotate(
        f"{cb_min}",
        xy=(cb_min, 0),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=16,
        clip_on=False,
    )
    ax.annotate(
        f"{cb_max}",
        xy=(cb_max, 0),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=16,
        clip_on=False,
    )
    ax.annotate(
        f"{cr_min}",
        xy=(0, cr_min),
        xytext=(-12, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=16,
        clip_on=False,
    )
    ax.annotate(
        f"{cr_max}",
        xy=(0, cr_max),
        xytext=(-12, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=16,
        clip_on=False,
    )


def save_ycbcr_distribution(img: Image.Image, mask: np.ndarray, y_target: int, out_path: Path) -> None:
    """Save a YCbCr plane with sky pixels plotted at a specific Y value."""
    ycbcr = np.array(img.convert("YCbCr"))
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]
    select = (mask == 1) & (y == y_target)

    cb_vals = cb[select]
    cr_vals = cr[select]

    base_rgb = ycrcb_base_plane(y_target)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.imshow(base_rgb, origin="lower", extent=[0, 255, 0, 255])
    if cb_vals.size > 0:
        cb_mean = float(cb_vals.mean())
        cb_std = float(cb_vals.std())
        cr_mean = float(cr_vals.mean())
        cr_std = float(cr_vals.std())
        cb_min = int(np.clip(cb_mean - 2 * cb_std, 0, 255))
        cb_max = int(np.clip(cb_mean + 2 * cb_std, 0, 255))
        cr_min = int(np.clip(cr_mean - 2 * cr_std, 0, 255))
        cr_max = int(np.clip(cr_mean + 2 * cr_std, 0, 255))
        ax.scatter(
            cb_vals,
            cr_vals,
            s=4,
            c="black",
            alpha=0.25,
            linewidths=0,
        )
        ax.axvline(cb_min, color="black", linestyle=":", linewidth=2)
        ax.axvline(cb_max, color="black", linestyle=":", linewidth=2)
        ax.axhline(cr_min, color="black", linestyle=":", linewidth=2)
        ax.axhline(cr_max, color="black", linestyle=":", linewidth=2)
        add_bounds_labels(ax, cb_min, cb_max, cr_min, cr_max)
    ax.set_title(f"YCbCr color field at Y={y_target}", fontsize=22)
    ax.set_xlabel("Cb", fontsize=22, labelpad=12)
    ax.set_ylabel("Cr", fontsize=22, labelpad=12)
    ax.set_xticks([0, 255])
    ax.set_yticks([0, 255])
    ax.tick_params(labelsize=16)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    fig.tight_layout()
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.92)
    fig.savefig(out_path)
    plt.close(fig)


def accumulate_density_from_masks(image_paths: list[Path], mask_dir: Path) -> dict[int, np.ndarray]:
    """Accumulate Cb/Cr densities for sky pixels across a dataset."""
    densities = {y: np.zeros((256, 256), dtype=np.float32) for y in Y_TARGETS}
    for img_path in tqdm(image_paths, desc="Accumulating density"):
        mask_path = mask_path_for_image(img_path, mask_dir)
        if not mask_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        mask = load_mask(mask_path)
        ycbcr = np.array(img.convert("YCbCr"))
        y = ycbcr[:, :, 0]
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]
        sky = mask == 1
        for y_target in Y_TARGETS:
            select = sky & (y == y_target)
            if not np.any(select):
                continue
            cb_vals = cb[select].astype(np.int32)
            cr_vals = cr[select].astype(np.int32)
            idx = cr_vals * 256 + cb_vals
            counts = np.bincount(idx, minlength=256 * 256).reshape(256, 256)
            densities[y_target] += counts
    return densities


def load_density(density_path: Path) -> dict[int, np.ndarray]:
    """Load densities from a compressed .npz file."""
    data = np.load(density_path)
    densities: dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("y"):
            y_target = int(key[1:])
            densities[y_target] = data[key]
    return densities


def save_ycbcr_density_map(density: np.ndarray, y_target: int, out_path: Path) -> None:
    """Save a YCbCr plane with a dataset-level density heatmap."""
    base_rgb = ycrcb_base_plane(y_target)

    heat = density.copy()
    if heat.max() > 0:
        heat = heat / heat.max()
    heat = heat ** 0.002

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.imshow(base_rgb, origin="lower", extent=[0, 255, 0, 255])
    if heat.max() > 0:
        heat = np.ma.masked_where(heat <= 0, heat)
        ax.imshow(
            heat,
            origin="lower",
            extent=[0, 255, 0, 255],
            cmap="gray_r",
            alpha=0.6,
        )
    if density.max() > 0:
        rows, cols = np.nonzero(density)
        weights = density[rows, cols]
        total = weights.sum()
        if total > 0:
            cb_mean = float((cols * weights).sum() / total)
            cr_mean = float((rows * weights).sum() / total)
            cb_var = float((weights * (cols - cb_mean) ** 2).sum() / total)
            cr_var = float((weights * (rows - cr_mean) ** 2).sum() / total)
            cb_std = cb_var ** 0.5
            cr_std = cr_var ** 0.5
            cb_min = int(np.clip(cb_mean - 2 * cb_std, 0, 255))
            cb_max = int(np.clip(cb_mean + 2 * cb_std, 0, 255))
            cr_min = int(np.clip(cr_mean - 2 * cr_std, 0, 255))
            cr_max = int(np.clip(cr_mean + 2 * cr_std, 0, 255))
            ax.axvline(cb_min, color="black", linestyle=":", linewidth=2)
            ax.axvline(cb_max, color="black", linestyle=":", linewidth=2)
            ax.axhline(cr_min, color="black", linestyle=":", linewidth=2)
            ax.axhline(cr_max, color="black", linestyle=":", linewidth=2)
            add_bounds_labels(ax, cb_min, cb_max, cr_min, cr_max)
    ax.set_title(f"YCbCr color field at Y={y_target}", fontsize=22)
    ax.set_xlabel("Cb", fontsize=22, labelpad=12)
    ax.set_ylabel("Cr", fontsize=22, labelpad=12)
    ax.set_xticks([0, 255])
    ax.set_yticks([0, 255])
    ax.tick_params(labelsize=16)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    fig.tight_layout()
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.92)
    fig.savefig(out_path)
    plt.close(fig)


def run_dataset(image_dir: Path) -> None:
    """Plot YCbCr density maps for a dataset."""
    if DENSITY_PATH.exists():
        densities = load_density(DENSITY_PATH)
    else:
        image_paths = list_images(image_dir)
        densities = accumulate_density_from_masks(image_paths, MASK_DIR)

    for y_target in Y_TARGETS:
        if y_target not in densities:
            continue
        out_path = OUTPUT_DIR / f"y{y_target}.png"
        save_ycbcr_density_map(densities[y_target], y_target, out_path)
        print(f"Saved dataset YCbCr map: {out_path}")


def main() -> None:
    run_dataset(Path(IMAGE_DIR))


if __name__ == "__main__":
    main()
