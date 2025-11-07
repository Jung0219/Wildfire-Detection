#!/usr/bin/env python3
"""Sample images that contain very small bounding boxes."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Pillow is required to draw bounding boxes. Install it via `pip install pillow`."
    ) from exc

DEFAULT_IMAGES_DIR = Path(
    "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/A/original/images"
)
DEFAULT_LABELS_DIR = Path(
    "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/A/original/labels"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "small_box_samples"
DEFAULT_BINS: Sequence[tuple[float, float]] = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass
class SampleCandidate:
    bin_index: int
    image_path: Path
    label_path: Path
    area_percent: float
    class_id: int
    box_index: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect example images that contain bounding boxes below 2.5% "
            "of the image area."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing the source images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help="Directory containing YOLO txt label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where sampled images and metadata will be written.",
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=5,
        help="Number of images to gather for each percentage bin.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing output directory before writing new samples.",
    )
    return parser.parse_args()


def assign_bin(area_percent: float, bins: Sequence[tuple[float, float]]) -> int | None:
    for idx, (low, high) in enumerate(bins):
        upper_inclusive = idx == len(bins) - 1
        in_range = low <= area_percent <= high if upper_inclusive else low <= area_percent < high
        if in_range:
            return idx
    return None


def resolve_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def yield_label_files(labels_dir: Path) -> Iterable[Path]:
    yield from sorted(labels_dir.glob("*.txt"))


def all_bins_filled(samples_by_bin: dict[int, list[SampleCandidate]], quota: int) -> bool:
    return all(len(bin_list) >= quota for bin_list in samples_by_bin.values())


def collect_samples(
    images_dir: Path,
    labels_dir: Path,
    bins: Sequence[tuple[float, float]],
    samples_per_bin: int,
    rng: random.Random,
) -> dict[int, list[SampleCandidate]]:
    samples_by_bin: dict[int, list[SampleCandidate]] = {i: [] for i in range(len(bins))}
    label_files = list(yield_label_files(labels_dir))
    rng.shuffle(label_files)

    for label_file in label_files:
        if all_bins_filled(samples_by_bin, samples_per_bin):
            break

        image_path = resolve_image(images_dir, label_file.stem)
        if image_path is None:
            continue

        raw_lines = [line for line in label_file.read_text().splitlines() if line.strip()]
        if len(raw_lines) != 1:
            continue

        pieces = raw_lines[0].split()
        if len(pieces) < 5:
            continue
        try:
            class_id = int(float(pieces[0]))
            x_center = float(pieces[1])
            y_center = float(pieces[2])
            width = float(pieces[3])
            height = float(pieces[4])
        except ValueError:
            continue

        area_percent = width * height * 100
        bin_index = assign_bin(area_percent, bins)
        if bin_index is None or len(samples_by_bin[bin_index]) >= samples_per_bin:
            continue

        samples_by_bin[bin_index].append(
            SampleCandidate(
                bin_index=bin_index,
                image_path=image_path,
                label_path=label_file,
                area_percent=area_percent,
                class_id=class_id,
                box_index=0,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return samples_by_bin


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_samples(
    samples_by_bin: dict[int, list[SampleCandidate]],
    bins: Sequence[tuple[float, float]],
    output_dir: Path,
) -> list[SampleCandidate]:
    chosen_samples: list[SampleCandidate] = []
    for idx, bin_candidates in samples_by_bin.items():
        bin_dir = output_dir / f"bin_{bins[idx][0]:.2f}_{bins[idx][1]:.2f}"
        bin_dir.mkdir(parents=True, exist_ok=True)
        for sample in bin_candidates:
            destination = bin_dir / sample.image_path.name
            draw_box_on_image(sample, destination)
            chosen_samples.append(sample)
    return chosen_samples


def draw_box_on_image(sample: SampleCandidate, destination: Path) -> None:
    with Image.open(sample.image_path) as img:
        image = img.convert("RGB")
    width, height = image.size
    box_width_px = sample.width * width
    box_height_px = sample.height * height
    x_center_px = sample.x_center * width
    y_center_px = sample.y_center * height

    half_w = box_width_px / 2
    half_h = box_height_px / 2
    x0 = max(0, x_center_px - half_w)
    y0 = max(0, y_center_px - half_h)
    x1 = min(width, x_center_px + half_w)
    y1 = min(height, y_center_px + half_h)

    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    image.save(destination)


def write_manifest(
    manifest_path: Path,
    samples: Sequence[SampleCandidate],
    bins: Sequence[tuple[float, float]],
) -> None:
    with manifest_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "bin_index",
                "bin_range_percent",
                "image_path",
                "label_path",
                "area_percent",
                "class_id",
                "box_index",
                "x_center_norm",
                "y_center_norm",
                "width_norm",
                "height_norm",
            ]
        )
        for sample in samples:
            bin_range = f"{bins[sample.bin_index][0]:.2f}-{bins[sample.bin_index][1]:.2f}"
            writer.writerow(
                [
                    sample.bin_index,
                    bin_range,
                    str(sample.image_path),
                    str(sample.label_path),
                    f"{sample.area_percent:.4f}",
                    sample.class_id,
                    sample.box_index,
                    f"{sample.x_center:.6f}",
                    f"{sample.y_center:.6f}",
                    f"{sample.width:.6f}",
                    f"{sample.height:.6f}",
                ]
            )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {args.images_dir}")
    if not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {args.labels_dir}")

    bins = DEFAULT_BINS
    samples_by_bin = collect_samples(
        args.images_dir,
        args.labels_dir,
        bins,
        args.samples_per_bin,
        rng,
    )
    total_samples = sum(len(bin_list) for bin_list in samples_by_bin.values())
    if total_samples == 0:
        raise RuntimeError("No candidates found in the specified bin ranges.")

    prepare_output_dir(args.output_dir, args.overwrite)
    chosen_samples = copy_samples(samples_by_bin, bins, args.output_dir)
    write_manifest(args.output_dir / "manifest.csv", chosen_samples, bins)

    for idx, bin_candidates in samples_by_bin.items():
        low, high = bins[idx]
        print(
            f"Bin {idx} ({low:.2f}% - {high:.2f}%): "
            f"{len(bin_candidates)}/{args.samples_per_bin} samples"
        )


if __name__ == "__main__":
    main()
