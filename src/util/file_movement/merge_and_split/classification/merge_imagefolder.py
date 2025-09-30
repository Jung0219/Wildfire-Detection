import shutil
import random
from pathlib import Path

# ==== CONFIGURATION ====
DATASETS = {
    "/lab/projects/fire_smoke_awr/data/classification/training/fp_mined/background": 1.0,
    "/lab/projects/fire_smoke_awr/data/classification/training/fp_mined/smoke": 1.0,
    "/lab/projects/fire_smoke_awr/data/classification/training/fp_mined/fire": 1.0
}
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/data/classification/training/fp_mined_split"
SEED = 42
DO_SPLIT = True    # toggle on/off split

# If splitting, define ratios. Example: 7:2:1
SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.2,
    "test": 0.0,
}
# =======================

def merge_datasets_imagefolder(datasets, output_dir, seed=42, split_ratios=None, do_split=True):
    random.seed(seed)
    output_dir = Path(output_dir)

    # Normalize split ratios
    if do_split and split_ratios:
        total = sum(split_ratios.values())
        split_ratios = {k: v/total for k, v in split_ratios.items()}
        for split in split_ratios:
            (output_dir / split).mkdir(parents=True, exist_ok=True)

    elif do_split:  # fallback if ratios not provided
        split_ratios = {"train": 0.8, "val": 0.2}
        for split in split_ratios:
            (output_dir / split).mkdir(parents=True, exist_ok=True)

    for dataset_path, fraction in datasets.items():
        dataset_path = Path(dataset_path)
        class_name = dataset_path.stem

        # collect images
        images = [p for p in dataset_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if not images:
            print(f"[WARN] No images in {dataset_path}")
            continue

        # fraction sampling
        if fraction >= 1.0:
            chosen = images
        else:
            n_samples = max(1, int(len(images) * fraction))
            chosen = random.sample(images, n_samples)

        random.shuffle(chosen)

        if do_split:
            n_total = len(chosen)
            split_points, prev = {}, 0
            for split, ratio in split_ratios.items():
                n_split = int(n_total * ratio)
                split_points[split] = chosen[prev:prev + n_split]
                prev += n_split
            # allocate any leftovers to train
            split_points["train"].extend(chosen[prev:])

            # create class subfolders and copy
            for split, imgs in split_points.items():
                outdir = output_dir / split / class_name
                outdir.mkdir(parents=True, exist_ok=True)
                prefix = f"{dataset_path.parent.stem}_{dataset_path.stem}"
                for img_path in imgs:
                    dst = outdir / f"{prefix}_{img_path.name}"
                    shutil.copy(img_path, dst)
            print(f"[INFO] {class_name}: " +
                  ", ".join([f"{split}={len(imgs)}" for split, imgs in split_points.items()]) +
                  f" out of {len(images)} ({fraction*100:.1f}% sampled)")

        else:
            outdir = output_dir / class_name
            outdir.mkdir(parents=True, exist_ok=True)
            prefix = f"{dataset_path.parent.stem}_{dataset_path.stem}"
            for img_path in chosen:
                dst = outdir / f"{prefix}_{img_path.name}"
                shutil.copy(img_path, dst)
            print(f"[INFO] {class_name}: {len(chosen)} images (no split) out of {len(images)} ({fraction*100:.1f}% sampled)")

    print(f"[INFO] Merge complete. Final dataset saved to {output_dir}")

if __name__ == "__main__":
    merge_datasets_imagefolder(DATASETS, OUTPUT_DIR, SEED, SPLIT_RATIOS, DO_SPLIT)
