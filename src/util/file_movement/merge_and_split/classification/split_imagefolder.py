import shutil
import random
from pathlib import Path

# ==== CONFIGURATION ====
INPUT_DIR = "/lab/projects/fire_smoke_awr/data/classification/datasets/train_gt+fp/augmented"         # dataset root with class subfolders
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/data/classification/training/train_gt+fp"   # where train/ val/ test/ will be created
SPLIT_RATIOS = {                       # specify your ratios here
    "train": 0.8,
    "val": 0.2,
    "test": 0.0
}
SEED = 42                              # reproducibility
# ========================

def split_dataset(input_dir, output_dir, split_ratios, seed=42):
    random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Check sum of ratios
    total = sum(split_ratios.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total})")

    # Create output dirs
    for split in split_ratios.keys():
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Process each class folder
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name

        # Create class subfolders in output
        for split in split_ratios.keys():
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        # Gather and shuffle images
        images = list(class_dir.glob("*.*"))
        random.shuffle(images)

        n_total = len(images)
        start_idx = 0
        splits = {}

        # Calculate indices for each split
        for split, ratio in split_ratios.items():
            n_split = int(ratio * n_total)
            splits[split] = images[start_idx:start_idx + n_split]
            start_idx += n_split

        # Handle remainder (assign leftover to last split)
        leftover = images[start_idx:]
        if leftover:
            last_split = list(split_ratios.keys())[-1]
            splits[last_split].extend(leftover)

        # Copy files
        for split, split_imgs in splits.items():
            for img_path in split_imgs:
                dst = output_dir / split / class_name / img_path.name
                shutil.copy(img_path, dst)

        counts = {k: len(v) for k, v in splits.items()}
        print(f"[INFO] {class_name}: {counts}")

    print(f"[INFO] Split complete. Saved to {output_dir}")

if __name__ == "__main__":
    split_dataset(INPUT_DIR, OUTPUT_DIR, SPLIT_RATIOS, SEED)
