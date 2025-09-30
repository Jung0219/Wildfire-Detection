import os
import shutil
import random

# ==== EDIT THESE VARIABLES ====
input_dir = '/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/ABCDE_early_fire_removed/dedup_phash10'
output_dir = '/lab/projects/fire_smoke_awr/data/detection/training/ABCDE_noEF'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
seed = 42
# ==============================

def split_dataset():
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    input_images_dir = os.path.join(input_dir, 'images/test')
    input_labels_dir = os.path.join(input_dir, 'labels/test')

    # Collect image files (only those with existing corresponding labels)
    images = [f for f in os.listdir(input_images_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 
              os.path.exists(os.path.join(input_labels_dir, os.path.splitext(f)[0] + '.txt'))]

    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    random.shuffle(images)

    num_images = len(images)
    train_end = int(train_ratio * num_images)
    val_end = train_end + int(val_ratio * num_images)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, files in splits.items():
        img_out_dir = os.path.join(output_dir, 'images', split)
        lbl_out_dir = os.path.join(output_dir, 'labels', split)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        for file in files:
            base = os.path.splitext(file)[0]
            img_src = os.path.join(input_images_dir, file)
            lbl_src = os.path.join(input_labels_dir, base + '.txt')

            shutil.copy2(img_src, os.path.join(img_out_dir, file))
            shutil.copy2(lbl_src, os.path.join(lbl_out_dir, base + '.txt'))

        print(f"{split.capitalize()} set: {len(files)} samples copied to 'images/{split}' and 'labels/{split}'")

if __name__ == "__main__":
    split_dataset()
