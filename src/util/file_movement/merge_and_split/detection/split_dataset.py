import os
import shutil
import random

# ==============================
# CONFIGURATION
# ==============================
CONFIG = {
    # Input parent dir containing images/test and labels/test
    "input_parent": "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130/early_fire",

    # Output parent dir where "dev" and "test" will be created
    "output_parent": "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire",

    # Ratio of dev set (e.g., 0.3 means 30% dev, 70% test)
    "dev_ratio": 0.5,

    # Random seed for reproducibility
    "seed": 42,
}
# ==============================


def make_dirs(base, split):
    """Create image and label subdirs for a split."""
    dirs = [
        os.path.join(base, split, "images", "test"),
        os.path.join(base, split, "labels", "test"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def split_dataset(cfg):
    img_dir = os.path.join(cfg["input_parent"], "images", "test")
    lbl_dir = os.path.join(cfg["input_parent"], "labels", "test")

    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    images.sort()

    random.seed(cfg["seed"])
    random.shuffle(images)

    n_dev = int(len(images) * cfg["dev_ratio"])
    dev_imgs = images[:n_dev]
    test_imgs = images[n_dev:]

    for split, split_imgs in [("dev", dev_imgs), ("test", test_imgs)]:
        make_dirs(cfg["output_parent"], split)
        for img in split_imgs:
            # Copy image
            src_img = os.path.join(img_dir, img)
            dst_img = os.path.join(cfg["output_parent"], split, "images", "test", img)
            shutil.copy2(src_img, dst_img)

            # Copy corresponding label
            lbl_file = os.path.splitext(img)[0] + ".txt"
            src_lbl = os.path.join(lbl_dir, lbl_file)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(cfg["output_parent"], split, "labels", "test", lbl_file)
                shutil.copy2(src_lbl, dst_lbl)

    print(f"Split complete: {len(dev_imgs)} in dev, {len(test_imgs)} in test.")


if __name__ == "__main__":
    split_dataset(CONFIG)
