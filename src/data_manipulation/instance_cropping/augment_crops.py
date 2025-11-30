#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Augmentation Script
- Takes an input folder of images
- Saves augmented images into an output folder
- Keeps the original resolution (no resizing)
- Number of augmentations per image is configurable
"""

import os
import cv2
from tqdm import tqdm
import albumentations as A

# ========= CONFIGURATION =========
CONFIG = {
    # Paths
    "INPUT_DIR": "/lab/projects/fire_smoke_awr/data/classification/datasets/AD_phash3_early_smoke/foreground",   # change this
    "OUTPUT_DIR": "/lab/projects/fire_smoke_awr/data/classification/datasets/AD_phash3_early_smoke/foreground", # change this

    # Augmentation settings
    "NUM_AUGMENTS": 4,   # number of augmented images per original
    "INPUT_EXTS": (".jpg", ".jpeg", ".png"),  # allowed image extensions

    # Augmentation pipeline
    "AUGMENT_PIPELINE": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
    ])
}   
# =================================

def augment_images():
    input_dir = CONFIG["INPUT_DIR"]
    output_dir = CONFIG["OUTPUT_DIR"]
    num_augments = CONFIG["NUM_AUGMENTS"]

    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(CONFIG["INPUT_EXTS"])]

    for img_name in tqdm(image_files, desc="Augmenting"):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        for i in range(num_augments):
            augmented = CONFIG["AUGMENT_PIPELINE"](image=image)
            aug_img = augmented["image"]

            # Save with size info in filename
            base, ext = os.path.splitext(img_name)
            aug_name = f"{base}_{h}x{w}_aug{i}{ext}"
            save_path = os.path.join(output_dir, aug_name)
            cv2.imwrite(save_path, aug_img)

if __name__ == "__main__":
    augment_images()
