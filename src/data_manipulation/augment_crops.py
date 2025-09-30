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
    "INPUT_DIR": "/lab/projects/fire_smoke_awr/data/classification/training/train_gt+fp/augmented/smoke",   # change this
    "OUTPUT_DIR": "/lab/projects/fire_smoke_awr/data/classification/training/train_gt+fp/augmented/smoke", # change this

    # Augmentation settings
    "NUM_AUGMENTS": 1,   # number of augmented images per original
    "INPUT_EXTS": (".jpg", ".jpeg", ".png"),  # allowed image extensions

    # Augmentation pipeline
    "AUGMENT_PIPELINE": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.6),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, alpha_coef=0.08, p=0.4),
        A.GaussNoise(p=0.4),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.CoarseDropout(max_holes=3, max_height=32, max_width=32, p=0.3)
        # No resize → keeps original resolution
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
