import os
import cv2
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def deduplicate_images_by_ssim(image_dir, csv_path, ssim_threshold=0.7, resize_shape=(224, 224)):
    image_dir = Path(image_dir)
    image_files = sorted([f for f in image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    kept_images = []
    kept_images_data = []

    for current_img_path in tqdm(image_files, desc="Deduplicating"):
        current_img = cv2.imread(str(current_img_path), cv2.IMREAD_GRAYSCALE)
        current_img = cv2.resize(current_img, resize_shape)

        is_duplicate = False
        for kept_img_data in kept_images_data:
            score = ssim(kept_img_data, current_img)
            if score >= ssim_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_images.append(current_img_path.name)
            kept_images_data.append(current_img)

    # Save kept image list only (no image copying)
    pd.DataFrame({"Kept Image Name": kept_images}).to_csv(csv_path, index=False)
    print(f"Deduplication complete. {len(kept_images)} images kept.")
    print(f"Kept image list saved to {csv_path}")

if __name__ == "__main__":
    # Set paths
    image_dir = "/lab/projects/fire_smoke_awr/data/original/D_fire/deduplicated/phash_hamming_20/images"
    csv_path = "d_fire_secondary_ssim_0.6.csv"
    ssim_threshold = 0.6  # You can adjust this

    deduplicate_images_by_ssim(image_dir, csv_path, ssim_threshold)
