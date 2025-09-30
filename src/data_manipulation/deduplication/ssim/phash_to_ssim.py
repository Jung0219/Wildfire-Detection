import os
import cv2
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def compute_ssim_between_consecutive_images(image_dir, output_csv, resize_shape=(224, 224)):
    image_dir = Path(image_dir)
    image_files = sorted([f for f in image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    ssim_results = []
    for i in tqdm(range(len(image_files) - 1), desc="Computing SSIM"):
        img1_path = image_files[i]
        img2_path = image_files[i + 1]

        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

        if resize_shape:
            img1 = cv2.resize(img1, resize_shape)
            img2 = cv2.resize(img2, resize_shape)

        score = ssim(img1, img2)
        ssim_results.append({
            "image_pair": f"{img1_path.name} vs {img2_path.name}",
            "ssim": score
        })

    # Save to CSV
    df = pd.DataFrame(ssim_results)
    df.to_csv(output_csv, index=False)
    print(f"SSIM results saved to {output_csv}")

if __name__ == "__main__":
    # Set your paths here
    image_dir = "/lab/projects/fire_smoke_awr/data/original/D_fire/deduplicated/phash_hamming_10/images"
    output_csv = "d_fire_hamming_10_ssim.csv"

    compute_ssim_between_consecutive_images(image_dir, output_csv)
