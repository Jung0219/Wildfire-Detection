import os
from PIL import Image
import imagehash
import pandas as pd
from tqdm import tqdm

# Parameters
image_dir = "/lab/projects/fire_smoke_awr/data/datasets/A/original/images"
csv_name = "A_phash_14.csv"
threshold = 14  # Hamming distance threshold

image_list = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]
# Initialize
kept = []
kept_hashes = []

for img_path in tqdm(image_list):
    img = Image.open(img_path).convert("L")
    img_hash = imagehash.phash(img)

    # Compare with all previously kept hashes
    is_duplicate = False
    for h in kept_hashes:
        if img_hash - h <= threshold:
            is_duplicate = True
            break

    if not is_duplicate:
        kept.append(os.path.basename(img_path))
        kept_hashes.append(img_hash)

# Save result
df_kept = pd.DataFrame({"Kept Image Name": kept})
df_kept.to_csv(csv_name, index=False)
print(f"Deduplicated: kept {len(kept)} out of {len(image_list)} images.")
