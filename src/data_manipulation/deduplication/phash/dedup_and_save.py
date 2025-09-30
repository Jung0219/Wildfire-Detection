import os
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm

# ==== CONFIGURATION ====
PARENT_DIR = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/ABCDE_early_fire_removed"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/ABCDE_early_fire_removed/dedup_phash10"
THRESHOLD  = 10  # Hamming distance threshold
# ========================

def deduplicate_with_labels(parent_dir, output_dir, threshold=10):
    input_img_dir = os.path.join(parent_dir, "images/test")
    input_lbl_dir = os.path.join(parent_dir, "labels/test")

    output_img_dir = os.path.join(output_dir, "images/test")
    output_lbl_dir = os.path.join(output_dir, "labels/test")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    image_list = [
        f for f in os.listdir(input_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    kept = []
    kept_hashes = []

    for fname in tqdm(image_list, desc="Deduplicating"):
        img_path = os.path.join(input_img_dir, fname)
        img = Image.open(img_path).convert("L")
        img_hash = imagehash.phash(img)

        # Compare with previously kept hashes
        is_duplicate = any((img_hash - h) <= threshold for h in kept_hashes)

        if not is_duplicate:
            kept.append(fname)
            kept_hashes.append(img_hash)

            # Copy image
            shutil.copy2(img_path, os.path.join(output_img_dir, fname))

            # Copy label if exists
            base_name, _ = os.path.splitext(fname)
            lbl_path = os.path.join(input_lbl_dir, base_name + ".txt")
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, os.path.join(output_lbl_dir, base_name + ".txt"))

    print(f"\n✅ Deduplicated: kept {len(kept)} out of {len(image_list)} images.")

# ==== RUN ====
deduplicate_with_labels(PARENT_DIR, OUTPUT_DIR, THRESHOLD)
