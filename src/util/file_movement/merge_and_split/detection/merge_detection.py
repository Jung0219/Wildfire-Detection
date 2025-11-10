import os
import shutil
from tqdm import tqdm

# === Provide list of parent folders here ===
parent_folders = [
    "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/A/deduplicated/phash10",
    "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/B/deduplicated/phash10",
    "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/C/deduplicated/phash10",
    "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/D/deduplicated/phash10",
    "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/datasets/E/deduplicated/phash10",
]
# === Destination merged folder ===
merged_dir = "/lab/biohpc/ComputerVisionAI/fire_smoke_awr/data/detection/processed/ABCDE_all/phash10"

merged_images_dir = os.path.join(merged_dir, "images")
merged_annos_dir = os.path.join(merged_dir, "labels")

# Create merged directories
os.makedirs(merged_images_dir, exist_ok=True)
os.makedirs(merged_annos_dir, exist_ok=True)

total_images = 0

for parent in parent_folders:
    images_dir = os.path.join(parent, "images")
    annos_dir = os.path.join(parent, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(annos_dir):
        raise RuntimeError(f"Missing required folders in '{parent}'.")

    image_files = os.listdir(images_dir)
    anno_files = os.listdir(annos_dir)

    # Copy images with progress bar
    print(f"\n📦 Copying images from {images_dir} ...")
    for fname in tqdm(image_files, desc=f"Images ({parent})"):
        src_img = os.path.join(images_dir, fname)
        dst_img = os.path.join(merged_images_dir, fname)
        if os.path.exists(dst_img):
            print(f"⚠️ Skipping duplicate image: {fname}")
            continue
        shutil.copy2(src_img, dst_img)
        total_images += 1

    # Copy annotations with progress bar
    print(f"📝 Copying annotations from {annos_dir} ...")
    for fname in tqdm(anno_files, desc=f"Annotations ({parent})"):
        src_anno = os.path.join(annos_dir, fname)
        dst_anno = os.path.join(merged_annos_dir, fname)
        if os.path.exists(dst_anno):
            print(f"⚠️ Skipping duplicate annotation: {fname}")
            continue
        shutil.copy2(src_anno, dst_anno)

print("\n✅ Merging complete.")
print("📷 Total images copied:", total_images)
print("📁 Merged images into:", merged_images_dir)
print("📁 Merged annotations into:", merged_annos_dir)
