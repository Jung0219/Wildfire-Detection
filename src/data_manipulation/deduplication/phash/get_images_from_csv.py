import os
import shutil
import pandas as pd

# Parameters (edit these)
parent_dir = "/lab/projects/fire_smoke_awr/data/datasets/B/original"
output_dir = "/lab/projects/fire_smoke_awr/data/datasets/B/deduplicated/phash14"
csv_path = "/lab/projects/fire_smoke_awr/B_phash_14.csv"
image_column_name = "Kept Image Name"  # column in CSV with image filenames

# Define source folders
image_dir = os.path.join(parent_dir, "images")
anno_dir = os.path.join(parent_dir, "annotations_yolo")

# Define destination folders
out_image_dir = os.path.join(output_dir, "images")
out_anno_dir = os.path.join(output_dir, "annotations_yolo")

# Create output directories if they don't exist
os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_anno_dir, exist_ok=True)

# Load list of kept image names from specified column
df = pd.read_csv(csv_path)
if image_column_name not in df.columns:
    raise ValueError(f"Column '{image_column_name}' not found in CSV.")

kept_images = df[image_column_name].tolist()

# Copy images and corresponding annotation files
copied_count = 0
for img_name in kept_images:
    # Image source and destination
    src_img = os.path.join(image_dir, img_name)
    dst_img = os.path.join(out_image_dir, img_name)

    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
        copied_count += 1
    else:
        print(f"Warning: Image not found: {src_img}")
        continue

    # Annotation source and destination (same basename, .txt extension)
    base_name, _ = os.path.splitext(img_name)
    anno_name = base_name + ".txt"
    src_anno = os.path.join(anno_dir, anno_name)
    dst_anno = os.path.join(out_anno_dir, anno_name)

    if os.path.exists(src_anno):
        shutil.copy2(src_anno, dst_anno)
    else:
        print(f"Warning: Annotation not found for {img_name}")

print(f"Copied {copied_count} images (with annotations where available) to {output_dir}")
