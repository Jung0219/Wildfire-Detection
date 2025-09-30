"""
- Loads a list of image filenames from a specified column in a CSV file
- Copies matching images from the source directory to a destination directory
- Creates the destination directory if it doesn't exist
- Useful for extracting a subset of images based on filtering or deduplication results
"""

import os
import shutil
import pandas as pd

# Parameters (edit these)
image_dir = "/lab/projects/fire_smoke_awr/data/datasets/.FASDD_CV/original/images"
output_dir = "/lab/projects/fire_smoke_awr/data/datasets/.FASDD_CV/deduplicated/phash10/images"
csv_path = "/lab/projects/fire_smoke_awr/FASDD_CV_phash_10.csv"
image_column_name = "Kept Image Name"  # Specify the column name to use for image filenames

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load list of kept image names from specified column
df = pd.read_csv(csv_path)
if image_column_name not in df.columns:
    raise ValueError(f"Column '{image_column_name}' not found in CSV.")

kept_images = df[image_column_name].tolist()

# Copy each kept image
for img_name in kept_images:
    src_path = os.path.join(image_dir, img_name)
    dst_path = os.path.join(output_dir, img_name)
    shutil.copy2(src_path, dst_path)

print(f"Copied {len(kept_images)} images to {output_dir}")
