import os
import shutil

"""
- Extracting annotations for a subset of images
- Takes a folder of image files (subset_img_dir)
- Finds matching annotation files (by filename + extension) in full_anno_dir
- Copies the matching annotation files to dest_anno_dir
"""

def copy_matching_annotations(subset_img_dir, full_anno_dir, dest_anno_dir, annotation_ext="txt"):
    os.makedirs(dest_anno_dir, exist_ok=True)

    image_files = [f for f in os.listdir(subset_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    copied = 0
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        anno_filename = base + f".{annotation_ext}"
        anno_path = os.path.join(full_anno_dir, anno_filename)
        if os.path.exists(anno_path):
            shutil.copy(anno_path, os.path.join(dest_anno_dir, anno_filename))
            copied += 1
        else:
            print(f"Warning: Annotation not found for {img_file}")

    print(f"Copied {copied} annotation files to: {dest_anno_dir}")

if __name__ == "__main__":
    # Set your paths directly here:
    full_anno_dir = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130/labels/test"
    subset_img_dir = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130/dedup_phash5/early_fire_hand_selected/images/test"
    dest_anno_dir = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130/dedup_phash5/early_fire_hand_selected/labels/test"
    annotation_ext = "txt"  # txt for yolo, xml for voc

    copy_matching_annotations(subset_img_dir, full_anno_dir, dest_anno_dir, annotation_ext)
