import os
import json
from PIL import Image
from tqdm import tqdm

# === Paths ===
image_dir = "/mnt/d/DB/Organized/D_fire/original/images"
label_dir = "/mnt/d/DB/Organized/D_fire/original/annotations_yolo"
output_json = "/mnt/d/DB/Organized/D_fire/original/annotations.json"

def convert_yolo_to_coco(image_dir, label_dir):
    images = []
    annotations = []
    category_mapping = {}  # YOLO class ID → COCO category ID
    next_category_id = 1
    image_id = 1
    annotation_id = 1

    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

    for label_file in tqdm(label_files, desc="Converting YOLO to COCO"):
        image_name = label_file.replace(".txt", ".jpg")  # adjust if using .png
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(image_path):
            continue  # skip if image doesn't exist

        with Image.open(image_path) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, w, h = map(float, parts)
            class_id = int(class_id)

            if class_id not in category_mapping:
                category_mapping[class_id] = next_category_id
                next_category_id += 1

            bbox_w = w * width
            bbox_h = h * height
            bbox_x = (x_center * width) - (bbox_w / 2)
            bbox_y = (y_center * height) - (bbox_h / 2)

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_mapping[class_id],
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    categories = [{"id": v, "name": str(k)} for k, v in category_mapping.items()]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

# === Run and save ===
coco_output = convert_yolo_to_coco(image_dir, label_dir)

with open(output_json, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"\n✅ COCO annotations saved to: {output_json}")
