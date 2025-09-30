import os
import json

# ==========================
#         CONFIG
# ==========================
COCO_JSON_PATH = "/lab/projects/fire_smoke_awr/data/datasets/.HPWREN_FigLib/jsons/annotations.json" 
YOLO_OUTPUT_DIR = "/lab/projects/fire_smoke_awr/data/datasets/.HPWREN_FigLib/labels"   

# ==========================
#    COCO to YOLO Convert
# ==========================
def convert_coco_to_yolo(input_json, output_dir):
    with open(input_json, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Build image ID to (width, height, file_name) map
    image_map = {
        img['id']: {
            'width': img['width'],
            'height': img['height'],
            'file_name': os.path.splitext(os.path.basename(img['file_name']))[0]
        }
        for img in data['images']
    }

    # Group annotations by image_id
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)

    for img_id, ann_list in annotations.items():
        if img_id not in image_map:
            continue

        img_info = image_map[img_id]
        w, h = img_info['width'], img_info['height']
        file_stem = img_info['file_name']
        label_path = os.path.join(output_dir, f"{file_stem}.txt")

        with open(label_path, 'w') as out_file:
            for ann in ann_list:
                cat_id = ann['category_id']  # Optionally subtract 1 if needed
                x, y, bw, bh = ann['bbox']
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h
                out_file.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

    print(f"YOLO annotations saved to: {output_dir}")

# Run the converter
convert_coco_to_yolo(COCO_JSON_PATH, YOLO_OUTPUT_DIR)
