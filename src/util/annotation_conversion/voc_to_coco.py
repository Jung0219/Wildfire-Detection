import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    annotations = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        annotations.append({
            "label": label,
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin]
        })

    return filename, width, height, annotations

def convert_to_coco(xml_dir, output_path):
    category_set = {}
    images = []
    annotations = []
    ann_id = 1
    image_id = 1

    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith(".xml")])

    for xml_file in tqdm(xml_files, desc="Converting VOC to COCO"):
        xml_path = os.path.join(xml_dir, xml_file)
        _, width, height, objects = parse_voc_xml(xml_path)
        filename = os.path.splitext(xml_file)[0] + ".jpg"

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for obj in objects:
            label = obj["label"]
            if label not in category_set:
                category_set[label] = len(category_set) + 1

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_set[label],
                "bbox": obj["bbox"],
                "area": obj["bbox"][2] * obj["bbox"][3],
                "iscrowd": 0,
            })
            ann_id += 1

        image_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name} for name, cid in category_set.items()
        ]
    }

    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=2)
    print(f"\n✅ COCO annotations saved to: {output_path}")

# === Example use ===
xml_dir = "/lab/projects/fire_smoke_awr/data/datasets/.foggia/annotations_voc"
output_json = "/lab/projects/fire_smoke_awr/data/datasets/.foggia/annotations.json"

convert_to_coco(xml_dir, output_json)
