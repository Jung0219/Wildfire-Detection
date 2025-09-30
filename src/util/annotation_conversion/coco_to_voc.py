import os
import json
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from tqdm import tqdm

# === EDIT HERE ===
coco_json_path = "/lab/projects/fire_smoke_awr/data/datasets/.foggia/annotations.json"
output_dir = "/lab/projects/fire_smoke_awr/data/datasets/.foggia/annotations_voc"

def create_voc_xml(image_info, annotations, categories, output_dir):
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = image_info["file_name"]

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_info["width"])
    ET.SubElement(size, "height").text = str(image_info["height"])
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    for ann in annotations:
        obj = ET.SubElement(annotation, "object")
        cat_name = categories[ann["category_id"]]
        bbox = ann["bbox"]  # [x, y, width, height]
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[0] + bbox[2])
        ymax = int(bbox[1] + bbox[3])

        ET.SubElement(obj, "name").text = cat_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    xml_str = ET.tostring(annotation, encoding="unicode")
    pretty_xml = parseString(xml_str).toprettyxml(indent="  ")
    xml_filename = os.path.splitext(image_info["file_name"])[0] + ".xml"
    with open(os.path.join(output_dir, xml_filename), "w") as f:
        f.write(pretty_xml)

def convert_coco_to_voc(coco_json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path) as f:
        data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    image_id_to_info = {img["id"]: img for img in data["images"]}
    image_id_to_anns = {img_id: [] for img_id in image_id_to_info}
    for ann in data["annotations"]:
        image_id_to_anns[ann["image_id"]].append(ann)

    for image_id in tqdm(image_id_to_info, desc="Converting to Pascal VOC"):
        img_info = image_id_to_info[image_id]
        anns = image_id_to_anns[image_id]
        create_voc_xml(img_info, anns, categories, output_dir)

    print(f"\n✅ Saved Pascal VOC annotations to: {output_dir}")

# === Run ===
convert_coco_to_voc(coco_json_path, output_dir)
