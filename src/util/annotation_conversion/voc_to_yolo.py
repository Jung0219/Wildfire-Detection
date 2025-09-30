import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Fixed class mapping: fire → 0, smoke → 1
class_mapping = {
    "fire": 0,
    "smoke": 1
}

def convert_voc_to_yolo(voc_dir, output_dir):
    voc_dir = Path(voc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for xml_file in voc_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                continue  # Skip unknown classes

            class_id = class_mapping[class_name]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Convert to YOLO normalized format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)

        output_txt = output_dir / f"{xml_file.stem}.txt"
        with open(output_txt, "w") as f:
            f.write("\n".join(yolo_lines))

        print(f"Converted {xml_file.name} → {output_txt.name}")

if __name__ == "__main__":
    # Edit your paths here:
    voc_dir = "/lab/projects/fire_smoke_awr/data/datasets/.fire_and_smoke_dataset/original/annotations_voc"
    output_dir = "/lab/projects/fire_smoke_awr/data/datasets/.fire_and_smoke_dataset/original/annotations_yolo"

    convert_voc_to_yolo(voc_dir, output_dir)
