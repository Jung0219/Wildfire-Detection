import os
import json
import xml.etree.ElementTree as ET
import fiftyone as fo
import fiftyone.core.labels as fol
from tqdm import tqdm

# === Set your paths here ===
parent = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/single_objects_lt_130/dedup_phash5"
dataset_name = "ABCDE_single_objects_lt_130_dedup_phash5"


# === Set your paths here ===
images_dir = f"{parent}/images/test"
anno_path = f"{parent}/labels/test"  # COCO file or VOC directory or YOLO directory
#anno_path = "/lab/projects/fire_smoke_awr/data/datasets/.HPWREN_FigLib/jsons/hpwren_target_test.json"
anno_type = "yolo"  # "coco", "voc", or "yolo"
class_list = ["fire", "smoke"]  # Example: ["fire", "smoke"] for YOLO, else keep as None

def load_coco_annotations(images_dir, anno_path):
    with open(anno_path) as f:
        coco = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    id_to_img = {img["id"]: img for img in coco["images"]}

    img_to_anns = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    samples = []
    for img_id, img_info in tqdm(id_to_img.items(), desc="Loading COCO"):
        filename = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]
        image_path = os.path.join(images_dir, filename)

        if not os.path.exists(image_path):
            continue

        detections = []
        for ann in img_to_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            label = cat_id_to_name[ann["category_id"]]
            detections.append(fol.Detection(label=label, bounding_box=[x/width, y/height, w/width, h/height]))

        sample = fo.Sample(filepath=image_path, ground_truth=fol.Detections(detections=detections))
        samples.append(sample)

    return samples

def load_voc_annotations(images_dir, anno_dir):
    samples = []
    for fname in tqdm(os.listdir(images_dir), desc="Loading VOC"):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        xml_file = os.path.join(anno_dir, base + ".xml")
        image_path = os.path.join(images_dir, fname)

        if not os.path.exists(xml_file) or not os.path.exists(image_path):
            continue

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)
        except:
            continue

        detections = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            detections.append(fol.Detection(label=label, bounding_box=[
                xmin/width, ymin/height, (xmax-xmin)/width, (ymax-ymin)/height
            ]))

        sample = fo.Sample(filepath=image_path, ground_truth=fol.Detections(detections=detections))
        samples.append(sample)

    return samples

def load_yolo_predictions(images_dir, preds_dir, class_list=None):
    samples = []
    for fname in tqdm(os.listdir(images_dir), desc="Loading YOLO Predictions"):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        base = os.path.splitext(fname)[0]
        txt_file = os.path.join(preds_dir, base + ".txt")
        image_path = os.path.join(images_dir, fname)

        if not os.path.exists(txt_file) or not os.path.exists(image_path):
            continue

        detections = []
        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, parts[:5])
                confidence = float(parts[5]) if len(parts) == 6 else None
                label = str(int(class_id)) if class_list is None else class_list[int(class_id)]
                bbox = [
                    x_center - width / 2,
                    y_center - height / 2,
                    width,
                    height,
                ]
                det = fol.Detection(label=label, bounding_box=bbox)
                if confidence is not None:
                    det.confidence = confidence
                detections.append(det)

        sample = fo.Sample(filepath=image_path, ground_truth=fol.Detections(detections=detections))
        samples.append(sample)

    return samples

if __name__ == "__main__":
    if anno_type == "coco":
        samples = load_coco_annotations(images_dir, anno_path)
    elif anno_type == "voc":
        samples = load_voc_annotations(images_dir, anno_path)
    elif anno_type == "yolo":
        samples = load_yolo_predictions(images_dir, anno_path, class_list)

    if dataset_name in fo.list_datasets():
        response = input(f"Dataset '{dataset_name}' already exists. Delete it? [y/N]: ").strip().lower()
        if response == "y":
            fo.delete_dataset(dataset_name)
            print("Old dataset deleted.")
        else:
            print("Aborting. Dataset was not deleted.")
            exit()  

    dataset = fo.Dataset(name=dataset_name, persistent=True)
    dataset.add_samples(samples)

    session = fo.launch_app(dataset)
    session.wait()
