import os
import argparse
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from composite_utils import process_image


def normalize_value(val):
    """Convert string 'None' (from YAML) into real Python None."""
    if isinstance(val, str) and val.lower() == "none":
        return None
    return val


def main():
    parser = argparse.ArgumentParser("Composite Driver Script (YAML-based)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Normalize values
    cfg = {k: normalize_value(v) for k, v in raw_cfg.items()}

    # Directories
    if "image_dir" in cfg:
        IMAGE_DIR = cfg["image_dir"]
    else:
        IMAGE_DIR = os.path.join(cfg["gt_dir"], "images/test")

    if "output_dir" in cfg:
        OUTPUT_DIR = cfg["output_dir"]
    else:
        OUTPUT_DIR = os.path.join(cfg["parent_dir"], "composites/run")

    composite_dir = os.path.join(OUTPUT_DIR, "composite_images")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if cfg.get("save_img", False):
        os.makedirs(composite_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO(cfg["model_path"])

    # Gather images
    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    # Process images
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        base, _ = os.path.splitext(img_name)

        out_txt = os.path.join(OUTPUT_DIR, base + ".txt")
        out_composite = os.path.join(composite_dir, base + "_composite.jpg")

        process_image(
            img_path=img_path,
            model=model,
            out_txt=out_txt,
            out_composite=out_composite,
            intermediate_size=cfg.get("intermediate_size", 780),
            conf_thresh=cfg.get("conf_thresh", None),
            size_thresh=cfg.get("size_thresh", None),
            nms_iou_thresh=cfg.get("nms_iou_thresh", None),
            save_img=cfg.get("save_img", False),
            postproc=cfg.get("postproc", "nms"),
        )

    print(f"✅ Finished processing {len(image_files)} images.")
    print(f"Results saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
