import os
from collections import defaultdict

# ==============================
# CONFIGURATION
# ==============================
CONFIG = {
    # Parent directory of the dataset (script will look for labels/test inside this)
    "parent_dir": "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/fp_mining/composites",

    # Relative path from parent_dir to the label files
    "labels_subdir": "",
}
# ==============================


def count_yolo_instances(parent_dir, labels_subdir):
    txt_dir = os.path.join(parent_dir, labels_subdir)
    if not os.path.isdir(txt_dir):
        raise FileNotFoundError(f"Labels directory not found: {txt_dir}")

    class_counts = defaultdict(int)

    for file in os.listdir(txt_dir):
        if file.endswith(".txt"):
            with open(os.path.join(txt_dir, file), "r") as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1

    return class_counts, txt_dir


if __name__ == "__main__":
    counts, used_dir = count_yolo_instances(CONFIG["parent_dir"], CONFIG["labels_subdir"])

    print(f"Counting instances in: {used_dir}")
    print("Class instance counts:")
    for cls_id, count in sorted(counts.items()):
        print(f"Class {cls_id}: {count}")
