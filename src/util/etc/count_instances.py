import os
import sys
from collections import defaultdict

# Set the parent directory here (or pass it as a CLI arg)
PARENT_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire"
LABELS_SUBDIR = "labels/test"


def count_yolo_instances(parent_dir):
    txt_dir = os.path.join(parent_dir, LABELS_SUBDIR)
    if not os.path.isdir(txt_dir):
        raise FileNotFoundError(f"Labels directory not found: {txt_dir}")

    class_counts = defaultdict(int)

    for fname in os.listdir(txt_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(txt_dir, fname)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    class_id = int(parts[0])
                except (ValueError, IndexError):
                    # skip malformed lines
                    continue
                class_counts[class_id] += 1

    return class_counts, txt_dir


if __name__ == "__main__":
    # allow overriding the parent dir via CLI: python count_instances.py /path/to/parent
    parent = sys.argv[1] if len(sys.argv) > 1 else PARENT_DIR

    counts, used_dir = count_yolo_instances(parent)

    print(f"Counting instances in: {used_dir}")
    if not counts:
        print("No instances found.")
        sys.exit(0)

    print("Class instance counts:")
    for cls_id, count in sorted(counts.items()):
        print(f"Class {cls_id}: {count}")
# filepath: /lab/projects/fire_smoke_awr/src/util/etc/count_instances.py
import os
import sys
from collections import defaultdict

# Set the parent directory here (or pass it as a CLI arg)
PARENT_DIR = "/lab/projects/fire_smoke_awr/data/detection/training/early_fire"
LABELS_SUBDIR = "labels/test"


def count_yolo_instances(parent_dir):
    txt_dir = os.path.join(parent_dir, LABELS_SUBDIR)
    if not os.path.isdir(txt_dir):
        raise FileNotFoundError(f"Labels directory not found: {txt_dir}")

    class_counts = defaultdict(int)

    for fname in os.listdir(txt_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(txt_dir, fname)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    class_id = int(parts[0])
                except (ValueError, IndexError):
                    # skip malformed lines
                    continue
                class_counts[class_id] += 1

    return class_counts, txt_dir


if __name__ == "__main__":
    # allow overriding the parent dir via CLI: python count_instances.py /path/to/parent
    parent = sys.argv[1] if len(sys.argv) > 1 else PARENT_DIR

    counts, used_dir = count_yolo_instances(parent)

    print(f"Counting instances in: {used_dir}")
    if not counts:
        print("No instances found.")
        sys.exit(0)

    print("Class instance counts:")
    for cls_id, count in sorted(counts.items()):
        print(f"Class {cls_id}: {count}")