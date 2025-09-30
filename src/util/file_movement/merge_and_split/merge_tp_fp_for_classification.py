import os
from glob import glob

# ========== CONFIG ==========
TP_DIR     = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/composites/objects_lt_100/tp"
FP_DIR     = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/baseline/A_phash10_single_objects/composites/objects_lt_100/fp"
OUTPUT_DIR = "/lab/projects/fire_smoke_awr/src/pipeline/2-stage/A_phash10_single_objects_lt_100_merged_labels"
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_labels(input_dir, class_id):
    """Return dict mapping filename -> list of relabeled lines"""
    data = {}
    for lf in glob(os.path.join(input_dir, "*.txt")):
        fname = os.path.basename(lf)
        with open(lf, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = str(class_id)
            new_lines.append(" ".join(parts) + "\n")
        data.setdefault(fname, []).extend(new_lines)
    return data

# collect TP and FP
tp_data = collect_labels(TP_DIR, class_id=1)
fp_data = collect_labels(FP_DIR, class_id=0)

# merge into one dictionary
merged = {}
for fname, lines in tp_data.items():
    merged.setdefault(fname, []).extend(lines)
for fname, lines in fp_data.items():
    merged.setdefault(fname, []).extend(lines)

# save merged files
for fname, lines in merged.items():
    out_path = os.path.join(OUTPUT_DIR, fname)
    with open(out_path, "w") as f:
        f.writelines(lines)

print(f"Merged {len(merged)} label files into {OUTPUT_DIR}")
