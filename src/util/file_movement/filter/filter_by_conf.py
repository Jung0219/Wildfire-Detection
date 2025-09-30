import os

# =========================
# CONFIGURATION
# =========================
LABEL_DIR = "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/composites"
OUTPUT_DIR = LABEL_DIR + "/conf_lt_0.3"
CONF_THRESHOLD = 0.3       # confidence cutoff
FILTER_MODE = "less"    # "greater" → keep conf >= threshold, "less" → keep conf <= threshold
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file_name in os.listdir(LABEL_DIR):
    if not file_name.endswith(".txt"):
        continue

    input_path = os.path.join(LABEL_DIR, file_name)
    output_path = os.path.join(OUTPUT_DIR, file_name)

    filtered_lines = []
    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                # skip malformed lines
                continue
            conf = float(parts[5])  # YOLO format: class x y w h conf

            if FILTER_MODE == "greater" and conf >= CONF_THRESHOLD:
                filtered_lines.append(line)
            elif FILTER_MODE == "less" and conf <= CONF_THRESHOLD:
                filtered_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(filtered_lines)

print(f"Filtering complete (mode={FILTER_MODE}). Results saved to {OUTPUT_DIR}")
