#!/bin/bash

# ================= CONFIG =================
SCRIPT="/lab/projects/fire_smoke_awr/src/data_manipulation/cropping/target_crop_dynamic_orig_scale.py"     # your Python script filename
ENV_PATH="/home/finn/.conda/envs/yolo/bin/python"  # adjust if needed
INTERMEDIATE_SIZES=(640 720 780 900 1024)  # list of sizes to test
# ==========================================

for SIZE in "${INTERMEDIATE_SIZES[@]}"; do
    echo "=============================="
    echo "Running with INTERMEDIATE_SIZE = $SIZE"
    echo "=============================="
    
    $ENV_PATH $SCRIPT --intermediate_size $SIZE
    
    echo "✅ Completed run with size $SIZE"
    echo
done

echo "🎯 All runs completed."
