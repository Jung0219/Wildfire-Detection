#!/bin/bash

# ================= CONFIG =================
SCRIPT="/lab/projects/fire_smoke_awr/experiments/cropping_experiment/scripts/target_crop_dynamic_window.py"     # your Python script filename
ENV_PATH="/home/finn/.conda/envs/yolo/bin/python"  # adjust if needed
INTERMEDIATE_SIZES=($(seq 650 50 1100) )  # list of sizes to test
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
