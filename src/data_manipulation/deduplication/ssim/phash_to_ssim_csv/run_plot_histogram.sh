#!/bin/bash

# Set the directory containing CSV files
CSV_DIR="/lab/projects/fire_smoke_awr/src/augmentation/deduplication/ssim/ssim_csv"

# Loop through all CSV files
for csv_file in "$CSV_DIR"/*.csv; do
    echo "Processing $csv_file..."
    python plot_histogram.py "$csv_file"
done

echo "All histograms generated."
