#!/bin/bash
# run_conf_size_grid.sh

CONF_VALUES=($(seq 0.3 0.1 0.6))
SIZE_VALUES=($(seq 100 10 150))

for conf in "${CONF_VALUES[@]}"; do
  for size in "${SIZE_VALUES[@]}"; do
    echo "=== Running with CONF_THRESH=$conf, SIZE_THRESH=$size ==="
    python /lab/projects/fire_smoke_awr/src/pipeline/multiresolution/auto_scripts/pipeline_auto_conf+box_size.py \
           --conf-top $conf \
           --size-top $size
  done
done
