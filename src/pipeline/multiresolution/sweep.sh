for THRESH in $(seq 0.1 0.2 0.9)
do
  RUN_NAME="ciounms_${THRESH}"
  OUTPUT_DIR="/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev/experiments/ciounms/${RUN_NAME}"

  cat > temp.yaml <<EOL
gt_dir: "/lab/projects/fire_smoke_awr/data/detection/test_sets/early_fire/dev"
parent_dir: "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/EF_dev"
model_path: "/lab/projects/fire_smoke_awr/outputs/yolo/detection/ABCDE_noEF/train/weights/best.pt"

intermediate_size: 780
conf_thresh: None
size_thresh: None
nms_iou_thresh: ${THRESH}
postproc: ciounms
save_img: false
output_dir: "${OUTPUT_DIR}"
EOL

  echo "Running with IoU threshold = $THRESH"
  python run.py --config temp.yaml
done
rm temp.yaml