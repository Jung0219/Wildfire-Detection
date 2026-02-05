# Multi-Resolution + Classifier Pipeline Plan

Goal: extend `mr+classifier.py` so that it wraps the multi-resolution pipeline (`src/pipeline/multiresolution/multiresolution.py`), runs detection, maps boxes back to the original image, and then runs a classifier (EVA or YOLO) on each detection before writing final labels.

## Inputs & Dependencies
- Multiresolution config (reuse existing config dict):
  - `GT_DIR`, `PARENT_DIR`, `YOLO_MODEL`, resizing parameters.
- Classifier config:
  - `CLASSIFIER` (`eva`/`yolo`), `CLASSIFIER_WEIGHTS`, `CLS_CONF_THRESH` (conf threshold for classifier probability if needed).
- Two-stage thresholds:
  - `CONF_LOW`, `CONF_HIGH` for gating detection confidence before classification.
- Output directories:
  - Reuse multiresolution composite/label paths, plus a final `FILTERED_LABEL_DIR`.

## Pipeline Overview
1. **Multiresolution Detection Pass**
   - Reuse functions from `multiresolution.py` to generate composites, run YOLO, remap boxes, and perform NMS.
   - Collect raw detections per image before writing them.
2. **Classifier Initialization**
   - Instantiate `YOLOClassifier` or `EVAClassifier` once with the desired weights/device.
3. **Post-processing Loop (per image)**
   - Convert raw detections to pixel coordinates on the original image.
   - Apply confidence gating:
     - Drop if `< CONF_LOW`.
     - Keep immediately if `≥ CONF_HIGH`.
     - For values in between, crop the region, run the classifier, and keep only if it predicts foreground.
4. **Output Writing**
   - Convert the kept pixel boxes back to YOLO normalized coordinates.
   - Write final `.txt` files to `FILTERED_LABEL_DIR`.
   - Optionally log counts per image (kept vs dropped, classifier decisions).

## Implementation Notes
- Integrate directly into `multiresolution.py` logic or build a wrapper that first calls the detection pipeline and receives detection outputs in memory before writing filtered files.
- Reuse XY conversion utilities from `multiresolution.py` or `two_stage/classify_region.py`.
- Use classifier wrappers from `src.pipeline.two_stage.classifiers`.
- Provide a module-level CONFIG merging both detection and classifier parameters.
- Maintain summaries (total detections, filtered via classifier, auto-kept).

## Testing
- Run the combined pipeline on a small image subset and compare outputs against the original multiresolution results to ensure coordinates still align.
- Verify classifier gating by forcing thresholds to extremes (only classifier vs. no classifier) to confirm both code paths work.
