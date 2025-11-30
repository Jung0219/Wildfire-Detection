## Classifier-Gated Detection Accuracy Script (design)

Goal: evaluate how a classifier refines detector outputs by filtering detections within a confidence band, saving true positives, and reporting metrics.

### Inputs (edit-at-the-top CONFIG)
- `PRED_DIR`: directory of detector predictions (YOLO txt alongside images; one file per image).
- `IMG_DIR`: images that align with `PRED_DIR` filenames.
- `GT_LABELS_DIR`: ground-truth labels for evaluation.
- `CLASSIFIER_WEIGHTS`: checkpoint path (e.g., EVA02) and transform mode (`letterbox`/`centerpad`).
- `CONF_MIN`, `CONF_MAX`: detector confidence band to send to the classifier.
- `CLS_POSITIVE_LABEL`: classifier output id treated as foreground (others => background).
- `DEVICE`: cpu/cuda device string.
- `OUTPUT_ROOT`: base output folder (e.g., `outputs/two_stage/classifier_accuracy_verification`), created if missing; all artifacts land under a timestamped/exp-named subdir.

### Processing flow
1. Load CONFIG and print resolved values for reproducibility.
2. Discover prediction files in `PRED_DIR`; skip empties, warn on missing images/labels.
3. For each prediction line (cls x y w h conf):
   - Keep boxes with `CONF_MIN <= conf <= CONF_MAX` as **candidates** for classification.
   - Boxes outside the band pass through unchanged into the refined predictions.
4. For each candidate box:
   - Crop a fixed `224x224` window centered on the YOLO box (convert xywh -> pixel center, clamp crop to image bounds), then apply classifier transform and run the classifier.
   - If classifier predicts `CLS_POSITIVE_LABEL`, keep the box and optionally attach classifier score; otherwise drop as FP.
5. Write refined predictions back to disk under `OUTPUT_ROOT/<run_name>/refined` mirroring `PRED_DIR` filenames. Also save band-only pre-classifier txts under `.../band_before` and post-classifier under `.../band_after` for comparison.
6. Evaluate (salvage band only):
   - Build two prediction sets scoped to the confidence band: (a) pre-classifier (original boxes within the band) and (b) post-classifier (band boxes after gating). Outside-band boxes are excluded from this comparison.
   - Run detection evaluation for these two sets against `GT_LABELS_DIR` and report mAP/precision/recall deltas, plus counts of TP/FP/FN within the band.
7. Save artifacts to `OUTPUT_ROOT/<run_name>`: refined txts, band-only before/after txts, `summary.json` (counts + band-scoped metrics + config), and optional histogram of classifier scores.

### CLI sketch
```
python -m src.pipeline.two_stage.classifier_accuracy
```
- Script lives in `src/pipeline/two_stage/classifier_accuracy.py` with module-level docstring and CONFIG block for the knobs above.
- Use `tqdm` over files/candidates; print finishing summaries like “Classified 432 boxes, kept 178”.
- Default experiment name: `classifier_accuracy_verification` stored under `OUTPUT_ROOT`.

### Edge cases & logging
- Handle empty prediction files; ensure missing images/labels are logged and skipped.
- Clamp boxes to image bounds before cropping.
- If no boxes fall in the confidence band, skip classification and still evaluate passthrough predictions.

### Next steps
- Implement the script per this outline, reuse `EVA02Classifier` helper for transforms, and wire evaluation call at the end.
