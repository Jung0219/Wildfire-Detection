## Extension Plan: Post-Multires Two-Stage Verification in `full_pipeline.py`

Goal: Keep the existing multiresolution block intact, then append a two-stage verification step that gates detections by confidence and validates the middle band with the classifier using centered 224x224 crops.

### Proposed Changes
- **Config**: Make both verification thresholds explicit (e.g., `VERIFICATION_CONF_LOW`, `VERIFICATION_CONF_HIGH`) and confirm `CLASSIFIER_CROP_SIZE=224` is clearly defined. No changes to the multires CONFIG; it must run exactly as-is.
- **Pipeline hook**: Immediately after the multiresolution block emits detections:
  - Stage 1: Confidence gate — drop boxes `< VERIFICATION_CONF_LOW`, auto-keep boxes `>= VERIFICATION_CONF_HIGH`, and route boxes in `[LOW, HIGH)` to the classifier.
  - Stage 2: For the mid-band only, run the classifier on a crop centered on the bounding box (ensure the box center lands in the middle of the 224x224 crop); keep if the classifier approves, discard if it does not.
- **Outputs**: Retain detector confidences for both auto-kept and classifier-kept boxes; discard others.
- **Logging**: Print counts for dropped-low, auto-kept-high, sent-to-classifier, and kept-after-classifier so the new block is observable.

### Validation
- Compare outputs with and without the verification block (set `LOW=HIGH` or set `HIGH=1.0`) to confirm the multiresolution detections are unchanged before verification.
- With a realistic threshold band, confirm:
  - Dropped count matches boxes `< LOW`.
  - Auto-kept count matches boxes `>= HIGH`.
  - Sent-to-classifier count matches boxes in `[LOW, HIGH)`.
  - Kept-after-classifier reflects classifier approvals only.
