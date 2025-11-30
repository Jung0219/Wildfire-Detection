# MR Modular Pipeline Structure (proposed)

This layout is for the refactored MR + classifier pipeline to support single-image calls and clean reuse.

## Suggested Directory Structure
- `config/`
  - `config.py` — dataclasses/defaults/env parsing for paths, thresholds, device, and flags.
- `models/`
  - `detector.py` — YOLO detector loader + predict wrapper.
  - `classifier.py` — secondary classifier loader/predict wrapper (YOLO or EVA02).
- `data/`
  - `loader.py` — image loading utilities (path -> np.ndarray) and base-name helpers.
- `preprocess/`
  - `composite.py` — pad/downscale vs composite generation + meta (skyline, anchor).
- `postprocess/`
  - `mapping.py` — map composite detections back to original coords.
  - `nms.py` — NMS utilities.
  - `labels.py` — format YOLO txt rows, optional stats (placeholder if needed).
- `classifier_stage/`
  - `gating.py` — confidence band gating + crop extraction + classifier call.
- `io/`
  - `save.py` — write labels, composites, debug images (guarded by flags).
- `api.py`
  - Single-image entrypoint: orchestrates load -> preprocess -> detect -> map -> gate -> return/save.
- `cli/`
  - `main.py` — folder runner using `api.py`, minimal argparse.
- `run/`
  - `run.py` — YAML-driven runner (editable path inside script) + example config.
- `tests/`
  - Unit tests for preprocess/mapping/gating with mocks and synthetic images.

## Implementation Plan (Entrypoints)
Three runnable variants are planned; do not implement until explicitly green-lit.
- **Raw YOLO run**: minimal script that loads images, runs `models.detector` directly on resized/letterboxed originals (no composite, no classifier), writes YOLO txt labels. Ideal for baseline metrics and regression checks.
- **Composite YOLO run**: builds the multiresolution composite via `preprocess.composite`, runs `models.detector`, maps with `postprocess.mapping`, and writes labels (optionally saves composites/debug). Mirrors `run/multiresolution` behavior without classifier gating.
- **Composite + two-stage run**: full pipeline: composite -> detector -> mapping -> optional NMS -> `classifier_stage.gating` for confidence band filtering and secondary classification, then saves filtered labels and any debug artifacts. This is the primary production path.

## Run Modes
- Each of the three variants should have `_single` and `_batch` entrypoints (or a mode flag) to support single-image and folder runs. Keep naming consistent across raw, composite, and composite-plus-two-stage scripts.
- House single-image scripts under `src/full_pipeline/run/single` and batch scripts under `src/full_pipeline/run/batch` (e.g., `raw_single.py`, `raw_batch.py`, `composite_single.py`, `composite_batch.py`, `two_stage_single.py`, `two_stage_batch.py`).

## Flow Overview
1) `data.loader` reads an image and base name.
2) `preprocess.composite` returns composite image + meta.
3) `models.detector` runs detection on composite.
4) `postprocess.mapping` converts boxes back to original coords; `postprocess.nms` filters if needed.
5) `classifier_stage.gating` applies confidence bands and secondary classifier on crops.
6) `io.save` optionally writes labels/composites/debug; `api.run_*` returns structured result.
7) `cli.main` / `run.run` loop over a folder and delegate per-image to `api`.

## Data/Control Flow Diagram (text)
```
image path -> data.loader -> image ndarray
               |-> base_name

image -> preprocess.composite -> composite image + meta
composite -> models.detector -> yolo_result
(yolo_result + meta + orig_shape) -> postprocess.mapping -> mapped dets
mapped dets --[optional nms]--> filtered dets
filtered dets + image -> classifier_stage.gating -> verified dets + counts

verified dets -> io.save.write_labels (if enabled)
composite      -> io.save.save_image (debug/composite flags)

api.run_mr_classifier_on_image orchestrates the above and returns PipelineResult
cli/main.py or run/run.py iterate files -> call api -> produce outputs
```
