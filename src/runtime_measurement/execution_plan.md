# Execution Plan: Inference Timing

Use this guide to instrument and run timing for the three single-image runners in `src/runtime_measurement/`. Each script already has `# start time` and `# end time` markers indicating where to bracket timing.

## Common prep
- You handle `config.yaml` edits (paths, weights, device, thresholds). Keep `data/` read-only; stage inputs under `outputs/runtime_measurement/inputs/` if needed.
- Use a high-resolution timer (`time.perf_counter()` or `time.time()`) and, on GPU, call `torch.cuda.synchronize()` immediately before stopping the timer to avoid async skew.
- Include warm-up iterations (not timed) to pay model load and CUDA init overhead; at least 1–2 images per run.
- Run each scenario multiple times (e.g., 3–5) and report mean/std along with per-image stats to smooth noise.
- Capture hardware context: GPU model, driver/CUDA, CPU/RAM; note if running CPU-only.

## Scenario A: Raw YOLO (`measure_raw.py`)
1) Use your `config.yaml` for `measure_raw.py` (image path, detector weights, output dir, device, confidence).
2) Start timing at the `# start time` marker before `detector.predict(...)`; `torch.cuda.synchronize()` before stopping.
3) End timing at the `# end time` marker immediately after prediction.
4) After the run, log per-image inference time, warm-up count, number of repeats, and the command used (e.g., `python -m src.runtime_measurement.measure_raw`).

## Scenario B: Composite Detector (`measure_one_stage.py`)
1) Use your `config.yaml` for `measure_one_stage.py` (image path, detector weights, intermediate size, anchor, output dir, device, confidence, NMS flag).
2) Start timing at the `# start time` marker before composite building (`prepare_image_for_detection(...)`); `torch.cuda.synchronize()` around detector predict.
3) End timing at the `# end time` marker after NMS/selection but before writing outputs.
4) Record composite build + detection duration; note warm-up count, repeats, and whether NMS was applied.

## Scenario C: Composite + Classifier (`measure_two_stage.py`)
1) Use your `config.yaml` for `measure_two_stage.py` (image path, detector weights, classifier weights/type, intermediate size, anchor, NMS, classifier thresholds/crop size, output dir, device).
2) Start timing at the `# start time` marker before composite build; `torch.cuda.synchronize()` around detector and classifier calls.
3) End timing at the `# end time` marker after classifier gating but before saving labels/composites.
4) Record total pipeline duration, warm-up count, repeats, and optionally sub-breakdowns (detector vs classifier) using interim timers.

## Running and collecting
- Execute each script with `python -m src.runtime_measurement.<script_name>`.
- Run multiple trials per scenario; average results and note std/variance.
- Store timings (per-image and summary) alongside the run outputs in `outputs/runtime_measurement/runs/<scenario>/`.
- Keep a short run log noting: command, config values, warm-up status, trial count, GPU sync usage, and any deviations (e.g., half precision, batch size changes).
