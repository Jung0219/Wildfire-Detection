# Runtime Measurement Plan

- **Objective**: quantify end-to-end runtime for the fire/smoke detectors and classifiers, capturing latency, throughput, and GPU/CPU utilization under representative workloads.
- **Inference entrypoints (per AGENTS.md guidance)**:
  1. Regular YOLO run: `python -m src.models.yolo.detection.predict`
  2. Composite-only detector run: `python -m src.full_pipeline.run.multiresolution`
  3. Composite + classifier run: `python -m src.full_pipeline.run.batch_run`
- **Models in scope**: YOLO detectors (e.g., `yolo11n.pt` or checkpoints in `weights/`), composite pipeline detector, and classifier weights referenced in the modular pipeline configs.
- **Datasets/inputs**: curated YOLO-format image sets in `data/` (read-only). Stage test batches in a new subfolder under `outputs/runtime_measurement/inputs/` to avoid mutating curated data.
- **Hardware targets**: prefer `cuda:5` unless coordinated otherwise; record GPU model, driver/CUDA versions, and CPU/memory info. Note if runs are CPU-only.

## Metrics to capture
- Per-image latency (mean/p50/p90/p99) and throughput (images/sec).
- Warm vs. steady-state timing (exclude first-batch cold start separately).
- GPU utilization, memory footprint, and CPU load during steady state.
- I/O overhead: image decode/load time and any pre/post-processing timing.

## Measurement procedure
1) **Environment prep**: log `python --version`, `pip freeze | grep 'torch\\|ultralytics'`, and `nvidia-smi -L` (or note CPU-only). Avoid modifying tracked files; keep weights in `weights/`.
2) **Dataset setup**: copy a representative sample (e.g., 500–2,000 images) into `outputs/runtime_measurement/inputs/<name>/images` with matching labels if needed. Document source split and class mix.
3) **Scenario A — regular YOLO**: time `src.models.yolo.detection.predict` (or the dedicated `src/runtime_measurement/measure_yolo.py` harness) around load, preprocess (if any), inference, NMS/write. Use `torch.cuda.synchronize()` immediately before and after timing windows to ensure accurate GPU durations. Capture warm-up iterations and per-image stats.
4) **Scenario B — composite detector**: time `src.full_pipeline.run.multiresolution` (or `src/runtime_measurement/measure_composite_detector.py`) around composite build, detector inference, mapping, NMS, and label writes. Use `torch.cuda.synchronize()` around GPU-bound sections. Keep intermediate size/anchor consistent across runs.
5) **Scenario C — composite + classifier**: time `src.full_pipeline.run.batch_run` (or `src/runtime_measurement/measure_composite_classifier.py`) for the full pipeline (composite + detector + mapping + classifier gating + saves). Use `torch.cuda.synchronize()` around detector and classifier steps; log classifier-specific overhead separately from detector time.
6) **Batch sizing sweep**: for each scenario, sweep batch sizes (e.g., 1, 4, 8, 16) where supported to observe throughput/latency trade-offs; keep image size consistent with training/eval settings.
7) **Concurrency check**: if applicable, test DataLoader workers or async I/O to see marginal gains; note any contention or diminishing returns.
8) **Resource logging**: sample `nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used --format=csv -l 1` during runs; optionally use `psutil` for CPU load.
9) **Repeatability**: run each configuration at least 3 times, discard outliers, and report mean/std. Fix seeds and deterministic flags where available.
10) **Outputs**: store raw logs and metrics in `outputs/runtime_measurement/runs/<date>_<config>/` (timing CSV/JSON, command used, `config_used.yaml` if applicable, and summary README).

## Reporting checklist
- Command(s) executed with exact args and paths.
- Hardware and environment snapshot.
- Timing summary (per-image latency percentiles, throughput), plus resource utilization plots or tables.
- Observed bottlenecks and proposed optimizations (e.g., smaller input size, half precision, batching, dataloader tweaks).
- Any deviations from default configs or dataset handling notes.
