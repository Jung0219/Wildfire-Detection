# Engineering Notes

## Core Layout
- `src/full_pipeline/` is the main framework; see `src/full_pipeline/STRUCTURE.md` for the module map and intended data flow.
- `src/full_pipeline/run/` holds the runnable entrypoints. `single_run.py` targets one image, `batch_run.py` iterates a folder, and `multiresolution.py` is for composite/multi-scale experiments. The `CONFIG` constant at the top of each script should point to your YAML.
- `config/config.py` defines `MRClassifierConfig` defaults plus helpers like `ensure_dir`; prefer changing YAML rather than hard-coding paths in code.
- Core modules live under `models/`, `preprocess/`, `postprocess/`, `classifier_stage/`, `data/`, and `io/`. Extend these components instead of bolting logic into the run scripts so the pipeline stays testable.

## Workflow Requests
- When we are planning work, do not create or modify code until explicitly instructed to start coding.

## Running the Pipeline
- Update the YAML referenced by `src/full_pipeline/run/*.py` (examples: `run/config.yaml`, `run/batch_run.yaml`) with your `image_dir` or `image_path`, `output_dir`, `detector_weights`, `classifier_weights`, thresholds (`conf_low`, `conf_high`, `nms_iou_thresh`), `classifier_crop_size`, `anchor_y_frac`, and flags (`save_debug`, `save_composites`, `device`).
- Run single-image inference with `python src/full_pipeline/run/single_run.py`; run folder inference with `python src/full_pipeline/run/batch_run.py`. Each script saves the resolved `args.yaml` next to outputs for reproducibility.
- Keep experiment-specific configs alongside outputs under `outputs/` and mirror names under `experiments/` when adding new study folders.
- Avoid committing user-specific absolute paths; if you change `CONFIG` defaults, leave a note in the YAML about expected path patterns.

## Coding Style & Conventions
- Use Python 3.10+, PEP 8 spacing (four spaces), snake_case functions/modules, PascalCase classes, and UPPER_CASE constants.
- Start every script with a concise module-level docstring describing its purpose and an invocation example (for example `python src/full_pipeline/run/single_run.py`).
- Place an editable CONFIG block immediately under the docstring; surface tunables (paths, thresholds, device flags) there rather than scattering literals.
- Annotate public functions with type hints and write docstrings for functions, classes, and methods to clarify arguments and return values. Keep inline comments for non-trivial logic only.
- Keep modules focused and favor small, pure functions for transformations; centralize I/O so pieces are mockable in tests. Document any non-obvious dependencies near the imports.
- Wrap substantial loops over images/files with `tqdm` and print lightweight summaries (counts processed, outputs written) so CLI logs tell the story without opening artifacts.

## Data and Access
- Treat `data/` as read-only; stage new drops under fresh subfolders and record provenance in experiment notes. Keep large weights in `weights/` but out of Git.
- Many historical paths point to `/lab/projects/fire_smoke_awr/...`; adjust locally but avoid committing environment-specific paths or credentials. GPU commands may need approval in this environment.
