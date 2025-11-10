# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the production packages. Core areas: `pipeline/` (inference flows, two-stage logic, sky detection), `analysis/` (dataset diagnostics), `evaluation/` (metric calculators), `data_manipulation/` (prep scripts), and `util/` (annotation and file helpers).
- `data/` holds curated detection/classification sets and should stay read-only; stage new assets under fresh subfolders.
- `experiments/` and `outputs/` capture reproducible studies and their artifacts—mirror subdirectory names between them to keep provenance obvious.
- `weights/` stores local checkpoints (for example `yolo11n.pt`). Keep large downloads out of Git and record their source in experiment notes.

## Build, Test, and Development Commands
- `python -m src.pipeline.two_stage.classify_region` executes the detector-plus-classifier cascade; adjust the config block for your image, label, and weight paths before running.
- `python -m src.evaluation.detection.eval_metrics` summarizes YOLO predictions into mAP/precision/recall and optional plots; set `GT_DIR`, `PRED_DIR`, and output knobs at the top of the file.
- `python -m src.analysis.analyze_boxes` produces bounding-box statistics and histograms in `outputs/analysis`, helping validate new dataset drops.
- GPU-facing commands such as `nvidia-smi` require explicit user approval for sandbox escalation—request permission before executing them.

## Coding Style & Naming Conventions
- Target Python 3.10+, PEP 8 spacing (four spaces), snake_case for functions/modules, PascalCase for classes, and UPPER_CASE for module-level constants.
- Favor type hints on public APIs (see `src/evaluation/detection/eval_metrics.py`) and keep helper names descriptive.
- Run `ruff check` and `ruff format` before raising a PR; keep imports clean and avoid committing notebook checkpoints.

## Testing Guidelines
- Introduce automated coverage under a new `tests/` package using `pytest`; structure cases to mirror the `src/` tree.
- Add smoke tests for new dataset utilities and deterministic evaluation helpers, mocking filesystem paths where possible.
- Before merging, execute the relevant evaluation script (for example `python -m src.evaluation.detection.metrics_at_conf`) and attach key metrics or plots to the PR.

## Commit & Pull Request Guidelines
- Recent commits are short (“update”, “experiments”); move toward imperative, component-scoped titles such as “Tune two-stage confidence gates”.
- Expand commit bodies with dataset versions, checkpoint names, and metric deltas so reviewers can audit experiment context.
- PRs should include: the problem statement, bullet summary of changes, commands executed with outcomes, linked issues, and any external asset locations.

## Configuration & Data Access
- Many scripts embed absolute paths rooted at `/lab/projects/fire_smoke_awr`; update these config blocks for your environment and avoid committing user-specific paths.
- Keep credentials and API keys outside the repo via environment variables or ignored `.env` files, and note any required secrets in the PR description.
