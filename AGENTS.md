<<<<<<< HEAD
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
=======
# Engineering Playbook

This project is an active research codebase. Optimize for readable experiments, fast iteration, and reproducible outputs rather than heavyweight abstractions.

## Coding Philosophy
- Stick to Python 3.10+ with PEP 8 spacing (four spaces) and snake_case functions/modules, PascalCase classes, and UPPER_CASE constants.
- Keep modules focused; prefer a few well-commented functions over sprawling classes. Annotate public functions with type hints and reserve inline comments for non-trivial logic.
- Every script must open with a concise module-level docstring that states what it does and includes a concrete invocation example (for example `python src/pipeline/two_stage/classify_region.py`).
- Write sufficient docstrings for functions, classes, and methods to clarify their purpose, arguments, and return values. This is crucial for making the code understandable.
- Follow the “edit-at-the-top” pattern: expose inputs, outputs, and tunable parameters through a module-level CONFIG block (plain dict/constant assignments—avoid `argparse` unless absolutely necessary).

## Script Authoring Checklist
- Place CONFIG immediately under the docstring. Use descriptive variable names (`IMG_DIR`, `PRED_DIR`, `CONF_THRESH`, etc.) so callers can tweak paths without searching through code.
- Wrap any substantial loop over images, files, or dataset rows with `tqdm` (or `tqdm.auto`) to surface progress. Pair it with lightweight `print` statements summarizing counts (processed files, detections kept, metrics saved) so terminal logs tell the story without opening artifacts.
- When adding new utilities, default to synchronous code that can be read in one pass; document any non-obvious dependencies (e.g., `fiftyone`, `torchvision`) near the import block.
- Prefer pure functions for transformations/conversions and keep I/O (loading/saving) centralized so scripts can be unit-tested with mocks.

## Dataset Handling
- Treat `data/` as read-only—stage new drops under fresh subfolders and record provenance in experiment notes.
- Most pipelines assume YOLO-style splits (`images/train`, `labels/train`, `images/val`, `labels/val`, `images/test`, `labels/test`). Preserve that scaffold whenever you curate datasets or generate crops so downstream scripts (crop generation, evaluation, composites) work without edits.
- Expect sibling `images/` and `labels/` directories with YOLO label text files; do not mix formats within the same run folder.
- Use `experiments/` for runnable study configs and mirror the directory name under `outputs/` to keep provenance obvious. Keep checkpoints in `weights/` (large binaries stay out of Git; log their source in experiment READMEs).

## Tooling & Quality Gates
- Run `ruff check` and `ruff format` before sharing changes. Keep imports clean and avoid committing notebook checkpoints or large generated binaries.
- Add targeted `pytest` coverage under `tests/` for deterministic utilities (conversion scripts, evaluation math). Mirror the `src/` structure so modules map cleanly to tests.
- When touching evaluation logic, run `python -m src.evaluation.detection.eval_metrics` (or the relevant helper such as `metrics_at_conf`) and capture key mAP/precision/recall numbers in your notes/PR.

## Logging & Reproducibility
- Print the resolved CONFIG at the start of each script so logs capture the exact paths and thresholds used.
- Emit finishing summaries (`Processed 1,204 images`, `Saved violin plots to …`, `Copied 892 annotations`) to make CLI output self-contained.
- Keep copies of YAML/JSON configs alongside experiment outputs (e.g., `config_used.yaml` in model folders) for reproducibility.

## Environment & Access
- Many historical paths point to `/lab/projects/fire_smoke_awr/...`; update them for your sandbox but never commit user-specific directories.
- Keep credentials/API keys outside the repo (environment variables, untracked `.env`). Document any required secrets in PR descriptions or experiment notes.
- GPU-facing commands (`nvidia-smi`, torch scripts requesting CUDA) may require sandbox escalation—request approval before running them in this environment.
- When allocating a GPU, stay off devices `cuda:0` and `cuda:1` so shared services stay stable; default to `cuda:5` unless you have coordination to use `2`, `3`, or `4`.
- When using FiftyOne, launch via `python -m src.util.fiftyone.load_dataset`, choose your dataset and port (5151 default), and tunnel the port (`ssh -L 5151:localhost:5151 …`) if you need remote access.

## Communication & PR Hygiene
- Write imperative, scope-focused commit messages (e.g., “Tune two-stage confidence gates”) and include dataset versions, checkpoints, and metric deltas in bodies.
- PRs should outline the problem, summarize changes, list commands executed with outcomes, reference issues/experiments, and note external assets (weights, datasets).

Following these guardrails keeps the research workflow nimble while making it easy for collaborators to understand, reproduce, and extend your work.
>>>>>>> 20e489c07ba06d3fcc44f6bcb36693049d328deb
