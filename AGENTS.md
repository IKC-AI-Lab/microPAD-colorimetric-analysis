# Repository Guidelines

## Project Structure & Module Organization
- `matlab_scripts/`: Main MATLAB pipeline scripts (`crop_micropad_papers.m`, `cut_concentration_rectangles.m`, `cut_elliptical_regions.m`, `extract_features.m`, `augment_dataset.m`) plus `helper_scripts/` utilities.
- Stage folders (sequential I/O): `1_dataset/` -> `2_micropad_papers/` -> `3_concentration_rectangles/` -> `4_elliptical_regions/` -> `5_extract_features/`.
- Augmented data: `augmented_1_dataset/`, `augmented_2_concentration_rectangles/`, `augmented_3_elliptical_regions/`.
- Image assets and raw datasets are intentionally ignored by Git (see `.gitignore`).

## Experimental Design
- **MicroPAD structure**: Each paper strip has 7 test zones (concentrations), each with 3 elliptical measurement regions.
- **Final design**: The 3 regions per zone will measure different chemicals (urea, creatinine, lactate).
- **Training phase**: All 3 regions contain the same chemical at the same concentration, yielding 3 replicate measurements per concentration level. This allows separate datasets for training ML models for each chemical.
- **File naming**: Stage 4 patches use `{base}_con{N}_rep{M}.{ext}` where `rep{M}` (M = 0, 1, 2) represents the 3 replicates.

## Build, Test, and Development Commands
- Run a stage (MATLAB): `matlab -batch "addpath('matlab_scripts'); crop_micropad_papers;"`
- Typical flow: `cut_concentration_rectangles`, `cut_elliptical_regions`, `extract_features('preset','robust','chemical','lactate')`.
- Augmentation: `matlab -batch "addpath('matlab_scripts'); augment_dataset('numAugmentations',5);"`
- Corner labels: pass `'exportCornerLabels', true` to `augment_dataset` when JSON training labels are required.
- Tests (if added): `matlab -batch "runtests('matlab_scripts')"`
- Code analysis: `matlab -batch "checkcode('matlab_scripts/script_name.m')"` (checks for unused variables, best practice violations, performance warnings)
Notes: Run from repo root or `matlab_scripts/`. Octave is not supported due to GUI/homography/Excel I/O.

## Coding Style & Naming Conventions
- MATLAB, 4-space indent, one public function per file. Use snake_case for files and functions (e.g., `extract_features.m`).
- Prefer name-value pairs for configuration; validate with `inputParser` or `arguments` blocks.
- Keep stage independence: each script reads `N_*` and writes `(N+1)_*`. Do not hardcode absolute paths.

## Testing Guidelines
- No formal suite yet. For changes, run a small subset per phone model and verify outputs and `coordinates.txt` integrity.
- If you add tests, place `test_*.m` under `matlab_scripts/tests/` using `matlab.unittest`; run with `runtests`.

## Commit & Pull Request Guidelines
- Use concise, imperative messages; Conventional Commits encouraged: `feat:`, `fix:`, `refactor:`, `docs:`.
- PRs should include: summary, affected stage(s), sample console output, and tiny before/after crops as attachments (do not commit images; use GitHub uploads). Link issues.
- Keep changes scoped; avoid reformatting large files. Do not commit raw datasets or generated images (they're ignored by default).
 - No AI/agent attribution: never add "Co-authored-by:" lines for Codex/AI, or phrases like "Generated with Codex" (or similar) in commit messages or files.
 - Preserve human authorship: do not inject bot signatures, agent headers, or metadata in commits; keep commit authors and footers human-only unless explicitly requested.

## Security & Configuration Tips
- `coordinates.txt` is written atomically; avoid manual edits. If corrupted, delete and rerun the stage.
- Ensure sufficient memory for `extract_features`; reduce batch size via `extract_features('batchSize',N)` if needed.

## Agent Execution Guidelines
- **Ask questions if stuck**: Do not add fallback algorithms instead; clarify requirements first
- No workarounds or fallbacks: implement direct solutions aligned with project constraints; do not add compatibility shims, alternative code paths, or temporary hacks
- Avoid overengineering and verbosity: keep changes minimal, focused, and idiomatic; do not add redundant layers, unused abstractions, or repetitive code
- Best practices first: follow all Coding Style & Naming Conventions above (snake_case, one public function per file, 4-space indent, name-value pairs, and validate with `inputParser`/`arguments`)
- Preserve stage independence and paths: read from `N_*`, write to `(N+1)_*`, and never hardcode absolute paths or alter the folder structure
- Do not add speculative features: no Octave fallbacks, GUI substitutes, optional modes, or extra outputs beyond the defined pipeline stages
- Minimize dependencies: do not introduce new toolboxes or third-party code unless explicitly requested; prefer MATLAB built-ins and existing `helper_scripts/`
- Fail fast over silent hacks: when inputs, data, or preconditions are missing, stop and request guidance instead of implementing workaround logic
- Keep interfaces stable: maintain filenames, data formats, and atomic writes (e.g., `coordinates.txt`) without unrequested changes
- Tests are minimal and targeted: if adding tests, place them under `matlab_scripts/tests/` using `matlab.unittest`; avoid placeholder or boilerplate tests
- Documentation changes are scoped: update only relevant sections and avoid broad reformatting unrelated to the task

## Comments & Documentation
- Keep descriptive and concise. Do not include code history, opinions, or obvious deductions.
