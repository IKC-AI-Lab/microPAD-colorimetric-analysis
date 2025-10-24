# Auto-Detection Refactor Task Tracker

## Legend
- [ ] Not started
- [~] In progress
- [x] Complete

## Task Backlog

### A. Augmentation Pipeline Refactor
- [x] A1. Emit augIdx=0 passthrough in `matlab_scripts/augment_dataset.m` by copying the Stage 1 capture, regenerating polygon and ellipse crops, and writing `*_aug_000_*` assets into the augmented folders.
- [x] A2. Update the synthetic branch (augIdx>0) to size backgrounds from the source image, regenerate polygons, and keep coordinate exports consistent with the current naming scheme.
- [x] A3. Restrict multi-scale generation to synthetic variants, emit them under `augmented_*/*/scales/<size>`, and flip the default `multiScale` flag to `false`.
- [ ] A4. Differentiate passthrough vs synthetic samples in logging and manifest generation to aid downstream consumers.

### B. Dataset Quality & Monitoring
- [ ] B1. Extend `verify_dataset_quality.m` to report real vs synthetic sample ratios based on the augmentation manifest and flag insufficient real coverage.

### C. Split Logic & Manifests
- [ ] C1. Refactor split creation so that when >=3 phone folders exist, entire phones are reserved for validation/test; warn and skip splitting when fewer than three phones are available.
- [ ] C2. Regenerate `dataset_manifest.json` entries to reflect the updated folder layout (including any scales subdirectories).

### D. Python Training Pipeline Alignment
- [ ] D1. Update `python/data/dataset.py` to consume `dataset_manifest.json`, maintain phone-based grouping, and return dict-style targets.
- [ ] D2. Adjust the training loop and loss code to accept the new dataset contract.
- [ ] D3. Add `tests/test_training_smoke.py` covering a minimal dataset->model->loss forward pass.

### E. Real-World Data Plan
- [ ] E1. Document the capture and annotation workflow needed to expand `1_dataset/<phone>` coverage across devices and lighting setups.
- [ ] E2. Schedule cross-checks comparing auto-detect outputs against manual polygons on real captures, recording any gaps.

### F. Integration & Fallback
- [ ] F1. Implement auto-detect fallback and logging in `cut_concentration_rectangles.m` so low-confidence detections revert to manual mode gracefully.
- [ ] F2. Outline a lightweight operator checklist (no new MATLAB test harness) for validating detections against manual polygons on a curated sample set.

## Notes
- Passthrough coordinate entries now use the `*_aug_000_*` naming convention, so downstream consumers should rely on filenames rather than legacy names.
- Add any new generated directories (e.g., per-scale folders) to `.gitignore` as needed.
- Re-run Stage 2-4 scripts in dry-run mode after major augmentation changes to confirm compatibility.

## Implementation Notes (Current State)
- `augment_dataset.m` recreates passthrough scenes as `*_aug_000` samples: the Stage 1 capture is copied under the new name, polygon crops and ellipse patches are regenerated from the original coordinates, and fresh coordinate rows are written for the augmented directories.
- Synthetic variants default to the source image dimensions unless a user-supplied background override is provided; optional multi-scale outputs are scoped to synthetic runs and saved under `augmented_1_dataset/<phone>/scales/scale<dim>/`.
- When non-overlapping placement fails, the pipeline now falls back to an overlap-tolerant placement so augments are still emitted (with a warning) instead of being skipped.
