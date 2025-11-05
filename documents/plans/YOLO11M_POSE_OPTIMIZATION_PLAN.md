## Phase 1 — Training Script Hygiene
- [x] Replace hard-coded GPU selection in `python_scripts/train_yolo.py` with a configurable constant sourced from CLI args or environment (eliminate `'0,2'` magic number).
- [x] Rename stage identifiers and experiment names to reflect the YOLOv11m pose workflow (e.g., `yolo11m_pose_stage2`) for clarity in logging and checkpoints.
- [x] Introduce named constants (or argparse defaults) for key hyperparameters currently inlined (e.g., stage-2 learning rate) to remove residual magic numbers.
- **Verification**:
  ```bash
  # Check help shows new defaults and options
  python train_yolo.py --help

  # Verify default GPU device selection
  conda run -n microPAD-python-env python -c "from train_yolo import DEFAULT_GPU_DEVICES; print(f'Default GPUs: {DEFAULT_GPU_DEVICES}')"

  # Verify environment variable override works
  CUDA_VISIBLE_DEVICES=0 conda run -n microPAD-python-env python -c "from train_yolo import DEFAULT_GPU_DEVICES; print(f'Override GPUs: {DEFAULT_GPU_DEVICES}')"

  # Verify script initialization
  conda run -n microPAD-python-env python -c "from train_yolo import YOLOTrainer; t = YOLOTrainer(); print('✅ All checks passed')"
  ```

## Phase 2 — CLI Flexibility and Best Practices
- [x] Expose CLI flags for loader workers, caching strategy, optimizer, and learning-rate schedule so workstation runs need no code edits.
  - Added `--workers` (default: 32)
  - Added `--cache` with options: True (RAM), False (disabled), disk (disk cache)
  - Added `--optimizer` with choices: SGD, Adam, AdamW, NAdam, RAdam, RMSProp, auto
  - Added `--cos-lr` for cosine learning rate scheduler
- [x] Fix export precision flag logic (pair `--half` / `--no-half`) to allow explicit FP32 export.
- [x] Ensure all user-facing descriptions avoid workstation-specific references.
- **Verification**:
  ```bash
  # Verify CLI accepts new arguments
  python train_yolo.py --help | grep -E "(workers|cache|optimizer|cos-lr|no-half)"

  # Test FP32 export flag parsing (will fail on missing weights, but that's expected)
  python train_yolo.py --export --weights dummy.pt --no-half 2>&1 | grep -E "(FP32|FP16|no-half)"

  # Test advanced training arguments
  python train_yolo.py --stage 1 --workers 16 --cache disk --optimizer AdamW --cos-lr --help
  ```

## Phase 3 — Workstation-Oriented Defaults (Applied on target machine)
- [x] Document recommended launch commands (batch sizes, patience, epochs) in README or accompanying ops doc, using symbolic placeholders instead of workstation identifiers.
- [x] Outline dataset preparation workflow emphasizing rerun of `prepare_yolo_dataset.py` post-migration (no absolute paths baked into repo files).
- [x] Capture validation/monitoring checklist (metrics thresholds, sync-bn enablement) in docs without hardware-specific constants.

### Recommended Training Commands

**Current Workstation Setup (2× RTX A6000):**
```bash
# Default training (uses CUDA_VISIBLE_DEVICES env or defaults to '0,2')
python train_yolo.py --stage 1

# Explicit GPU selection via environment variable
CUDA_VISIBLE_DEVICES=0,2 python train_yolo.py --stage 1 --batch 64 --workers 32

# Single GPU training (if other GPU is busy)
CUDA_VISIBLE_DEVICES=0 python train_yolo.py --stage 1 --batch 32 --workers 16

# Advanced: Custom optimizer and scheduler
python train_yolo.py --stage 1 --optimizer AdamW --cos-lr --batch 64 --epochs 150
```

**Dataset Preparation Workflow:**
1. After cloning repo or dataset changes, regenerate YOLO configs:
   ```bash
   cd python_scripts
   python prepare_yolo_dataset.py
   ```
2. This ensures dataset paths are absolute and machine-specific (not baked into repo)

**Validation & Monitoring:**
- **Target metrics:** OKS mAP@50 > 0.85, Detection rate > 85%
- **Early stopping:** Default patience = 20 epochs (Stage 1), 15 epochs (Stage 2)
- **Checkpoint saving:** Every 10 epochs + best model
- **Monitoring:** `tensorboard --logdir training_runs` or check `results.png` plots

- **Verification**: Run training commands without code edits; all hyperparameters configurable via CLI or environment variables.
