# Workstation Setup Guide for YOLOv11 Training

Quick start guide for setting up and running YOLOv11 training on the workstation with dual A6000 GPUs.

## Hardware Specifications
- **CPUs**: 64 cores / 128 threads
- **RAM**: 256GB
- **GPUs**: Dual NVIDIA A6000 (48GB each, NVLink)
- **CUDA**: 12.1+ required

## Prerequisites
- Miniconda or Anaconda installed
- NVIDIA drivers and CUDA 12.1+ installed
- Git (to clone repository or copy files)

## Setup Steps

### 1. Copy Files to Workstation

Transfer these directories from development machine to workstation:

```bash
# Required directories
microPAD-colorimetric-analysis/
├── python_scripts/           # Training scripts and configs
├── augmented_1_dataset/      # Training data (168 images + labels)
└── documents/plans/          # (Optional) Implementation plan
```

**Estimated transfer size**: ~500MB (depending on augmented dataset size)

### 2. Create Python Environment

```bash
# Create conda environment
conda create -n microPAD-python-env python=3.10 -y
conda activate microPAD-python-env

# Install CUDA-enabled PyTorch (CRITICAL for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
cd microPAD-colorimetric-analysis/python_scripts
pip install ultralytics onnx onnxsim onnxruntime-gpu numpy scipy pillow opencv-python matplotlib psutil pyyaml requests

# Verify installation
yolo checks
```

**Expected output**: GPU detection should show both A6000 GPUs.

### 3. Verify Dataset

```bash
# Check dataset structure
cd ../augmented_1_dataset
ls -l train.txt val.txt
head -3 train.txt

# Verify label files exist
ls iphone_11/labels/ | head -5

# Expected output:
# train.txt: 126 lines (iphone_11, iphone_15, realme_c55)
# val.txt: 42 lines (samsung_a75)
# labels/: .txt files with YOLO segmentation format
```

### 4. Run Training

```bash
cd ../python_scripts

# Stage 1: Synthetic data pretraining (RECOMMENDED START)
python train_yolo.py --stage 1

# Alternative: Custom parameters
python train_yolo.py --stage 1 --epochs 200 --batch 96 --device 0,1
```

**Training will start immediately with default parameters:**
- Model: `yolo11n-seg.pt` (auto-downloaded)
- Epochs: 150
- Batch size: 128 (dual GPU)
- Image size: 640×640
- Early stopping: 20 epochs patience
- Results: `micropad_detection/yolo11n_synth/`

**Expected training time**: 1-2 hours

### 5. Monitor Training Progress

**Option 1: Check results plot**
```bash
# Results updated every epoch
xdg-open micropad_detection/yolo11n_synth/results.png
# or open manually: micropad_detection/yolo11n_synth/results.png
```

**Option 2: TensorBoard (optional)**
```bash
pip install tensorboard
tensorboard --logdir micropad_detection/yolo11n_synth
# Open browser to http://localhost:6006
```

**Option 3: Live console output**
Training prints progress every epoch:
```
Epoch    GPU_mem   box_loss   seg_loss   cls_loss  Instances       Size
  1/150      15.2G      1.234      0.876      0.543         7        640
  ...
```

### 6. Validate Model

After training completes (or during training to check intermediate weights):

```bash
# Validate best checkpoint
python train_yolo.py --validate --weights ../micropad_detection/yolo11n_synth/weights/best.pt

# Expected metrics (target):
# Mask mAP@50: > 0.85
# Detection rate: > 85%
```

### 7. Export Models

Export trained model for MATLAB and Android deployment:

```bash
# Export to both ONNX (MATLAB) and TFLite (Android)
python train_yolo.py --export --weights ../micropad_detection/yolo11n_synth/weights/best.pt

# Exported files location:
# micropad_detection/yolo11n_synth/weights/best.onnx
# micropad_detection/yolo11n_synth/weights/best_saved_model/  (TFLite)
```

**Copy exported models back to development machine** for MATLAB/Android integration.

## Training Outputs

After successful training, you'll have:

```
micropad_detection/yolo11n_synth/
├── weights/
│   ├── best.pt              # Best checkpoint (highest mAP)
│   ├── last.pt              # Latest checkpoint
│   ├── best.onnx            # ONNX export (after export step)
│   └── best_saved_model/    # TFLite export (after export step)
├── results.png              # Training curves
├── confusion_matrix.png     # Class confusion matrix
├── val_batch0_labels.jpg    # Validation samples with labels
├── val_batch0_pred.jpg      # Validation samples with predictions
└── args.yaml                # Training arguments used
```

## Troubleshooting

### Out of Memory (OOM) Error

Reduce batch size:
```bash
python train_yolo.py --stage 1 --batch 64
# or try --batch 32
```

### Slow Training / GPUs Not Detected

Check GPU utilization:
```bash
# Run in separate terminal
watch -n 1 nvidia-smi

# Expected: Both GPUs at 95%+ utilization during training
```

Verify CUDA-enabled PyTorch:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Expected output:
# CUDA available: True
# GPU count: 2
```

If CUDA not available, reinstall PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Training Crashes or Hangs

Check system resources:
```bash
# CPU/RAM usage
htop

# GPU memory
nvidia-smi

# Disk space (need ~10GB for checkpoints)
df -h .
```

### Poor Validation Metrics (mAP < 0.85)

**Possible causes:**
1. **Insufficient training**: Let it train longer (increase `--epochs`)
2. **Dataset issues**: Verify labels are correct format
3. **Hyperparameter tuning**: Try different learning rates or batch sizes

**Debug steps:**
```bash
# Check label format
head -1 ../augmented_1_dataset/iphone_11/labels/*.txt
# Expected: 0 x1 y1 x2 y2 x3 y3 x4 y4 (9 space-separated values)

# Visualize predictions on validation set
# Check: micropad_detection/yolo11n_synth/val_batch0_pred.jpg
```

## Next Steps After Training

1. **Validate performance**: Check mAP@50 > 0.85
2. **Export models**: Run export step (Section 7)
3. **Copy to development machine**:
   ```bash
   # Files to copy back:
   micropad_detection/yolo11n_synth/weights/best.pt
   micropad_detection/yolo11n_synth/weights/best.onnx
   micropad_detection/yolo11n_synth/weights/best_saved_model/
   micropad_detection/yolo11n_synth/results.png
   ```
4. **Phase 4**: Integrate ONNX model into MATLAB (`detect_quads_yolo.m`)
5. **Phase 5**: Integrate TFLite model into Android app

## Stage 2: Fine-Tuning (Future)

Once manual labels are available:

```bash
# 1. Create mixed dataset config
# (Follow instructions in documents/plans/AI_DETECTION_PLAN.md Phase 2)

# 2. Fine-tune from Stage 1 checkpoint
python train_yolo.py --stage 2 \
    --weights ../micropad_detection/yolo11n_synth/weights/best.pt \
    --epochs 80 \
    --batch 96 \
    --lr0 0.01

# 3. Validate and export Stage 2 model
python train_yolo.py --validate --weights ../micropad_detection/yolo11n_mixed/weights/best.pt
python train_yolo.py --export --weights ../micropad_detection/yolo11n_mixed/weights/best.pt
```

## Quick Reference

### Essential Commands

```bash
# Setup
conda activate microPAD-python-env
cd microPAD-colorimetric-analysis/python_scripts

# Train Stage 1
python train_yolo.py --stage 1

# Validate
python train_yolo.py --validate --weights ../micropad_detection/yolo11n_synth/weights/best.pt

# Export
python train_yolo.py --export --weights ../micropad_detection/yolo11n_synth/weights/best.pt

# Monitor GPUs
nvidia-smi -l 1
```

### File Locations

- **Scripts**: `python_scripts/train_yolo.py`
- **Dataset**: `augmented_1_dataset/`
- **Config**: `python_scripts/configs/micropad_synth.yaml`
- **Results**: `micropad_detection/yolo11n_synth/`
- **Weights**: `micropad_detection/yolo11n_synth/weights/best.pt`

## Support

For issues or questions:
1. Check `python_scripts/README.md` for detailed documentation
2. Review `documents/plans/AI_DETECTION_PLAN.md` for implementation plan
3. Check YOLO docs: https://docs.ultralytics.com/
4. Verify dataset with `prepare_yolo_dataset.py`

---

**Last Updated**: 2025-10-29
**Target Hardware**: Dual A6000 GPUs, 256GB RAM
**Expected Time**: 1-2 hours for Stage 1 training
