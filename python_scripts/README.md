# microPAD Python Scripts

Python scripts for YOLOv11 training and inference for microPAD quadrilateral auto-detection.

## Pipeline Overview

The microPAD analysis pipeline consists of **4 stages**:

```
1_dataset (raw images)
    -> cut_micropads.m
2_micropads (cropped regions with rotation)
    -> cut_elliptical_regions.m
3_elliptical_regions (elliptical patches)
    -> extract_features.m
4_extract_features (feature tables)
```

**Augmented data pipeline:**
```
1_dataset
    -> augment_dataset.m
augmented_1_dataset (synthetic scenes)
augmented_2_micropads (transformed polygons)
augmented_3_elliptical_regions (transformed ellipses)
```

**YOLOv11 training uses:** `augmented_1_dataset` for synthetic training data generation.

## Environment Setup

### Prerequisites
- Miniconda or Anaconda installed
- CUDA 12.1+ (for GPU training on workstation with A6000)

### Installation

1. **Create conda environment:**
   ```bash
   conda create -n microPAD-python-env python=3.10 -y
   conda activate microPAD-python-env
   ```

2. **Install CPU version (development):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install GPU version (workstation with A6000):**
   ```bash
   # Install CUDA-enabled PyTorch first
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Then install remaining dependencies
   pip install ultralytics onnx onnxsim onnxruntime-gpu
   ```

4. **Verify installation:**
   ```bash
   yolo checks
   ```

## Dataset Preparation

### Current Dataset Status
- **Total images**: 168 (574 originally, using 168 for initial training)
- **Train images**: 126 (iphone_11, iphone_15, realme_c55)
- **Val images**: 42 (samsung_a75)
- **Classes**: 1 (concentration_zone)

### Prepare Dataset
```bash
python prepare_yolo_dataset.py
```

This script:
- Creates `train.txt` and `val.txt` in `augmented_1_dataset/`
- Generates YOLO config at `configs/micropad_synth.yaml`
- Uses 3 phones for training, 1 phone for validation
- Verifies all label files exist

## Training

### Phase 3.3: Train YOLOv11n-seg

**On workstation with dual A6000 GPUs:**

#### Using train_yolo.py (Recommended)

The `train_yolo.py` script provides a comprehensive CLI interface for training, validation, and export:

```bash
conda activate microPAD-python-env
cd python_scripts

# Stage 1: Train on synthetic data (automatic pretraining)
python train_yolo.py --stage 1

# Stage 2: Fine-tune with manual labels (when available)
python train_yolo.py --stage 2 --weights ../micropad_detection/yolo11n_synth/weights/best.pt

# Validate trained model
python train_yolo.py --validate --weights ../micropad_detection/yolo11n_synth/weights/best.pt

# Export to ONNX and TFLite
python train_yolo.py --export --weights ../micropad_detection/yolo11n_synth/weights/best.pt
```

**Custom training parameters:**
```bash
# Custom epochs, batch size, single GPU
python train_yolo.py --stage 1 --epochs 200 --batch 16 --device 0

# Try different resolution (640, 800, or 960)
python train_yolo.py --stage 1 --imgsz 800 --batch 28

# Export with INT8 quantization for Android
python train_yolo.py --export --weights best.pt --formats tflite --int8
```

#### Using YOLO CLI directly (Alternative)

```bash
conda activate microPAD-python-env

# Stage 1: Train on synthetic data
yolo segment train \
    model=yolo11n-seg.pt \
    data=python_scripts/configs/micropad_synth.yaml \
    epochs=150 \
    imgsz=960 \
    batch=24 \
    device=0,1 \
    project=micropad_detection \
    name=yolo11n_synth \
    patience=20
```

**Training parameters explained:**
- `model=yolo11n-seg.pt`: YOLOv11-nano segmentation (smallest, fastest)
- `epochs=150`: Maximum training epochs (Stage 1), 80 (Stage 2)
- `imgsz=960`: Input image size (960x960 for better detail)
- `batch=24`: Batch size optimized for 960 resolution on dual A6000 GPUs
- `device=0,1`: Use both A6000 GPUs
- `patience=20`: Early stopping patience (15 for Stage 2)

**Expected training time:**
- ~2-3 hours on dual A6000 (48GB each) at 960 resolution
- ~1-2 hours at 640 resolution (if using lower resolution)

**Target metrics:**
- Mask mAP@50 > 0.85
- Detection rate > 85% on validation set

### Monitor Training

```bash
# TensorBoard (if installed)
tensorboard --logdir micropad_detection/yolo11n_synth

# Or view results in:
# micropad_detection/yolo11n_synth/results.png
```

## Export Models

### Phase 3.4: Export for Deployment

#### Using train_yolo.py (Recommended)

```bash
# Export to ONNX (MATLAB) and TFLite (Android) with one command
python train_yolo.py --export --weights ../micropad_detection/yolo11n_synth/weights/best.pt

# Export only ONNX
python train_yolo.py --export --weights best.pt --formats onnx

# Export TFLite with INT8 quantization (if FP16 > 50ms)
python train_yolo.py --export --weights best.pt --formats tflite --int8
```

#### Using YOLO CLI directly (Alternative)

```bash
# 1. ONNX for MATLAB
yolo export \
    model=micropad_detection/yolo11n_synth/weights/best.pt \
    format=onnx \
    imgsz=960 \
    simplify=True

# 2. TFLite for Android (FP16)
yolo export \
    model=micropad_detection/yolo11n_synth/weights/best.pt \
    format=tflite \
    imgsz=960 \
    half=True

# 3. INT8 quantization (if FP16 inference > 50ms)
yolo export \
    model=micropad_detection/yolo11n_synth/weights/best.pt \
    format=tflite \
    imgsz=960 \
    int8=True
```

**Exported files:**
- `best.onnx`: For MATLAB integration
- `best_fp16.tflite` or `best_saved_model/`: For Android (start with FP16)
- `best_int8.tflite`: For Android (if FP16 too slow)

## Project Structure

```
python_scripts/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── prepare_yolo_dataset.py      # Dataset preparation script
├── train_yolo.py                # Comprehensive training/export CLI
├── configs/
│   └── micropad_synth.yaml      # YOLO dataset config (synthetic only)
│   └── micropad_mixed.yaml      # [Future] Mixed synthetic + manual labels
└── [future scripts]
    ├── evaluate_model.py        # Evaluation script
    └── visualize_predictions.py # Visualization tools
```

## Next Steps

After successful training:

1. ✅ **Phase 3.1-3.2**: Environment and dataset setup (DONE)
2. **Phase 3.3**: Train model on workstation
3. **Phase 3.4**: Export models for deployment
4. **Phase 4**: MATLAB integration (PyTorch model via Python interface)
5. **Phase 5**: Android integration (TFLite inference)
6. **Phase 6**: Validation and deployment

## Pipeline Integration

**MATLAB Integration:**
- MATLAB scripts use PyTorch models directly via Python interface
- No ONNX export needed for MATLAB (uses native PyTorch inference)
- Detection function: `detectQuadsYOLO.m` (calls Python YOLO model)

**Android Integration:**
- Export to TFLite format for mobile deployment
- Target inference time: <50ms on Android devices
- Quantization options: FP16 (default) or INT8 (if needed for speed)

## Troubleshooting

### Out of Memory (OOM) Error
Reduce batch size or image resolution:
```bash
# Reduce batch size
python train_yolo.py --stage 1 --batch 16

# Or reduce image resolution
python train_yolo.py --stage 1 --imgsz 640 --batch 32
```

### Slow Training
- Ensure CUDA-enabled PyTorch is installed
- Check GPU utilization: `nvidia-smi`
- Verify both GPUs are being used: `device=0,1`

### Label Format Issues
Run dataset preparation again:
```bash
python prepare_yolo_dataset.py
```

Check label format (9 values per line for YOLO, 10 for MATLAB coordinates):
```bash
head -1 augmented_1_dataset/iphone_11/labels/IMG_0957_aug_001.txt
# YOLO format: 0 x1 y1 x2 y2 x3 y3 x4 y4 (9 values, normalized)

# MATLAB coordinate format (2_micropads/phone/coordinates.txt):
# image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation (10 values, pixel coords)
```

## References

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [AI_DETECTION_PLAN.md](../documents/plans/AI_DETECTION_PLAN.md)
- [CLAUDE.md](../CLAUDE.md)
