# microPAD Python Scripts

Python scripts for YOLOv11 training and inference for microPAD quadrilateral auto-detection.

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

```bash
conda activate microPAD-python-env

# Stage 1: Train on synthetic data
yolo segment train \
    model=yolo11n-seg.pt \
    data=python_scripts/configs/micropad_synth.yaml \
    epochs=150 \
    imgsz=640 \
    batch=128 \
    device=0,1 \
    project=micropad_detection \
    name=yolo11n_synth \
    patience=20
```

**Training parameters explained:**
- `model=yolo11n-seg.pt`: YOLOv11-nano segmentation (smallest, fastest)
- `epochs=150`: Maximum training epochs
- `imgsz=640`: Input image size
- `batch=128`: Batch size (adjust based on GPU memory)
- `device=0,1`: Use both A6000 GPUs
- `patience=20`: Early stopping patience

**Expected training time:**
- ~1-2 hours on dual A6000 (48GB each)

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

After training completes, export models for MATLAB and Android:

```bash
# 1. ONNX for MATLAB
yolo export \
    model=micropad_detection/yolo11n_synth/weights/best.pt \
    format=onnx \
    imgsz=640 \
    simplify=True

# 2. TFLite for Android (FP16)
yolo export \
    model=micropad_detection/yolo11n_synth/weights/best.pt \
    format=tflite \
    imgsz=640 \
    half=True

# 3. INT8 quantization (if FP16 inference > 50ms)
yolo export \
    model=micropad_detection/yolo11n_synth/weights/best.pt \
    format=tflite \
    imgsz=640 \
    int8=True
```

**Exported files:**
- `best.onnx`: For MATLAB integration
- `best_fp16.tflite`: For Android (start with this)
- `best_int8.tflite`: For Android (if FP16 too slow)

## Project Structure

```
python_scripts/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── prepare_yolo_dataset.py      # Dataset preparation script
├── configs/
│   └── micropad_synth.yaml      # YOLO dataset config
└── [future scripts]
    ├── train_yolo.py            # Training wrapper (optional)
    ├── evaluate_model.py        # Evaluation script
    └── visualize_predictions.py # Visualization tools
```

## Next Steps

After successful training:

1. ✅ **Phase 3.1-3.2**: Environment and dataset setup (DONE)
2. **Phase 3.3**: Train model on workstation
3. **Phase 3.4**: Export ONNX/TFLite models
4. **Phase 4**: MATLAB integration (`detect_quads_yolo.m`)
5. **Phase 5**: Android integration (TFLite inference)
6. **Phase 6**: Validation and deployment

## Troubleshooting

### Out of Memory (OOM) Error
Reduce batch size:
```bash
batch=64  # or batch=32
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

Check label format (9 values per line):
```bash
head -1 augmented_1_dataset/iphone_11/labels/IMG_0957_aug_001.txt
# Expected: 0 x1 y1 x2 y2 x3 y3 x4 y4
```

## References

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [AI_DETECTION_PLAN.md](../documents/plans/AI_DETECTION_PLAN.md)
- [CLAUDE.md](../CLAUDE.md)
