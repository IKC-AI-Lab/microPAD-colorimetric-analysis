# Quadrilateral Auto-Detection Implementation Plan

**Last Updated:** 2025-10-28
**Current Phase:** Phase 3 In Progress (2/4 tasks complete)
**Overall Progress:** Phase 1 complete (8/8); Phase 2 skipped; Phase 3 setup done (2/4)
**Architecture:** YOLOv11n Instance Segmentation (Ultralytics) + Enhanced Post-Processing

## Project Overview

Implement AI-based auto-detection of concentration rectangles for microPAD analysis, achieving <3px corner accuracy on Android devices using state-of-the-art YOLOv11n instance segmentation with enhanced mask-to-quad conversion.

**Hardware:** dual A6000 (48GB each, NVLink), 256GB RAM
**Input Images:** High-resolution smartphone photos (4032×3024 iPhone, similar for Android)
**Target Accuracy:** 95% of corners within 3 pixels (IoU > 0.95)
**Model Size:** ~5MB (YOLOv11n-seg)
**Inference Time:** <50ms on budget Android devices (target optimization during implementation)
**Architecture Rationale:** YOLOv11n-seg achieves 83.1% mask precision with 22% fewer parameters than YOLOv8m. Instance segmentation handles irregular quadrilaterals (perspective distortion, paper defects, occlusion) better than OBB. Mature ONNX/TFLite export for Android/MATLAB deployment.

## Project Context

Integrates with existing MATLAB pipeline (`CLAUDE.md`):

**Pipeline:**
```
1_dataset → 2_micropad_papers → 3_concentration_rectangles → 4_elliptical_regions → 5_extract_features
```

**AI Detection Target:**
- Replace manual polygon selection in `cut_concentration_rectangles.m` (Stage 3)
- Current: User manually defines 7 concentration regions per paper
- Goal: AI auto-detects all 7 regions with <3px corner accuracy

**Dataset:**
- Four phone directories: `iphone_11/`, `iphone_15/`, `realme_c55/`, `samsung_a75/`
- No Python infrastructure exists yet
- No augmented training data generated yet

**Related Documentation:**
- `CLAUDE.md`: Main project documentation and coding standards
- `AGENTS.md`: Multi-agent workflow documentation

---

## Status Legend
- [ ] Not started
- [🔄] In progress
- [✅] Completed
- [⚠️] Blocked/needs attention

---

## Phase 1: Refactor `augment_dataset.m` for Segmentation Training

### 1.1 Enhanced Perspective & Camera Parameters
- [✅] **File:** `matlab_scripts/augment_dataset.m` (lines 69-75)
- [✅] **Task:** Increase camera perspective ranges for extreme viewing angles
- [✅] **Implementation:**
  ```matlab
  CAMERA = struct( ...
      'maxAngleDeg', 60, ...           % Increased from 45°
      'xRange', [-0.8, 0.8], ...       % Increased from [-0.5, 0.5]
      'yRange', [-0.8, 0.8], ...       % Increased from [-0.5, 0.5]
      'zRange', [1.2, 3.0], ...        % Widened from [1.4, 2.6]
      'coverageCenter', 0.97, ...
      'coverageOffcenter', 0.90);      % Reduced from 0.95
  ```
- [✅] **Test:** Generate 10 samples, verify polygon corners visible

---

### 1.2 Multi-Scale Scene Generation
- [✅] **File:** `matlab_scripts/augment_dataset.m` (lines 158-160, 687-709)
- [✅] **Task:** Generate each augmentation at multiple scales
- [✅] **Parameters:**
  ```matlab
  addParameter(parser, 'multiScale', true, @islogical);
  addParameter(parser, 'scales', [640, 800, 1024], ...
      @(x) validateattributes(x, {'numeric'}, {'vector', 'positive', 'integer'}));
  ```
- [✅] **Output:** `synthetic_XXX_scale640.jpg`, `synthetic_XXX_scale800.jpg`, `synthetic_XXX_scale1024.jpg`
- [✅] **Test:** Verify polygon coordinates scale correctly

---

### 1.3 Export YOLOv11 Segmentation Labels
- [✅] **File:** `matlab_scripts/augment_dataset.m`
- [✅] **Task:** Replace old JSON label export with YOLOv11 segmentation polygon format

#### 1.3.1 Remove Old JSON Label Export
- [✅] **Cleanup:** Remove deprecated corner label functions
- [✅] **Functions to Remove:**
  - `export_corner_labels()` (lines ~2713-2886)
  - `generate_gaussian_targets()` (if exists)
  - `compute_subpixel_offsets()` (if exists)
  - MAT heatmap export logic
- [✅] **Parameters to Remove:**
  - `exportCornerLabels` parameter
  - Any heatmap-related configuration
- [✅] **Remove Calls:** Search and remove all calls to `export_corner_labels()`

#### 1.3.2 Add YOLOv11 Segmentation Label Export Function
- [✅] **New Function:** Replace `export_corner_labels()` (lines 3213-3306) with simplified version:
  ```matlab
  function export_yolo_segmentation_labels(outputDir, imageName, polygons, imageSize)
      % Export YOLOv11 segmentation format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)

      labelDir = fullfile(outputDir, 'labels');
      if ~isfolder(labelDir), mkdir(labelDir); end

      labelPath = fullfile(labelDir, [imageName '.txt']);
      tmpPath = tempname(labelDir);
      fid = fopen(tmpPath, 'wt');

      for i = 1:numel(polygons)
          quad = polygons{i};  % 4×2 vertices

          % Order corners clockwise from top-left
          quad = order_corners_clockwise(quad);

          % Normalize to [0, 1]
          normQuad = quad ./ [imageSize(2), imageSize(1)];

          % Write: 0 x1 y1 x2 y2 x3 y3 x4 y4
          fprintf(fid, '0');  % class_id always 0 (concentration zone)
          fprintf(fid, ' %.6f %.6f', normQuad');
          fprintf(fid, '\n');
      end

      fclose(fid);
      movefile(tmpPath, labelPath, 'f');
  end
  ```
- [✅] **Note:** Removed distractor class support - use standard augmentation instead of synthetic distractors

#### 1.3.3 Integration
- [✅] **Rename Parameter:** Change `exportCornerLabels` → `exportYOLOLabels` (line 155)
- [✅] **Replace Calls:** Replace `export_corner_labels()` calls (lines 564, 931, 960) with:
  ```matlab
  if cfg.exportYOLOLabels
      export_yolo_segmentation_labels(outputDir, imageName, polygons, imageSize);
  end
  ```
- [✅] **Test:** Run `augment_dataset('numAugmentations', 3, 'rngSeed', 42)` and verify:
  - `augmented_1_dataset/labels/*.txt` files exist
  - Each line format: `0 x1 y1 x2 y2 x3 y3 x4 y4` (9 values, space-separated)
  - All coordinates in [0, 1] range

---

### 1.4 Background Texture Pooling & Rotation
- [✅] **File:** `matlab_scripts/augment_dataset.m` (lines 78-83, 1180-1335)
- [✅] **Task:** Full rotation coverage with texture pooling
- [✅] **Configuration:**
  ```matlab
  ROTATION_RANGE = [0, 360];

  TEXTURE = struct( ...
      'poolSize', 16, ...
      'poolRefreshInterval', 25, ...
      'poolShiftPixels', 48, ...
      'poolScaleRange', [0.9, 1.1], ...
      'poolFlipProbability', 0.15);
  ```
- [✅] **Test:** Generate 50 scenes, verify rotation coverage

---

### 1.5 Optimize Artifact Sharpness & Density
- [✅] **File:** `matlab_scripts/augment_dataset.m` (lines 96-112, 1356-1513)
- [✅] **Task:** Sharp artifacts matching polygon sharpness
- [✅] **Changes:**
  - Removed individual artifact blur
  - Changed upscaling from bilinear to nearest-neighbor
  - Artifacts sharp by default, scene-wide blur only (25% probability)
- [✅] **Configuration:**
  ```matlab
  ARTIFACTS = struct( ...
      'countRange', [5, 40], ...
      'sizeRangePercent', [0.01, 0.75], ...
      'minSizePixels', 3);
  ```
- [✅] **Test:** Verify artifacts have sharp edges

---

### 1.6 Polygon-Shaped Distractors
- [✅] **File:** `matlab_scripts/augment_dataset.m` (lines 120-130, 1106-1196)
- [✅] **Task:** Add synthetic polygon distractors (false positives for training)
- [✅] **Configuration:**
  ```matlab
  DISTRACTOR_POLYGONS = struct( ...
      'enabled', true, ...
      'minCount', 1, ...
      'maxCount', 10, ...
      'sizeScaleRange', [0.5, 1.5], ...
      'maxPlacementAttempts', 30, ...
      'brightnessOffsetRange', [-20, 20], ...
      'contrastScaleRange', [0.9, 1.15], ...
      'noiseStd', 6);
  ```
- [✅] **Implementation:** 1-10 distractor polygons per image with varied sizes
- [✅] **Test:** Verify distractor count and size variation

---

### 1.7 Add Extreme Edge Cases
- [✅] **File:** `matlab_scripts/augment_dataset.m` (lines 160, 377-390, 676-680)
- [✅] **Task:** Generate 10% samples with extreme conditions
- [✅] **Parameter:**
  ```matlab
  addParameter(parser, 'extremeCasesProbability', 0.10, ...
      @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
  ```
- [✅] **Extreme Conditions:**
  - Very low lighting (brightness × 0.4-0.6)
  - High viewing angle (maxAngleDeg: 75°, zRange: [0.8, 4.0])
  - Small/large polygons (extended z range)
- [✅] **Test:** Generate 100 samples, verify ~10 are extreme

---

### 1.8 Performance Optimization Results
- [✅] **Task:** Optimize memory and I/O
- [✅] **Optimizations:**
  - Background synthesis: Single-precision with texture pooling
  - Artifact masks: Unit-square normalization
  - Motion blur: PSF caching
  - Poisson-disk polygon placement
- [✅] **Results:**
  - Throughput: 3.0s → 1.0s per augmentation (3x speedup)
  - Peak memory: 8GB → 2GB (4x reduction)

---

## Phase 2: Real Dataset Curation & Synthetic Generation

### 2.1 Manual Labeling Sprint
- [ ] **Task:** Annotate 50 images per phone directory (`iphone_11`, `iphone_15`, `realme_c55`, `samsung_a75`) covering lighting, motion blur, and partial occlusion cases.
- [ ] **Tooling:** Use CVAT polygon mode or Labelme; export YOLO segmentation polygons with TL-TR-BR-BL ordering.
- [ ] **Output:** Store images under `1_dataset/labels_manual/images/` and labels under `1_dataset/labels_manual/labels/`.
- [ ] **QA:** Review 10% of annotations with a second pass; document issues in `documents/qa/manual_labeling.md`.

### 2.2 Domain Gap Audit
- [ ] **Task:** Compare manual labels with synthetic augmentations; adjust augmentation parameters when histogram or color drift exceeds tolerance.
- [ ] **Checklist:** Brightness histogram overlap, white balance deltaE < 5, blur kernel estimate, perspective angle distribution.
- [ ] **Deliverables:** Update augmentation presets accordingly and record findings in `documents/reports/domain_gap_summary.md`.

### 2.3 Synthetic Data Generation
- [ ] **Task:** Generate 24,000+ training samples (8,000 base × 3 scales)
- [ ] **Command:**
  ```matlab
  for seed = 1:10
      fprintf('=== Seed %d ===\\n', seed);
      augment_dataset('numAugmentations', 10, ...
                      'rngSeed', seed * 42, ...
                      'multiScale', true, ...
                      'photometricAugmentation', true, ...
                      'independentRotation', false, ...
                      'extremeCasesProbability', 0.10, ...
                      'exportYOLOLabels', true);
  end
  ```
- [ ] **Expected Output:**
  - 80 papers × 10 augmentations × 10 seeds = 8,000 base images
  - 8,000 × 3 scales = 24,000 training samples
- [ ] **Storage:** ~50GB (2MB per image)

## Phase 3: YOLOv11 Training Pipeline

### 3.1 Environment Setup
- [✅] **Task:** Create dedicated Python environment and install Ultralytics YOLOv11.
- [✅] **Implementation:**
  - Created conda environment: `microPAD-python-env` with Python 3.10.19
  - Installed Ultralytics 8.3.221, PyTorch 2.9.0, ONNX tools
  - Created `python_scripts/requirements.txt` for workstation deployment
  - Location: `C:\Users\veyse\miniconda3\envs\microPAD-python-env`
- [✅] **Verified:** `yolo checks` passed (CPU version, GPU version for workstation)

### 3.2 Dataset Configuration
- [✅] **Files Created:**
  - `python_scripts/configs/micropad_synth.yaml`: synthetic-only split
  - `augmented_1_dataset/train.txt`: 126 images (iphone_11, iphone_15, realme_c55)
  - `augmented_1_dataset/val.txt`: 42 images (samsung_a75)
- [✅] **Configuration:**
  ```yaml
  path: C:\Users\veyse\Documents\GitHub\microPAD-colorimetric-analysis\augmented_1_dataset
  train: train.txt
  val: val.txt
  nc: 1
  names: ['concentration_zone']
  ```
- [✅] **Script Created:** `python_scripts/prepare_yolo_dataset.py` for automated setup
- [✅] **Validation Strategy:** Reserved samsung_a75 (1 phone) for validation, 3 phones for training

### 3.3 Training Schedule
- [ ] **Task:** Train synthetic baseline then fine-tune with mixed data.
- [ ] **Commands:**
  ```bash
  # Stage 1: Synthetic pretraining
  yolo segment train model=yolo11n-seg.pt data=configs/micropad_synth.yaml \
      epochs=150 imgsz=640 batch=128 device=0,1 \
      project=micropad_detection name=yolo11n_synth patience=20

  # Stage 2: Fine-tune with real images
  yolo segment train model=micropad_detection/yolo11n_synth/weights/best.pt \
      data=configs/micropad_mixed.yaml epochs=80 imgsz=640 batch=96 device=0,1 \
      project=micropad_detection name=yolo11n_mixed patience=15 lr0=0.01
  ```
- [ ] **Note:** Tune hyperparameters (batch size, epochs, learning rate) during training based on validation curves. Target mask mAP@50 > 0.85.

### 3.4 Export Artifacts
- [ ] **Task:** Export trained weights for MATLAB and Android.
- [ ] **Commands:**
  ```bash
  # ONNX for MATLAB
  yolo export model=micropad_detection/yolo11n_mixed/weights/best.pt \
      format=onnx imgsz=640 simplify=True

  # TFLite for Android (start with FP16, try INT8 if speed insufficient)
  yolo export model=micropad_detection/yolo11n_mixed/weights/best.pt \
      format=tflite imgsz=640 half=True
  ```
- [ ] **Validation:** Load ONNX in MATLAB (`importONNXNetwork`) and run on test image to verify output shape.
- [ ] **Note:** Benchmark inference speed during Phase 4/5 integration. Optimize quantization (INT8) only if FP16 inference >50ms.

## Phase 4: MATLAB Integration

### 4.1 ONNX Inference Wrapper
- [ ] **File:** `matlab_scripts/detect_quads_yolo.m`
- [ ] **Function Signature:**
  ```matlab
  function [quads, confidences] = detect_quads_yolo(img, modelPath, confThreshold)
      % Inputs:
      %   img: RGB image (H×W×3, uint8 or single)
      %   modelPath: Path to YOLO ONNX model (default: 'models/yolo11n_best.onnx')
      %   confThreshold: Minimum confidence (default: 0.5)
      % Outputs:
      %   quads: Detected quadrilaterals (N×4×2) in image coordinates
      %   confidences: Confidence scores per detection (N×1)
  ```
- [ ] **Implementation:** Load ONNX model, resize input to 640×640, run inference, parse segmentation masks, convert to quads via `mask_to_quad.m`
- [ ] **Test:** Run on synthetic and real images, verify detection rate >85%

### 4.2 Post-Processing: Mask-to-Quad Conversion
- [ ] **File:** `matlab_scripts/mask_to_quad.m`
- [ ] **Function Signature:**
  ```matlab
  function [quad, confidence] = mask_to_quad(mask)
      % Convert binary mask to 4-vertex quadrilateral
      % Input: mask (H×W logical or single [0-1])
      % Output: quad (4×2), confidence (scalar [0-1])
  ```
- [ ] **Algorithm:**
  1. Threshold mask if probability map (>0.5)
  2. Morphological cleanup: closing + hole filling
  3. Extract largest contour (`bwboundaries`)
  4. Simplify to 4-6 points (`reducepoly` or `approxPolyDP`)
  5. If 4 points: use directly. If >4: fit min-area rectangle (PCA)
  6. Order clockwise from top-left
  7. Compute confidence from mask quality (area ratio, shape regularity)
  8. Return empty if confidence <0.6

- [ ] **Note:** Implement sub-pixel refinement (Harris corner detector) only if initial testing shows >3px errors

### 4.3 Refactor `cut_concentration_rectangles.m`
- [ ] **File:** `matlab_scripts/cut_concentration_rectangles.m`
- [ ] **Add Parameters:**
  ```matlab
  parser.addParameter('autoDetect', false, @islogical);
  parser.addParameter('detectionModel', 'models/yolo11n_best.onnx', @ischar);
  parser.addParameter('minConfidence', 0.6, @(x) x>=0 && x<=1);
  ```
- [ ] **Modify `getInitialPolygons()` function:**
  ```matlab
  if cfg.autoDetect
      [detectedQuads, confidences] = detect_quads_yolo(img, cfg.detectionModel, cfg.minConfidence);
      if ~isempty(detectedQuads)
          fprintf('Auto-detected %d regions (avg conf: %.2f)\n', size(detectedQuads,1), mean(confidences));
          polygonVertices = detectedQuads;
          return;
      end
      warning('Auto-detection found no regions, falling back to manual mode.');
  end
  % Continue with manual polygon selection...
  ```
- [ ] **Test:** Run on 10 real images, verify auto-detection works or falls back gracefully

## Phase 5: Android Integration

### 5.1 Create Android Project Structure
- [ ] **Task:** Set up Android Studio project
- [ ] **Directory:**
  ```
  android/
  ├── app/
  │   ├── src/main/
  │   │   ├── java/com/micropad/
  │   │   │   ├── QuadDetector.kt
  │   │   │   └── CameraActivity.kt
  │   │   ├── assets/
  │   │   │   └── yolov8n_best_int8.tflite
  │   │   └── res/
  │   └── build.gradle
  └── build.gradle
  ```
- [ ] **Dependencies:** Add to `app/build.gradle`
  ```gradle
  dependencies {
      implementation 'com.github.ultralytics:ultralytics-android:1.0.0'
      implementation 'org.tensorflow:tensorflow-lite:2.13.0'
      implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
      implementation 'androidx.camera:camera-core:1.3.0'
      implementation 'androidx.camera:camera-camera2:1.3.0'
      implementation 'androidx.camera:camera-lifecycle:1.3.0'
  }
  ```

### 5.2 TFLite Inference Engine
- [ ] **File:** `android/app/src/main/java/com/micropad/QuadDetector.kt`
- [ ] **Implementation:**
  ```kotlin
  class QuadDetector(context: Context) {
      private val interpreter: Interpreter

      init {
          val options = Interpreter.Options().apply {
              addDelegate(GpuDelegate())  // Try GPU, fallback to CPU
              setNumThreads(4)
          }
          interpreter = Interpreter(loadModel(context, "yolo11n_best.tflite"), options)
      }

      fun detectQuads(bitmap: Bitmap): List<Quad> {
          val input = preprocessImage(bitmap)  // Resize to 640×640, normalize
          val output = runInference(input)
          val masks = parseMasks(output)
          return masks.map { maskToQuad(it) }.filter { it.confidence > 0.6 }
      }
  }
  ```
- [ ] **Note:** Optimize inference speed during testing. If >50ms, try NNAPI delegate or INT8 quantization
- [ ] **Test:** Benchmark on target Android device, verify detection matches MATLAB output

### 5.3 Camera Integration
- [ ] **File:** `android/app/src/main/java/com/micropad/CameraActivity.kt`
- [ ] **Task:** Real-time camera preview with quad overlay
- [ ] **Features:**
  - CameraX API
  - Real-time inference (every N frames)
  - Quad overlay rendering
  - Capture button (enabled when quads detected)
- [ ] **Test:**
  - [ ] Camera preview works
  - [ ] Quads detected in real-time
  - [ ] Overlay renders correctly
  - [ ] Capture saves image with coordinates


---

## Phase 6: Validation & Deployment

### 6.1 Model Validation
- [ ] **Task:** Validate model performance on real data
- [ ] **Metrics to Verify:**
  - Detection rate: >85% on real test images (n=20 manually labeled)
  - Corner accuracy: 95% of corners within 3px (visual inspection + IoU calculation)
  - MATLAB inference time: <100ms on 4032×3024 images
  - Android inference time: <50ms on target devices (Samsung A75, Realme C55)
- [ ] **Cross-Phone Test:** Run inference on all 4 phone datasets, verify generalization

### 6.2 Deployment
- [ ] **MATLAB Package:**
  - Archive models: `models/yolo11n_micropad/best.onnx`
  - Scripts: `detect_quads_yolo.m`, `mask_to_quad.m`, updated `cut_concentration_rectangles.m`
  - Update `CLAUDE.md` with auto-detect usage examples
- [ ] **Android Release:**
  - Build signed APK with ProGuard
  - Test on minimum 2 physical devices
  - Verify real-time detection works (<50ms inference)

### 6.3 Future Improvements (Optional - Implement if Performance Insufficient)
- [ ] **If detection rate <85%:** Implement test-time augmentation (TTA)
  - Run inference on original + flipped + brightness-adjusted images
  - Ensemble predictions via weighted fusion
  - Expected gain: +2-5% detection rate
- [ ] **If false positive rate >10%:** Hard negative mining
  - Collect false positives from validation set
  - Retrain for 20 epochs with collected negatives
  - Expected gain: -50% false positive rate
- [ ] **For continuous improvement:** Active learning loop
  - Deploy telemetry in Android app
  - Collect low-confidence predictions (<0.7)
  - Manually label 50-100 edge cases per iteration
  - Fine-tune and re-deploy

---

## Progress Tracking

### Overall Status
- [✅] Phase 1: Refactor `augment_dataset.m` (8/8 tasks complete, 100%)
  - [✅] 1.1-1.8 Complete (all tasks finished)
- [⚠️] Phase 2: Dataset Curation & Synthetic Generation (SKIPPED - using existing synthetic data)
  - Phase 2.1-2.2: Deferred (optional manual labeling for fine-tuning)
  - Phase 2.3: Already complete (synthetic data exists in `augmented_1_dataset/`)
- [🔄] Phase 3: YOLOv11 Training (2/4 tasks complete, 50%)
  - [✅] 3.1 Environment Setup (complete)
  - [✅] 3.2 Dataset Configuration (complete)
  - [ ] 3.3 Training Schedule (pending - run on workstation)
  - [ ] 3.4 Export Artifacts (pending)
- [ ] Phase 4: MATLAB Integration (0/3 tasks)
- [ ] Phase 5: Android Integration (0/3 tasks)
- [ ] Phase 6: Validation & Deployment (0/3 tasks)

### Key Milestones
- [✅] Augmentation refactor complete (Phase 1 finished)
- [✅] YOLO label export implemented (Phase 1.3 complete)
- [✅] Python environment setup (Phase 3.1 complete)
- [✅] Dataset configuration ready (Phase 3.2 complete)
- [ ] **NEXT:** Train YOLOv11n-seg on workstation (Phase 3.3)
- [ ] Train YOLOv11n-seg: mask mAP@50 > 0.85
- [ ] Export ONNX/TFLite models
- [ ] MATLAB auto-detect functional
- [ ] Android app with real-time detection

---

## Implementation Notes

### Dataset Structure (After Phase 2)
```
augmented_1_dataset/
├── images/
│   ├── synthetic_001_scale640.jpg
│   ├── synthetic_001_scale800.jpg
│   ├── synthetic_001_scale1024.jpg
│   └── ...
└── labels/
    ├── synthetic_001_scale640.txt
    ├── synthetic_001_scale800.txt
    ├── synthetic_001_scale1024.txt
    └── ...
```

### YOLO Label Format
```
0 x1 y1 x2 y2 x3 y3 x4 y4
```
- `class_id`: 0 (concentration zone only)
- Coordinates: Normalized [0, 1] (divide by image width/height)
- Order: TL, TR, BR, BL (clockwise from top-left)
- Format: Space-separated, one polygon per line

### Model Architecture: YOLOv11n-seg

**Why YOLOv11n-seg over alternatives:**
- **vs YOLOv8:** 22% fewer parameters, better mask precision (83.1%)
- **vs OBB:** Handles irregular quadrilaterals (perspective distortion, paper defects)
- **vs RT-DETR:** Better mobile deployment path, smaller model size
- **vs SAM/MobileSAM:** No prompt required, faster inference, trained end-to-end

**Architecture:**
- Backbone: CSPDarknet with C3k2 blocks (lightweight feature extractor)
- Neck: Enhanced PAN-FPN (multi-scale feature fusion)
- Head: Decoupled detection + segmentation heads
- Output: Instance masks → converted to quadrilateral polygons via post-processing

---

## Critical Success Factors

### Technical Requirements
1. Corner accuracy: 95% within 3 pixels
2. Detection rate: >85% on real images (with manual fallback for remaining 15%)
3. Inference speed: MATLAB <100ms, Android <50ms
4. Model size: <5MB
5. Cross-phone generalization: Works on all 4 phone datasets

### Failure Modes & Mitigation
| Failure Mode | Mitigation Strategy |
|-------------|---------------------|
| Auto-detection fails | Graceful fallback to manual polygon selection |
| Inference too slow | Try INT8 quantization (if FP16 >50ms) |
| Poor generalization | Cross-phone validation, collect more real labels |
| False positives | Hard negative mining (Phase 6.3) |
| Low lighting failures | Already mitigated by augmentation (Phase 1.7 extreme cases) |

---

**Project Lead:** Veysel Y. Yilmaz
**Last Updated:** 2025-10-28
**Version:** 3.1.0 (YOLOv11n-seg, Python Setup Complete)
**Repository:** microPAD-colorimetric-analysis
**Branch:** claude/auto-detect-polygons

**Next Action:** Run Phase 3.3 training on workstation with A6000 GPUs
