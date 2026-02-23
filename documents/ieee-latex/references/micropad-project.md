# microPAD Colorimetric Analysis Project Reference

Project-specific facts, pipeline details, and novelty contributions for writing papers about this codebase. All information extracted from the actual codebase -- not fabricated.

## Table of Contents

1. [Project Identity](#project-identity)
2. [Novelty Contributions](#novelty-contributions)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Feature Extraction System](#feature-extraction-system)
5. [YOLO Detection System](#yolo-detection-system)
6. [Augmentation Pipeline](#augmentation-pipeline)
7. [Available Demo Images](#available-demo-images)
8. [Group Publication History](#group-publication-history)
9. [Suggested Paper Titles](#suggested-paper-titles)

---

## Project Identity

- **Author**: Veysel Yusuf Yilmaz (Y250230002)
- **Program**: M.Sc. Electrical and Electronics Engineering
- **Advisor**: Prof. Dr. Volkan Kilic
- **Institution**: Izmir Katip Celebi University (IKCU), Izmir, Turkey
- **Target analytes**: Urea, Creatinine, Lactate (simultaneous multi-analyte)
- **Sample type**: Biological fluids (sweat, saliva, tears)
- **uPAD design**: 7 test zones per strip, 3 elliptical regions per zone

## Novelty Contributions

When writing Introduction "This work" paragraph or Abstract, frame novelty around these verified contributions that address specific literature gaps:

### 1. YOLO-Based Keypoint Detection for Automatic ROI (Gap 4)
- First application of YOLO pose keypoint detection for uPAD quad vertex localization
- YOLO26-pose model (not bounding box -- keypoints for precise quad corners)
- Desktop: YOLO26s-pose at 1280x1280; Mobile: YOLO26n-pose at 640x640
- Eliminates manual ROI selection dependency
- Novel synthetic data augmentation for training without manual labeling

### 2. White Reference Paper Normalization (Gap 3)
- Paper area surrounding detection zones used as illumination reference
- Features: paper_R, paper_G, paper_B ratios; paper_tempK (estimated color temperature)
- Enhanced normalization: L_corrected, a_corrected, b_corrected via Lab distance
- delta_E_from_paper: perceptually uniform illumination-invariant metric
- No controlled lighting or flash required

### 3. Multi-Analyte Simultaneous Detection (Gap 5)
- 3 analytes per strip: Region 1=Urea, Region 2=Creatinine, Region 3=Lactate
- Training paradigm: all 3 regions = same chemical (replicate measurements)
- Deployment paradigm: each region = different analyte
- 7 concentration levels per analyte (0-6 scale)

### 4. Comprehensive Feature Engineering (Gap 1 - Regression)
- 150+ features across 13 categories (robust preset: ~80 features)
- Color spaces: RGB, HSV, CIE L*a*b*, chromaticity
- Texture: GLCM (correlation, contrast, energy, homogeneity), entropy, gradients
- Spatial: radial profiles, spatial distribution, frequency energy
- Paper normalization: illumination-compensated features
- Designed for regression (continuous concentration prediction)

### 5. Synthetic Data Augmentation for Detector Training (Unique)
- Procedural background generation (5 surface types)
- Physical paper damage simulation (3 damage profiles)
- Photometric augmentation (gamma, white balance, sensor noise, JPEG artifacts)
- Grid-based O(1) collision detection for scene composition
- ~1.0s per augmented image, reproducible via RNG seed

## Pipeline Architecture

### 4-Stage Sequential Pipeline

```
Stage 1: 1_dataset/           Raw smartphone images
    |
    v  cut_micropads.m (+ optional YOLO detection)
    |
Stage 2: 2_micropads/         Cropped paper strips
    |                         coordinates.txt: image conc x1 y1 x2 y2 x3 y3 x4 y4 rotation
    v  cut_micropads.m (perspective correction)
    |
Stage 3: 3_concentration_regions/   Polygonal test zones (con_0 through con_6)
    |                                coordinates.txt: image conc replicate x y semiMajor semiMinor angle
    v  cut_micropads.m (ellipse extraction)
    |
Stage 4: 4_elliptical_regions/     Elliptical patches (3 replicates per zone)
    |
    v  extract_features.m
    |
Output: Excel files with ~80 features per patch + concentration labels
```

### Key Design Principles
- **Stage independence**: Read from N_*, write to (N+1)_*
- **Phone-based organization**: Subdirectories per smartphone model
- **Consolidated coordinates**: Phone-level coordinates.txt files
- **Atomic writes**: Temp file + movefile pattern prevents corruption
- **EXIF handling**: imread_raw() inverts 90-degree EXIF rotations

### Smartphone Models in Dataset

| Model | OS | Resolution | Role |
|-------|-----|-----------|------|
| iPhone 11 | iOS | 4032x3024 | Training |
| iPhone 15 | iOS | 4032x3024 | Training |
| Samsung A75 | Android | 4000x3000 | Validation |
| Realme C55 | Android | 4000x3000 | Training |

## Feature Extraction System

### Feature Presets

| Preset | Count | Use Case |
|--------|-------|----------|
| minimal | ~30 | Quick prototyping |
| robust | ~80 | Recommended for papers |
| full | ~150+ | Exploratory/ablation studies |
| custom | Variable | User-defined selection |

### Feature Categories (Robust Preset)

1. **RGB/HSV/Lab basics**: R, G, B, H, S, V, L, a, b (median, IQR)
2. **Color ratios**: RG_ratio, RB_ratio, GB_ratio (lighting-invariant)
3. **Chromaticity**: r/g_chromaticity, chroma_magnitude, dominant_chroma
4. **Color uniformity**: RGB_CV, saturation/value/L/chroma_uniformity
5. **Illuminant invariant**: ab_magnitude, ab_angle, hue_circular_mean
6. **Texture (GLCM)**: stripe_correlation, contrast, energy, homogeneity
7. **Entropy + gradients**: entropyValue, L/a/b_gradient_mean/std/max
8. **Robust statistics**: Median and IQR for L*, a*, b*, S, V
9. **Spatial distribution**: L/a/b_spatial_std, radial_L_gradient, spatial_uniformity
10. **Radial profile**: radial_L_inner/outer/ratio, radial_chroma/saturation_slope
11. **Concentration metrics**: saturation_range, chroma_intensity/max, Lab ranges
12. **Frequency energy**: fft_low_energy, fft_high_energy, fft_band_contrast
13. **Paper normalization**: R/G/B_paper_ratio, L/a/b_corrected, delta_E_from_paper

### MATLAB Commands
```matlab
extract_features('chemical', 'lactate', 'preset', 'robust')
extract_features('trainTestSplit', true, 'testSize', 0.25)
```

## YOLO Detection System

### Model Architecture
- **Model**: YOLO26-pose (keypoint detection, not bounding box)
- **Task**: Detect 4 quad vertices of each concentration zone
- **Label format**: `class_id x1 y1 vis1 x2 y2 vis2 x3 y3 vis3 x4 y4 vis4` (normalized)
- **Vertex order**: Clockwise from top-left

### Training Configuration
- Desktop: YOLO26s-pose, 1280x1280, 200 epochs, AdamW, batch 32
- Mobile: YOLO26n-pose, 640x640 (for TFLite deployment)
- Patience: 20 (early stopping)
- Target: mAP@50 > 0.85, detection rate > 85%

### MATLAB Integration
```matlab
cut_micropads('useAIDetection', true)                          % Desktop
cut_micropads('useAIDetection', true, 'useMobileModel', true)  % Mobile
```

### YOLO26 Advantages (Over YOLOv11)
- 43% faster CPU inference (critical for mobile)
- RLE (Residual Log-Likelihood Estimation) for accurate keypoints
- MuSGD optimizer option (hybrid SGD/Muon)
- NMS-free end-to-end inference

## Augmentation Pipeline

### Two-Stage Architecture

**Stage 1 -- MATLAB (augment_dataset.m)**: Unique augmentations
- Geometric: perspective transforms (+-60 pitch/yaw), rotation, scaling
- 5 procedural backgrounds: uniform, speckled, laminate, skin, texture pooling
- Photometric: gamma (0.92-1.08), white balance jitter, sensor noise, JPEG artifacts
- Physical: paper damage (3 profiles: minimalWarp 30%, cornerChew 45%, sideCollapse 25%)
- Occlusions: lines, blobs, fingers
- Stains, specular highlights, edge feathering (50% hard/soft)

**Stage 2 -- YOLO runtime (train_yolo.py)**: Standard augmentations
- HSV jitter: hue +-1.5%, sat +-70%, val +-40%
- Mosaic: 100%, Scale: +-50%, Translate: +-10%
- Horizontal flip: 50%, Random erasing: 40%

### Performance
- ~1.0s per augmented image (3x speedup from v1)
- Grid-based O(1) collision detection
- Background texture pooling (16 cached variants per type)

### MATLAB Commands
```matlab
augment_dataset('numAugmentations', 5, 'rngSeed', 42)
augment_dataset('numAugmentations', 10, 'photometricAugmentation', true)
```

## Available Demo Images

Located in `demo_images/` -- copy to `figures/` for papers:

| File | Shows | Use For |
|------|-------|---------|
| stage1_original_image.jpeg | Raw smartphone photo | System overview figure |
| stage2_micropad.jpeg | Cropped paper strip | Pipeline stage 2 |
| stage3_elliptical_region_1.jpeg | Individual elliptical patch | Pipeline stage 3/4 |
| white_referenced_pixels_on_rectangle.png | White reference strategy | Methods figure |
| augmented_dataset_1.jpg | Full synthetic scene | Augmentation figure |
| augmented_micropad_1.jpeg | Augmented concentration region | Augmentation detail |
| augmented_micropad_2.jpeg | Additional augmented example | Augmentation detail |
| augmented_elliptical_region1.jpeg | Augmented ellipse patch | Augmentation detail |
| augmented_elliptical_region_2.jpeg | Additional augmented ellipse | Augmentation detail |

**Concentration series** (for color change grid figure):
- `3_concentration_regions/iphone_11/con_0/` through `con_6/`
- Shows actual color gradient across 7 concentration levels

## Group Publication History

15 papers (2016-2024) in `documents/previous pubs/`. Key evolution:

| Year | Topic | Significance |
|------|-------|-------------|
| 2016 | Dye detection in water | First smartphone colorimetry work |
| 2017 | ML-based colorimetric detection | First ML classifiers (SVM, RF, KNN) |
| 2018 | Bisphenol-A, water quality, quantification | Multiple applications, portable spectrometer |
| 2021 | Offline platform, ML glucose, non-enzymatic | Firebase, TFLite, ML on uPADs |
| 2022 | DL lactate, H2O2, food spoilage, glucose | Deep learning transition, embedded models |
| 2024 | AI regression multi-analyte, food spoilage | Regression approach, ChemiCheck app |

**Key self-citations** (always include in bibliography):
- Basturk et al. 2024 (regression, multi-analyte, ChemiCheck)
- Yuzer et al. 2022 (DL lactate, smartphone embedded)
- Mercan et al. 2021 (ML glucose, uPAD, different reagents)
- Mutlu et al. 2017 (first smartphone ML colorimetry)

## Suggested Paper Titles

Based on group's naming patterns and project novelty:

1. "YOLO-Based Automatic Region of Interest Detection for Smartphone Colorimetric Analysis with Microfluidic Paper-Based Analytical Devices"
2. "Smartphone-Embedded Deep Learning for Illumination-Invariant Colorimetric Quantification of Multiple Analytes Using $\mu$PADs"
3. "Automatic Keypoint Detection and White Reference Normalization for Robust Smartphone-Based Colorimetric Sensing on $\mu$PADs"
4. "AI-Enhanced Smartphone Platform with Synthetic Data Augmentation for Colorimetric Multi-Analyte Detection in Biological Fluids"

Pattern: `[Technology]-Based [AI Method] for [Task] of [Analytes] with [Device] in [Sample]`
