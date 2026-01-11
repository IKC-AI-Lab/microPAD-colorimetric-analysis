# microPAD colorimetric analysis

A MATLAB-based colorimetric analysis pipeline for microfluidic paper-based analytical devices (microPADs). Processes smartphone images of test strips to extract color features and predict biomarker concentrations (urea, creatinine, lactate) using machine learning.

![Pipeline Overview](demo_images/stage1_original_image.jpeg)

## Table of Contents
- [Requirements](#requirements)
- [Dataset Structure](#dataset-structure)
- [Pipeline Stages](#pipeline-stages)
- [Quick Start](#quick-start)
- [Data Augmentation](#data-augmentation)
- [Helper Scripts](#helper-scripts)
- [Directory Layout](#directory-layout)
- [Tips and Troubleshooting](#tips-and-troubleshooting)
- [Output Example](#output-example)
- [Future Work: Android Smartphone Application](#future-work-android-smartphone-application)

---

## Requirements

- **MATLAB R2020a or later** with Image Processing Toolbox
- Run scripts from the project folder or `matlab_scripts/` folder

---

## Dataset Structure

### Phone Models & Lighting Conditions

The dataset is organized by **phone model** (e.g., `iphone_11/`, `samsung_a75/`, `realme_c55/`). Each phone captures **7 images per microPAD**, representing different lighting combinations using 3 laboratory lamps:

| Lighting ID | Lamp 1 | Lamp 2 | Lamp 3 | Description |
|-------------|--------|--------|--------|-------------|
| **light_1** | ✓ | ✗ | ✗ | Only lamp 1 |
| **light_2** | ✗ | ✓ | ✗ | Only lamp 2 |
| **light_3** | ✗ | ✗ | ✓ | Only lamp 3 |
| **light_4** | ✓ | ✓ | ✗ | Lamps 1 + 2 |
| **light_5** | ✓ | ✗ | ✓ | Lamps 1 + 3 |
| **light_6** | ✗ | ✓ | ✓ | Lamps 2 + 3 |
| **light_7** | ✓ | ✓ | ✓ | All lamps |

This lighting variation helps the model work well under different real-world lighting conditions.

### microPAD Structure

Each microPAD paper strip contains **7 test zones** (concentrations), and each test zone has **3 elliptical measurement regions**:

- **Final design:** The 3 regions will contain different chemicals (urea, creatinine, lactate)
- **Training phase:** All 3 regions contain the same chemical at the same concentration, providing 3 replicate measurements per concentration level
- This allows separate datasets to be collected for training machine learning models for each chemical

---

## Pipeline Stages

The pipeline processes images through **4 sequential stages**. Each stage reads from folder `N_*` and writes to `(N+1)_*`:

### **Stage 1 → 2: Cut microPAD Concentration Regions**

**Script:** `cut_micropads.m`

**Input:** Raw smartphone photos in `1_dataset/{phone_model}/`
**Output:** Individual concentration rectangles in `2_micropads/{phone_model}/con_{N}/`

![Stage 1 to 2](demo_images/stage2_micropad.jpeg)

Processes raw smartphone images in a single step: applies optional rotation, uses AI-powered quad detection (YOLOv11 pose keypoints) to locate test zones, and crops individual concentration regions. Combines the functionality of the previous two-step process into one streamlined workflow.

**Key Features:**
- AI-based auto-detection of test zones (YOLOv11s pose keypoints)
- Graceful fallback to manual quad selection if auto-detection fails
- Interactive rotation control with cumulative rotation memory
- Saves 10-column coordinate format: `image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation`

---

### **Stage 2 → 3: Extract Elliptical Patches**

**Script:** `cut_elliptical_regions.m`

**Input:** Concentration rectangles from `2_micropads/`
**Output:** Elliptical patches in `3_elliptical_regions/{phone_model}/con_{N}/`

![Stage 2 to 3](demo_images/stage3_elliptical_region_1.jpeg)

Extracts three elliptical measurement regions from each test zone. In the final microPAD design, these three regions will contain different chemicals (urea, creatinine, lactate). For training purposes, each experiment fills all three regions with the same concentration of a single chemical, providing three replicate measurements per concentration level.

---

### **Stage 3 → 4: Feature Extraction**

**Script:** `extract_features.m`

**Input:**
- Concentration rectangles from `2_micropads/` (used as paper/white reference)
- Elliptical patches from `3_elliptical_regions/`

**Output:** Excel feature tables in `4_extract_features/`

![White Reference Strategy](demo_images/white_referenced_pixels_on_rectangle.png)

Extracts 80+ colorimetric features from elliptical test zones while compensating for varying lighting conditions using the white paper as a reference. This enables robust biomarker concentration prediction under different smartphone cameras and lighting environments.

---

#### **White Reference Strategy**

The pipeline automatically samples white paper pixels **outside the three elliptical test regions** within each concentration rectangle (shown in green above). These reference pixels provide:

1. **Illuminant estimation** - Determines color temperature of the lighting (2500-10000K)
2. **Chromatic adaptation** - Normalizes color channels to compensate for lighting color cast
3. **Reflectance calculation** - Converts RGB values to relative reflectance ratios
4. **Delta-E baseline** - Measures color difference from paper white point in perceptually-uniform Lab space

This approach makes features **lighting-invariant** across different phone models and laboratory conditions.

---

#### **Available Presets**

```matlab
extract_features('preset','robust','chemical','lactate')  % Recommended
extract_features('preset','minimal')  % Fast, essential features only
extract_features('preset','full')     % All 150+ features
extract_features('preset','custom')   % Interactive feature group selection dialog
```

| Preset | Feature Count | Description |
|--------|---------------|-------------|
| **minimal** | ~30 | Essential color ratios and basic statistics |
| **robust** | ~80 | Comprehensive set balancing accuracy and speed (recommended) |
| **full** | ~150 | All available features including advanced texture analysis |
| **custom** | Variable | User-defined via interactive dialog or struct |

---

#### **Extracted Features (Robust Preset)**

**Color Normalization (Paper-Referenced)**
- RGB ratios relative to paper white point (R/R_paper, G/G_paper, B/B_paper)
- Lab color corrected for paper baseline (L*, a*, b* shifts)
- Delta-E color difference from paper (perceptually uniform)
- Chromatic adaptation factors (von Kries-style)

**Lighting-Invariant Color Ratios**
- Red/Green, Red/Blue, Green/Blue channel ratios
- Hue-based features immune to brightness changes
- Normalized chroma and saturation metrics

**Color Space Statistics**
- RGB: Mean, median, standard deviation per channel
- HSV: Hue, saturation, value distributions
- Lab: Lightness (L*), red-green (a*), blue-yellow (b*) statistics

**Texture and Spatial Features**
- Edge sharpness (gradient magnitude)
- Color uniformity (coefficient of variation)
- Spatial frequency patterns (FFT-based)
- Local color gradients

**Advanced Colorimetry**
- Estimated illuminant color temperature (Kelvin)
- Color purity and dominance wavelength
- Entropy and histogram statistics

---

#### **Output Format**

Excel files with one row per elliptical measurement (replicate):

| PhoneType | ImageName | RG_ratio | delta_E_from_paper | R_paper_ratio | Label |
|-----------|-----------|----------|-------------------|---------------|-------|
| iphone_11 | IMG_0957_aug_000_con_0.png | 0.461 | 32.04 | 0.723 | 0 |
| iphone_11 | IMG_0957_aug_000_con_0.png | 0.460 | 31.51 | 0.719 | 0 |
| iphone_11 | IMG_0957_aug_000_con_0.png | 0.462 | 32.23 | 0.728 | 0 |

- Each concentration zone contributes **3 rows** (3 elliptical replicates)
- `Label` column = known concentration (0-6) for supervised learning
- Automatically exports train/test splits (70/30 default, stratified by image)

---

#### **Parameters**

```matlab
% Chemical name (used in output filename)
extract_features('chemical', 'urea')

% White reference override (if auto-detection fails)
extract_features('paperTempK', 6000)

% Train/test split control
extract_features('trainTestSplit', true, 'testSize', 0.25, 'randomSeed', 42)

% Custom feature selection without dialog
customFeatures = struct('ColorRatios', true, 'PaperNormalization', true);
extract_features('preset', 'custom', 'features', customFeatures, 'useDialog', false)
```

---

#### **Machine Learning Integration**

The extracted features are designed for training AI models that will:
1. **Predict biomarker concentrations** from smartphone images
2. **Run on Android smartphones** via embedded TensorFlow Lite models
3. **Auto-detect test zones** using YOLOv11 segmentation (already integrated in MATLAB pipeline)

**Typical ML workflow:**
```matlab
% 1. Extract features in MATLAB
extract_features('preset','robust','chemical','lactate')

% 2. Load train/test splits in Python
train_df = pd.read_excel('4_extract_features/robust_lactate_train_features.xlsx')
test_df = pd.read_excel('4_extract_features/robust_lactate_test_features.xlsx')

% 3. Train regression model
X_train = train_df.drop(['PhoneType','ImageName','Label'], axis=1)
y_train = train_df['Label']
model = RandomForestRegressor()
model.fit(X_train, y_train)

% 4. Export for Android deployment
# Convert to TensorFlow Lite for smartphone inference
```

---

## Quick Start

Run these commands **in order** from the project folder:

### Full Pipeline (from raw images)

```bash
# Stage 1→2: Cut microPAD concentration regions (with AI detection)
matlab -batch "addpath('matlab_scripts'); cut_micropads;"

# Stage 2→3: Extract elliptical regions
matlab -batch "addpath('matlab_scripts'); cut_elliptical_regions;"

# Stage 3→4: Extract features (change 'lactate' to your chemical name)
matlab -batch "addpath('matlab_scripts'); extract_features('preset','robust','chemical','lactate');"
```

### Typical Workflow (starting from Stage 2)

If you already have concentration rectangles in `2_micropads/`:

```bash
matlab -batch "addpath('matlab_scripts'); cut_elliptical_regions;"
matlab -batch "addpath('matlab_scripts'); extract_features('preset','robust','chemical','lactate');"
```

---

## Data Augmentation

**Script:** `augment_dataset.m` (optional, recommended for AI training)

Generates synthetic training data by transforming real microPAD images and concentration quadrilaterals and composing them onto procedural backgrounds. Essential for training robust quad detection models for the Android smartphone application.

![Augmented Dataset](demo_images/augmented_dataset_1.png)
*Synthetic scene with transformed microPAD under simulated lighting*

![Augmented microPAD](demo_images/augmented_micropad_1.png)
*Augmented concentration region with perspective distortion*

![Augmented Ellipse](demo_images/augmented_elliptical_region1.png)
*Augmented elliptical patch preserving colorimetric properties*

---

### **Why Augmentation is Critical**

The final Android application will use AI-based **auto-detection** to locate test zones in smartphone photos. Augmentation **multiplies the training dataset** by 5-10x without requiring additional physical experiments, generating diverse viewpoints, lighting conditions, and backgrounds.

**MATLAB augmentation** creates synthetic image transformations following the standard 1→2→3 pipeline structure. **Python scripts** then prepare YOLO training labels from the MATLAB-generated coordinates.

---

### **Transformation Pipeline**

**Geometric Transformations (per region)**
1. **3D perspective projection** - Simulates camera viewing angles (+/-60 deg pitch/yaw)
2. **Rotation** - Random orientation (0-360 deg)
3. **Spatial placement** - Random non-overlapping positions via grid acceleration
4. **Optional per-region rotation** - Independent orientation for each concentration zone

**Photometric Augmentation**
- **Brightness/contrast** - Global illumination variation (±5-10%)
- **White balance jitter** - Per-channel gains simulating different lighting (0.92-1.08x)
- **Saturation adjustment** - Color intensity variation (0.94-1.06x)
- **Gamma correction** - Exposure simulation (0.92-1.08)

**Background Generation (Procedural)**
- **Uniform surfaces** - Clean laboratory benches (220 +/- 15 RGB)
- **Speckled textures** - Granite, composite materials
- **Laminate** - High-contrast white/black surfaces (245 or 30 RGB)
- **Skin tones** - Human hand/arm backgrounds (HSV-based)
- **Texture pooling** - 16 cached variants per surface type with random shifts, flips, and scale jitter to avoid regeneration artifacts

**Distractor Artifacts (5-40 per image)**
- **Shapes**: Ellipses, rectangles, quadrilaterals, triangles, lines
- **Sizes**: 1-75% of image diagonal (allows partial occlusions)
- **Placement**: Unconstrained (can extend beyond frame)
- **Sharpness**: Sharp by default, matching concentration rectangle behavior
- **Rendering**: Nearest-neighbor interpolation to preserve crisp edges
- **Purpose**: Train quad detector to ignore false positives while maintaining realistic appearance

**Distractor Quads (Synthetic Look-Alikes)**
- **Count**: Random 1-10 per image (independent of real quad count)
- **Size Variation**: Each distractor randomly scaled 50%-150% of source quad size
- **Appearance**: Synthesized from real quads with jittered colors and textures
- **Placement**: Non-overlapping collision detection, respects spacing constraints
- **Purpose**: Train AI to distinguish real test zones from similar-looking objects by position/context rather than appearance alone

**Blur and Occlusions (Optional)**
- **Scene-wide blur** - Applied to entire image (quads + artifacts) when enabled
- **Motion blur** - Camera shake simulation (15% probability)
- **Gaussian blur** - Focus variation (25% probability, sigma 0.25-0.65px)
- **Thin occlusions** - Hair/strap-like artifacts across test zones (disabled by default)

**Paper Damage Augmentation (Physical Defects)**
- **Damage probability** - 50% of samples show realistic paper wear
- **Three damage profiles:**
  - `minimalWarp` (30%): Subtle warping from handling (projective jitter + edge bending)
  - `cornerChew` (45%): Corner clips, tears, and edge fraying
  - `sideCollapse` (25%): Heavy side damage (bites, tapered edges)
- **Protected regions** - Ellipse measurement zones are never damaged
- **Three-phase pipeline:**
  - Phase 1: Base warp & shear (projective jitter, nonlinear edge bending)
  - Phase 2: Structural cuts (corner clips, tears, side bites, tapered edges)
  - Phase 3: Edge wear & thickness (wave noise, fraying, shadows)
- **Realistic effects:**
  - **Corner clips** - Triangular cuts from 1-3 corners (6-22% of edge length)
  - **Corner tears** - Jagged irregular cuts with 4-6 vertices and random jitter
  - **Side bites** - Circular concave cuts from folding/mishandling (8-28% of edge)
  - **Tapered edges** - Diagonal cuts creating trapezoid shapes (10-30% taper)
  - **Edge wave noise** - Sinusoidal perimeter distortion (1.5-3 cycles, 1-5% amplitude)
  - **Fiber fraying** - Micro-chipping along cut edges (1-3px blobs)
  - **Thickness shadows** - Depth cues at removal boundaries (15% darkening, 8px fade)
- **Guard mechanisms:**
  - Ellipse zones + 12px margin protected by dilation
  - Bridge paths connecting ellipses to centroid (prevent isolated regions)
  - Core quad region (60% inner area) enforces maxAreaRemoval limit
- **Purpose**: Train AI to handle real-world paper degradation from storage, handling, and environmental factors

![Damaged microPAD Example](demo_images/damaged_micropad_cornerchew.png)
*Example: cornerChew profile with corner tear + side bite preserving ellipse regions*

---

### **Usage Examples**

```matlab
% Basic usage: 5 augmented versions per paper (6 total with original)
augment_dataset('numAugmentations', 5, 'rngSeed', 42)

% High-augmentation mode for deep learning (10x data)
augment_dataset('numAugmentations', 10, 'photometricAugmentation', true)

% Enable all augmentation features
augment_dataset('numAugmentations', 5, ...
                'photometricAugmentation', true, ...
                'blurProbability', 0.30, ...
                'motionBlurProbability', 0.20, ...
                'occlusionProbability', 0.15, ...
                'independentRotation', true)

% Enable paper damage with higher probability
augment_dataset('numAugmentations', 5, ...
                'paperDamageProbability', 0.7, ...
                'damageSeed', 42)

% Customize damage profile weights (prefer minimal damage)
weights = struct('minimalWarp', 0.60, 'cornerChew', 0.30, 'sideCollapse', 0.10);
augment_dataset('numAugmentations', 5, ...
                'damageProfileWeights', weights, ...
                'maxAreaRemovalFraction', 0.30)

% Fast mode: disable expensive features
augment_dataset('numAugmentations', 3, ...
                'photometricAugmentation', false, ...
                'independentRotation', false)

% Reproducible augmentation for ablation studies
augment_dataset('numAugmentations', 5, 'rngSeed', 12345)

% Emit JSON labels for CornerNet-style training
augment_dataset('numAugmentations', 5, 'exportCornerLabels', true)
```

---

### **Advanced: Paper Damage Configuration**

Fine-tune damage realism by overriding default parameters:

```matlab
% Increase damage frequency
augment_dataset('paperDamageProbability', 0.7)  % 70% of samples damaged

% Prefer minimal damage (subtle warping only)
weights = struct('minimalWarp', 0.60, 'cornerChew', 0.30, 'sideCollapse', 0.10);
augment_dataset('damageProfileWeights', weights)

% Allow more aggressive cuts (up to 50% removal per ellipse/micropad)
augment_dataset('maxAreaRemovalFraction', 0.50)

% Deterministic damage for ablation studies
augment_dataset('damageSeed', 42)
```

**Damage Parameters:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `paperDamageProbability` | 0.5 | 0-1 | Fraction of quads with damage |
| `damageProfileWeights` | See below | Sums to 1.0 | Profile selection probabilities |
| `maxAreaRemovalFraction` | 0.40 | 0-1 | Max removable fraction (per ellipse or per micropad fallback) |
| `damageSeed` | (random) | Any integer | RNG seed for reproducible damage |

**Default Profile Weights:**
- `minimalWarp`: 30% (subtle handling distortion only)
- `cornerChew`: 45% (corner damage + cuts + wear)
- `sideCollapse`: 25% (heavy side damage, bites dominate)

**Damage Operation Ranges (Advanced):**

These internal constants control damage severity and can be found in `PAPER_DAMAGE` struct (lines 130-141):

| Operation | Parameter | Default Range | Effect |
|-----------|-----------|---------------|--------|
| Corner clips | `cornerClipRange` | [0.06, 0.22] | Fraction of shorter edge |
| Side bites | `sideBiteRange` | [0.08, 0.28] | Fraction of side length |
| Tapered edges | `taperStrengthRange` | [0.10, 0.30] | Fraction of perpendicular dim |
| Edge waves | `edgeWaveAmplitudeRange` | [0.01, 0.05] | Fraction of min dimension |
| Edge waves | `edgeWaveFrequencyRange` | [1.5, 3.0] | Cycles along perimeter |
| Max operations | `maxOperations` | 3 | Max structural cuts per quad |

---

### **Performance**

**Speed**: ~1.0 second per augmented image (3x faster than v1)
- Grid-based spatial acceleration (O(1) collision detection vs O(n^2))
- Simplified quad warping (nearest-neighbor vs bilinear)
- Background texture pooling (reuses 4 procedural types with cached surfaces instead of regenerating each frame)
- ROI-based meshgrid allocation for ellipse protection (5-10× memory reduction)
- Fast approximate signed distance for edge noise (3-5× faster than bwdist)
- Precomputed profile sampling arrays (eliminates fieldnames() overhead)
- Size-based bypass for small quads (<200px, <150px for edge noise)

**Memory**: Low overhead
- Processes one paper at a time
- Temporary buffers released after each scene

**Scalability**: Handles large datasets
- Automatic background expansion if quads don't fit
- Graceful degradation on positioning failures

---

### **Input/Output Structure**

**Inputs:**
- `1_dataset/{phone}/` - Original smartphone images
- `2_micropads/{phone}/coordinates.txt` - Quad vertices (required)
- `3_elliptical_regions/{phone}/coordinates.txt` - Ellipse parameters (optional)

**Outputs:**
- `augmented_1_dataset/{phone}/` - Full synthetic scenes
- `augmented_2_micropads/{phone}/con_{N}/` - Transformed concentration regions + coordinates.txt
- `augmented_3_elliptical_regions/{phone}/con_{N}/` - Transformed elliptical patches + coordinates.txt (if input ellipses exist)

**YOLO Training Labels:** Generated by Python scripts from MATLAB coordinates (see Python pipeline documentation)

**Naming convention:**
- Original: `paper_name_aug_000.png` (identity transformations)
- Augmented: `paper_name_aug_001.png`, `paper_name_aug_002.png`, etc.

---

### **Integration with ML Pipeline**

**For Quad Detection (YOLO, Faster R-CNN)**
```python
# 1. Generate augmented data in MATLAB
augment_dataset('numAugmentations', 5)

# 2. Prepare YOLO dataset with Python (reads MATLAB coordinates, creates labels)
python python_scripts/prepare_yolo_dataset.py

# 3. Train model (desktop: yolo11s @ 1280px, mobile: yolo11n @ 640px)
python python_scripts/train_yolo.py           # Desktop model
python python_scripts/train_yolo.py --mobile  # Mobile model
```

**For Concentration Prediction (Regression Models)**
```matlab
% Extract features from augmented data
extract_features('preset','robust','chemical','lactate')
% Output: 4_extract_features/ with (N+1) × original sample count
```

**For Android Deployment**
1. Train quad detector on `augmented_1_dataset/`
2. Export to TensorFlow Lite (.tflite)
3. Train concentration predictor on features from augmented ellipses
4. Embed both models in Android app

---

### **Important Notes**

**Ellipse coordinate propagation:**
- If `3_elliptical_regions/coordinates.txt` is missing, augmentation still runs but only produces stages 1-2
- To generate augmented ellipses without pre-existing coordinates, run `cut_elliptical_regions.m` on `augmented_2_micropads/` after augmentation

**Coordinate preservation:**
- All transformations (perspective, rotation, translation) are recorded in output coordinates.txt files
- Ellipse transformations use conic section mathematics to preserve accuracy
- Coordinates are validated (degenerate ellipses/quads are skipped with warnings)

**Quality control:**
- Original version (aug_000) uses identity transformations for debugging
- Photometric augmentation is color-safe (preserves relative hue relationships)
- At most one blur type is applied per image (prevents over-softening)

---

## Helper Scripts

Three utility scripts in `matlab_scripts/helper_scripts/` for recreating images and checking quality:

### **extract_images_from_coordinates.m**

Recreates all processed images from `coordinates.txt` files. Instead of saving thousands of image files, you only need to keep small text files with coordinates.

**Why this matters:**
- You don't need to save processed images (stages 2-4) - just keep the `coordinates.txt` files
- Anyone can recreate the exact same images from the coordinates
- Saves storage space (gigabytes → kilobytes)

**Usage:**
```matlab
addpath('matlab_scripts/helper_scripts');
extract_images_from_coordinates();
```

Run this script to recreate any missing processed images from your saved coordinates.

---

### **preview_overlays.m**

Opens a viewer that shows how well your coordinates match the actual images. Displays quads and ellipses overlaid on the original photos.

**Use this to:**
- Check if your annotations are accurate
- Find mistakes before extracting features
- Verify your work after editing `coordinates.txt` files

**Usage:**
```matlab
addpath('matlab_scripts/helper_scripts');
preview_overlays();  % Press 'n' to navigate, 'q' to quit
```

---

### **preview_augmented_overlays.m**

Same as `preview_overlays.m`, but for checking the augmented (synthetic) dataset quality.

**Usage:**
```matlab
addpath('matlab_scripts/helper_scripts');
preview_augmented_overlays();
```

---

## Directory Layout

```
microPAD-colorimetric-analysis/
├── matlab_scripts/          # Main processing scripts
│   ├── cut_micropads.m      # Stage 1→2 (AI detection + cropping)
│   ├── cut_elliptical_regions.m  # Stage 2→3
│   ├── extract_features.m   # Stage 3→4
│   ├── augment_dataset.m    # Data augmentation (optional)
│   └── helper_scripts/      # Utility functions
├── 1_dataset/               # Raw smartphone photos
│   ├── iphone_11/
│   ├── iphone_15/
│   ├── realme_c55/
│   └── samsung_a75/
├── 2_micropads/             # Concentration regions + coordinates.txt
├── 3_elliptical_regions/    # Elliptical patches + coordinates.txt
├── 4_extract_features/      # Feature tables (.xlsx)
├── augmented_1_dataset/     # (Optional) Synthetic scenes
├── augmented_2_micropads/
├── augmented_3_elliptical_regions/
└── demo_images/             # Visual examples for documentation
```

**Note:** The processed images are not saved to version control (only the `coordinates.txt` files are saved).

---

## Tips and Troubleshooting

### Coordinate Files

Each stage saves a `coordinates.txt` file with position and shape information:
- **Stage 2 (2_micropads):** Corner points (4 vertices) + rotation for each test zone
- **Stage 3 (3_elliptical_regions):** Ellipse center, semi-major/minor axes, and rotation angle

**If corrupted:** Delete the file and re-run the stage to create a new one.

### Memory Issues

If MATLAB runs out of memory, process fewer images at once:

```matlab
extract_features('preset', 'robust', 'batchSize', 10)
```

### Running from the Wrong Folder

If you see warnings about missing folders, make sure you're running scripts from:
- The main project folder, OR
- The `matlab_scripts/` folder

### Testing Before Full Processing

Before processing all images, test with 2-3 sample images to make sure everything works correctly.

---

## Output Example

Sample rows from `4_extract_features/robust_lactate_features.xlsx` (showing subset of 80+ feature columns):

| PhoneType  | ImageName                   | RG_ratio | RB_ratio | GB_ratio | L       | a       | b      | delta_E_from_paper | R_paper_ratio | Label |
|------------|-----------------------------|----------|----------|----------|---------|---------|--------|--------------------|---------------|-------|
| iphone_11  | IMG_0957_aug_000_con_0.png | 0.461449 | 0.360812 | 0.781910 | 53.9623 | 10.3421 | 18.952 | 32.0361            | 0.7234        | 0     |
| iphone_11  | IMG_0957_aug_000_con_0.png | 0.460265 | 0.365116 | 0.793272 | 54.0647 | 10.1204 | 19.124 | 31.5070            | 0.7189        | 0     |
| iphone_15  | IMG_1234_aug_000_con_3.png | 0.483148 | 0.388192 | 0.803463 | 55.1563 | 11.8934 | 20.456 | 30.2742            | 0.7512        | 3     |
| realme_c55 | IMG_5678_aug_000_con_5.png | 0.492301 | 0.395421 | 0.810234 | 56.2341 | 12.4567 | 21.345 | 29.1234            | 0.7634        | 5     |
| samsung_a75| IMG_9012_aug_000_con_6.png | 0.501234 | 0.402345 | 0.821345 | 57.3456 | 13.2345 | 22.456 | 28.3456            | 0.7789        | 6     |

**Notes:**
- Each row represents one elliptical region measurement (replicate)
- Multiple rows with the same concentration indicate replicate measurements from the 3 elliptical regions within that test zone
- `Label` column shows the known concentration (0-6) for training the model
- During training, all 3 replicates per test zone contain the same chemical at the same concentration
- Full table has 80+ columns (only some shown above)
- Can split data into separate training and testing files

---

## Future Work: Android Smartphone Application

This MATLAB pipeline serves as the **data preparation and training infrastructure** for an Android smartphone application that will:

1. **Capture microPAD photos** using the smartphone camera
2. **Auto-detect test zones** using quad detection AI (YOLOv11 pose keypoints - already integrated in MATLAB)
3. **Predict biomarker concentrations** (urea, creatinine, lactate) using regression models (trained on features from `4_extract_features/`)
4. **Display results** to the user in real-time

### **AI Model Training Workflow**

```
MATLAB Pipeline (this repository)
    ↓
augmented_1_dataset/ → Train quad detector (YOLOv11s pose keypoints)
    ↓
4_extract_features/ → Train concentration predictor (Random Forest/XGBoost)
    ↓
Export models (PyTorch for MATLAB, TFLite for Android)
    ↓
Android Application (separate repository, coming soon)
```

### **Key Features of Android App**

- **Real-time detection** - Auto-locate test zones in live camera feed (using YOLOv11s-pose model)
- **Lighting compensation** - White reference strategy (same as MATLAB pipeline)
- **Multi-biomarker support** - Separate models for urea, creatinine, lactate
- **Offline inference** - Embedded TensorFlow Lite models (no internet required)
- **Result history** - Save and track measurements over time

### **Current Status**

✅ **Completed**: MATLAB data preparation pipeline (4-stage pipeline)
✅ **Completed**: AI quad detection training (YOLOv11s-pose)
✅ **Completed**: MATLAB integration with graceful fallback to manual selection
📋 **Planned**: Android application development with TFLite deployment

Stay tuned for the Android app repository link!
