# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Orchestration and Workflow

### Hybrid Task Delegation

**Implement directly (Claude):**
- Single-file edits, bug fixes with clear scope
- Parameter changes, simple additions (<5 steps)
- Quick fixes after code review

**Delegate to coder agents:**
- Multi-file refactors (3+ files)
- New features spanning multiple functions
- Complex algorithmic changes
- Cross-language integration points

### Standard Workflow

**For complex tasks (3+ phases, 4+ files, cross-language):**
1. **Plan** → Use `plan-writer` agent to create `documents/plans/[TASK]_PLAN.md`
2. **Implement** → Delegate to `matlab-coder` or `python-coder` agent (or Claude for simple parts)
3. **Review** → Delegate to `matlab-code-reviewer` or `python-code-reviewer` agent
4. **Iterate** → Simple fixes by Claude directly; complex fixes back to coder agent
5. **Complete** → Mark phase done, ask user about deleting plan file

**For simple tasks:** Claude implements directly without plan or delegation.

### Task Delegation Patterns
```
Plans                    → plan-writer agent
Complex MATLAB work      → matlab-coder agent
Complex Python work      → python-coder agent
Simple tasks             → Claude directly
MATLAB review            → matlab-code-reviewer agent
Python review            → python-code-reviewer agent
```

### Review-Fix-Verify Cycle
After reviewer identifies issues:
1. **Simple fixes** (1-2 lines, obvious changes) → Claude implements directly
2. **Complex fixes** (multi-function, algorithmic) → Delegate back to coder agent
3. Re-review until clean
4. Optionally run MATLAB Code Analyzer for final verification

**Critical:** If uncertain about requirements or approach → ASK USER. Never create fallback code.

## Self-Maintenance Rules

**CRITICAL:** These rules MUST be followed to keep documentation accurate.

1. **Update CLAUDE.md on structure changes:** When adding, removing, renaming, or reorganizing files/scripts, update this file immediately to reflect the changes.

2. **Helper script organization:** When adding new MATLAB functions:
   - Add to an existing helper script in `matlab_scripts/helper_scripts/` if the function belongs to an existing functionality class
   - Create a new helper script only if it represents a distinct functionality class not covered by existing helpers
   - Follow the module pattern: `function module = module_name()` returning a struct of function handles

## Code Quality

- Fix root causes, not symptoms
- Keep code simple and direct
- No overengineering, no backward compatibility code (project not in production)
- No new MATLAB scripts unless explicitly requested
- Variable names: descriptive nouns; Functions: verb phrases; Constants: ALL_CAPS

## Project Overview

MATLAB-based colorimetric analysis pipeline for microPAD analysis. Processes raw images through 4 stages to extract features for concentration prediction.

**microPAD design:** 7 test zones per strip, 3 elliptical regions per zone (Urea, Creatinine, Lactate).

### Pipeline
```
1_dataset → cut_micropads.m → 2_micropads → cut_elliptical_regions.m → 3_elliptical_regions → extract_features.m → 4_extract_features
```

**Augmentation:** `augment_dataset.m` creates synthetic training data with YOLO labels.

### Augmentation Architecture

The augmentation pipeline uses a two-stage approach:

**MATLAB (augment_dataset.m) - Synthetic Data Generation:**
- Geometric: perspective transforms, rotation, random placement, variable scaling
- Background: 5 procedural surface types, artifacts, adaptive shadows (all backgrounds)
- Unique photometric: gamma correction, color temperature, sensor noise, JPEG artifacts
- Physical: paper damage, occlusions (lines/blobs/fingers), stains, specular highlights, edge feathering (50% hard/soft edges)

**YOLO (train_yolo.py) - Runtime Augmentation:**
- HSV: hue ±1.5%, saturation ±70%, value ±40%
- Mosaic: 80% (4 images combined)
- Scale: ±50%, Translate: ±10%
- Horizontal flip: 50%, Random erasing: 40%

This separation avoids redundancy: MATLAB handles unique augmentations (noise profiles, JPEG artifacts, geometric transforms), while YOLO handles HSV jitter at runtime.

### Helper Scripts (`matlab_scripts/helper_scripts/`)

Modular utility functions organized by functionality. Each returns a struct of function handles.

| Script | Purpose | Used By |
|--------|---------|---------|
| `geometry_transform.m` | Geometry + homography operations (quad/ellipse transforms, projective math) | cut_micropads, augment_dataset |
| `feature_pipeline.m` | Feature registry + output (definitions, presets, Excel export, train/test split) | extract_features |
| `augmentation_synthesis.m` | Synthetic data generation (backgrounds, distractors, artifacts, shadows, stains, specular highlights) | augment_dataset |
| `occlusion_utils.m` | Occlusion generation (lines, blobs, fingers) for augmentation | augment_dataset |
| `coordinate_io.m` | **Authoritative source** for coordinate file I/O (parsing, atomic writes, format validation) | all scripts handling coordinates |
| `image_io.m` | Image loading with EXIF handling, motion blur, edge feathering | cut_micropads, augment_dataset, extract_features |
| `mask_utils.m` | Quad/ellipse mask creation with caching | cut_micropads, extract_features |
| `path_utils.m` | Path resolution and folder operations | all main scripts |
| `color_analysis.m` | Color space conversions, paper detection | extract_features |
| `yolo_integration.m` | YOLO model inference via Python subprocess | cut_micropads |
| `micropad_ui.m` | GUI components for cut_micropads | cut_micropads |
| `file_io_manager.m` | Image cropping/saving (delegates coordinate I/O to coordinate_io.m) | cut_micropads |

**Standalone utilities (not module pattern):**
- `preview_overlays.m` - Visualize quad overlays on images
- `preview_augmented_overlays.m` - Visualize augmented data results
- `extract_images_from_coordinates.m` - Extract image patches from coordinates

### Key Principles
- **Stage Independence:** Read from `N_*`, write to `(N+1)_*`
- **Phone-based Organization:** Subdirectories per phone model
- **Consolidated Coordinates:** Phone-level `coordinates.txt` files
- **AI Detection:** YOLOv8 pose keypoint detection for test zones

## File Formats

**Stage 2 coordinates.txt:** `image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation`
**Stage 3 coordinates.txt:** `image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle`

**YOLO labels:** `class_id x1 y1 vis1 x2 y2 vis2 x3 y3 vis3 x4 y4 vis4` (normalized, clockwise from top-left)

## MATLAB-Python Separation

**MATLAB:** Data processing pipeline, coordinate generation, feature extraction
**Python:** AI model training/inference, YOLO label generation
**Interface:** Subprocess communication via `detect_quads.py`

### Model Training (`python_scripts/train_yolo.py`)

Three-tier preset system:

| Preset | Model | Resolution | Batch | Use Case |
|--------|-------|------------|-------|----------|
| **Medium** | yolov8m-pose | 640px | 32 | **DEFAULT** - High accuracy for production |
| **Small** | yolov8s-pose | 640px | 48 | Balanced speed/accuracy |
| **Nano** | yolov8n-pose | 640px | 64 | Fast training/inference |

**Training commands:**
```bash
# Zero-config training (uses medium preset)
python train_yolo.py

# Explicit preset selection
python train_yolo.py --medium  # High accuracy (DEFAULT)
python train_yolo.py --small   # Balanced speed/accuracy
python train_yolo.py --nano    # Fast inference

# Override batch size
python train_yolo.py --medium --batch 48
python train_yolo.py --small --batch 64
```

**Key flags:**
- `--medium/--small/--nano`: Select preset (mutually exclusive)
- `--name`: Custom experiment name (auto-generated if omitted)
- `--validate`: Validate trained model
- `--export`: Export to TFLite for deployment
- `--optimizer`: Choose optimizer (AdamW default)

**MATLAB inference:**
```matlab
% Default (medium model - auto-detects inference size from filename)
cut_micropads('useAIDetection', true)

% Small or nano model
cut_micropads('useAIDetection', true, 'detectionModel', 'models/yolov8s-micropad-pose-640.pt')
cut_micropads('useAIDetection', true, 'detectionModel', 'models/yolov8n-micropad-pose-640.pt')

% Custom model with explicit inference size
cut_micropads('useAIDetection', true, 'detectionModel', 'path/to/model.pt', 'inferenceSize', 640)
```

## Critical Patterns

### Atomic Coordinate Writes
```matlab
tmpPath = tempname(targetFolder);
fid = fopen(tmpPath, 'wt');
% write data
fclose(fid);
movefile(tmpPath, coordPath, 'f');
```

### Geometry Constraints
- Ellipse: `semiMajorAxis >= semiMinorAxis` (swap and rotate 90° if needed)
- Quad: clockwise vertex order from top-left

### MATLAB Standards
```matlab
% Input validation
parser = inputParser();
addParameter(parser, 'numSquares', 7, @(x) validateattributes(x, {'numeric'}, {'scalar','integer','>=',1}));
parse(parser, varargin{:});

% Error handling
error('scriptName:errorType', 'Message: %s', details);

% Performance: vectorize, pre-allocate arrays
```

### Python Standards
```python
# Type hints required
def process(image: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    """Google-style docstring."""
    pass

# Reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

## Common Issues

- **Path resolution:** Run from `matlab_scripts/` or project root
- **EXIF rotation:** Re-run GUI to update rotation in coordinates.txt
- **Corrupt coordinates:** Delete file and regenerate
- **Memory errors:** Use `extract_features('batchSize', 5)`
