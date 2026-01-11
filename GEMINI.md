# GEMINI.md

Guidance for Gemini when working with this repository.

## Orchestration and Workflow

### Task Management

**Simple Tasks:**
- Implement directly using `replace`, `write_file`, or `run_shell_command`.
- Quick fixes, single-file edits, or parameter changes.

**Complex Tasks (Multi-file, Refactoring, System Analysis):**
1. **Investigate** → Use `codebase_investigator` to map dependencies and understand the current architecture.
2. **Plan** → Use `write_todos` to break down the task into tracked subtasks.
3. **Implement** → Execute changes iteratively, updating the todo list as progress is made.
4. **Verify** → Run relevant tests or verification scripts (e.g., `detect_quads.py` or MATLAB scripts if accessible/requested).

### Workflow Patterns

**Planning:**
- Use `write_todos` for any task requiring more than 2 steps.
- For complex architectural changes, draft a plan in `documents/plans/` if requested, similar to the `plan-writer` workflow.

**Implementation:**
- **MATLAB:** Follow the existing module pattern and helper script organization.
- **Python:** Adhere to type hinting and Google-style docstrings.

**Review:**
- Self-correct using `run_shell_command` to check syntax or run linters (if available).
- Always verify file content with `read_file` before and after modification.

## Self-Maintenance Rules

**CRITICAL:** These rules MUST be followed to keep documentation accurate.

1. **Update GEMINI.md (and CLAUDE.md if consistent) on structure changes:** When adding, removing, renaming, or reorganizing files/scripts, update this file immediately to reflect the changes.

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

### Helper Scripts (`matlab_scripts/helper_scripts/`)

Modular utility functions organized by functionality. Each returns a struct of function handles.

| Script | Purpose | Used By |
|--------|---------|---------|
| `geometry_transform.m` | Geometry + homography operations (quad/ellipse transforms, projective math) | cut_micropads, augment_dataset |
| `feature_pipeline.m` | Feature registry + output (definitions, presets, Excel export, train/test split) | extract_features |
| `augmentation_synthesis.m` | Synthetic data generation (backgrounds, distractors, artifacts) | augment_dataset |
| `coordinate_io.m` | Coordinate file I/O (quad/ellipse formats) | cut_micropads, extract_features |
| `image_io.m` | Image loading with EXIF handling | cut_micropads, augment_dataset, extract_features |
| `mask_utils.m` | Quad/ellipse mask creation with caching | cut_micropads, extract_features |
| `path_utils.m` | Path resolution and folder operations | all main scripts |
| `color_analysis.m` | Color space conversions, paper detection | extract_features |
| `yolo_integration.m` | YOLO model inference via Python subprocess | cut_micropads |
| `micropad_ui.m` | GUI components for cut_micropads | cut_micropads |
| `file_io_manager.m` | File operations and atomic writes | cut_micropads, augment_dataset |

**Standalone utilities (not module pattern):**
- `preview_overlays.m` - Visualize quad overlays on images
- `preview_augmented_overlays.m` - Visualize augmented data results
- `extract_images_from_coordinates.m` - Extract image patches from coordinates

### Key Principles
- **Stage Independence:** Read from `N_*`, write to `(N+1)_*`
- **Phone-based Organization:** Subdirectories per phone model
- **Consolidated Coordinates:** Phone-level `coordinates.txt` files
- **AI Detection:** YOLOv11 pose keypoint detection for test zones

## File Formats

**Stage 2 coordinates.txt:** `image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation`
**Stage 3 coordinates.txt:** `image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle`

**YOLO labels:** `class_id x1 y1 vis1 x2 y2 vis2 x3 y3 vis3 x4 y4 vis4` (normalized, clockwise from top-left)

## MATLAB-Python Separation

**MATLAB:** Data processing pipeline, coordinate generation, feature extraction
**Python:** AI model training/inference, YOLO label generation
**Interface:** Subprocess communication via `detect_quads.py`

### Model Training (`python_scripts/train_yolo.py`)

Two model configurations available:

| Mode | Model | Resolution | Command |
|------|-------|------------|---------|
| Desktop | yolo11s-pose | 1280x1280 | `python train_yolo.py` |
| Mobile | yolo11n-pose | 640x640 | `python train_yolo.py --mobile` |

**Key flags:**
- `--mobile`: Use mobile-optimized settings (nano model, 640px, reduced augmentation)
- `--name`: Custom experiment name (auto-generated if omitted)
- `--validate`: Validate trained model
- `--export`: Export to TFLite for deployment

**MATLAB inference:**
```matlab
% Desktop model (default)
cut_micropads('useAIDetection', true)

% Mobile model
cut_micropads('useAIDetection', true, 'useMobileModel', true)
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
