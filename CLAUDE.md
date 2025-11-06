# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orchestration and Task Management

When handling complex multi-phase implementations, follow this orchestration workflow:

### Plan Management

**When to create a plan file:**
- Task requires 3+ distinct phases or stages
- Modifies 4+ files across different directories
- Involves both MATLAB and Python code
- Restructures core pipeline architecture
- User expects to work across multiple sessions
- User explicitly requests implementation plan

**Plan file location:** All plans stored in `documents/plans/` with naming convention `[TASK_NAME]_PLAN.md` (uppercase, underscores)

**When NOT to create a plan:**
- Simple single-file edits (bug fixes, parameter tweaks)
- Trivial additions (new utility function, documentation update)
- Tasks completable in <5 steps

**Creating a plan:**
Use the `plan-writer` agent to create structured implementation plans with checkboxes and progress tracking.

**Plan Philosophy:**
Plans are **high-order strategic documents**, not detailed implementation guides. They specify:
- **WHAT** needs to be done (clear task objectives)
- **WHERE** it needs to be done (file paths, integration points)
- **WHY** it needs to be done (rationale, context)
- **HOW TO VERIFY** it works (test cases, success criteria)

Plans should **NOT** include:
- Complete code implementations (that's the coder agents' job)
- Detailed function bodies or algorithms
- Line-by-line code snippets

**Separation of concerns:**
- `plan-writer` creates high-level roadmap with clear objectives
- `matlab-coder` and `python-coder` implement the actual code
- Code reviewers verify implementation quality

### Task Delegation and Workflow

**Standard workflow for complex tasks:**

1. **Plan Phase** (if complexity criteria met)
   - Delegate to `plan-writer` agent to create plan in `documents/plans/`
   - Review plan with user before proceeding

2. **Implementation Phase**
   - Delegate MATLAB work to `matlab-coder` agent
   - Delegate Python work to `python-coder` agent
   - Agents implement **entire phase** (all steps within a phase) before review
   - Agents perform quick sanity checks only (no self-review)

3. **Independent Review Phase** (MANDATORY after each phase completes)
   - Delegate MATLAB code review to `matlab-code-reviewer` agent
   - Delegate Python code review to `python-code-reviewer` agent
   - Reviewers assess **entire phase** as a complete system
   - Reviewers identify issues but do NOT fix them

4. **Iteration Phase**
   - If issues found: Send back to appropriate coder agent with fix instructions
   - Repeat review after fixes
   - Continue until review is clean

5. **Completion**
   - Mark phase complete in plan file only after passing review
   - After entire plan completes: Ask user whether to delete plan file

### Critical Rules for Orchestration

**NEVER make low-confidence assumptions:**
- If uncertain about requirements → ASK USER
- If stuck on implementation approach → ASK USER
- If multiple valid approaches exist → ASK USER which to use
- Never create fallback code or workarounds due to uncertainty

**Quality assurance:**
- Implement complete phases before review (e.g., Phase 1: steps 1.1, 1.2, 1.3 → then review)
- Review entire phase as a cohesive system, not individual steps
- No phase marked complete without passing review
- Issues found in review MUST be fixed before proceeding
- Re-review after fixes until clean

**Delegation patterns:**
```
MATLAB scripts → matlab-coder
Python code → python-coder
Plan creation/updates → plan-writer
MATLAB review → matlab-code-reviewer (via Task tool)
Python review → python-code-reviewer (via Task tool)
```

### Example Orchestration Flow

```
User: "Implement feature X that requires MATLAB and Python changes"

1. Check complexity → 3 phases, cross-language → CREATE PLAN
2. Delegate to plan-writer: Create plan in documents/plans/FEATURE_X_PLAN.md
3. Review plan with user → Get approval
4. For each phase in plan:
   a. Delegate to matlab-coder or python-coder: Implement ALL steps in phase
   b. After entire phase implemented, delegate to respective code-reviewer
   c. If issues found → Send back to coder with fixes → Re-review
   d. Mark phase complete only after clean review
5. After all phases complete: "Feature X complete. Delete documents/plans/FEATURE_X_PLAN.md? (Yes/No)"
```

## Code Quality Standards

**CRITICAL IMPLEMENTATION RULES:**
- **NEVER create workarounds** - Always fix root causes, not symptoms
- **NEVER use fallback patterns** - Implement proper solutions that handle edge cases correctly
- **NEVER write overengineered code** - Keep implementations simple, direct, and maintainable
- **NEVER add redundant or verbose code** - Every line must serve a clear purpose
- **NEVER create new MATLAB scripts** - Only create new scripts when explicitly requested by the user
- **NEVER add backward compatibility code** - This project is in active development and not production-ready until all work is complete; do not add migration logic, compatibility shims, or handling for "old versions"
- **ALWAYS use best practices** for MATLAB and this project's architecture
- **ALWAYS consider side effects** of changes across the entire pipeline
- **ALWAYS implement bulletproof solutions** that handle edge cases without defensive bloat
- **ALWAYS analyze root causes** before proposing solutions
- **ASK QUESTIONS if stuck** - Do not add fallback algorithms instead; clarify requirements first

**Development vs Production Philosophy:**
- This codebase is **NOT production-ready** until explicitly stated by the user
- Code runs in development mode (with and without AI detection enabled)
- Do not add migration paths for "previous versions" that don't exist in production
- Use current formats exclusively; error out cleanly on invalid data instead of attempting migration
- Backward compatibility is only needed when code is actually deployed to production with existing users

**Naming and Documentation Standards:**
- Variable names: descriptive nouns without opinions or history
- Function names: verb phrases describing actions
- Constants: ALL_CAPS with units if applicable
- Comments: state facts only, no opinions or historical notes
- Use git history for tracking changes, not code comments

When fixing bugs:
1. Identify the root cause, not just the symptom
2. Fix the underlying issue, not the manifestation
3. Ensure the fix doesn't introduce technical debt
4. Validate the fix handles all edge cases naturally

## Project Overview

This is a MATLAB-based colorimetric analysis pipeline for microPAD (microfluidic Paper-based Analytical Device) analysis. The pipeline processes raw microPAD images through multiple stages to extract colorimetric features for concentration prediction.

### Experimental Design

**Final microPAD design:** Each paper strip contains 7 test zones. Each test zone has 3 elliptical regions designed for 3 different chemicals:
1. **Region 1:** Urea
2. **Region 2:** Creatinine
3. **Region 3:** Lactate

**Training data collection:** To train machine learning models for each chemical separately, experiments are conducted where all 3 elliptical regions in each test zone are filled with the **same concentration** of a **single chemical**. This provides 3 replicate measurements per concentration level, resulting in separate datasets for urea, creatinine, and lactate.

## Pipeline Architecture

The codebase implements a **4-stage sequential colorimetric analysis pipeline** where each stage consumes the output of the previous stage:

### Stage Flow
```
1_dataset (raw images)
    -> cut_micropads.m
2_micropads (polygonal concentration regions + rotation)
    -> cut_elliptical_regions.m
3_elliptical_regions (elliptical patches + coordinates)
    -> extract_features.m
4_extract_features (feature tables)
```

Additionally, `augment_dataset.m` creates synthetic training data:
```
1_dataset (original images) + 2_micropads (coordinates)
    -> augment_dataset.m
augmented_1_dataset (synthetic scenes)
augmented_2_micropads (transformed polygons)
augmented_3_elliptical_regions (transformed ellipses)
```
### Key Design Principles

1. **Stage Independence**: Each script reads from `N_*` and writes to `(N+1)_*`
2. **Phone-based Organization**: Data is organized by phone model subdirectories (e.g., `iphone_11/`, `samsung_a75/`)
3. **Concentration Folders**: Stages 2-3 use `con_0/`, `con_1/`, etc. subfolders
4. **Consolidated Coordinates**: Coordinate files are stored at phone-level (not per-image) to avoid duplication
5. **AI-Powered Detection**: Stage 1→2 uses YOLOv11 pose keypoint detection for auto-detection of test zones

## Running the Pipeline

### Individual Stages
All scripts use dynamic project root resolution (searches up to 5 directory levels):

```matlab
% Stage 1→2: Cut microPAD concentration regions with AI detection
cd matlab_scripts
cut_micropads('numSquares', 7)  % 7 regions per strip, YOLOv11 pose auto-detection

% Stage 2→3: Extract elliptical patches from concentration regions
cut_elliptical_regions()

% Stage 3→4: Extract features from elliptical patches
extract_features('preset', 'robust', 'chemical', 'lactate')

% Data Augmentation (generates synthetic training data)
augment_dataset('numAugmentations', 5, 'rngSeed', 42)
```

### Common Parameters
- All scripts accept name-value pairs for configuration
- `saveCoordinates`: Write coordinate metadata files
- **Output Format**: All pipeline stages (2-4) output PNG exclusively for lossless compression and colorimetric accuracy

## File Naming Conventions

### Coordinate Files
- **Stage 2 (2_micropads)**: `coordinates.txt` format: `image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation` (4 polygon vertices + rotation in degrees)
- **Stage 3 (3_elliptical_regions)**: `coordinates.txt` format: `image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle`

### Output Files
- Stage 2 crops: `{base}_con_{N}.{ext}` in `con_{N}/` folders
- Stage 3 patches: `{base}_con{N}_rep{M}.{ext}` in `con_{N}/` folders
  - Note: `rep{M}` represents replicate measurements (M = 0, 1, 2). In the final microPAD design, these correspond to three different chemicals (urea, creatinine, lactate). During training, all 3 replicates contain the same chemical at the same concentration.
- Stage 4 features: `{chemical}_features.xlsx`

### YOLO Label Files (Augmentation Output)
When `augment_dataset.m` is run with `exportYOLOLabels=true`, it generates YOLOv11 pose keypoint labels for AI polygon detection:
- **Format**: `{imageName}.txt` in `labels/` subdirectory
- **Structure**: One line per polygon, space-separated values

**Format:**
```
class_id x1 y1 vis1 x2 y2 vis2 x3 y3 vis3 x4 y4 vis4
```
- `class_id`: Always 0 (concentration zone - single class)
- `x1 y1 ... x4 y4`: Normalized keypoint coordinates [0, 1] (divide by image width/height)
- `vis1 ... vis4`: Visibility flags, always 2 (visible)
- Keypoint order: Clockwise from top-left (TL, TR, BR, BL)

**Example label file (`augmented_1_dataset/phoneName/labels/synthetic_001.txt`):**
```
0 0.234567 0.156789 2 0.345678 0.167890 2 0.356789 0.278901 2 0.245678 0.267890 2
0 0.456789 0.389012 2 0.567890 0.400123 2 0.578901 0.511234 2 0.467890 0.500123 2
```

**Loading (Python with Ultralytics):**
```python
from ultralytics import YOLO

# Train YOLOv11 pose model
model = YOLO('yolo11m-pose.pt')
model.train(data='micropad_synth.yaml', epochs=150, imgsz=960)
```

**Dataset YAML configuration:**
```yaml
path: /path/to/augmented_1_dataset
train: train.txt
val: val.txt
nc: 1
names: ['concentration_zone']
```

## MATLAB-Python Separation of Concerns

**MATLAB Responsibilities (Data Processing):**
- Stage pipeline (1→2→3 and augmented variants)
- Image transformations and geometry
- Coordinate file generation (coordinates.txt)
- Feature extraction for ML training

**Python Responsibilities (AI Model Training/Inference):**
- YOLO label generation (reads MATLAB coordinates, creates normalized polygon labels)
- Model training and evaluation
- Inference helper scripts (called by MATLAB for predictions)

**Interface:**
- MATLAB calls Python subprocess (`detect_quads.py`) for AI predictions
- Python reads MATLAB coordinates (`coordinates.txt`) for label generation
- No direct MATLAB-Python API coupling (subprocess-based communication)

**Key Principle:** MATLAB scripts should be agnostic to AI model training details. AI training format changes (e.g., switching from YOLO to Faster R-CNN) should only require Python script modifications, not MATLAB changes.

## Critical Implementation Details

### Coordinate File Management
**All coordinate files use atomic write pattern** (temp file + move) to prevent corruption:
```matlab
tmpPath = tempname(targetFolder);
fid = fopen(tmpPath, 'wt');
% ... write data ...
fclose(fid);
movefile(tmpPath, coordPath, 'f');
```

**No duplicate rows per image**: Scripts automatically filter existing entries before appending new coordinates.

### Image Orientation Handling
All pipeline scripts use an `imread_raw()` helper that simply defers to MATLAB's `imread()` so we always work with raw sensor pixels. We intentionally ignore EXIF orientation metadata and instead record any manual rotation adjustments in `coordinates.txt` (10th column in `2_micropads/coordinates.txt`). Downstream steps apply those rotations when needed.

### Geometry and Projection
- **Stage 2 (cut_micropads.m)**:
  - YOLOv11 pose keypoint detection for polygon corners
  - Interactive rotation control with cumulative rotation memory
  - Saves rotation as 10th column in coordinates.txt
  - AI detection automatically re-runs after rotation changes
- **Stage 3 (cut_elliptical_regions.m)**: Ellipse geometry with axis-aligned bounding box calculation
  - Rotation angle in degrees (-180 to 180), clockwise from horizontal
  - Constraint: `semiMajorAxis >= semiMinorAxis` (enforced for extract_features.m compatibility)

### Augmentation Strategy
`augment_dataset.m` generates synthetic training data for polygon detection training by:
1. Reading polygon coordinates from `2_micropads/coordinates.txt` (10-column format with rotation)
2. Applying shared perspective and rotation per region (+/-60 deg pitch/yaw, 0-360 deg roll)
3. Optional independent rotation per region
4. Placing regions via grid-accelerated random sampling (guaranteed non-overlapping)
5. Compositing onto procedural backgrounds drawn from uniform, speckled, laminate, and skin surface pools (cached variants with jitter)
6. Adding moderate-density distractor artifacts (5-40 geometric shapes per image)
7. Adding polygon-shaped distractors (1-10 synthetic look-alikes with random sizes 0.5×-1.5× of real polygons)
8. Applying color-safe photometric augmentation (brightness/contrast, white balance, saturation/gamma)
9. Optionally adding at most one blur type (motion or Gaussian) and thin occlusions

**Performance optimizations (v2):**
- Grid-based spatial acceleration for O(1) collision detection (vs O(n^2))
- Simplified polygon warping without alpha blending (3x faster)
- Background texture pooling (reuses 4 procedural types from cached pools instead of regenerating each frame)
- Artifact density: 5-40 per image (optimized from 1-100)
- Artifact rendering: Sharp by default with nearest-neighbor upscaling (matches polygon sharpness)
- Scene-wide blur: Optional 25% probability applied uniformly to entire image
- Overall speedup: 3x (1.0s vs 3.0s per augmented image)

**Key parameters:**
- `numAugmentations`: number of synthetic versions per original
- `independentRotation`: enable per-polygon rotation
- `occlusionProbability`: thin occlusions across polygons
- `exportYOLOLabels`: emit YOLOv11 pose keypoint labels (normalized corner coordinates with visibility flags)
- `backgroundWidth/Height`: output dimensions

### Interactive GUI Sessions
All interactive scripts maintain **persistent memory** across images in the same folder:
- Stores last-used positions, rotations, or ellipse parameters
- Scales stored positions when image dimensions change
- First image establishes baseline; subsequent images use saved settings

### Batch Processing (extract_features.m)
- Reads polygon images from `augmented_2_micropads/` and ellipse coordinates from `augmented_3_elliptical_regions/`
- Adaptive batch sizing based on dataset size and available memory
- Small datasets (<10 files): process all at once
- Medium datasets (10-50): 3-batch split
- Large datasets (>50): dynamic batch size calculation
- Memory threshold: 80% to prevent out-of-memory errors

## Feature Extraction

### Presets
- **minimal**: Essential color ratios and basic statistics
- **robust**: Comprehensive feature set (recommended)
- **full**: All available features
- **custom**: User-defined feature groups via dialog or struct

### Output Format
- Excel (.xlsx) with train/test split option
- Optional Label column for downstream ML
- Feature names follow registry convention: `GroupName_FeatureName`

## Testing and Development

### Adding New Features
1. Register feature in feature registry (lines 100-300 in extract_features.m)
2. Implement extraction function with signature: `function val = extract_{feature}(patchImg, ellipseParams, ...)`
3. Add to appropriate feature group preset

### Debugging Tips
- Check `coordinates.txt` for malformed entries (causes silent failures)
- Verify polygon vertex order (clockwise from top-left)
- Verify rotation column (10th field) in `2_micropads/coordinates.txt`
- Confirm ellipse axes constraint: major >= minor
- Script complexity (approximate line counts):
  - cut_micropads.m: ~2500 lines (AI detection, rotation control, interactive GUI)
  - cut_elliptical_regions.m: ~1457 lines
  - extract_features.m: ~4440 lines (largest, most complex)
  - augment_dataset.m: ~2730 lines (label export, transforms, texture pooling)

## Common Issues

### Path Resolution Failures
Scripts search up to 5 directory levels for project root. If warnings appear:
```
Could not find input folder "X_*". Using current directory as project root.
```
Solution: Run from `matlab_scripts/` or project root directory.

### EXIF Orientation Issues
If images appear incorrectly rotated, re-run the GUI stage to update the stored rotation in `coordinates.txt`. The pipeline relies solely on that metadata and does not modify image files in place.

### Coordinate File Corruption
Atomic writes prevent most corruption, but if coordinates.txt is malformed:
1. Delete the corrupted file
2. Re-run the stage to regenerate from scratch
3. Check for trailing newlines or mixed delimiters

### Memory Errors in Feature Extraction
If `extract_features.m` runs out of memory:
```matlab
% Reduce batch size explicitly (overrides adaptive sizing)
extract_features('preset', 'robust', 'batchSize', 5)
```
