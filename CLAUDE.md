# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Workflow

**For complex tasks (3+ phases, 4+ files, cross-language):**
1. **Plan** → Use `plan-writer` agent to create `documents/plans/[TASK]_PLAN.md`
2. **Implement** → Claude implements entire phase directly
3. **Review** → Delegate to `matlab-code-reviewer` or `python-code-reviewer` agent
4. **Iterate** → Fix issues based on review, re-review until clean
5. **Complete** → Mark phase done, ask user about deleting plan file

**For simple tasks:** Implement directly without plan.

**Delegation:**
- Plans → `plan-writer` agent
- MATLAB/Python review → respective reviewer agents
- All implementation → Claude directly

**Critical:** If uncertain about requirements or approach → ASK USER. Never create fallback code.

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
- Polygon: clockwise vertex order from top-left

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
