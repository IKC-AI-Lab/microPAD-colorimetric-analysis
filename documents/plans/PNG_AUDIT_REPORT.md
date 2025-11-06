# PNG Standardization Audit Report

**Date:** 2025-11-06
**Status:** Phase 1 Complete - Ready for Phase 2 Implementation

---

## Executive Summary

This audit identified all image format-related code across the microPAD colorimetric analysis pipeline. The codebase currently supports mixed JPEG/PNG formats through `preserveFormat` and `jpegQuality` parameters in 7 MATLAB scripts and format filters in 1 Python script.

**Key Findings:**
- 7 MATLAB files require format parameter removal
- 1 Python file requires extension filter update
- 2 documentation files require format reference updates
- 4 `determineOutputExtension()` functions need hardcoding to PNG
- 11 `imwrite()` calls have JPEG-specific logic
- 6 `imread_raw()` helpers (no changes needed - already format-agnostic)

---

## MATLAB Pipeline Files

### 1. cut_micropads.m (~2500 lines)
**Location:** `matlab_scripts/cut_micropads.m`

**Format-Related Code:**

| Line | Code | Current Behavior | Migration Action |
|------|------|------------------|------------------|
| 23 | Comment mentioning `preserveFormat` / `jpegQuality` | Documentation | Update comment to PNG-only |
| 94-95 | `SUPPORTED_FORMATS = {'.jpg','.jpeg','.png',...}` | Format whitelist | Keep (used for input validation) |
| 181-185 | `createConfiguration()` signature with `preserveFormat, jpegQuality` | Parameter passing | Remove both parameters |
| 192-193 | `addParameter('preserveFormat'...)` / `addParameter('jpegQuality'...)` | Parser definition | Remove both parameters |
| 223-224 | `cfg.output.preserveFormat = ...` / `cfg.output.jpegQuality = ...` | Config storage | Remove both assignments |
| 2285 | `outExt = determineOutputExtension(extOrig, cfg.output.supportedFormats, cfg.output.preserveFormat)` | Format selection | Replace with `outExt = '.png'` |
| 2524-2530 | `saveImageWithFormat()` function | JPEG quality branching | Simplify to `imwrite(img, outPath)` |
| 2525-2526 | `if strcmpi(outExt, '.jpg')...imwrite(...'jpg', 'Quality', cfg.output.jpegQuality)` | JPEG write | Remove branch |
| 2532-2538 | `determineOutputExtension()` function | Format logic | Delete entire function |
| 2536 | `outExt = '.jpg'` (fallback) | Default to JPEG | N/A (function deleted) |
| 2662 | `tmpImgPath = ...sprintf('%s_micropad_detect.jpg'...)` | Temp file for AI detection | Change to `.png` |

**Dependencies:**
- Calls Python subprocess `detect_quads.py` (format-agnostic - no changes needed)
- Writes `coordinates.txt` (format-agnostic - no changes needed)

**Success Criteria:**
- Script runs without `preserveFormat` or `jpegQuality` parameters
- Output directory `2_micropads/` contains only `.png` files
- Passing deprecated parameters throws error: `"Error: JPEG format no longer supported. Pipeline outputs PNG exclusively."`

---

### 2. cut_elliptical_regions.m (~1457 lines)
**Location:** `matlab_scripts/cut_elliptical_regions.m`

**Format-Related Code:**

| Line | Code | Current Behavior | Migration Action |
|------|------|------------------|------------------|
| 7-8 | Comment mentioning `preserveFormat` / `jpegQuality` | Documentation | Update to PNG-only |
| 48 | `ALLOWED_IMAGE_EXTENSIONS = {'*.jpg', '*.jpeg', '*.png',...}` | Input filters | Keep (input validation) |
| 55 | `SUPPORTED_FORMATS = {'.jpg','.jpeg','.png',...}` | Format whitelist | Keep (input validation) |
| 129 | `createConfiguration()` signature with `preserveFormat, jpegQuality` | Parameter passing | Remove both parameters |
| 143-144 | `addParameter('preserveFormat'...)` / `addParameter('jpegQuality'...)` | Parser definition | Remove both parameters |
| 154-158 | Quality warning: `if ~preserveFormat && jpegQuality < 95` | Validation logic | Remove entire warning block |
| 192-193 | `cfg.output.preserveFormat = ...` / `cfg.output.jpegQuality = ...` | Config storage | Remove both assignments |
| 1203 | `outExt = determineOutputExtension(extOrig, supported, cfg.output.preserveFormat)` | Format selection | Replace with `outExt = '.png'` |
| 1541-1548 | `saveImageWithFormat()` function | JPEG quality branching | Simplify to `imwrite(img, outPath)` |
| 1543-1544 | `if any(strcmp(outExt, {'.jpg','.jpeg'}))...imwrite(...'JPEG', 'Quality'...)` | JPEG write | Remove branch |
| 1550-1557 | `determineOutputExtension()` function | Format logic | Delete entire function |
| 1555 | `outExt = '.jpeg'` (fallback) | Default to JPEG | N/A (function deleted) |

**Dependencies:**
- Reads from `2_micropads/coordinates.txt` (format-agnostic)
- Writes to `3_elliptical_regions/coordinates.txt` (format-agnostic)

**Success Criteria:**
- Output directory `3_elliptical_regions/` contains only `.png` files
- `imfinfo()` confirms PNG format with no EXIF metadata
- No warnings about JPEG quality in console

---

### 3. augment_dataset.m (~2730 lines)
**Location:** `matlab_scripts/augment_dataset.m`

**Format-Related Code:**

| Line | Code | Current Behavior | Migration Action |
|------|------|------------------|------------------|
| 76 | `SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png',...}` | Input validation | Keep (for reading 1_dataset) |
| 209 | `cfg.jpegQuality = JPEG_QUALITY` | Config storage | Remove assignment |
| 439-442 | `if any(strcmpi(imgExt, {'.jpg', '.jpeg'}))...imwrite(...'JPEG', 'Quality'...)` | Conditional JPEG write | Remove condition, use `imwrite(stage1Img, sceneOutPath)` |
| 479 | `imwrite(polygonImg, polygonOutPath)` | Generic write | No change (already format-agnostic) |
| 514 | `imwrite(patchImg, patchOutPath)` | Generic write | No change |
| 728 | `imwrite(augPolygonImg, polygonOutPath)` | Generic write | No change |
| 846 | `imwrite(patchImg, patchOutPath)` | Generic write | No change |
| 900 | `sceneFileName = sprintf('%s%s', sceneName, '.jpg')` | Hardcoded JPEG extension | Change to `'.png'` |
| 902 | `imwrite(background, sceneOutPath, 'JPEG', 'Quality', cfg.jpegQuality)` | JPEG write | Change to `imwrite(background, sceneOutPath)` |

**Note:** Lines 439-442 handle re-encoding fallback when file copy fails. This should preserve PNG for PNG inputs and output PNG for all augmented images.

**Dependencies:**
- YOLO label export (format-agnostic - references `.png` extensions in generated labels)

**Success Criteria:**
- Directories `augmented_1_dataset/`, `augmented_2_micropads/`, `augmented_3_elliptical_regions/` contain only `.png`
- YOLO label files correctly reference `.png` image names
- `imfinfo()` confirms PNG format

---

### 4. extract_features.m (~4440 lines)
**Location:** `matlab_scripts/extract_features.m`

**Format-Related Code:**

| Line | Code | Current Behavior | Migration Action |
|------|------|------------------|------------------|
| 2151 | `img = imread_raw(imageName)` | Image loading | No change (already format-agnostic) |
| 2703-2704 | `outPath = fullfile(outDir, [base '_mask.png']); imwrite(outImg, outPath)` | Debug output | No change (already PNG) |
| 3158 | `validExts = {'.jpg', '.jpeg', '.png',...}` | Input validation | Change to `{'.png'}` only |
| 4345-4350 | `imread_raw()` helper function | Image loading | No change (already format-agnostic) |

**Success Criteria:**
- Script processes PNG inputs from stages 2-3 without errors
- Feature extraction Excel output generated successfully
- No console warnings about unexpected formats

---

### 5. helper_scripts/extract_images_from_coordinates.m
**Location:** `matlab_scripts/helper_scripts/extract_images_from_coordinates.m`

**Format-Related Code:**

| Line | Code | Current Behavior | Migration Action |
|------|------|------------------|------------------|
| 24-25 | Comment mentioning `preserveFormat` / `jpegQuality` | Documentation | Update to PNG-only |
| 56 | Example with `jpegQuality` | Documentation | Remove example |
| 67-68 | `addParameter('preserveFormat'...)` / `addParameter('jpegQuality'...)` | Parser definition | Remove both parameters |
| 75 | Function call passing `preserveFormat, jpegQuality` | Parameter passing | Remove both parameters |
| 166 | `createConfiguration()` signature with `preserveFormat, jpegQuality` | Function signature | Remove both parameters |
| 173-174 | Validation for `preserveFormat` / `jpegQuality` | Parameter validation | Remove both validations |
| 188-189 | `cfg.output = struct('preserveFormat', ..., 'jpegQuality', ...)` | Config storage | Remove both fields |
| 190 | `supportedFormats` in config | Format whitelist | Keep (input validation) |
| 191 | `allowedImageExtensions` in config | Input filters | Keep (input validation) |
| 246, 281, 403 | `determine_output_extension(...preserveFormat)` | Format selection | Replace with `'.png'` |
| 566-585 | Format switch statement for imwrite | JPEG/PNG/BMP/TIFF branching | Simplify to `imwrite(img, outPath)` |
| 577 | `imwrite(img, outPath, 'JPEG', 'Quality', cfg.output.jpegQuality)` | JPEG write | Remove |
| 578-584 | PNG/BMP/TIFF specific writes | Format branching | Remove (generic imwrite handles PNG) |
| 596 | `outExt = '.jpeg'` (fallback) | Default to JPEG | Change to `'.png'` |
| 619, 767, 796 | `validExts` arrays with JPEG | Validation lists | Keep (input validation) |

**Success Criteria:**
- Helper executes without format parameters
- Output uses PNG exclusively

---

### 6. helper_scripts/preview_overlays.m
**Location:** `matlab_scripts/helper_scripts/preview_overlays.m`

**Format-Related Code:**

| Line | Code | Current Behavior | Migration Action |
|------|------|------------------|------------------|
| 30 | `SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png',...}` | Input validation | Keep (for reading various inputs) |
| 159 | `img = imread_raw(entry.imagePath)` | Image loading | No change (format-agnostic) |
| 986-991 | `imread_raw()` helper function | Image loading | No change (format-agnostic) |

**Success Criteria:**
- Preview tool loads PNG files without errors
- No format-specific assumptions

---

### 7. helper_scripts/preview_augmented_overlays.m
**Location:** `matlab_scripts/helper_scripts/preview_augmented_overlays.m`

**Format-Related Code:**

| Line | Code | Current Behavior | Migration Action |
|------|------|------------------|------------------|
| 31 | `SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png',...}` | Input validation | Keep (for reading various inputs) |
| 176 | `img = imread_raw(entry.imagePath)` | Image loading | No change (format-agnostic) |
| 260-263 | Comment explaining extension handling | Documentation | Update to mention PNG outputs |
| 530-535 | `imread_raw()` helper function | Image loading | No change (format-agnostic) |

**Success Criteria:**
- Preview tool loads PNG augmented images
- No format-specific errors

---

## Python Scripts

### 1. prepare_yolo_dataset.py
**Location:** `python_scripts/prepare_yolo_dataset.py`

**Format-Related Code:**

| Line | Code | Current Behavior | Migration Action |
|------|------|------------------|------------------|
| 59 | `for ext in ["*.jpg", "*.jpeg", "*.png"]:` | File enumeration | Change to `["*.png"]` |
| 101 | `for ext in ["*.jpg", "*.jpeg", "*.png"]:` | File enumeration | Change to `["*.png"]` |
| 180 | `for ext in ['.jpg', '.jpeg', '.png']:` | Extension check | Change to `['.png']` |

**Dependencies:**
- Consumes YOLO labels from `augmented_1_dataset/` (format-agnostic - just references image names)

**Success Criteria:**
- Script enumerates PNG files without errors
- No "missing file" warnings for JPEG extensions
- Generated train/val splits reference only PNG images

---

### 2. detect_quads.py
**Location:** `python_scripts/detect_quads.py`

**Format-Related Code:**
- **None found** - Script uses PIL/OpenCV image loading which automatically handles formats
- Input validation may check extensions but doesn't restrict to JPEG/PNG specifically

**Success Criteria:**
- Script processes PNG inputs from MATLAB calls
- Returns valid JSON coordinates

---

## Documentation Files

### 1. README.md
**Location:** `README.md`

**Format References:**

| Line | Content | Migration Action |
|------|---------|------------------|
| 5 | `![Pipeline Overview](demo_images/stage1_original_image.jpeg)` | Keep (stage 1 unchanged) |
| 67 | `![Stage 1 to 2](demo_images/stage2_micropad.jpeg)` | Change to `.png` (if demo file updated) |
| 86 | `![Stage 2 to 3](demo_images/stage3_elliptical_region_1.jpeg)` | Change to `.png` |
| 176-178 | Example table rows with `.jpeg` extensions | Update to `.png` |
| 267, 270, 273 | Augmented demo image paths with JPEG | Update to `.png` |
| 393-394 | File naming examples with `.jpg` | Update to `.png` |
| 561-565 | Feature extraction table with `.jpeg` | Update to `.png` |

**Success Criteria:**
- All examples reference `.png` for stages 2-4
- Stage 1 raw images retain original format references

---

### 2. CLAUDE.md
**Location:** `CLAUDE.md`

**Format References:**

| Line | Content | Migration Action |
|------|---------|------------------|
| 216-217 | Common parameters: `preserveFormat`, `jpegQuality` | Remove both parameters |

**Success Criteria:**
- No mention of format parameters in pipeline documentation
- PNG-only workflow documented

---

## Migration Checklist by Priority

### Critical Path (Phase 2 Required)

**High Priority:**
1. `cut_micropads.m` - Remove `preserveFormat`, `jpegQuality`, hardcode `.png`
2. `cut_elliptical_regions.m` - Remove format parameters, hardcode `.png`
3. `augment_dataset.m` - Remove `jpegQuality`, hardcode `.png` outputs
4. `extract_features.m` - Update `validExts` to PNG-only

**Medium Priority:**
5. `helper_scripts/extract_images_from_coordinates.m` - Remove format parameters
6. `python_scripts/prepare_yolo_dataset.py` - Update extension filters to PNG-only

**Low Priority (Phase 3):**
7. `helper_scripts/preview_overlays.m` - Verify PNG compatibility
8. `helper_scripts/preview_augmented_overlays.m` - Verify PNG compatibility
9. `README.md` - Update examples and image references
10. `CLAUDE.md` - Update parameter documentation

---

## Format Logic Functions to Modify

### Functions to DELETE:
1. `cut_micropads.m:2532` - `determineOutputExtension()`
2. `cut_elliptical_regions.m:1550` - `determineOutputExtension()`

### Functions to SIMPLIFY:
1. `cut_micropads.m:2524` - `saveImageWithFormat()` → Remove JPEG branch
2. `cut_elliptical_regions.m:1541` - `saveImageWithFormat()` → Remove JPEG branch
3. `extract_images_from_coordinates.m:566` - Format switch → Generic `imwrite()`

### Constants to UPDATE:
1. `augment_dataset.m:209` - Remove `cfg.jpegQuality`
2. `augment_dataset.m:900` - Change `'.jpg'` → `'.png'`

---

## Validation Strategy

### Per-File Testing:
After migrating each MATLAB script:
```matlab
% Test with default parameters
cut_micropads('numSquares', 7)
cut_elliptical_regions()
augment_dataset('numAugmentations', 2)
extract_features('preset', 'robust', 'chemical', 'lactate')

% Verify deprecated parameters throw errors
cut_micropads('preserveFormat', true)  % Should error
cut_elliptical_regions('jpegQuality', 90)  % Should error
```

### Format Verification:
```matlab
% Check output is PNG
files = dir('2_micropads/phone_name/con_0/*.png');
info = imfinfo(files(1).name);
assert(strcmp(info.Format, 'png'))
assert(~isfield(info, 'Orientation'))  % No EXIF metadata
```

### Python Integration:
```bash
python python_scripts/prepare_yolo_dataset.py
# Verify: No "file not found" warnings for JPEG
# Verify: train.txt, val.txt reference only PNG files
```

---

## Estimated Impact

**Files Modified:** 9 total (7 MATLAB, 1 Python, 2 docs)
**Lines Changed:** ~50 lines removed, ~15 lines modified
**Functions Deleted:** 2
**Parameters Removed:** 2 (`preserveFormat`, `jpegQuality`)

**Breaking Changes:**
- Users passing `preserveFormat` or `jpegQuality` will get errors (intentional)
- Existing JPEG outputs in user directories remain untouched
- Stage 1 raw images unaffected (can be any format)

---

## Next Steps (Phase 2)

1. Implement MATLAB migrations in priority order:
   - `cut_micropads.m`
   - `cut_elliptical_regions.m`
   - `augment_dataset.m`
   - `extract_features.m`
   - Helper scripts

2. Add error handling for deprecated parameters

3. Run integration tests on small sample dataset

4. Proceed to Phase 3 (Python + documentation)

---

**Audit Completed:** 2025-11-06
**Ready for Phase 2:** ✅
