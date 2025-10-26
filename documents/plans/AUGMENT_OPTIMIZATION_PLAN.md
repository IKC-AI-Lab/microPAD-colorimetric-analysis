\# augment\_dataset.m Performance Optimization Plan



&nbsp;  \*\*Last Updated:\*\* 2025-01-31

&nbsp;  \*\*Target File:\*\* matlab\_scripts/augment\_dataset.m (2,736 lines)

&nbsp;  \*\*Current Performance:\*\* ~1.0s per augmented image (optimized from 3.0s)

&nbsp;  \*\*Goal:\*\* Further optimize memory usage and I/O bottlenecks

&nbsp;  \*\*Overall Progress:\*\* 2/18 tasks (11%)



&nbsp;  ## Project Context



&nbsp;  This optimization plan targets the already-optimized augment\_dataset.m script that generates synthetic training data

&nbsp;  for AI polygon detection. The script was recently optimized from 2,225 to 2,736 lines with significant performance

&nbsp;  improvements (3.0s → 1.0s per image).



&nbsp;  \*\*Related Documentation:\*\*

&nbsp;  - `AI\_DETECTION\_PLAN.md`: Main AI detection implementation plan (Phase 1 complete)

&nbsp;  - `CLAUDE.md`: Project coding standards and architecture

&nbsp;  - `AGENTS.md`: Multi-agent workflow documentation



&nbsp;  \*\*Current State:\*\*

&nbsp;  - Phase 1 of AI\_DETECTION\_PLAN complete (8/8 tasks)

&nbsp;  - Background types reduced from 7 to 4

&nbsp;  - Grid-based collision detection implemented

&nbsp;  - Artifact density reduced from 1-100 to 1-20

&nbsp;  - Overall speedup: 3x (3.0s → 1.0s per image)



&nbsp;  \*\*Remaining Bottlenecks:\*\*

&nbsp;  1. Background synthesis: Heavy double-precision operations

&nbsp;  2. Artifact masks: Full-resolution meshgrids cause memory spikes

&nbsp;  3. Corner label export: JSON serialization of large heatmap arrays

&nbsp;  4. Unused configuration blocks: Maintenance overhead



&nbsp;  ---



&nbsp;  ## Status Legend

&nbsp;  - \[ ] Not started

&nbsp;  - \[🔄] In progress

&nbsp;  - \[✅] Completed

&nbsp;  - \[⚠️] Blocked/needs attention

&nbsp;  - \[🔍] Needs review



&nbsp;  ---



&nbsp;  ## Phase 1: High-Impact Optimizations (Critical Bottlenecks)



&nbsp;  ### 1.1 Corner Label Export - Switch to MAT Format

&nbsp;  - \[ ] \*\*Priority:\*\* CRITICAL (100x I/O speedup, 95% storage reduction)

&nbsp;  - \[ ] \*\*File:\*\* `matlab\_scripts/augment\_dataset.m` (lines 2384-2506)

&nbsp;  - \[ ] \*\*Complexity:\*\* Medium

&nbsp;  - \[ ] \*\*Risk:\*\* Medium (changes output format, requires downstream Python loader update)

&nbsp;  - \[ ] \*\*Current Issue:\*\*

&nbsp;    - Serializes 4×H/4×W/4 float32 heatmaps to JSON (tens of MB per label)

&nbsp;    - JSON encoding is slow (100-500ms per label)

&nbsp;    - Storage: ~12GB for 24,000 labels (uncompressed JSON)

&nbsp;  - \[ ] \*\*Solution:\*\* Export heatmaps to compressed MAT files while keeping JSON metadata references

&nbsp;    - Store heatmaps in compressed MAT files (.mat with -v7.3 HDF5 format)

&nbsp;    - Keep JSON for metadata only (corners, image\_name, file references)

&nbsp;    - Expected: ~2GB storage (95% reduction), <5ms I/O per label (100x faster)

&nbsp;  - \[ ] \*\*Changes Required:\*\*

&nbsp;    ```matlab

&nbsp;    % BEFORE (lines 2384-2437):

&nbsp;    function export\_corner\_labels(outputDir, imageName, polygons, imageSize)

&nbsp;        % ... setup ...

&nbsp;        labels = struct();

&nbsp;        labels.quads = \[];



&nbsp;        for i = 1:numel(polygons)

&nbsp;            heatmaps = generate\_gaussian\_targets(quad, imageSize, 3);

&nbsp;            offsets = compute\_subpixel\_offsets(quad, imageSize);



&nbsp;            quadStruct = struct( ...

&nbsp;                'heatmaps', heatmaps, ...  % LARGE ARRAYS IN JSON!

&nbsp;                'offsets', offsets, ...

&nbsp;                ...);

&nbsp;        end



&nbsp;        jsonStr = jsonencode(labels, 'PrettyPrint', true);

&nbsp;        fprintf(fid, '%s', jsonStr);

&nbsp;    end



&nbsp;    % AFTER (optimized):

&nbsp;    function export\_corner\_labels(outputDir, imageName, polygons, imageSize)

&nbsp;        labelDir = fullfile(outputDir, 'labels');

&nbsp;        if ~isfolder(labelDir), mkdir(labelDir); end



&nbsp;        labelPath = fullfile(labelDir, \[imageName '.json']);

&nbsp;        heatmapPath = fullfile(labelDir, \[imageName '\_heatmaps.mat']);



&nbsp;        % JSON: metadata only (corners, image info, file references)

&nbsp;        metadata = struct();

&nbsp;        metadata.image\_size = imageSize;

&nbsp;        metadata.image\_name = imageName;

&nbsp;        metadata.heatmap\_file = \[imageName '\_heatmaps.mat'];

&nbsp;        metadata.quads = \[];



&nbsp;        % MAT: all heatmaps and offsets for this image

&nbsp;        allHeatmaps = {};

&nbsp;        allOffsets = {};



&nbsp;        for i = 1:numel(polygons)

&nbsp;            quad = polygons{i};

&nbsp;            quad = order\_corners\_clockwise(quad);



&nbsp;            heatmaps = generate\_gaussian\_targets(quad, imageSize, 3);

&nbsp;            offsets = compute\_subpixel\_offsets(quad, imageSize);



&nbsp;            allHeatmaps{i} = single(heatmaps);  % Store as single precision

&nbsp;            allOffsets{i} = single(offsets);



&nbsp;            % JSON: corners and metadata only

&nbsp;            quadStruct = struct( ...

&nbsp;                'quad\_id', i, ...

&nbsp;                'corners', quad, ...

&nbsp;                'corners\_normalized', quad ./ \[imageSize(2), imageSize(1)], ...

&nbsp;                'embedding\_id', i);



&nbsp;            if isempty(metadata.quads)

&nbsp;                metadata.quads = quadStruct;

&nbsp;            else

&nbsp;                metadata.quads(end+1) = quadStruct;

&nbsp;            end

&nbsp;        end



&nbsp;        % Write MAT file (atomic pattern, compressed)

&nbsp;        tmpPath = tempname(labelDir);

&nbsp;        save(tmpPath, 'allHeatmaps', 'allOffsets', '-v7.3');  % HDF5 compression

&nbsp;        movefile(tmpPath, heatmapPath, 'f');



&nbsp;        % Write JSON (atomic pattern)

&nbsp;        tmpPath = tempname(labelDir);

&nbsp;        fid = fopen(tmpPath, 'w');

&nbsp;        if fid < 0

&nbsp;            error('augmentDataset:jsonWrite', 'Cannot write label file: %s', labelPath);

&nbsp;        end

&nbsp;        jsonStr = jsonencode(metadata, 'PrettyPrint', true);

&nbsp;        fprintf(fid, '%s', jsonStr);

&nbsp;        fclose(fid);

&nbsp;        movefile(tmpPath, labelPath, 'f');

&nbsp;    end

&nbsp;    ```

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Generate 10 samples, verify JSON + MAT files created

&nbsp;    - \[ ] Check JSON file size (<5KB, metadata only)

&nbsp;    - \[ ] Check MAT file size (~100KB compressed vs ~10MB JSON)

&nbsp;    - \[ ] Verify heatmaps can be loaded with `load(matfile)`

&nbsp;    - \[ ] Python compatibility: Update dataset loader to read MAT files

&nbsp;  - \[ ] \*\*Downstream Impact:\*\*

&nbsp;    - \[ ] Update `python/data/dataset.py` (Phase 3.2) to load MAT files:

&nbsp;      ```python

&nbsp;      from scipy.io import loadmat



&nbsp;      # Load metadata

&nbsp;      with open(label\_path) as f:

&nbsp;          metadata = json.load(f)



&nbsp;      # Load heatmaps from MAT file

&nbsp;      mat\_path = label\_dir / metadata\['heatmap\_file']

&nbsp;      mat\_data = loadmat(mat\_path)

&nbsp;      heatmaps = mat\_data\['allHeatmaps']

&nbsp;      offsets = mat\_data\['allOffsets']

&nbsp;      ```



&nbsp;  ---



&nbsp;  ### 1.2 Artifact Masks - Normalize to Unit Square

&nbsp;  - \[ ] \*\*Priority:\*\* HIGH (eliminates multi-GB memory spikes)

&nbsp;  - \[ ] \*\*File:\*\* `matlab\_scripts/augment\_dataset.m` (lines 1253-1451, function `add\_sparse\_artifacts`)

&nbsp;  - \[ ] \*\*Complexity:\*\* Medium

&nbsp;  - \[ ] \*\*Risk:\*\* Medium (changes mask generation logic)

&nbsp;  - \[ ] \*\*Current Issue:\*\*

&nbsp;    - Creates full-resolution meshgrids for every artifact (lines 1313-1410)

&nbsp;    - When `artifactSize` approaches image diagonal (4000-5000 px), meshgrid allocates multi-GB arrays

&nbsp;    - Example: 5000×5000 meshgrid = 200MB per artifact × 30 artifacts = 6GB temporary memory

&nbsp;  - \[ ] \*\*Solution:\*\*

&nbsp;    - Generate each shape in normalized unit square (36×36 or 64×64)

&nbsp;    - Use `poly2mask` on unit square

&nbsp;    - Scale/warp final mask to target size with `imresize`

&nbsp;    - Expected: 64×64 unit mask = 32KB vs 5000×5000 = 200MB (6000x reduction)

&nbsp;  - \[ ] \*\*Implementation Strategy:\*\*

&nbsp;    ```matlab

&nbsp;    % BEFORE (lines 1311-1322, ellipse example):

&nbsp;    \[X, Y] = meshgrid(1:artifactSize, 1:artifactSize);  % HUGE ALLOCATION

&nbsp;    centerX = artifactSize / 2;

&nbsp;    centerY = artifactSize / 2;

&nbsp;    % ... rotation math ...

&nbsp;    mask = (xRot / radiusA).^2 + (yRot / radiusB).^2 <= 1;

&nbsp;    mask = imgaussfilt(double(mask), artifactCfg.ellipseBlurSigma);



&nbsp;    % AFTER (optimized):

&nbsp;    UNIT\_SIZE = 64;  % Small unit square

&nbsp;    \[X, Y] = meshgrid(1:UNIT\_SIZE, 1:UNIT\_SIZE);  % 32KB allocation

&nbsp;    centerX = UNIT\_SIZE / 2;

&nbsp;    centerY = UNIT\_SIZE / 2;

&nbsp;    % ... rotation math (same, but on unit coordinates) ...

&nbsp;    unitMask = (xRot / radiusA).^2 + (yRot / radiusB).^2 <= 1;



&nbsp;    % Apply blur to unit mask (cheap)

&nbsp;    unitMask = imgaussfilt(double(unitMask), artifactCfg.ellipseBlurSigma);



&nbsp;    % Scale to target size (hardware-accelerated)

&nbsp;    mask = imresize(unitMask, \[artifactSize, artifactSize], 'bilinear');

&nbsp;    ```

&nbsp;  - \[ ] \*\*Changes Required:\*\*

&nbsp;    - \[ ] Define `UNIT\_SIZE = 64` constant at function start (line 1253)

&nbsp;    - \[ ] Refactor ellipse mask generation (lines 1311-1322)

&nbsp;    - \[ ] Refactor rectangle mask generation (lines 1324-1337)

&nbsp;    - \[ ] Refactor quadrilateral mask generation (lines 1339-1366)

&nbsp;    - \[ ] Refactor triangle mask generation (lines 1368-1393)

&nbsp;    - \[ ] Line artifact can stay as-is (already uses distance formula)

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Generate 100 artifacts, monitor peak memory usage (<500MB)

&nbsp;    - \[ ] Visual comparison: unit-square masks vs original (should be identical after resize)

&nbsp;    - \[ ] Edge case: Very small artifacts (artifactSize < 10)

&nbsp;    - \[ ] Edge case: Very large artifacts (artifactSize > diagonal)

&nbsp;  - \[ ] \*\*Performance Validation:\*\*

&nbsp;    - \[ ] Benchmark memory: Before vs After (expect 90% reduction in peak)

&nbsp;    - \[ ] Benchmark time: Imresize is GPU-accelerated, should be faster than meshgrid



&nbsp;  ---



&nbsp;  ### 1.3 Background Synthesis - Single Precision

&nbsp;  - \[ ] \*\*Priority:\*\* HIGH (50% memory reduction, 50% faster convolution)

&nbsp;  - \[ ] \*\*File:\*\* `matlab\_scripts/augment\_dataset.m` (lines 1138-1252)

&nbsp;  - \[ ] \*\*Complexity:\*\* Low

&nbsp;  - \[ ] \*\*Risk:\*\* Low (transparent change, same visual output)

&nbsp;  - \[ ] \*\*Current Issue:\*\*

&nbsp;    - All texture generation uses double precision (`randn()` defaults to double)

&nbsp;    - Gaussian filtering (`imgaussfilt`) operates on double precision

&nbsp;    - Example: 4000×3000 double noise = 91MB vs single = 46MB

&nbsp;  - \[x] \*\*Solution:\*\* Single-precision background path and pooled textures implemented 2025-01-31

&nbsp;    - Cast `randn()` outputs to `single` immediately

&nbsp;    - Use `single()` for all intermediate texture arrays

&nbsp;    - Reuse a preallocated `single` buffer for each noise layer (write in-place rather than allocating new height×width arrays)

&nbsp;    - Final composite to uint8 remains unchanged

&nbsp;    - Expected: 50% memory reduction, 30-50% faster Gaussian filtering

&nbsp;  - \[x] \*\*Changes Implemented:\*\*

&nbsp;    ```matlab

&nbsp;    % BEFORE (lines 1149-1158):

&nbsp;    baseRGB = textureCfg.uniformBaseRGB + randi(\[...], \[1, 3]);

&nbsp;    texture = randn(height, width) \* (...);  % DOUBLE PRECISION



&nbsp;    % ... speckled ...

&nbsp;    highFreqNoise = randn(height, width) \* textureCfg.speckleHighFreq;

&nbsp;    lowFreqNoise = imgaussfilt(randn(height, width), 8) \* textureCfg.speckleLowFreq;



&nbsp;    % AFTER (optimized):

&nbsp;    baseRGB = textureCfg.uniformBaseRGB + randi(\[...], \[1, 3]);

&nbsp;    texture = single(randn(height, width)) \* single(...);  % SINGLE PRECISION



&nbsp;    % ... speckled ...

&nbsp;    highFreqNoise = single(randn(height, width)) \* single(textureCfg.speckleHighFreq);

&nbsp;    lowFreqNoise = imgaussfilt(single(randn(height, width)), 8) \* single(textureCfg.speckleLowFreq);

&nbsp;    ```

&nbsp;  - \[x] \*\*Functions Updated:\*\*

&nbsp;    - \[x] `generate\_realistic\_lab\_surface()` - lines 1149-1158 (uniform, speckled)

&nbsp;    - \[x] `generate\_laminate\_texture()` - line 1206

&nbsp;    - \[x] `generate\_skin\_texture()` - lines 1215-1219

&nbsp;    - \[x] `add\_lighting\_gradient()` - lines 1235-1240 (meshgrid + projection)

&nbsp;    - \[x] Preallocate a shared `single` buffer (e.g., `texture = zeros(height, width, 'single')`) and overwrite it with each noise component before final conversion

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Visual comparison: single vs double backgrounds (should be identical)

&nbsp;    - \[ ] Check final image is still uint8 (no degradation)

&nbsp;    - \[ ] Verify no overflow/underflow issues with single precision

&nbsp;  - \[ ] \*\*Performance Validation:\*\*

&nbsp;    - \[ ] Measure memory: Peak RAM usage during background generation

&nbsp;    - \[ ] Measure time: Gaussian filter speedup (expect 30-50% faster)



&nbsp;  ---



&nbsp;  ## Phase 2: Medium-Impact Optimizations (Memory + Speed)



&nbsp;  ### 2.1 Artifact Blur Softening - Separable Convolution

&nbsp;  - \[ ] \*\*Priority:\*\* MEDIUM (3-5x faster blur, 80% less memory)

&nbsp;  - \[ ] \*\*File:\*\* `matlab\_scripts/augment\_dataset.m` (lines 1322, 1337, 1366, 1393, 1410)

&nbsp;  - \[ ] \*\*Complexity:\*\* Low

&nbsp;  - \[ ] \*\*Risk:\*\* Low (transparent mathematical optimization)

&nbsp;  - \[ ] \*\*Current Issue:\*\*

&nbsp;    - `imgaussfilt(mask, sigma)` operates on full artifact-size masks

&nbsp;    - 2D Gaussian convolution is O(N²) in kernel size

&nbsp;    - Large masks (5000×5000) with blur take 100-200ms each

&nbsp;  - \[ ] \*\*Solution:\*\*

&nbsp;    - Apply `imgaussfilt` to unit-square mask (64×64) before scaling

&nbsp;    - Separable 1-D convolution is faster than 2-D

&nbsp;    - Upscale preserves softness

&nbsp;    - Expected: 3-5x faster blur (operates on 64×64 instead of 5000×5000)

&nbsp;  - \[ ] \*\*Implementation:\*\*

&nbsp;    ```matlab

&nbsp;    % BEFORE (line 1322):

&nbsp;    mask = (xRot / radiusA).^2 + (yRot / radiusB).^2 <= 1;

&nbsp;    mask = imgaussfilt(double(mask), artifactCfg.ellipseBlurSigma);  % EXPENSIVE ON LARGE MASK



&nbsp;    % AFTER (optimized - already done if Phase 1.2 implemented):

&nbsp;    unitMask = (xRot / radiusA).^2 + (yRot / radiusB).^2 <= 1;

&nbsp;    unitMask = imgaussfilt(double(unitMask), artifactCfg.ellipseBlurSigma);  % CHEAP ON 64×64

&nbsp;    mask = imresize(unitMask, \[artifactSize, artifactSize], 'bilinear');

&nbsp;    ```

&nbsp;  - \[ ] \*\*Note:\*\* This optimization is automatically achieved by implementing Phase 1.2

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Visual comparison: blur-then-resize vs resize-then-blur

&nbsp;    - \[ ] Verify softness is preserved in final composited image

&nbsp;  - \[ ] \*\*Performance Validation:\*\*

&nbsp;    - \[ ] Benchmark blur time: 64×64 vs 5000×5000 (expect 3-5x speedup)



&nbsp;  ---



&nbsp;  ### 2.2 Motion Blur PSF Caching

&nbsp;  - \[ ] \*\*Priority:\*\* MEDIUM (10x speedup when blur enabled)

&nbsp;  - \[ ] \*\*File:\*\* `matlab\_scripts/augment\_dataset.m` (lines 2311-2317, function `apply\_motion\_blur`)

&nbsp;  - \[ ] \*\*Complexity:\*\* Low

&nbsp;  - \[ ] \*\*Risk:\*\* Low (transparent caching)

&nbsp;  - \[ ] \*\*Current Issue:\*\*

&nbsp;    - `fspecial('motion', len, ang)` recomputes identical kernels

&nbsp;    - Called 25% of images (blurProbability = 0.25)

&nbsp;    - Same kernel likely used multiple times (len/ang have limited range)

&nbsp;  - \[ ] \*\*Solution:\*\*

&nbsp;    - Use persistent cache map keyed by `(len, round(ang))`

&nbsp;    - Cache PSF results across function calls

&nbsp;    - Expected: 10x speedup for repeated kernels

&nbsp;  - \[ ] \*\*Implementation:\*\*

&nbsp;    ```matlab

&nbsp;    % BEFORE (lines 2311-2317):

&nbsp;    function img = apply\_motion\_blur(img)

&nbsp;        len = 4 + randi(4);            % 5-8 px

&nbsp;        ang = rand() \* 180;            % degrees

&nbsp;        psf = fspecial('motion', len, ang);  % RECOMPUTES EVERY TIME

&nbsp;        img = imfilter(img, psf, 'replicate');

&nbsp;    end



&nbsp;    % AFTER (optimized):

&nbsp;    function img = apply\_motion\_blur(img)

&nbsp;        persistent psfCache

&nbsp;        if isempty(psfCache)

&nbsp;            psfCache = containers.Map('KeyType', 'char', 'ValueType', 'any');

&nbsp;        end



&nbsp;        len = 4 + randi(4);            % 5-8 px

&nbsp;        ang = rand() \* 180;            % degrees

&nbsp;        angRounded = round(ang);       % Round to nearest degree



&nbsp;        % Create cache key

&nbsp;        cacheKey = sprintf('%d\_%d', len, angRounded);



&nbsp;        % Check cache

&nbsp;        if isKey(psfCache, cacheKey)

&nbsp;            psf = psfCache(cacheKey);

&nbsp;        else

&nbsp;            psf = fspecial('motion', len, angRounded);

&nbsp;            psfCache(cacheKey) = psf;

&nbsp;        end



&nbsp;        img = imfilter(img, psf, 'replicate');

&nbsp;    end

&nbsp;    ```

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Generate 1000 blurred images, verify cache hits increase over time

&nbsp;    - \[ ] Check cache size doesn't grow unbounded (len: 5-8, ang: 0-180 = ~720 entries max)

&nbsp;    - \[ ] Visual comparison: cached vs fresh PSF (should be identical)

&nbsp;  - \[ ] \*\*Performance Validation:\*\*

&nbsp;    - \[ ] Measure `fspecial` time vs cache lookup (expect 10x speedup)

&nbsp;    - \[ ] Monitor cache hit rate (expect >80% after 100 images)



&nbsp;  ---



&nbsp;  ## Phase 3: Low-Impact Optimizations (Maintenance)



&nbsp;  ### 3.1 Remove Unused Configuration Blocks

&nbsp;  - \[ ] \*\*Priority:\*\* LOW (code clarity only, no performance impact)

&nbsp;  - \[ ] \*\*File:\*\* `matlab\_scripts/augment\_dataset.m` (lines 125-138)

&nbsp;  - \[ ] \*\*Complexity:\*\* Low

&nbsp;  - \[ ] \*\*Risk:\*\* Low (dead code removal)

&nbsp;  - \[ ] \*\*Current Issue:\*\*

&nbsp;    - `CORNER\_OCCLUSION` and `EDGE\_DEGRADATION` structs defined (lines 125-138)

&nbsp;    - Configuration logged but never used in pipeline

&nbsp;    - Helper functions mentioned in AI\_DETECTION\_PLAN (lines 97-116) never implemented

&nbsp;    - Creates maintenance confusion

&nbsp;  - \[ ] \*\*Solution:\*\*

&nbsp;    - Option A: Implement the helper functions (adds complexity)

&nbsp;    - Option B: Remove unused config blocks (simplifies code)

&nbsp;    - Recommendation: Option B (defer implementation to Phase 1 of AI\_DETECTION\_PLAN if needed)

&nbsp;  - \[ ] \*\*Changes Required:\*\*

&nbsp;    ```matlab

&nbsp;    % REMOVE lines 125-138:

&nbsp;    % === CORNER ROBUSTNESS AUGMENTATION ===

&nbsp;    CORNER\_OCCLUSION = struct( ...

&nbsp;        'probability', 0.15, ...

&nbsp;        'occlusionTypes', {{'finger', 'shadow', 'small\_object'}}, ...

&nbsp;        'sizeRange', \[15, 40], ...

&nbsp;        'maxCornersPerPolygon', 2, ...

&nbsp;        'intensityRange', \[-80, -30]);



&nbsp;    EDGE\_DEGRADATION = struct( ...

&nbsp;        'probability', 0.25, ...

&nbsp;        'blurTypes', {{'gaussian', 'motion'}}, ...

&nbsp;        'blurRadiusRange', \[1.5, 4.0], ...

&nbsp;        'affectsEdgesOnly', true, ...

&nbsp;        'edgeWidth', 10);

&nbsp;    ```

&nbsp;  - \[ ] \*\*Alternative:\*\* Add comment explaining deferred implementation

&nbsp;    ```matlab

&nbsp;    % === CORNER ROBUSTNESS AUGMENTATION (DEFERRED) ===

&nbsp;    % These features are planned for AI\_DETECTION\_PLAN Phase 1.2

&nbsp;    % but not yet implemented. See AI\_DETECTION\_PLAN.md lines 75-119.

&nbsp;    % CORNER\_OCCLUSION = struct(...);  % Placeholder

&nbsp;    % EDGE\_DEGRADATION = struct(...);  % Placeholder

&nbsp;    ```

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Verify script runs after removal (no references to these structs)

&nbsp;    - \[ ] Grep for `CORNER\_OCCLUSION` and `EDGE\_DEGRADATION` usage (should be none)



&nbsp;  ---



&nbsp;  ### 3.2 Poisson-Disk Placement Evaluation

&nbsp;  - \[ ] \*\*Priority:\*\* MEDIUM (reduce placement retries without touching image quality)

&nbsp;  - \[ ] \*\*File:\*\* `matlab\_scripts/augment\_dataset.m` (lines 2158-2234, function `place\_polygons\_nonoverlapping`)

&nbsp;  - \[ ] \*\*Complexity:\*\* Medium

&nbsp;  - \[ ] \*\*Risk:\*\* Low (deterministic sampling with existing spacing guarantees)

&nbsp;  - \[ ] \*\*Current Issue:\*\*

&nbsp;    - Rejection sampling still loops up to 50 times when polygons are large or clustered near the background margin

&nbsp;    - Fallback path (positioningOverlap warning) fires in ~8% of synthetic scenes, allowing controlled overlaps

&nbsp;    - Placement runtime spikes and fallback path fires under dense layouts despite grid acceleration

&nbsp;  - \[x] \*\*Solution:\*\* Poisson-disk placement integrated with grid-aware fallback (2025-01-31)

&nbsp;    - Combine Bridson-style Poisson seeds with best-candidate scoring to maximize spacing variety

&nbsp;    - Preserve largest-first ordering while evaluating random fallbacks for dense layouts

&nbsp;    - Keep legacy overlap path only as last-resort guard (unchanged)

&nbsp;  - \[x] \*\*Implementation Summary:\*\*

&nbsp;    ```matlab

&nbsp;    candidates = generate_poisson_disk_points(...);
&nbsp;    for bbox = largest_first(polygons)
&nbsp;        best = bestCandidate(candidates, bbox, maxAttempts);
&nbsp;        if isempty(best)
&nbsp;            error("fallback");
&nbsp;        end
&nbsp;        commit(best);  % updates spatial grid + marks candidate as used
&nbsp;    end

&nbsp;    ```

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Worst case: seven large polygons nearly spanning the background (expect no fallback)

&nbsp;    - \[ ] Random batches: track average attempts per polygon (target \<5)

&nbsp;    - \[ ] Verify margin/min-spacing invariants and downstream coordinate correctness

&nbsp;  - \[ ] \*\*Performance Validation:\*\*

&nbsp;    - \[ ] Profile placement runtime before/after (expect \>=20% improvement)

&nbsp;    - \[ ] Observe fallback rate (target \<1% of scenes)

&nbsp;    - \[ ] Confirm no regression in occlusion logic and ellipse alignment

&nbsp;  ### 3.3 Background Texture Pool (Optional)

&nbsp;  - \[ ] \*\*Priority:\*\* LOW (cache after single-precision path is stable)

&nbsp;  - \[ ] \*\*File:\*\* `matlab\_scripts/augment\_dataset.m` (lines 1138-1252, `generate_realistic_lab_surface`)

&nbsp;  - \[ ] \*\*Complexity:\*\* Medium

&nbsp;  - \[ ] \*\*Risk:\*\* Low (pooled assets stay in single precision; compositing still adds stochastic variation)

&nbsp;  - \[ ] \*\*Current Issue:\*\*

&nbsp;    - Even after single-precision, each augmentation rebuilds 4K noise fields per background

&nbsp;    - Procedural parameters overlap, so many textures are visually similar but expensive to regenerate

&nbsp;    - No reuse between augmentations within the same MATLAB session

&nbsp;  - \[x] \*\*Solution:\*\* Background texture pooling with scheduled refresh integrated (2025-01-31)

&nbsp;    - Lazily populate a capped single-precision pool (auto-sized per resolution to stay \<512 MB total)

&nbsp;    - For each scene, draw a texture, apply lightweight jitters (shift/flip/scale), and keep the base for reuse

&nbsp;    - Refresh entries by clearing slots after N uses so the next borrower regenerates with new noise

&nbsp;  - \[x] \*\*Implementation Summary:\*\*

&nbsp;    ```matlab

&nbsp;    persistent poolState
&nbsp;    if isempty(poolState) || texture_pool_config_changed(poolState, w, h, cfg.texture)
&nbsp;        poolState = initialize_background_texture_pool(w, h, cfg.texture);
&nbsp;    end
&nbsp;    texture = borrow_background_texture(surfaceType, w, h, cfg.texture);

&nbsp;    ```

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Visual audit: ensure pooled backgrounds remain diverse (SSIM \<0.90 across 50 samples)

&nbsp;    - \[ ] Memory profile: peak usage unchanged (pool stored as single precision)

&nbsp;    - \[ ] Verify deterministic behaviour when `rngSeed` is supplied

&nbsp;  - \[ ] \*\*Performance Validation:\*\*

&nbsp;    - \[ ] Benchmark background generation time before/after warm-up (expect additional 20-25% reduction)

&nbsp;    - \[ ] Measure throughput over 200 scenes (expect smoother runtimes due to cache hits)

&nbsp;    - \[ ] Confirm no regression in artifact placement or photometric augmentation



&nbsp;  ---



&nbsp;  ## Phase 4: Validation \& Testing



&nbsp;  ### 4.1 Performance Benchmarking

&nbsp;  - \[ ] \*\*Task:\*\* Measure optimization impact

&nbsp;  - \[ ] \*\*Baseline Metrics (before optimization):\*\*

&nbsp;    - \[ ] Time per augmented image: ~1.0s

&nbsp;    - \[ ] Peak memory usage: ~8GB (with large artifacts)

&nbsp;    - \[ ] JSON label I/O: ~200ms per label

&nbsp;    - \[ ] Background generation: ~150ms

&nbsp;    - \[ ] Artifact generation: ~300ms

&nbsp;  - \[ ] \*\*Target Metrics (after optimization):\*\*

&nbsp;    - \[ ] Time per augmented image: <0.7s (30% improvement)

&nbsp;    - \[ ] Peak memory usage: <2GB (75% reduction)

&nbsp;    - \[ ] MAT label I/O: <5ms per label (40x faster)

&nbsp;    - \[ ] Background generation: <100ms (33% faster)

&nbsp;    - \[ ] Artifact generation: <150ms (50% faster)

&nbsp;  - \[ ] \*\*Benchmark Script:\*\*

&nbsp;    ```matlab

&nbsp;    function results = benchmark\_augmentation()

&nbsp;        % Run augmentation with profiling

&nbsp;        profile on



&nbsp;        tic;

&nbsp;        augment\_dataset('numAugmentations', 10, 'rngSeed', 42);

&nbsp;        totalTime = toc;



&nbsp;        profile viewer

&nbsp;        p = profile('info');



&nbsp;        results = struct();

&nbsp;        results.totalTime = totalTime;

&nbsp;        results.timePerImage = totalTime / 10;

&nbsp;        results.profileData = p;



&nbsp;        % Memory profiling (requires manual monitoring)

&nbsp;        fprintf('Manually monitor Task Manager during run\\n');

&nbsp;    end

&nbsp;    ```

&nbsp;  - \[ ] \*\*Run Benchmark:\*\*

&nbsp;    - \[ ] Before optimizations: `results\_before = benchmark\_augmentation()`

&nbsp;    - \[ ] After Phase 1: `results\_phase1 = benchmark\_augmentation()`

&nbsp;    - \[ ] After Phase 2: `results\_phase2 = benchmark\_augmentation()`

&nbsp;    - \[ ] After Phase 3: `results\_final = benchmark\_augmentation()`



&nbsp;  ---



&nbsp;  ### 4.2 Visual Quality Validation

&nbsp;  - \[ ] \*\*Task:\*\* Ensure optimizations don't degrade output quality

&nbsp;  - \[ ] \*\*Test Cases:\*\*

&nbsp;    - \[ ] Generate 100 samples before and after optimizations

&nbsp;    - \[ ] Visual comparison: Check for artifacts, blur differences, color shifts

&nbsp;    - \[ ] Load MAT files in Python: Verify heatmaps match JSON baseline

&nbsp;    - \[ ] Corner detection accuracy: Run on validation set (expect no change)

&nbsp;  - \[ ] \*\*Validation Script:\*\*

&nbsp;    ```matlab

&nbsp;    function validate\_quality(beforeDir, afterDir, numSamples)

&nbsp;        % Compare images from before/after optimization

&nbsp;        beforeFiles = dir(fullfile(beforeDir, '\*.jpg'));

&nbsp;        afterFiles = dir(fullfile(afterDir, '\*.jpg'));



&nbsp;        indices = randperm(numel(beforeFiles), min(numSamples, numel(beforeFiles)));



&nbsp;        for i = indices

&nbsp;            imgBefore = imread(fullfile(beforeFiles(i).folder, beforeFiles(i).name));

&nbsp;            imgAfter = imread(fullfile(afterFiles(i).folder, afterFiles(i).name));



&nbsp;            % Compute SSIM (structural similarity)

&nbsp;            ssimVal = ssim(imgBefore, imgAfter);



&nbsp;            if ssimVal < 0.95

&nbsp;                warning('Low SSIM (%.3f) for %s', ssimVal, beforeFiles(i).name);

&nbsp;            end

&nbsp;        end



&nbsp;        fprintf('Quality validation complete\\n');

&nbsp;    end

&nbsp;    ```



&nbsp;  ---



&nbsp;  ### 4.3 Code Review \& Testing

&nbsp;  - \[ ] \*\*Task:\*\* Review optimized code before marking complete

&nbsp;  - \[ ] \*\*Checklist:\*\*

&nbsp;    - \[ ] All functions maintain atomic write pattern (tempfile + movefile)

&nbsp;    - \[ ] Error handling preserved for file I/O

&nbsp;    - \[ ] Memory allocations are properly released (no leaks)

&nbsp;    - \[ ] Single precision conversions don't cause overflow

&nbsp;    - \[ ] Cache implementations are thread-safe (if needed)

&nbsp;    - \[ ] Comments updated to reflect optimizations

&nbsp;  - \[ ] \*\*Testing Strategy:\*\*

&nbsp;    - \[ ] Unit tests for each optimized function

&nbsp;    - \[ ] Integration test: Full pipeline run (100 images)

&nbsp;    - \[ ] Edge cases: Tiny images (<100×100), huge images (>8000×6000)

&nbsp;    - \[ ] Memory stress test: Generate 1000 images, monitor RAM



&nbsp;  ---



&nbsp;  ## Phase 5: Documentation \& Deployment



&nbsp;  ### 5.1 Update AI\_DETECTION\_PLAN.md

&nbsp;  - \[ ] \*\*Task:\*\* Document optimization results in AI\_DETECTION\_PLAN

&nbsp;  - \[ ] \*\*Add Section:\*\* "Phase 1.9: Performance Optimization Results"

&nbsp;    ```markdown

&nbsp;    ### 1.9 Performance Optimization Results

&nbsp;    - \[✅] \*\*Task:\*\* Optimize augment\_dataset.m for memory and I/O

&nbsp;    - \[✅] \*\*Optimizations Applied:\*\*

&nbsp;      - Corner label export: JSON → MAT format (100x I/O speedup)

&nbsp;      - Artifact masks: Unit-square normalization (90% memory reduction)

&nbsp;      - Background synthesis: Single precision (50% memory reduction)

&nbsp;      - Motion blur: PSF caching (10x speedup)

&nbsp;    - \[✅] \*\*Results:\*\*

&nbsp;      - Time per image: 1.0s → 0.7s (30% faster)

&nbsp;      - Peak memory: 8GB → 2GB (75% reduction)

&nbsp;      - Label I/O: 200ms → 5ms (40x faster)

&nbsp;    - \[✅] \*\*Test:\*\* Validated on 100 samples, no visual degradation (SSIM > 0.98)

&nbsp;    ```



&nbsp;  ---



&nbsp;  ### 5.2 Update CLAUDE.md

&nbsp;  - \[ ] \*\*Task:\*\* Document new MAT file format for corner labels

&nbsp;  - \[ ] \*\*Add to "File Naming Conventions" section:\*\*

&nbsp;    ```markdown

&nbsp;    ### Corner Label Files (Augmentation Output)

&nbsp;    - \*\*JSON\*\*: `{imageName}.json` - Metadata only (corners, image info)

&nbsp;    - \*\*MAT\*\*: `{imageName}\_heatmaps.mat` - Heatmaps and offsets (HDF5 compressed)

&nbsp;    - \*\*Format:\*\*

&nbsp;      - JSON: `{"image\_name": "...", "heatmap\_file": "...\_heatmaps.mat", "quads": \[...]}`

&nbsp;      - MAT: `allHeatmaps{quadIdx}(4, H/4, W/4)` - single precision Gaussian targets

&nbsp;      - MAT: `allOffsets{quadIdx}(4, 2)` - single precision sub-pixel offsets

&nbsp;    - \*\*Loading (MATLAB):\*\*

&nbsp;      ```matlab

&nbsp;      data = load('image\_001\_heatmaps.mat');

&nbsp;      heatmaps = data.allHeatmaps{1};  % First quad

&nbsp;      ```

&nbsp;    - \*\*Loading (Python):\*\*

&nbsp;      ```python

&nbsp;      from scipy.io import loadmat

&nbsp;      data = loadmat('image\_001\_heatmaps.mat')

&nbsp;      heatmaps = data\['allHeatmaps']\[0]\[0]  # First quad

&nbsp;      ```

&nbsp;    ```



&nbsp;  ---



&nbsp;  ### 5.3 Create Performance Report

&nbsp;  - \[ ] \*\*Task:\*\* Document optimization methodology and results

&nbsp;  - \[ ] \*\*File:\*\* `documents/AUGMENT\_OPTIMIZATION\_REPORT.md`

&nbsp;  - \[ ] \*\*Sections:\*\*

&nbsp;    - Executive Summary

&nbsp;    - Methodology (profiling, bottleneck identification)

&nbsp;    - Optimizations Implemented (before/after code snippets)

&nbsp;    - Performance Results (tables, graphs)

&nbsp;    - Visual Quality Validation (SSIM scores, sample images)

&nbsp;    - Lessons Learned

&nbsp;  - \[ ] \*\*Include:\*\*

&nbsp;    - Profiling screenshots (MATLAB profiler)

&nbsp;    - Memory usage graphs (before/after)

&nbsp;    - Benchmark tables (time, memory, storage)



&nbsp;  ---



&nbsp;  ## Progress Tracking



&nbsp;  ### Overall Status

&nbsp;  - \[ ] Phase 1: High-Impact Optimizations (1/3 tasks) - \*\*IN PROGRESS\*\*

&nbsp;  - \[ ] Phase 2: Medium-Impact Optimizations (0/2 tasks) - \*\*NOT STARTED\*\*

&nbsp;  - \[ ] Phase 3: Low-Impact Optimizations (1/2 tasks) - \*\*IN PROGRESS\*\*

&nbsp;  - \[ ] Phase 4: Validation \& Testing (0/3 tasks) - \*\*NOT STARTED\*\*

&nbsp;  - \[ ] Phase 5: Documentation \& Deployment (0/3 tasks) - \*\*NOT STARTED\*\*



&nbsp;  ### Key Milestones

&nbsp;  - \[ ] Phase 1 complete: Critical bottlenecks eliminated

&nbsp;  - \[ ] Phase 2 complete: Memory optimizations validated

&nbsp;  - \[ ] Phase 4 complete: Quality validation passed (SSIM > 0.95)

&nbsp;  - \[ ] Phase 5 complete: Documentation updated, optimization report published



&nbsp;  ### Estimated Timeline

&nbsp;  - Phase 1: 4-6 hours (complex changes, testing)

&nbsp;  - Phase 2: 2-3 hours (simpler optimizations)

&nbsp;  - Phase 3: 1 hour (cleanup only)








