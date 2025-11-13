function augment_dataset(varargin)
    %% microPAD Colorimetric Analysis — Dataset Augmentation Tool
    %% Generates synthetic training datasets from microPAD paper images for polygon detection
    %% Author: Veysel Y. Yilmaz
    %
    % FEATURES:
    % - Procedural textured backgrounds (uniform, speckled, laminate, skin)
    % - Perspective and rotation transformations
    % - Random spatial placement with uniform distribution
    % - Independent rotation per concentration region
    % - Collision detection to prevent overlap
    % - Optional photometric augmentation, white-balance jitter, and blur
    % - Optional thin occlusions over polygons (hair/strap-like) for robustness
    % - Variable-density distractor artifacts (1-20 per image, unconstrained placement)
    % - Polygon-shaped distractor generation for detection robustness
    %
    % Generates synthetic training data by applying geometric and photometric
    % transformations to microPAD paper images and their labeled concentration regions.
    %
    % PIPELINE:
    % 1. Copy real captures from 1_dataset/ into augmented_1_dataset/ (passthrough)
    % 2. Load polygon coordinates from 2_micropads/
    % 3. Load ellipse coordinates from 3_elliptical_regions/ (optional)
    % 4. Generate N synthetic augmentations per paper (augIdx = 1..N)
    % 5. Write outputs to augmented_* directories
    %
    % TRANSFORMATION ORDER (applied to each concentration region):
    %   a) Shared perspective transformation (same for all regions from one paper)
    %   b) Shared rotation (same for all regions from one paper)
    %   c) Independent rotation (unique per region)
    %   d) Random spatial translation (Gaussian-distributed, center-biased)
    %   e) Composite onto procedural background
    %
    % OUTPUT STRUCTURE:
    %   augmented_1_dataset/[phone]/           - Real copies + synthetic scenes
    %   augmented_2_micropads/[phone]/con_*/   - Polygon crops + coordinates.txt
    %   augmented_3_elliptical_regions/[phone]/con_*/ - Elliptical patches + coordinates.txt
    %
    % IMPORTANT: If you change backgroundWidth/backgroundHeight parameters mid-session,
    % run 'clear functions' to reset the internal texture cache.
    %
    % Parameters (Name-Value):
    % - 'numAugmentations' (positive integer, default 10): synthetic versions per paper
    %   Note: Real captures are always copied; synthetic scenes are labelled *_aug_XXX
    % - 'rngSeed' (numeric, optional): for reproducibility
    % - 'phones' (cellstr/string array): subset of phones to process
    % - 'backgroundWidth' (positive integer, default 4000): synthetic background width override
    % - 'backgroundHeight' (positive integer, default 3000): synthetic background height override
    % - 'scenePrefix' (char/string, default 'synthetic'): synthetic filename prefix
    % - 'photometricAugmentation' (logical, default true): enable color/lighting variation
    % - 'blurProbability' (0-1, default 0.25): fraction of samples with Gaussian blur
    % - 'motionBlurProbability' (0-1, default 0.15): fraction of samples with motion blur
    % - 'occlusionProbability' (0-1, default 0.0): fraction of samples with thin occlusions
    % - 'independentRotation' (logical, default true): enable per-polygon rotation
    % - 'extremeCasesProbability' (0-1, default 0.10): fraction using extreme viewpoints
    % - 'enableDistractorPolygons' (logical, default true): add synthetic look-alike distractors
    % - 'distractorMultiplier' (numeric, default 0.6): scale factor for distractor count
    % - 'distractorMaxCount' (integer, default 6): maximum distractors per scene
    % - 'paperDamageProbability' (0-1, default 0.5): fraction of polygons with physical defects
    % - 'damageSeed' (numeric, optional): RNG seed for reproducible damage patterns
    % - 'damageProfileWeights' (struct, optional): custom damage profile probabilities
    %     Default: struct('minimalWarp',0.30, 'cornerChew',0.45, 'sideCollapse',0.25)
    % - 'maxAreaRemovalFraction' (0-1, default 0.40): max removable fraction (per ellipse or micropad fallback)
    %
    % PAPER DAMAGE AUGMENTATION:
    %   Simulates realistic paper defects from storage, handling, and environmental factors.
    %   Three damage profiles with different severity levels:
    %     - minimalWarp (30%):   Subtle projective jitter + edge bending only
    %     - cornerChew (45%):    Corner clips/tears + edge cuts + surface wear
    %     - sideCollapse (25%):  Heavy side damage (bites dominate over corners)
    %
    %   Three-phase pipeline:
    %     Phase 0: Prepare ellipse guard masks and core protection zone
    %     Phase 1: Base warp & shear (projective jitter, nonlinear edge bending)
    %     Phase 2: Structural cuts (corner clips, tears, side bites, tapered edges)
    %     Phase 3: Edge wear & thickness (wave noise, fraying, shadows)
    %
    %   Protected regions (never damaged):
    %     - Inner ellipse cores sized by (1 - maxAreaRemovalFraction) when labels exist
    %     - Bridge paths connecting ellipses to polygon centroid (prevent islands)
    %     - Core polygon fallback when ellipses are missing (same area fraction)
    %
    % Examples:
    %   augment_dataset('numAugmentations', 5, 'rngSeed', 42)
    %   augment_dataset('phones', {'iphone_11'}, 'photometricAugmentation', false)
    %   augment_dataset('paperDamageProbability', 0.7, 'damageSeed', 42)
    %   augment_dataset('damageProfileWeights', struct('minimalWarp',0.5,'cornerChew',0.3,'sideCollapse',0.2))
    %   augment_dataset('maxAreaRemovalFraction', 0.3)

    %% =====================================================================
    %% CONFIGURATION CONSTANTS
    %% =====================================================================
    DEFAULT_INPUT_STAGE1 = '1_dataset';
    DEFAULT_INPUT_STAGE2 = '2_micropads';
    DEFAULT_INPUT_STAGE3_COORDS = '3_elliptical_regions';
    DEFAULT_OUTPUT_STAGE1 = 'augmented_1_dataset';
    DEFAULT_OUTPUT_STAGE2 = 'augmented_2_micropads';
    DEFAULT_OUTPUT_STAGE3 = 'augmented_3_elliptical_regions';

    COORDINATE_FILENAME = 'coordinates.txt';
    CONCENTRATION_PREFIX = 'con_';
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    MIN_VALID_POLYGON_AREA = 100;  % square pixels

    % Camera/transformation parameters
    CAMERA = struct( ...
        'maxAngleDeg', 60, ...
        'xRange', [-0.8, 0.8], ...
        'yRange', [-0.8, 0.8], ...
        'zRange', [1.2, 3.0], ...
        'coverageCenter', 0.97, ...
        'coverageOffcenter', 0.90);

    ROTATION_RANGE = [0, 360];  % degrees

    % Background generation parameters
    TEXTURE = struct( ...
        'speckleHighFreq', 35, ...
        'speckleLowFreq', 20, ...
        'uniformBaseRGB', [220, 218, 215], ...
        'uniformVariation', 15, ...
        'uniformNoiseRange', [10, 25], ...  % uint8 intensity units
        'poolSize', 16, ...
        'poolRefreshInterval', 25, ...  % images before texture refresh
        'poolShiftPixels', 48, ...  % pixels
        'poolScaleRange', [0.9, 1.1], ...
        'poolFlipProbability', 0.15, ...
        'laminateNoiseStrength', 5, ...
        'skinLowFreqStrength', 6, ...
        'skinMidFreqStrength', 2, ...
        'skinHighFreqStrength', 1, ...
        'poolMaxMemoryMB', 512);

    % Artifact generation parameters
    ARTIFACTS = struct( ...
        'unitMaskSize', 64, ...  % pixels
        'countRange', [5, 40], ...
        'sizeRangePercent', [0.01, 0.75], ...  % fraction of background dimensions
        'minSizePixels', 3, ...  % pixels
        'overhangMargin', 0.5, ...  % fraction of artifact size
        'lineWidthRatio', 0.02, ...  % fraction of artifact size
        'lineRotationPadding', 10, ...  % pixels
        'ellipseRadiusARange', [0.4, 0.7], ...  % fraction of unitMaskSize
        'ellipseRadiusBRange', [0.3, 0.6], ...  % fraction of unitMaskSize
        'rectangleSizeRange', [0.5, 0.9], ...  % fraction of unitMaskSize
        'quadSizeRange', [0.5, 0.9], ...  % fraction of unitMaskSize
        'quadPerturbation', 0.15, ...  % fraction of quad size
        'triangleSizeRange', [0.6, 0.9], ...  % fraction of unitMaskSize
        'lineIntensityRange', [-80, -40], ...  % uint8 intensity units
        'blobDarkIntensityRange', [-60, -30], ...  % uint8 intensity units
        'blobLightIntensityRange', [20, 50]);  % uint8 intensity units

    % Polygon placement parameters
    PLACEMENT = struct( ...
        'margin', 50, ...  % pixels from edge
        'minSpacing', 30, ...  % pixels between regions
        'maxOverlapRetries', 5);

    % Distractor polygon parameters (synthetic look-alikes)
    DISTRACTOR_POLYGONS = struct( ...
        'enabled', true, ...
        'minCount', 1, ...
        'maxCount', 10, ...
        'sizeScaleRange', [0.5, 1.5], ...  % fraction of original polygon size
        'maxPlacementAttempts', 30, ...
        'brightnessOffsetRange', [-20, 20], ...  % uint8 intensity units
        'contrastScaleRange', [0.9, 1.15], ...
        'noiseStd', 6, ...  % uint8 intensity units
        'typeWeights', [1, 1, 1], ...
        'outlineWidthRange', [1.5, 4.0], ...  % pixels
        'textureGainRange', [0.06, 0.18], ...  % normalized modulation strength
        'textureSurfaceTypes', [1, 2, 3, 4]);        % Background texture primitives to reuse

    % Paper damage augmentation parameters
    PAPER_DAMAGE = struct( ...
        'probability', 0.3, ...  % Fraction of polygons with damage
        'profileWeights', struct('minimalWarp', 0.30, 'cornerChew', 0.45, 'sideCollapse', 0.25), ...
        'cornerClipRange', [0.06, 0.22], ...  % Fraction of shorter side
        'sideBiteRange', [0.08, 0.28], ...  % Fraction of side length
        'taperStrengthRange', [0.10, 0.30], ...  % Fraction of perpendicular dimension
        'edgeWaveAmplitudeRange', [0.01, 0.05], ...  % Fraction of min dimension
        'edgeWaveFrequencyRange', [1.5, 3.0], ...  % Cycles along perimeter
        'maxOperations', 3, ...  % Max destructive ops per polygon
        'maxAreaRemovalFraction', 0.40);  % Fraction allowed to be removed per ellipse (or micropad fallback)

    %% =====================================================================
    %% INPUT PARSING
    %% =====================================================================
    parser = inputParser();
    parser.FunctionName = mfilename;

    addParameter(parser, 'numAugmentations', 10, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>=',1}));
    addParameter(parser, 'rngSeed', [], @(n) isempty(n) || isnumeric(n));
    addParameter(parser, 'phones', {}, @(c) iscellstr(c) || isstring(c));
    addParameter(parser, 'backgroundWidth', 4000, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>',0}));
    addParameter(parser, 'backgroundHeight', 3000, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>',0}));
    addParameter(parser, 'scenePrefix', 'synthetic', @(s) validateattributes(s, {'char','string'}, {'scalartext'}));
    addParameter(parser, 'photometricAugmentation', true, @islogical);
    addParameter(parser, 'blurProbability', 0.25, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'motionBlurProbability', 0.15, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'occlusionProbability', 0.0, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'independentRotation', true, @islogical);
    addParameter(parser, 'extremeCasesProbability', 0.10, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    addParameter(parser, 'enableDistractorPolygons', true, @islogical);
    addParameter(parser, 'distractorMultiplier', 0.6, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0}));
    addParameter(parser, 'distractorMaxCount', 6, @(x) validateattributes(x, {'numeric'}, {'scalar','integer','>=',0}));
    addParameter(parser, 'paperDamageProbability', [], @(n) isempty(n) || (isnumeric(n) && isscalar(n) && n >= 0 && n <= 1));
    addParameter(parser, 'damageSeed', [], @(n) isempty(n) || (isnumeric(n) && isscalar(n) && isfinite(n)));
    addParameter(parser, 'damageProfileWeights', [], @(s) isempty(s) || isstruct(s));
    addParameter(parser, 'maxAreaRemovalFraction', [], @(n) isempty(n) || (isnumeric(n) && isscalar(n) && n >= 0 && n <= 1));

    parse(parser, varargin{:});
    opts = parser.Results;

    % Set random seed
    if isempty(opts.rngSeed)
        rng('shuffle');
    else
        rng(opts.rngSeed);
    end

    % Determine which optional parameters were provided explicitly
    defaultsUsed = parser.UsingDefaults;
    if ~iscell(defaultsUsed)
        defaultsUsed = cellstr(defaultsUsed);
    end
    customBgWidth = ~ismember('backgroundWidth', defaultsUsed);
    customBgHeight = ~ismember('backgroundHeight', defaultsUsed);
    customScenePrefix = ~ismember('scenePrefix', defaultsUsed);

    % Build configuration
    cfg = struct();
    cfg.numAugmentations = opts.numAugmentations;
    cfg.backgroundOverride = struct( ...
        'useWidth', customBgWidth, ...
        'useHeight', customBgHeight, ...
        'width', opts.backgroundWidth, ...
        'height', opts.backgroundHeight);
    cfg.scenePrefix = char(opts.scenePrefix);
    cfg.useScenePrefix = customScenePrefix;
    if ~cfg.useScenePrefix || isempty(cfg.scenePrefix)
        cfg.scenePrefix = '';
        cfg.useScenePrefix = false;
    end
    cfg.photometricAugmentation = opts.photometricAugmentation;
    cfg.blurProbability = opts.blurProbability;
    cfg.motionBlurProbability = opts.motionBlurProbability;
    cfg.occlusionProbability = opts.occlusionProbability;
    cfg.independentRotation = opts.independentRotation;
    cfg.files = struct('coordinates', COORDINATE_FILENAME);
    cfg.concPrefix = CONCENTRATION_PREFIX;
    cfg.supportedFormats = SUPPORTED_FORMATS;
    cfg.camera = CAMERA;
    cfg.rotationRange = ROTATION_RANGE;
    cfg.minValidPolygonArea = MIN_VALID_POLYGON_AREA;
    cfg.texture = TEXTURE;
    cfg.artifacts = ARTIFACTS;
    cfg.placement = PLACEMENT;
    cfg.extremeCasesProbability = opts.extremeCasesProbability;
    cfg.distractors = DISTRACTOR_POLYGONS;
    cfg.distractors.enabled = cfg.distractors.enabled && opts.enableDistractorPolygons;
    cfg.distractors.multiplier = max(0, opts.distractorMultiplier);
    if opts.distractorMaxCount >= 0
        cfg.distractors.maxCount = max(opts.distractorMaxCount, 0);
    end
    if cfg.distractors.maxCount > 0
        cfg.distractors.maxCount = max(cfg.distractors.minCount, cfg.distractors.maxCount);
    else
        cfg.distractors.minCount = 0;
    end

    % Paper damage configuration with precomputed profile sampling
    cfg.damage = PAPER_DAMAGE;
    if ~isempty(opts.paperDamageProbability)
        cfg.damage.probability = opts.paperDamageProbability;
    end
    if ~isempty(opts.maxAreaRemovalFraction)
        cfg.damage.maxAreaRemovalFraction = opts.maxAreaRemovalFraction;
    end
    if ~isempty(opts.damageProfileWeights)
        % Validate custom weights sum to ~1.0
        customWeights = opts.damageProfileWeights;
        expectedFields = {'minimalWarp', 'cornerChew', 'sideCollapse'};
        actualFields = fieldnames(customWeights);
        if ~isequal(sort(actualFields), sort(expectedFields))
            error('augmentDataset:invalidWeights', ...
                  'Custom damage profile weights must contain exactly these fields: %s', ...
                  strjoin(expectedFields, ', '));
        end
        fieldNames = fieldnames(customWeights);
        weightSum = 0;
        for i = 1:numel(fieldNames)
            fieldVal = customWeights.(fieldNames{i});
            if fieldVal < 0
                error('augmentDataset:invalidWeights', ...
                      'Damage profile weight "%s" must be non-negative (got %.4f)', ...
                      fieldNames{i}, fieldVal);
            end
            weightSum = weightSum + fieldVal;
        end
        if abs(weightSum - 1.0) > 1e-6
            error('augmentDataset:invalidWeights', ...
                  'Custom damage profile weights must sum to 1.0 (got %.4f)', weightSum);
        end
        cfg.damage.profileWeights = customWeights;
    end
    
    % Precompute profile sampling arrays for performance
    cfg.damage.profileNames = fieldnames(cfg.damage.profileWeights);
    profileWeightValues = zeros(numel(cfg.damage.profileNames), 1);
    for i = 1:numel(cfg.damage.profileNames)
        profileWeightValues(i) = cfg.damage.profileWeights.(cfg.damage.profileNames{i});
    end
    cfg.damage.profileCumWeights = cumsum(profileWeightValues);
    
    % Validate range bounds
    if cfg.damage.cornerClipRange(2) <= cfg.damage.cornerClipRange(1)
        error('augmentDataset:invalidRange', 'cornerClipRange upper bound must exceed lower bound');
    end
    if cfg.damage.sideBiteRange(2) <= cfg.damage.sideBiteRange(1)
        error('augmentDataset:invalidRange', 'sideBiteRange upper bound must exceed lower bound');
    end
    if cfg.damage.taperStrengthRange(2) <= cfg.damage.taperStrengthRange(1)
        error('augmentDataset:invalidRange', 'taperStrengthRange upper bound must exceed lower bound');
    end
    if cfg.damage.edgeWaveAmplitudeRange(2) <= cfg.damage.edgeWaveAmplitudeRange(1)
        error('augmentDataset:invalidRange', 'edgeWaveAmplitudeRange upper bound must exceed lower bound');
    end
    if cfg.damage.edgeWaveFrequencyRange(2) <= cfg.damage.edgeWaveFrequencyRange(1)
        error('augmentDataset:invalidRange', 'edgeWaveFrequencyRange upper bound must exceed lower bound');
    end
    cfg.damage.rngSeed = opts.damageSeed;

    % Resolve paths
    projectRoot = find_project_root(DEFAULT_INPUT_STAGE1);
    cfg.projectRoot = projectRoot;
    cfg.paths = struct( ...
        'stage1Input', fullfile(projectRoot, DEFAULT_INPUT_STAGE1), ...
        'stage2Coords', fullfile(projectRoot, DEFAULT_INPUT_STAGE2), ...
        'stage3Coords', fullfile(projectRoot, DEFAULT_INPUT_STAGE3_COORDS), ...
        'ellipseCoords', fullfile(projectRoot, DEFAULT_INPUT_STAGE3_COORDS), ...
        'stage1Output', DEFAULT_OUTPUT_STAGE1, ...
        'stage2Output', DEFAULT_OUTPUT_STAGE2, ...
        'stage3Output', DEFAULT_OUTPUT_STAGE3);

    % Validate inputs exist
    if ~isfolder(cfg.paths.stage1Input)
        warning('augmentDataset:missingStage1', ...
            'Stage 1 input not found: %s. Passthrough copies will be skipped.', ...
            cfg.paths.stage1Input);
    end
    if ~isfolder(cfg.paths.stage2Coords)
        error('augmentDataset:missingCoords', 'Stage 2 coordinates folder not found: %s', cfg.paths.stage2Coords);
    end
    if ~isfolder(cfg.paths.ellipseCoords)
        fprintf('Note: Elliptical regions folder not found (%s) - ellipse processing will be skipped\n', ...
                cfg.paths.ellipseCoords);
    end

    % Get phone list
    requestedPhones = string(opts.phones);
    requestedPhones = requestedPhones(requestedPhones ~= "");

    phoneList = list_phones(cfg.paths.stage1Input);
    if isempty(phoneList)
        error('augmentDataset:noPhones', 'No phone folders found in %s', cfg.paths.stage1Input);
    end

    % Validate configuration consistency
    if cfg.independentRotation && cfg.extremeCasesProbability > 0.5
        warning('augmentDataset:config', ...
            'Independent rotation + high extreme cases may generate too-difficult samples');
    end

    % Process each phone
    fprintf('\n=== Augmentation Configuration ===\n');
    fprintf('Camera perspective: %.0f° max angle, X=[%.1f,%.1f], Y=[%.1f,%.1f], Z=[%.1f,%.1f]\n', ...
        cfg.camera.maxAngleDeg, cfg.camera.xRange, cfg.camera.yRange, cfg.camera.zRange);
    fprintf('Artifacts: %d-%d per image\n', ...
        cfg.artifacts.countRange(1), cfg.artifacts.countRange(2));
    fprintf('Extreme cases: %.0f%% probability\n', cfg.extremeCasesProbability*100);
    fprintf('Augmentations per paper: %d\n', cfg.numAugmentations);
    widthStr = 'source width';
    if cfg.backgroundOverride.useWidth
        widthStr = sprintf('%d px', cfg.backgroundOverride.width);
    end
    heightStr = 'source height';
    if cfg.backgroundOverride.useHeight
        heightStr = sprintf('%d px', cfg.backgroundOverride.height);
    end
    if cfg.backgroundOverride.useWidth || cfg.backgroundOverride.useHeight
        fprintf('Background override: width=%s, height=%s\n', widthStr, heightStr);
    else
        fprintf('Background size: matches source image dimensions\n');
    end
    if cfg.useScenePrefix
        fprintf('Scene prefix: %s\n', cfg.scenePrefix);
    else
        fprintf('Scene prefix: (none)\n');
    end
    fprintf('Backgrounds: 4 types (uniform, speckled, laminate, skin)\n');
    fprintf('Photometric augmentation: %s\n', char(string(cfg.photometricAugmentation)));
    fprintf('Blur probability: %.1f%%\n', cfg.blurProbability * 100);
    fprintf('==================================\n');

    for i = 1:numel(phoneList)
        phoneName = phoneList{i};
        if ~isempty(requestedPhones) && ~any(strcmpi(requestedPhones, phoneName))
            continue;
        end
        fprintf('\n=== Processing phone: %s ===\n', phoneName);
        augment_phone(phoneName, cfg);
    end

    fprintf('\n=== Augmentation Complete ===\n');
end

%% -------------------------------------------------------------------------
function augment_phone(phoneName, cfg)
    % Main processing loop for a single phone
    % Strategy: For each paper, generate N augmented versions

    % Define phone-specific paths
    stage1PhoneDir = fullfile(cfg.paths.stage1Input, phoneName);
    stage2PhoneCoords = fullfile(cfg.paths.stage2Coords, phoneName, cfg.files.coordinates);
    ellipsePhoneCoords = fullfile(cfg.paths.ellipseCoords, phoneName, cfg.files.coordinates);

    % Validate stage 1 images exist
    if ~isfolder(stage1PhoneDir)
        warning('augmentDataset:missingPhone', 'Stage 1 folder not found for %s', phoneName);
        return;
    end

    % Load polygon coordinates from stage 2 (required)
    if ~isfile(stage2PhoneCoords)
        warning('augmentDataset:noPolygonCoords', 'No polygon coordinates for %s. Skipping.', phoneName);
        return;
    end

    polygonEntries = read_polygon_coordinates(stage2PhoneCoords);
    if isempty(polygonEntries)
        warning('augmentDataset:emptyPolygons', 'No valid polygon entries for %s', phoneName);
        return;
    end

    % Load ellipse coordinates from stage 3 (optional)
    ellipseEntries = struct('image', {}, 'concentration', {}, 'replicate', {}, ...
                            'center', {}, 'semiMajor', {}, 'semiMinor', {}, 'rotation', {});
    hasEllipses = false;
    if isfile(ellipsePhoneCoords)
        ellipseEntries = read_ellipse_coordinates(ellipsePhoneCoords);
        hasEllipses = ~isempty(ellipseEntries);
    else
        fprintf('  [INFO] No ellipse coordinates. Will process polygons only.\n');
    end

    % Build lookup structures
    ellipseMap = group_ellipses_by_parent(ellipseEntries, hasEllipses);
    paperGroups = group_polygons_by_image(polygonEntries);

    % Create output directories
    stage1PhoneOut = fullfile(cfg.projectRoot, cfg.paths.stage1Output, phoneName);
    stage2PhoneOut = fullfile(cfg.projectRoot, cfg.paths.stage2Output, phoneName);
    stage3PhoneOut = fullfile(cfg.projectRoot, cfg.paths.stage3Output, phoneName);
    ensure_folder(stage1PhoneOut);
    ensure_folder(stage2PhoneOut);
    % Only create stage3 output folder if we have ellipse data
    if hasEllipses
        ensure_folder(stage3PhoneOut);
    end

    % Get unique paper names
    paperNames = keys(paperGroups);
    fprintf('  Total papers: %d\n', numel(paperNames));
    fprintf('  Total polygons: %d\n', numel(polygonEntries));
    fprintf('  Total ellipses: %d\n', numel(ellipseEntries));

    % Process each paper
    for paperIdx = 1:numel(paperNames)
        paperBase = paperNames{paperIdx};
        fprintf('  -> Paper %d/%d: %s\n', paperIdx, numel(paperNames), paperBase);

        % Find stage 1 image
        imgPath = find_stage1_image(stage1PhoneDir, paperBase, cfg.supportedFormats);
        if isempty(imgPath)
            warning('augmentDataset:missingImage', 'Stage 1 image not found for %s', paperBase);
            continue;
        end

        % Load image once (using imread_raw to handle EXIF orientation)
        stage1Img = imread_raw(imgPath);

        % Convert grayscale to RGB (synthetic backgrounds are always RGB)
        if size(stage1Img, 3) == 1
            stage1Img = repmat(stage1Img, [1, 1, 3]);
        end

        imgExt = '.png';

        % Get all polygons from this paper
        polygons = paperGroups(paperBase);

        % Emit passthrough sample (augIdx = 0) with original geometry
        emit_passthrough_sample(paperBase, imgPath, stage1Img, polygons, ellipseMap, ...
                                 hasEllipses, stage1PhoneOut, stage2PhoneOut, ...
                                 stage3PhoneOut, cfg);

        % Generate synthetic augmentations only
        if cfg.numAugmentations < 1
            continue;
        end
        for augIdx = 1:cfg.numAugmentations
            augment_single_paper(paperBase, imgExt, stage1Img, polygons, ellipseMap, ...
                                 hasEllipses, augIdx, stage1PhoneOut, stage2PhoneOut, ...
                                 stage3PhoneOut, paperIdx, phoneName, cfg);
        end
    end

    % Paper damage summary logging
    if cfg.damage.probability > 0
        fprintf('  Paper damage applied per polygon with %.0f%% probability\n', cfg.damage.probability * 100);
    end

end

%% -------------------------------------------------------------------------
function emit_passthrough_sample(paperBase, ~, stage1Img, polygons, ellipseMap, ...
                                 hasEllipses, stage1PhoneOut, stage2PhoneOut, ...
                                 stage3PhoneOut, cfg)
    % Generate aug_000 assets by reusing original captures without augmentation

    imgExt = '.png';
    if cfg.useScenePrefix
        baseSceneId = sprintf('%s_%s', cfg.scenePrefix, paperBase);
    else
        baseSceneId = paperBase;
    end
    sceneName = sprintf('%s_aug_%03d', baseSceneId, 0);
    sceneFileName = sprintf('%s%s', sceneName, imgExt);

    % Save to phone directory
    sceneOutPath = fullfile(stage1PhoneOut, sceneFileName);

    % Always re-encode to PNG format (cannot copy JPEG bytes with .png extension)
    try
        imwrite(stage1Img, sceneOutPath);
    catch writeErr
        error('augmentDataset:passthroughSceneWrite', ...
              'Cannot emit passthrough scene %s: %s', sceneOutPath, writeErr.message);
    end

    stage2Coords = cell(numel(polygons), 1);
    % Pre-allocate for 3 ellipses per polygon (experimental design)
    maxEllipsesPerPolygon = 3;
    stage3Coords = cell(max(1, numel(polygons) * maxEllipsesPerPolygon), 1);
    polyCount = 0;
    s2Count = 0;
    s3Count = 0;

    for idx = 1:numel(polygons)
        poly = polygons(idx);
        origVertices = double(poly.vertices);

        if ~is_valid_polygon(origVertices, cfg.minValidPolygonArea)
            warning('augmentDataset:passthroughInvalidPolygon', ...
                    '  ! Polygon %s con %d invalid for passthrough. Skipping.', ...
                    paperBase, poly.concentration);
            continue;
        end

        [polygonImg, ~] = extract_polygon_masked(stage1Img, origVertices);
        if isempty(polygonImg)
            warning('augmentDataset:passthroughEmptyCrop', ...
                    '  ! Polygon %s con %d produced empty crop.', ...
                    paperBase, poly.concentration);
            continue;
        end

        concDir = fullfile(stage2PhoneOut, sprintf('%s%d', cfg.concPrefix, poly.concentration));
        ensure_folder(concDir);
        polygonFileName = sprintf('%s_%s%d%s', sceneName, cfg.concPrefix, poly.concentration, imgExt);
        polygonOutPath = fullfile(concDir, polygonFileName);
        imwrite(polygonImg, polygonOutPath);

        polyCount = polyCount + 1;

        s2Count = s2Count + 1;
        stage2Coords{s2Count} = struct( ...
            'image', polygonFileName, ...
            'concentration', poly.concentration, ...
            'vertices', origVertices, ...
            'rotation', poly.rotation);

        ellipseKey = sprintf('%s#%d', paperBase, poly.concentration);
        if hasEllipses && isKey(ellipseMap, ellipseKey)
            ellipseList = ellipseMap(ellipseKey);
            for eIdx = 1:numel(ellipseList)
                ellipseIn = ellipseList(eIdx);
                ellipseGeom = struct( ...
                    'center', ellipseIn.center, ...
                    'semiMajor', ellipseIn.semiMajor, ...
                    'semiMinor', ellipseIn.semiMinor, ...
                    'rotation', ellipseIn.rotation);

                [patchImg, patchValid] = crop_ellipse_patch(polygonImg, ellipseGeom);
                if ~patchValid || isempty(patchImg)
                    warning('augmentDataset:passthroughPatchInvalid', ...
                            '  ! Ellipse %s con %d rep %d invalid for passthrough.', ...
                            paperBase, poly.concentration, ellipseIn.replicate);
                    continue;
                end

                ellipseDir = fullfile(stage3PhoneOut, sprintf('%s%d', cfg.concPrefix, poly.concentration));
                ensure_folder(ellipseDir);
                patchFileName = sprintf('%s_%s%d_rep%d%s', sceneName, cfg.concPrefix, ...
                                        poly.concentration, ellipseIn.replicate, imgExt);
                patchOutPath = fullfile(ellipseDir, patchFileName);
                imwrite(patchImg, patchOutPath);

                s3Count = s3Count + 1;
                if s3Count > numel(stage3Coords)
                    stage3Coords = [stage3Coords; cell(s3Count, 1)]; %#ok<AGROW> % Rare case: more ellipses than expected
                end
                stage3Coords{s3Count} = struct( ...
                    'image', polygonFileName, ...
                    'concentration', poly.concentration, ...
                    'replicate', ellipseIn.replicate, ...
                    'center', ellipseGeom.center, ...
                    'semiMajor', ellipseGeom.semiMajor, ...
                    'semiMinor', ellipseGeom.semiMinor, ...
                    'rotation', ellipseGeom.rotation);
            end
        end
    end

    if polyCount == 0
        warning('augmentDataset:passthroughNoPolygons', ...
                '  ! No valid polygons for passthrough %s. Removing scene.', sceneName);
        if exist(sceneOutPath, 'file') == 2
            delete(sceneOutPath);
        end
        return;
    end

    stage2Coords = stage2Coords(1:s2Count);
    stage3Coords = stage3Coords(1:s3Count);

    write_stage2_coordinates(stage2Coords, stage2PhoneOut, cfg.files.coordinates);
    if s3Count > 0
        write_stage3_coordinates(stage3Coords, stage3PhoneOut, cfg.files.coordinates);
    end

    fprintf('     Passthrough: %s (%d polygons, %d ellipses)\n', ...
            sceneFileName, numel(stage2Coords), numel(stage3Coords));
end

%% -------------------------------------------------------------------------
function augment_single_paper(paperBase, imgExt, stage1Img, polygons, ellipseMap, ...
                               hasEllipses, augIdx, stage1PhoneOut, stage2PhoneOut, ...
                               stage3PhoneOut, paperIdx, phoneName, cfg)
    % Generate one augmented version of a paper with all its concentration regions

    [origHeight, origWidth, ~] = size(stage1Img);

    % Sample transformation (same for all regions in this augmentation)
    if rand() < cfg.extremeCasesProbability
        % Extreme camera viewpoint
        extremeCamera = cfg.camera;
        extremeCamera.maxAngleDeg = 75;
        extremeCamera.zRange = [0.8, 4.0];
        viewParams = sample_viewpoint(extremeCamera);
    else
        % Normal camera viewpoint
        viewParams = sample_viewpoint(cfg.camera);
    end
    tformPersp = compute_homography(size(stage1Img), viewParams, cfg.camera);
    rotAngle = rand_range(cfg.rotationRange);
    tformRot = centered_rotation_tform(size(stage1Img), rotAngle);

    % Pre-allocate coordinate accumulators
    % Stage 2: one entry per polygon (upper bound = validCount after validation)
    % Stage 3: 3 ellipses per polygon (experimental design)
    maxPolygons = numel(polygons);
    maxEllipsesPerPolygon = 3;
    stage2Coords = cell(maxPolygons, 1);
    stage3Coords = cell(maxPolygons * maxEllipsesPerPolygon, 1);
    s2Count = 0;
    s3Count = 0;

    % Cache deterministic hashes for reproducible per-polygon damage sampling
    phoneHash = stable_string_hash(phoneName);
    paperHash = stable_string_hash(paperBase);

    % Temporary storage for transformed polygon crops and their properties
    transformedRegions = cell(numel(polygons), 1);
    validCount = 0;

    % Transform all polygons and extract crops
    for polyIdx = 1:numel(polygons)
        polyEntry = polygons(polyIdx);
        concentration = polyEntry.concentration;
        origVertices = polyEntry.vertices;

        % Apply shared perspective transformation
        augVertices = transform_polygon(origVertices, tformPersp);
        augVertices = transform_polygon(augVertices, tformRot);

        % Apply independent rotation per polygon if enabled
        if cfg.independentRotation
            independentRotAngle = rand_range(cfg.rotationRange);
            tformIndepRot = centered_rotation_tform(size(stage1Img), independentRotAngle);
        else
            independentRotAngle = 0;
            tformIndepRot = affine2d(eye(3));
        end
        augVertices = transform_polygon(augVertices, tformIndepRot);

        % Validate transformed polygon
        if ~is_valid_polygon(augVertices, cfg.minValidPolygonArea)
            warning('augmentDataset:degeneratePolygon', ...
                    '  ! Polygon %s con %d degenerate after transform. Skipping.', ...
                    paperBase, concentration);
            continue;
        end

        % Extract polygon content with masking
        [polygonContent, contentBbox] = extract_polygon_masked(stage1Img, origVertices);

        % Transform extracted content to match augmented shape
        augPolygonImg = transform_polygon_content(polygonContent, ...
                                                  origVertices, augVertices, contentBbox);

        % Convert augmented vertices to cropped coordinate space (shared by damage + ellipse transforms)
        minXCrop = min(augVertices(:,1));
        minYCrop = min(augVertices(:,2));
        augVerticesCrop = augVertices - [minXCrop, minYCrop];

        % Transform ellipse annotations into the augmented crop frame (guard bands + stage 3)
        ellipseAugList = [];
        ellipseKey = sprintf('%s#%d', paperBase, concentration);
        if hasEllipses && isKey(ellipseMap, ellipseKey)
            rawEllipseList = ellipseMap(ellipseKey);
            ellipseAugList = transform_region_ellipses( ...
                rawEllipseList, paperBase, concentration, origVertices, contentBbox, ...
                augVertices, minXCrop, minYCrop, tformPersp, tformRot, ...
                tformIndepRot, rotAngle + independentRotAngle);
        end

        % Apply paper damage augmentation per polygon using configured probability
        if cfg.damage.probability > 0
            % Extract alpha mask from transformed polygon
            polygonAlpha = any(augPolygonImg > 0, 3);  % Non-zero in any channel

            % Apply damage (with deterministic RNG if seed provided)
            damageRng = [];
            if ~isempty(cfg.damage.rngSeed)
                % Create deterministic seed combining global seed + paper + concentration + augmentation
                % Multipliers ensure no collisions: 10000 papers × 100 concentrations × 100 augmentations
                damageRng = cfg.damage.rngSeed + phoneHash + paperHash + ...
                            paperIdx * 10000 + concentration * 100 + augIdx;
            end

            [damagedRGB, ~, ~, ~, damageCornersCrop] = apply_paper_damage( ...
                augPolygonImg, polygonAlpha, augVerticesCrop, ellipseAugList, cfg.damage, damageRng);

            % Replace augPolygonImg and propagate updated quadrilateral vertices
            augPolygonImg = damagedRGB;
            augVertices = damageCornersCrop + [minXCrop, minYCrop];
        end

        % Store transformed region for later composition
        validCount = validCount + 1;
        transformedRegions{validCount} = struct( ...
            'concentration', concentration, ...
            'augVertices', augVertices, ...
            'augPolygonImg', augPolygonImg, ...
            'contentBbox', contentBbox, ...
            'origVertices', origVertices, ...
            'independentRotAngle', independentRotAngle, ...
            'originalRotation', polyEntry.rotation, ...
            'augEllipses', ellipseAugList);
    end

    % Trim to valid regions
    transformedRegions = transformedRegions(1:validCount);

    if validCount == 0
        warning('augmentDataset:noValidRegions', '  ! No valid regions for %s aug %d', paperBase, augIdx);
        return;
    end

    % Compute individual polygon bounding boxes for random placement
    polygonBboxes = cell(validCount, 1);
    for i = 1:validCount
        verts = transformedRegions{i}.augVertices;
        polygonBboxes{i} = struct( ...
            'minX', min(verts(:,1)), ...
            'maxX', max(verts(:,1)), ...
            'minY', min(verts(:,2)), ...
            'maxY', max(verts(:,2)), ...
            'width', max(verts(:,1)) - min(verts(:,1)), ...
            'height', max(verts(:,2)) - min(verts(:,2)));
    end

    % Start with default background size (real capture resolution)
    bgWidth = origWidth;
    bgHeight = origHeight;
    if cfg.backgroundOverride.useWidth
        bgWidth = cfg.backgroundOverride.width;
    end
    if cfg.backgroundOverride.useHeight
        bgHeight = cfg.backgroundOverride.height;
    end

    % Place polygons at random non-overlapping positions
    % Calculate polygon density to adapt spacing requirements
    totalPolygonArea = 0;
    for i = 1:validCount
        bbox = polygonBboxes{i};
        totalPolygonArea = totalPolygonArea + (bbox.width * bbox.height);
    end
    randomPositions = place_polygons_nonoverlapping(polygonBboxes, ...
                                                     bgWidth, bgHeight, ...
                                                     cfg.placement.margin, ...
                                                     cfg.placement.minSpacing, ...
                                                     cfg.placement.maxOverlapRetries);

    % Generate realistic background with final size
    background = generate_realistic_lab_surface(bgWidth, bgHeight, cfg.texture, cfg.artifacts);

    % Composite each region onto background and save outputs
    if cfg.useScenePrefix
        baseSceneId = sprintf('%s_%s', cfg.scenePrefix, paperBase);
    else
        baseSceneId = paperBase;
    end
    sceneName = sprintf('%s_aug_%03d', baseSceneId, augIdx);
    scenePolygons = cell(validCount, 1);
    occupiedBboxes = zeros(validCount, 4);
    polygonIdx = 0;

    for i = 1:validCount
        region = transformedRegions{i};
        concentration = region.concentration;
        augVertices = region.augVertices;
        augPolygonImg = region.augPolygonImg;

        % Get random position for this polygon
        pos = randomPositions{i};

        % Compute offset to move polygon from its current position to random position
        % Random position specifies the top-left corner of the bounding box
        currentMinX = polygonBboxes{i}.minX;
        currentMinY = polygonBboxes{i}.minY;
        offsetX = pos.x - currentMinX;
        offsetY = pos.y - currentMinY;

        % Translate vertices to background coordinates
        sceneVertices = augVertices + [offsetX, offsetY];

        % Composite onto background
        background = composite_to_background(background, augPolygonImg, sceneVertices);

        % Paper lies flat on surface; no shadows needed
        minSceneX = min(sceneVertices(:,1));
        minSceneY = min(sceneVertices(:,2));
        maxSceneX = max(sceneVertices(:,1));
        maxSceneY = max(sceneVertices(:,2));
        occupiedBboxes(i, :) = [minSceneX, minSceneY, maxSceneX, maxSceneY];

        % Save polygon crop (stage 2 output)
        concDirOut = fullfile(stage2PhoneOut, sprintf('%s%d', cfg.concPrefix, concentration));
        ensure_folder(concDirOut);

        polygonFileName = sprintf('%s_%s%d%s', sceneName, cfg.concPrefix, concentration, imgExt);
        polygonOutPath = fullfile(concDirOut, polygonFileName);
        imwrite(augPolygonImg, polygonOutPath);

        % Record stage 2 coordinates (polygon in scene)
        % Compute total applied rotation and update saved rotation field
        totalAppliedRotation = rotAngle + region.independentRotAngle;
        augmentedRotation = normalizeAngle(region.originalRotation - totalAppliedRotation);

        s2Count = s2Count + 1;
        stage2Coords{s2Count} = struct( ...
            'image', polygonFileName, ...
            'concentration', concentration, ...
            'vertices', sceneVertices, ...
            'rotation', augmentedRotation);

        % Track polygon in scene space for optional occlusions
        polygonIdx = polygonIdx + 1;
        scenePolygons{polygonIdx} = sceneVertices;

        % Process ellipses for this concentration (stage 3)
        ellipseCropList = region.augEllipses;
        if hasEllipses && ~isempty(ellipseCropList)
            for eIdx = 1:numel(ellipseCropList)
                ellipseCrop = ellipseCropList(eIdx);
                replicateId = ellipseCrop.replicate;

                % Extract ellipse patch
                [patchImg, patchValid] = crop_ellipse_patch(augPolygonImg, ellipseCrop);
                if ~patchValid
                    warning('augmentDataset:patchInvalid', ...
                            '  ! Ellipse patch %s con %d rep %d invalid. Skipping.', ...
                            paperBase, concentration, replicateId);
                    continue;
                end

                % Save ellipse patch (stage 3 output)
                ellipseConcDir = fullfile(stage3PhoneOut, sprintf('%s%d', cfg.concPrefix, concentration));
                ensure_folder(ellipseConcDir);
                patchFileName = sprintf('%s_%s%d_rep%d%s', sceneName, cfg.concPrefix, ...
                                        concentration, replicateId, imgExt);
                patchOutPath = fullfile(ellipseConcDir, patchFileName);
                imwrite(patchImg, patchOutPath);

                % Record stage 3 coordinates (ellipse in polygon-crop space)
                s3Count = s3Count + 1;
                if s3Count > numel(stage3Coords)
                    stage3Coords = [stage3Coords; cell(s3Count, 1)]; %#ok<AGROW> % Rare case: more ellipses than expected
                end
                stage3Coords{s3Count} = struct( ...
                    'image', polygonFileName, ...
                    'concentration', concentration, ...
                    'replicate', replicateId, ...
                    'center', ellipseCrop.center, ...
                    'semiMajor', ellipseCrop.semiMajor, ...
                    'semiMinor', ellipseCrop.semiMinor, ...
                    'rotation', ellipseCrop.rotation);
            end
        end

    end

    % Trim scenePolygons to actual size
    scenePolygons = scenePolygons(1:polygonIdx);

    additionalDistractors = 0;
    if cfg.distractors.enabled && cfg.distractors.multiplier > 0
        [background, additionalDistractors] = add_polygon_distractors(background, transformedRegions, polygonBboxes, occupiedBboxes, cfg);
    end

    % Optional: add thin occlusions (e.g., hair/strap-like) over polygons
    if cfg.occlusionProbability > 0 && ~isempty(scenePolygons)
        background = add_polygon_occlusions(background, scenePolygons, cfg.occlusionProbability);
    end

    % Apply photometric augmentation and non-overlapping blur before saving
    if cfg.photometricAugmentation
        % Phase 1.7: Extreme photometric conditions (low lighting)
        if rand() < cfg.extremeCasesProbability
            background = apply_photometric_augmentation(background, 'extreme');
        else
            background = apply_photometric_augmentation(background, 'subtle');
        end
    end

    % Ensure at most one blur type is applied to avoid double-softening
    blurApplied = false;
    if cfg.motionBlurProbability > 0 && rand() < cfg.motionBlurProbability
        background = apply_motion_blur(background);
        blurApplied = true;
    end
    if ~blurApplied && cfg.blurProbability > 0 && rand() < cfg.blurProbability
        blurSigma = 0.25 + rand() * 0.40;  % [0.25, 0.65] pixels - very subtle
        background = imgaussfilt(background, blurSigma);
    end

    % Save synthetic scene (stage 1 output)
    sceneFileName = sprintf('%s%s', sceneName, '.png');
    sceneOutPath = fullfile(stage1PhoneOut, sceneFileName);
    imwrite(background, sceneOutPath);

    % Trim coordinate arrays to actual size
    stage2Coords = stage2Coords(1:s2Count);
    stage3Coords = stage3Coords(1:s3Count);

    % Write coordinates
    write_stage2_coordinates(stage2Coords, stage2PhoneOut, cfg.files.coordinates);
    if s3Count > 0
        write_stage3_coordinates(stage3Coords, stage3PhoneOut, cfg.files.coordinates);
    end

    fprintf('     Generated: %s (%d polygons, %d ellipses, %d distractors)\n', ...
            sceneFileName, numel(stage2Coords), numel(stage3Coords), additionalDistractors);
end

%% =========================================================================
%% CORE PROCESSING FUNCTIONS
%% =========================================================================

function [content, bbox] = extract_polygon_masked(img, vertices)
    % Extract polygon region with masking to avoid black pixels.
    % Optimization: restrict poly2mask to the local bbox instead of the full frame.

    [imgH, imgW, numChannels] = size(img);

    if isempty(vertices) || any(~isfinite(vertices(:)))
        content = zeros(0, 0, numChannels, 'like', img);
        bbox = [1, 1, 0, 0];
        return;
    end

    minX = floor(min(vertices(:,1)));
    maxX = ceil(max(vertices(:,1)));
    minY = floor(min(vertices(:,2)));
    maxY = ceil(max(vertices(:,2)));

    minX = max(1, minX);
    minY = max(1, minY);
    maxX = min(imgW, maxX);
    maxY = min(imgH, maxY);

    if maxX < minX || maxY < minY
        content = zeros(0, 0, numChannels, 'like', img);
        bbox = [1, 1, 0, 0];
        return;
    end

    bboxWidth = maxX - minX + 1;
    bboxHeight = maxY - minY + 1;

    relVerts = vertices - [minX - 1, minY - 1];

    mask = poly2mask(relVerts(:,1), relVerts(:,2), bboxHeight, bboxWidth);
    if ~any(mask(:))
        content = zeros(0, 0, numChannels, 'like', img);
        bbox = [1, 1, 0, 0];
        return;
    end

    bboxContent = img(minY:maxY, minX:maxX, :);
    if numChannels == 3
        content = bboxContent .* cast(repmat(mask, [1, 1, 3]), 'like', bboxContent);
    else
        content = bboxContent .* cast(mask, 'like', bboxContent);
    end

    bbox = [minX, minY, bboxWidth, bboxHeight];
end

function augImg = transform_polygon_content(content, origVerts, augVerts, bbox)
    % Transform polygon content to match target geometry

    % Convert vertices to bbox-relative coordinates
    origVertsRel = origVerts - [bbox(1) - 1, bbox(2) - 1];
    minX = min(augVerts(:,1));
    minY = min(augVerts(:,2));
    augVertsRel = augVerts - [minX, minY];

    % Compute projective transformation
    tform = fitgeotrans(origVertsRel, augVertsRel, 'projective');

    % Determine output dimensions
    outWidth = ceil(max(augVertsRel(:,1)) - min(augVertsRel(:,1)) + 1);
    outHeight = ceil(max(augVertsRel(:,2)) - min(augVertsRel(:,2)) + 1);
    outRef = imref2d([outHeight, outWidth]);

    % Apply transformation
    augImg = imwarp(content, tform, 'OutputView', outRef, ...
                    'InterpolationMethod', 'linear', 'FillValues', 0);
end

function ellipseCropList = transform_region_ellipses(ellipseList, paperBase, concentration, ...
                                                     origVertices, contentBbox, augVertices, ...
                                                     minXCrop, minYCrop, tformPersp, tformRot, ...
                                                     tformIndepRot, totalAppliedRotation)
    % Transform ellipse annotations into the augmented crop coordinate frame

    if isempty(ellipseList)
        ellipseCropList = struct('replicate', {}, 'center', {}, ...
                                 'semiMajor', {}, 'semiMinor', {}, 'rotation', {});
        return;
    end

    % Pre-allocate for all ellipses (trim to actual count after loop)
    maxEllipses = numel(ellipseList);
    ellipseCropList = repmat(struct('replicate', [], 'center', [], ...
                                    'semiMajor', [], 'semiMinor', [], 'rotation', []), ...
                             maxEllipses, 1);
    validCount = 0;

    for idx = 1:numel(ellipseList)
        ellipseIn = ellipseList(idx);

        % Map ellipse from crop space to original image coordinates
        ellipseInImageSpace = map_ellipse_crop_to_image(ellipseIn, contentBbox);

        % Validate ellipse lies inside the original polygon before augmentations
        if ~inpolygon(ellipseInImageSpace.center(1), ellipseInImageSpace.center(2), ...
                      origVertices(:,1), origVertices(:,2))
            warning('augmentDataset:ellipseOutsideOriginal', ...
                    '  ! Ellipse %s con %d rep %d outside original polygon. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Apply shared perspective + rotation + independent rotation transforms
        ellipseAug = transform_ellipse(ellipseInImageSpace, tformPersp);
        if ~ellipseAug.valid
            warning('augmentDataset:ellipseInvalid1', ...
                    '  ! Ellipse %s con %d rep %d invalid after perspective. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        ellipseAug = transform_ellipse(ellipseAug, tformRot);
        if ~ellipseAug.valid
            warning('augmentDataset:ellipseInvalid2', ...
                    '  ! Ellipse %s con %d rep %d invalid after shared rotation. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        ellipseAug = transform_ellipse(ellipseAug, tformIndepRot);
        if ~ellipseAug.valid
            warning('augmentDataset:ellipseInvalid3', ...
                    '  ! Ellipse %s con %d rep %d invalid after independent rotation. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Ensure ellipse stays inside the augmented polygon footprint
        if ~inpolygon(ellipseAug.center(1), ellipseAug.center(2), ...
                      augVertices(:,1), augVertices(:,2))
            warning('augmentDataset:ellipseOutside', ...
                    '  ! Ellipse %s con %d rep %d outside transformed polygon. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Convert to polygon-crop coordinates
        ellipseCrop = ellipseAug;
        ellipseCrop.center = ellipseAug.center - [minXCrop, minYCrop];

        if ~isfinite(ellipseCrop.semiMajor) || ~isfinite(ellipseCrop.semiMinor) || ...
           ellipseCrop.semiMajor <= 0 || ellipseCrop.semiMinor <= 0
            warning('augmentDataset:invalidAxes', ...
                    '  ! Ellipse %s con %d rep %d has invalid axes (major=%.4f, minor=%.4f). Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate, ...
                    ellipseCrop.semiMajor, ellipseCrop.semiMinor);
            continue;
        end

        % Enforce semiMajor >= semiMinor convention
        if ellipseCrop.semiMajor < ellipseCrop.semiMinor
            tmp = ellipseCrop.semiMajor;
            ellipseCrop.semiMajor = ellipseCrop.semiMinor;
            ellipseCrop.semiMinor = tmp;
            ellipseCrop.rotation = ellipseCrop.rotation + 90;
        end

        % Adjust rotation relative to the augmented patch reference frame
        ellipseCrop.rotation = normalizeAngle(ellipseCrop.rotation - totalAppliedRotation);

        validCount = validCount + 1;
        ellipseCropList(validCount) = struct( ...
            'replicate', ellipseIn.replicate, ...
            'center', ellipseCrop.center, ...
            'semiMajor', ellipseCrop.semiMajor, ...
            'semiMinor', ellipseCrop.semiMinor, ...
            'rotation', ellipseCrop.rotation);
    end

    % Trim to actual valid ellipse count
    ellipseCropList = ellipseCropList(1:validCount);
end

function bg = composite_to_background(bg, polygonImg, sceneVerts)
    % Composite transformed polygon onto background using per-channel alpha blending

    % Compute target region in background
    minX = max(1, floor(min(sceneVerts(:,1))));
    maxX = min(size(bg, 2), ceil(max(sceneVerts(:,1))));
    minY = max(1, floor(min(sceneVerts(:,2))));
    maxY = min(size(bg, 1), ceil(max(sceneVerts(:,2))));

    targetWidth = maxX - minX + 1;
    targetHeight = maxY - minY + 1;

    % Guard: degenerate target (outside image or zero area)
    if targetWidth < 1 || targetHeight < 1
        return;
    end

    % Resize polygon to target size only when necessary. In the common case the
    % warped patch already matches the target bbox; skip extra resampling to
    % preserve edges and save time.
    [patchH, patchW, ~] = size(polygonImg);
    if patchH == targetHeight && patchW == targetWidth
        resized = polygonImg;
    else
        % Use nearest-neighbor to prevent color bleeding across masked boundaries
        resized = imresize(polygonImg, [targetHeight, targetWidth], 'nearest');
    end

    % Create mask for target region
    vertsTarget = sceneVerts - [minX - 1, minY - 1];
    targetMask = poly2mask(vertsTarget(:,1), vertsTarget(:,2), targetHeight, targetWidth);

    % If mask is empty, nothing to composite
    if ~any(targetMask(:))
        return;
    end

    % Use the synthesized patch mask to support hollow distractors
    patchMask = any(resized > 0, 3);
    effectiveMask = targetMask & patchMask;
    if ~any(effectiveMask(:))
        return;
    end

    % Composite per-channel using arithmetic (no logical linearization)
    bgRegion = bg(minY:maxY, minX:maxX, :);

    % Prepare alpha in double for stable math
    alpha = double(effectiveMask);

    % All images are RGB at this point (converted on load)
    numChannels = size(bgRegion, 3);
    for c = 1:numChannels
        R = double(bgRegion(:,:,c));
        F = double(resized(:,:,c));
        bgRegion(:,:,c) = uint8(R .* (1 - alpha) + F .* alpha);
    end

    bg(minY:maxY, minX:maxX, :) = bgRegion;
end

function [bg, placedCount] = add_polygon_distractors(bg, regions, polygonBboxes, occupiedBboxes, cfg)
    % Inject additional polygon-shaped distractors matching source geometry statistics.

    distractorCfg = cfg.distractors;
    if isempty(regions) || ~distractorCfg.enabled
        placedCount = 0;
        return;
    end

    targetCount = randi([distractorCfg.minCount, distractorCfg.maxCount]);
    if targetCount <= 0
        placedCount = 0;
        return;
    end

    [bgHeight, bgWidth, ~] = size(bg);
    if bgHeight < 1 || bgWidth < 1
        placedCount = 0;
        return;
    end

    numSource = numel(regions);
    maxBboxes = targetCount + size(occupiedBboxes, 1);
    allBboxes = zeros(maxBboxes, 4);
    bboxCount = 0;

    if ~isempty(occupiedBboxes)
        bboxCount = size(occupiedBboxes, 1);
        allBboxes(1:bboxCount, :) = double(occupiedBboxes);
    end

    placedCount = 0;
    for k = 1:targetCount
        srcIdx = randi(numSource);
        region = regions{srcIdx};
        bboxInfo = polygonBboxes{srcIdx};
        templatePatch = region.augPolygonImg;

        if isempty(templatePatch) || bboxInfo.width <= 0 || bboxInfo.height <= 0
            continue;
        end

        patchType = sample_distractor_type(distractorCfg);
        patch = synthesize_distractor_patch(templatePatch, cfg.texture, distractorCfg, patchType);
        if isempty(patch)
            continue;
        end

        patch = jitter_polygon_patch(patch, distractorCfg);

        % Apply random uniform scaling to distractor
        scaleRange = distractorCfg.sizeScaleRange;
        scaleFactor = scaleRange(1) + rand() * diff(scaleRange);
        [patch, localVerts] = scale_distractor_patch(patch, region.augVertices, bboxInfo, scaleFactor);

        if isempty(patch)
            continue;
        end

        % Compute scaled bbox dimensions
        scaledWidth = round(bboxInfo.width * scaleFactor);
        scaledHeight = round(bboxInfo.height * scaleFactor);
        if scaledWidth <= 0 || scaledHeight <= 0
            continue;
        end

        bboxStruct = struct('width', scaledWidth, 'height', scaledHeight);
        for attempt = 1:distractorCfg.maxPlacementAttempts
            [xCandidate, yCandidate] = random_top_left(bboxStruct, cfg.placement.margin, bgWidth, bgHeight);
            if ~isfinite(xCandidate) || ~isfinite(yCandidate)
                continue;
            end

            candidateBbox = [xCandidate, yCandidate, xCandidate + scaledWidth, yCandidate + scaledHeight];

            % Check collision with all placed bboxes (O(n) iteration acceptable for small distractor counts)
            hasConflict = false;
            for j = 1:bboxCount
                if bboxes_overlap(candidateBbox, allBboxes(j, :), cfg.placement.minSpacing)
                    hasConflict = true;
                    break;
                end
            end
            if hasConflict
                continue;
            end

            sceneVerts = localVerts + [xCandidate, yCandidate];

            bg = composite_to_background(bg, patch, sceneVerts);
            bboxCount = bboxCount + 1;
            allBboxes(bboxCount, :) = candidateBbox;
            placedCount = placedCount + 1;
            break;
        end
    end
end

function patchType = sample_distractor_type(distractorCfg)
    % Sample distractor rendering style using configured weights.

    weights = [1, 1, 1];
    if isfield(distractorCfg, 'typeWeights')
        candidate = double(distractorCfg.typeWeights(:)');
        candidate = candidate(isfinite(candidate) & candidate >= 0);
        if ~isempty(candidate)
            limit = min(3, numel(candidate));
            weights(1:limit) = candidate(1:limit);
        end
    end

    totalWeight = sum(weights);
    if totalWeight <= 0
        weights = [1, 1, 1];
        totalWeight = 3;
    end

    cumulative = cumsum(weights);
    r = rand() * totalWeight;
    patchType = find(r <= cumulative, 1, 'first');
    if isempty(patchType)
        patchType = 2;
    end
end

function patch = synthesize_distractor_patch(templatePatch, textureCfg, distractorCfg, patchType)
    % Create a synthetic distractor polygon using the original mask as a template.

    if isempty(templatePatch)
        patch = templatePatch;
        return;
    end

    mask = any(templatePatch > 0, 3);
    if ~any(mask(:))
        patch = [];
        return;
    end

    numChannels = size(templatePatch, 3);
    baseColor = sample_distractor_color(textureCfg, numChannels);
    if nargin < 4 || isempty(patchType) || ~ismember(patchType, 1:3)
        patchType = 2;
    end

    maskFloat = single(mask);
    baseColorNorm = single(baseColor) / 255;
    [height, width, ~] = size(templatePatch);
    patchFloat = zeros(height, width, numChannels, 'single');

    switch patchType
        case 1  % Outline only
            outlineMask = compute_outline_mask(mask, distractorCfg);
            if ~any(outlineMask(:))
                patch = [];
                return;
            end
            outlineFloat = single(outlineMask);
            strokeScale = 1 + 0.12 * (single(rand(1, numChannels)) - 0.5);
            strokeScale = max(0.6, min(1.4, strokeScale));
            for c = 1:numChannels
                patchFloat(:,:,c) = outlineFloat * (baseColorNorm(c) * strokeScale(c));
            end
            activeMask = outlineMask;

        case 3  % Textured fill
            texture = synthesize_distractor_texture(mask, textureCfg, distractorCfg);
            channelScale = 1 + 0.10 * (single(rand(1, numChannels)) - 0.5);
            channelScale = max(0.7, min(1.3, channelScale));
            for c = 1:numChannels
                modulation = texture * channelScale(c);
                patchFloat(:,:,c) = (baseColorNorm(c) + modulation) .* maskFloat;
            end
            activeMask = mask;

        otherwise  % Solid fill
            for c = 1:numChannels
                patchFloat(:,:,c) = maskFloat * baseColorNorm(c);
            end
            activeMask = mask;
    end

    patch = finalize_distractor_patch(patchFloat, activeMask, templatePatch);
end

function outlineMask = compute_outline_mask(mask, distractorCfg)
    % Compute an outline mask from the filled polygon mask.

    thickness = sample_outline_width(distractorCfg);
    outlineMask = bwperim(mask);
    if thickness > 1
        radius = max(0, thickness - 1);
        se = strel('disk', radius, 0);
        outlineMask = imdilate(outlineMask, se);
        outlineMask = outlineMask & mask;
    end

    if ~any(outlineMask(:))
        outlineMask = mask;
    end
end

function thickness = sample_outline_width(distractorCfg)
    % Sample outline stroke thickness in pixels.

    range = resolve_range(distractorCfg, 'outlineWidthRange', [1.5, 4.0], 1);
    widthVal = sample_range_value(range);
    thickness = max(1, round(widthVal));
end

function baseColor = sample_distractor_color(textureCfg, numChannels)
    if nargin < 2 || isempty(numChannels)
        numChannels = 3;
    end

    switch randi(4)
        case 1  % Uniform surface
            baseRGB = textureCfg.uniformBaseRGB + randi([-textureCfg.uniformVariation, textureCfg.uniformVariation], [1, 3]);
        case 2  % Speckled surface
            baseGray = 160 + randi([-25, 25]);
            baseRGB = [baseGray, baseGray, baseGray] + randi([-5, 5], [1, 3]);
        case 3  % Laminate surface
            if rand() < 0.5
                baseRGB = [245, 245, 245] + randi([-5, 5], [1, 3]);
            else
                baseRGB = [30, 30, 30] + randi([-5, 5], [1, 3]);
            end
        otherwise  % Skin-like hues
            hsv = [0.03 + rand() * 0.07, 0.25 + rand() * 0.35, 0.55 + rand() * 0.35];
            baseRGB = round(255 * hsv2rgb(hsv));
    end

    baseRGB = max(80, min(220, baseRGB));

    if numChannels ~= numel(baseRGB)
        if numChannels < numel(baseRGB)
            baseColor = baseRGB(1:numChannels);
        else
            baseColor = repmat(baseRGB(end), 1, numChannels);
            baseColor(1:numel(baseRGB)) = baseRGB;
        end
    else
        baseColor = baseRGB;
    end
end

function jittered = jitter_polygon_patch(patch, distractorCfg)
    % Apply lightweight photometric jitter while preserving mask boundaries.
    %
    % Inputs:
    %   patch - RGB image (uint8)
    %   distractorCfg - Configuration struct

    if isempty(patch)
        jittered = patch;
        return;
    end

    mask = any(patch > 0, 3);
    if ~any(mask(:))
        jittered = patch;
        return;
    end

    patchFloat = im2single(patch);

    contrastRange = resolve_range(distractorCfg, 'contrastScaleRange', [1, 1], 0);
    contrastScale = sample_range_value(contrastRange);

    brightnessRange = resolve_range(distractorCfg, 'brightnessOffsetRange', [0, 0]);
    brightnessOffset = sample_range_value(brightnessRange) / 255;

    patchFloat = (patchFloat - 0.5) * contrastScale + 0.5 + brightnessOffset;

    if isfield(distractorCfg, 'noiseStd') && distractorCfg.noiseStd > 0
        sigma = distractorCfg.noiseStd / 255;
        patchFloat = patchFloat + sigma * randn(size(patchFloat), 'like', patchFloat);
    end

    mask3 = repmat(single(mask), [1, 1, size(patchFloat, 3)]);
    patchFloat = min(1, max(0, patchFloat .* mask3));

    jittered = cast_patch_like_template(patchFloat, patch);
end

function patch = finalize_distractor_patch(patchFloat, activeMask, templatePatch)
    if isempty(activeMask) || ~any(activeMask(:))
        patch = [];
        return;
    end

    mask3 = repmat(single(activeMask), [1, 1, size(patchFloat, 3)]);
    patchFloat = min(1, max(0, patchFloat .* mask3));

    patch = cast_patch_like_template(patchFloat, templatePatch);
end

function patch = cast_patch_like_template(patchFloat, templatePatch)
    if isa(templatePatch, 'uint8')
        patch = im2uint8(patchFloat);
    elseif isa(templatePatch, 'uint16')
        patch = im2uint16(patchFloat);
    elseif isa(templatePatch, 'single')
        patch = patchFloat;
    else
        patch = cast(patchFloat, 'like', templatePatch);
    end
end

function texture = synthesize_distractor_texture(mask, textureCfg, distractorCfg)
    [height, width] = size(mask);

    surfaceTypes = 1:4;
    if isfield(distractorCfg, 'textureSurfaceTypes')
        candidate = unique(round(double(distractorCfg.textureSurfaceTypes(:)')));
        candidate = candidate(isfinite(candidate) & candidate >= 1 & candidate <= 4);
        if ~isempty(candidate)
            surfaceTypes = candidate;
        end
    end
    surfaceType = surfaceTypes(randi(numel(surfaceTypes)));

    texture = generate_surface_texture_base(surfaceType, width, height, textureCfg);
    texture = single(texture);

    activeVals = texture(mask);
    if isempty(activeVals)
        activeVals = single(randn(height * width, 1));
    end
    textureMean = mean(activeVals);
    textureStd = std(activeVals);
    if ~isfinite(textureStd) || textureStd < eps
        textureStd = 1;
    end

    texture = (texture - single(textureMean)) / single(textureStd);

    gainRange = resolve_range(distractorCfg, 'textureGainRange', [0.06, 0.18], 0);
    gain = sample_range_value(gainRange);

    texture = texture * single(gain);
end

function range = resolve_range(cfg, fieldName, defaultRange, minValue)
    if nargin < 4
        minValue = -inf;
    end

    range = defaultRange;
    if isfield(cfg, fieldName)
        values = double(cfg.(fieldName)(:).');
        values = values(isfinite(values));
        if isempty(values)
            range = defaultRange;
        elseif numel(values) >= 2
            range = sort(values(1:2));
        else
            range = [values(1), values(1)];
        end
    end

    range = max(minValue, range);
    if numel(range) < 2
        range = [range(1), range(1)];
    elseif range(1) > range(2)
        range(2) = range(1);
    end
end

function value = sample_range_value(range)
    range = range(:).';
    if isempty(range)
        value = 0;
        return;
    end

    if isscalar(range) || range(2) <= range(1)
        value = range(1);
    else
        value = range(1) + rand() * (range(2) - range(1));
    end
end

function [scaledPatch, scaledLocalVerts] = scale_distractor_patch(patch, vertices, bboxInfo, scaleFactor)
    % Apply uniform scaling to distractor patch and vertices.
    % Scales patch via imresize and adjusts vertices to match new dimensions.

    if isempty(patch) || scaleFactor <= 0
        scaledPatch = [];
        scaledLocalVerts = [];
        return;
    end

    % Resize patch image
    [origHeight, origWidth, ~] = size(patch);
    newHeight = round(origHeight * scaleFactor);
    newWidth = round(origWidth * scaleFactor);

    if newHeight < 1 || newWidth < 1
        scaledPatch = [];
        scaledLocalVerts = [];
        return;
    end

    scaledPatch = imresize(patch, [newHeight, newWidth], 'nearest');

    % Scale vertices relative to bbox origin
    localVerts = vertices - [bboxInfo.minX, bboxInfo.minY];
    scaledLocalVerts = localVerts * scaleFactor;
end

function [patchImg, isValid] = crop_ellipse_patch(polygonImg, ellipse)
    % Crop elliptical patch from polygon image
    [imgHeight, imgWidth, ~] = size(polygonImg);
    bbox = ellipse_bounding_box(ellipse);

    x1 = max(1, floor(bbox(1)));
    y1 = max(1, floor(bbox(2)));
    x2 = min(imgWidth, ceil(bbox(3)));
    y2 = min(imgHeight, ceil(bbox(4)));

    if x1 > x2 || y1 > y2
        patchImg = [];
        isValid = false;
        return;
    end

    patchImg = polygonImg(y1:y2, x1:x2, :);
    [h, w, ~] = size(patchImg);

    % Create ellipse mask
    [X, Y] = meshgrid(1:w, 1:h);
    cx = ellipse.center(1) - x1 + 1;
    cy = ellipse.center(2) - y1 + 1;

    theta = deg2rad(ellipse.rotation);
    dx = X - cx;
    dy = Y - cy;
    xRot =  dx * cos(theta) + dy * sin(theta);
    yRot = -dx * sin(theta) + dy * cos(theta);
    mask = (xRot / ellipse.semiMajor).^2 + (yRot / ellipse.semiMinor).^2 <= 1;

    if ~any(mask(:))
        patchImg = [];
        isValid = false;
        return;
    end

    % Zero out pixels outside the ellipse per-channel (all images are RGB)
    numChannels = size(patchImg, 3);
    for c = 1:numChannels
        plane = patchImg(:,:,c);
        plane(~mask) = 0;
        patchImg(:,:,c) = plane;
    end
    isValid = true;
end

function bbox = ellipse_bounding_box(ellipse)
    % Compute axis-aligned bounding box for rotated ellipse
    theta = deg2rad(ellipse.rotation);
    a = ellipse.semiMajor;
    b = ellipse.semiMinor;
    dx = sqrt((a * cos(theta))^2 + (b * sin(theta))^2);
    dy = sqrt((a * sin(theta))^2 + (b * cos(theta))^2);
    xc = ellipse.center(1);
    yc = ellipse.center(2);
    bbox = [xc - dx, yc - dy, xc + dx, yc + dy];
end

%% =========================================================================
%% BACKGROUND GENERATION (PROCEDURAL TEXTURES)
%% =========================================================================

function bg = generate_realistic_lab_surface(width, height, textureCfg, artifactCfg)
    % Generate realistic lab surface backgrounds with pooled single-precision textures
    width = max(1, round(width));
    height = max(1, round(height));

    surfaceType = randi(4);
    texture = borrow_background_texture(surfaceType, width, height, textureCfg);

    switch surfaceType
        case 1  % Uniform surface
            baseRGB = textureCfg.uniformBaseRGB + randi([-textureCfg.uniformVariation, textureCfg.uniformVariation], [1, 3]);
            noiseAmplitude = textureCfg.uniformNoiseRange(1) + rand() * diff(textureCfg.uniformNoiseRange);
            texture = texture .* single(noiseAmplitude);
        case 2  % Speckled surface
            baseGray = 160 + randi([-25, 25]);
            baseRGB = [baseGray, baseGray, baseGray] + randi([-5, 5], [1, 3]);
        case 3  % Laminate surface
            if rand() < 0.5
                baseRGB = [245, 245, 245] + randi([-5, 5], [1, 3]);
            else
                baseRGB = [30, 30, 30] + randi([-5, 5], [1, 3]);
            end
        otherwise  % Skin texture
            h = 0.03 + rand() * 0.07;
            s = 0.25 + rand() * 0.35;
            v = 0.55 + rand() * 0.35;
            baseRGB = round(255 * hsv2rgb([h, s, v]));
    end

    baseRGB = max(100, min(230, baseRGB));

    numChannels = 3;  % RGB backgrounds only
    bgSingle = repmat(reshape(single(baseRGB), [1, 1, numChannels]), [height, width, 1]);
    for c = 1:numChannels
        bgSingle(:,:,c) = bgSingle(:,:,c) + texture;
    end

    if rand() < 0.60
        bgSingle = add_lighting_gradient(bgSingle, width, height);
    end

    bg = clamp_uint8(bgSingle);
    bg = add_sparse_artifacts(bg, width, height, artifactCfg);
end

function texture = borrow_background_texture(surfaceType, width, height, textureCfg)
    persistent poolState

    if isempty(poolState) || texture_pool_config_changed(poolState, width, height, textureCfg)
        if ~isempty(poolState)
            oldWidth = poolState.width;
            oldHeight = poolState.height;
            widthDiff = abs(width - oldWidth) / max(oldWidth, 1);
            heightDiff = abs(height - oldHeight) / max(oldHeight, 1);
            if widthDiff > 0.01 || heightDiff > 0.01
                warning('augmentDataset:poolDimensionChange', ...
                    'Background dimensions changed from %dx%d to %dx%d. Texture pool reset. Run ''clear functions'' to avoid this warning.', ...
                    oldWidth, oldHeight, width, height);
            end
        end
        poolState = initialize_background_texture_pool(width, height, textureCfg);
    end

    entry = poolState.surface(surfaceType);
    if entry.cursor > entry.poolSize
        entry.order = randperm(entry.poolSize);
        entry.cursor = 1;
    end

    slot = entry.order(entry.cursor);
    entry.cursor = entry.cursor + 1;

    baseTexture = entry.textures{slot};
    if isempty(baseTexture)
        baseTexture = generate_surface_texture_base(surfaceType, width, height, textureCfg);
        entry.textures{slot} = baseTexture;
    end

    texture = apply_texture_pool_jitter(baseTexture, poolState);

    entry.usage(slot) = entry.usage(slot) + 1;
    if entry.usage(slot) >= poolState.refreshInterval
        entry.textures{slot} = [];
        entry.usage(slot) = 0;
    end

    poolState.surface(surfaceType) = entry;
end

function poolState = initialize_background_texture_pool(width, height, textureCfg)
    requestedPoolSize = max(1, round(textureCfg.poolSize));
    bytesPerTexture = max(1, double(width) * double(height) * 4);
    surfaces = 4;
    maxPoolBytes = textureCfg.poolMaxMemoryMB * 1024 * 1024;
    maxPerSurface = max(1, floor((maxPoolBytes / surfaces) / bytesPerTexture));
    poolSize = min(requestedPoolSize, maxPerSurface);
    refreshInterval = max(1, round(textureCfg.poolRefreshInterval));

    surfaceTemplate = struct( ...
        'textures', {cell(poolSize, 1)}, ...
        'usage', zeros(poolSize, 1, 'uint32'), ...
        'order', randperm(poolSize), ...
        'cursor', 1, ...
        'poolSize', poolSize);

    poolState = struct();
    poolState.width = width;
    poolState.height = height;
    poolState.cfgSnapshot = textureCfg;
    poolState.poolSize = poolSize;
    poolState.refreshInterval = refreshInterval;
    poolState.shiftPixels = max(0, round(textureCfg.poolShiftPixels));
    poolState.scaleRange = sort(textureCfg.poolScaleRange);
    if numel(poolState.scaleRange) ~= 2 || any(~isfinite(poolState.scaleRange))
        poolState.scaleRange = [1, 1];
    end
    poolState.flipProb = max(0, min(1, textureCfg.poolFlipProbability));
    poolState.surface = repmat(surfaceTemplate, 1, 4);
    for surfaceType = 1:4
        entry = poolState.surface(surfaceType);
        entry.order = randperm(entry.poolSize);
        entry.cursor = 1;
        entry.usage(:) = 0;
        poolState.surface(surfaceType) = entry;
    end
end

function changed = texture_pool_config_changed(poolState, width, height, textureCfg)
    changed = poolState.width ~= width || poolState.height ~= height || ~isequal(poolState.cfgSnapshot, textureCfg);
end

function texture = generate_surface_texture_base(surfaceType, width, height, textureCfg)
    height = max(1, round(double(height)));
    width = max(1, round(double(width)));

    persistent randBuffer1 randBuffer2 bufferSize
    if isempty(bufferSize) || any(bufferSize ~= [height, width])
        randBuffer1 = zeros(height, width, 'single');
        randBuffer2 = zeros(height, width, 'single');
        bufferSize = [height, width];
    end

    switch surfaceType
        case 1  % Uniform noise baseline (scaled per sample)
            randBuffer1(:) = single(randn(height, width));
            texture = randBuffer1;
        case 2  % Speckled surface (high + low frequency noise)
            randBuffer1(:) = single(randn(height, width));
            randBuffer1 = randBuffer1 .* single(textureCfg.speckleHighFreq);

            randBuffer2(:) = single(randn(height, width));
            randBuffer2 = imgaussfilt(randBuffer2, 8);
            randBuffer2 = randBuffer2 .* single(textureCfg.speckleLowFreq);

            texture = randBuffer1 + randBuffer2;
        case 3  % Laminate grain
            texture = generate_laminate_texture(width, height, textureCfg);
        case 4  % Skin-like microtexture
            texture = generate_skin_texture(width, height, textureCfg);
        otherwise
            randBuffer1(:) = single(randn(height, width));
            texture = randBuffer1;
    end
end

function texture = apply_texture_pool_jitter(baseTexture, poolState)
    % Apply random jitter to pooled texture
    texture = baseTexture;

    if poolState.shiftPixels > 0
        shiftX = randi([-poolState.shiftPixels, poolState.shiftPixels]);
        shiftY = randi([-poolState.shiftPixels, poolState.shiftPixels]);
        if shiftX ~= 0 || shiftY ~= 0
            texture = circshift(texture, [shiftY, shiftX]);
        end
    end

    if poolState.flipProb > 0
        if rand() < poolState.flipProb
            texture = flip(texture, 2);
        end
        if rand() < poolState.flipProb
            texture = flip(texture, 1);
        end
    end

    scaleRange = poolState.scaleRange;
    if numel(scaleRange) == 2 && scaleRange(2) > scaleRange(1)
        scale = scaleRange(1) + rand() * (scaleRange(2) - scaleRange(1));
        texture = texture .* single(scale);
    end
end

function texture = generate_laminate_texture(width, height, textureCfg)
    % Generate high-contrast laminate surface with subtle noise (single precision).

    width = max(1, round(width));
    height = max(1, round(height));

    texture = single(randn(height, width)) .* single(textureCfg.laminateNoiseStrength);
end

function texture = generate_skin_texture(width, height, textureCfg)
    % Generate subtle skin-like microtexture (single precision).

    width = max(1, round(width));
    height = max(1, round(height));

    lowFreq = imgaussfilt(single(randn(height, width)), 12) .* single(textureCfg.skinLowFreqStrength);
    midFreq = imgaussfilt(single(randn(height, width)), 3) .* single(textureCfg.skinMidFreqStrength);
    highFreq = single(randn(height, width)) .* single(textureCfg.skinHighFreqStrength);

    texture = lowFreq + midFreq + highFreq;
end

function bg = add_lighting_gradient(bg, width, height)
    % Add simple linear lighting gradient to simulate directional lighting (single-aware).

    width = max(1, round(width));
    height = max(1, round(height));
    if width < 50 || height < 50
        return;
    end

    lightAngle = rand() * 2 * pi;
    xAxis = single(0:(width - 1));
    yAxis = single(0:(height - 1));
    if width > 1
        xAxis = xAxis / single(width - 1);
    else
        xAxis = zeros(size(xAxis), 'single');
    end
    if height > 1
        yAxis = yAxis / single(height - 1);
    else
        yAxis = zeros(size(yAxis), 'single');
    end

    [Ygrid, Xgrid] = ndgrid(yAxis, xAxis);
    projection = Xgrid .* single(cos(lightAngle)) + Ygrid .* single(sin(lightAngle));

    gradientStrength = single(0.05 + rand() * 0.05);
    gradient = single(1) - gradientStrength/2 + projection .* gradientStrength;
    gradient = max(single(0.90), min(single(1.10), gradient));

    for c = 1:size(bg, 3)
        bg(:,:,c) = bg(:,:,c) .* gradient;
    end
end

function bg = add_sparse_artifacts(bg, width, height, artifactCfg)
    % Add variable-density artifacts anywhere on background for robust detection training
    %
    % OPTIMIZATION: Ellipses/lines use unit-square normalization (default 64x64 defined
    % in artifactCfg.unitMaskSize) to avoid large meshgrid allocations. Polygonal artifacts
    % render directly at target resolution so corner geometry remains crisp.
    %
    % Artifacts: rectangles, quadrilaterals, triangles, ellipses, lines
    % Count: configurable via artifactCfg.countRange (default 5-30)
    % Size: 1-100% of image diagonal (allows artifacts larger than frame)
    % Placement: unconstrained (artifacts can extend beyond boundaries for uniform spatial distribution)

    % Quick guard for tiny backgrounds
    width = max(1, round(width));
    height = max(1, round(height));
    if width < 8 || height < 8
        return;
    end

    % Number of artifacts: variable (1-100 by default)
    numArtifacts = artifactCfg.countRange(1) + randi(diff(artifactCfg.countRange) + 1);

    % Image diagonal for relative sizing (allows artifacts larger than image dimensions)
    diagSize = sqrt(width^2 + height^2);

    if isfield(artifactCfg, 'unitMaskSize') && ~isempty(artifactCfg.unitMaskSize)
        unitMaskSize = max(8, round(double(artifactCfg.unitMaskSize)));
    else
        unitMaskSize = 64;
    end
    unitCoords = linspace(0, 1, unitMaskSize);
    [unitGridX, unitGridY] = meshgrid(unitCoords, unitCoords);
    unitCenteredX = unitGridX - 0.5;
    unitCenteredY = unitGridY - 0.5;

    for i = 1:numArtifacts
        % Select artifact type (equal probability)
        artifactTypeRand = rand();
        if artifactTypeRand < 0.20
            artifactType = 'rectangle';
        elseif artifactTypeRand < 0.40
            artifactType = 'quadrilateral';
        elseif artifactTypeRand < 0.60
            artifactType = 'triangle';
        elseif artifactTypeRand < 0.80
            artifactType = 'ellipse';
        else
            artifactType = 'line';
        end

        % Uniform size: 1-100% of image diagonal (allows artifacts larger than frame)
        artifactSize = round(diagSize * (artifactCfg.sizeRangePercent(1) + rand() * diff(artifactCfg.sizeRangePercent)));
        artifactSize = max(artifactCfg.minSizePixels, artifactSize);

        % Lines: use artifactSize as length, add smaller width
        if strcmp(artifactType, 'line')
            lineLength = artifactSize;
            lineWidth = max(1, round(artifactSize * artifactCfg.lineWidthRatio));
            artifactSize = lineLength + artifactCfg.lineRotationPadding;
        end

        % Unconstrained random placement (artifacts can extend beyond frame boundaries)
        % Overhang margin creates partial artifacts at edges and uniform spatial distribution
        margin = round(artifactSize * artifactCfg.overhangMargin);
        xMin = 1 - margin;
        xMax = width + margin;
        yMin = 1 - margin;
        yMax = height + margin;

        x = randi([xMin, xMax]);
        y = randi([yMin, yMax]);

        % Create artifact mask; polygons draw directly at target resolution to keep sharp edges
        mask = [];
        unitMask = [];
        switch artifactType
            case 'ellipse'
                radiusAFraction = 0.5 * (artifactCfg.ellipseRadiusARange(1) + rand() * diff(artifactCfg.ellipseRadiusARange));
                radiusBFraction = 0.5 * (artifactCfg.ellipseRadiusBRange(1) + rand() * diff(artifactCfg.ellipseRadiusBRange));
                radiusAFraction = max(radiusAFraction, 1e-3);
                radiusBFraction = max(radiusBFraction, 1e-3);
                angle = rand() * pi;
                cosTheta = cos(angle);
                sinTheta = sin(angle);
                xRot = unitCenteredX * cosTheta - unitCenteredY * sinTheta;
                yRot = unitCenteredX * sinTheta + unitCenteredY * cosTheta;
                unitMask = single((xRot / radiusAFraction).^2 + (yRot / radiusBFraction).^2 <= 1);

            case 'rectangle'
                rectWidthFraction = artifactCfg.rectangleSizeRange(1) + rand() * diff(artifactCfg.rectangleSizeRange);
                rectHeightFraction = artifactCfg.rectangleSizeRange(1) + rand() * diff(artifactCfg.rectangleSizeRange);
                rectHalfWidth = max(rectWidthFraction * artifactSize / 2, 0.5);
                rectHalfHeight = max(rectHeightFraction * artifactSize / 2, 0.5);
                angle = rand() * pi;
                cosTheta = cos(angle);
                sinTheta = sin(angle);
                baseVerts = [
                    -rectHalfWidth, -rectHalfHeight;
                    rectHalfWidth, -rectHalfHeight;
                    rectHalfWidth,  rectHalfHeight;
                    -rectHalfWidth,  rectHalfHeight];
                rotMatrix = [cosTheta, -sinTheta; sinTheta, cosTheta];
                rotatedVerts = baseVerts * rotMatrix';
                centerPix = [(artifactSize + 1) / 2, (artifactSize + 1) / 2];
                verticesPix = rotatedVerts + centerPix;
                mask = generate_polygon_mask(verticesPix, artifactSize);

            case 'quadrilateral'
                baseWidthFraction = artifactCfg.quadSizeRange(1) + rand() * diff(artifactCfg.quadSizeRange);
                baseHeightFraction = artifactCfg.quadSizeRange(1) + rand() * diff(artifactCfg.quadSizeRange);
                perturbFraction = artifactCfg.quadPerturbation;
                halfWidth = max(baseWidthFraction / 2, 1e-3);
                halfHeight = max(baseHeightFraction / 2, 1e-3);
                verticesNorm = [
                    0.5 - halfWidth + (rand()-0.5) * perturbFraction, 0.5 - halfHeight + (rand()-0.5) * perturbFraction;
                    0.5 + halfWidth + (rand()-0.5) * perturbFraction, 0.5 - halfHeight + (rand()-0.5) * perturbFraction;
                    0.5 + halfWidth + (rand()-0.5) * perturbFraction, 0.5 + halfHeight + (rand()-0.5) * perturbFraction;
                    0.5 - halfWidth + (rand()-0.5) * perturbFraction, 0.5 + halfHeight + (rand()-0.5) * perturbFraction
                ];
                centeredVerts = (verticesNorm - 0.5) * (artifactSize - 1);
                centerPix = [(artifactSize + 1) / 2, (artifactSize + 1) / 2];
                verticesPix = centeredVerts + centerPix;
                mask = generate_polygon_mask(verticesPix, artifactSize);

            case 'triangle'
                baseSizeFraction = artifactCfg.triangleSizeRange(1) + rand() * diff(artifactCfg.triangleSizeRange);
                radius = max(baseSizeFraction * (artifactSize - 1) / 2, 0.5);
                angle = rand() * 2 * pi;
                verticesNorm = [
                    cos(angle),           sin(angle);
                    cos(angle + 2*pi/3),  sin(angle + 2*pi/3);
                    cos(angle + 4*pi/3),  sin(angle + 4*pi/3)
                ];
                centeredVerts = radius * verticesNorm;
                centerPix = [(artifactSize + 1) / 2, (artifactSize + 1) / 2];
                verticesPix = centeredVerts + centerPix;
                mask = generate_polygon_mask(verticesPix, artifactSize);

            otherwise  % 'line'
                angle = rand() * pi;
                cosTheta = cos(angle);
                sinTheta = sin(angle);
                lengthNorm = min(1, lineLength / artifactSize);
                halfLengthNorm = max(lengthNorm / 2, 1e-3);
                halfWidthNorm = max(lineWidth / artifactSize, 1 / artifactSize);
                xRot = unitCenteredX * cosTheta - unitCenteredY * sinTheta;
                yRot = unitCenteredX * sinTheta + unitCenteredY * cosTheta;
                lineCore = (abs(xRot) <= halfLengthNorm) & (abs(yRot) <= halfWidthNorm);
                unitMask = single(lineCore);
        end

        if isempty(mask) && isempty(unitMask)
            continue;
        end

        if isempty(mask)
            mask = imresize(unitMask, [artifactSize, artifactSize], 'nearest');
            mask = max(mask, single(0));
            mask = min(mask, single(1));
        end
        if ~any(mask(:))
            continue;
        end

        % Random intensity: darker or lighter
        % Lines tend to be darker; blobs can be either
        if strcmp(artifactType, 'line')
            intensity = artifactCfg.lineIntensityRange(1) + randi(diff(artifactCfg.lineIntensityRange) + 1);
        else
            if rand() < 0.5
                intensity = artifactCfg.blobDarkIntensityRange(1) + randi(diff(artifactCfg.blobDarkIntensityRange) + 1);
            else
                intensity = artifactCfg.blobLightIntensityRange(1) + randi(diff(artifactCfg.blobLightIntensityRange) + 1);
            end
        end

        % Blend into background, handling artifacts that extend beyond frame boundaries
        % Compute valid intersection between artifact bbox and image bounds
        xStart = max(1, x);
        yStart = max(1, y);
        xEnd = min(width, x + artifactSize - 1);
        yEnd = min(height, y + artifactSize - 1);

        % Validate intersection exists
        if xEnd < xStart || yEnd < yStart
            continue;  % Artifact completely outside bounds
        end

        % Compute corresponding mask region (offset if artifact starts outside frame)
        maskXStart = max(1, 2 - x);  % Offset into mask if x < 1
        maskYStart = max(1, 2 - y);  % Offset into mask if y < 1
        maskXEnd = maskXStart + (xEnd - xStart);
        maskYEnd = maskYStart + (yEnd - yStart);

        % Blend artifact into background
        maskRegion = single(mask(maskYStart:maskYEnd, maskXStart:maskXEnd));
        intensitySingle = single(intensity);
        numChannels = size(bg, 3);
        for c = 1:numChannels
            region = single(bg(yStart:yEnd, xStart:xEnd, c));
            region = region + maskRegion .* intensitySingle;
            bg(yStart:yEnd, xStart:xEnd, c) = clamp_uint8(region);
        end
    end
end

function mask = generate_polygon_mask(verticesPix, targetSize)
    % Rasterize polygon vertices expressed in pixel coordinates into a binary mask.
    if isempty(verticesPix) || size(verticesPix, 2) ~= 2
        mask = [];
        return;
    end

    verticesPix = double(verticesPix);
    polyMask = poly2mask(verticesPix(:,1), verticesPix(:,2), targetSize, targetSize);
    if ~any(polyMask(:))
        mask = [];
        return;
    end

    mask = single(polyMask);
end

function bg = add_polygon_occlusions(bg, scenePolygons, probability)
    % Draw thin occlusions (e.g., hair/strap-like) across polygons with some probability.
    % Each occlusion is a soft line that slightly darkens or lightens the image beneath.

    if probability <= 0
        return;
    end

    [imgH, imgW, ~] = size(bg);

    for i = 1:numel(scenePolygons)
        if rand() >= probability
            continue;
        end

        verts = scenePolygons{i};
        if isempty(verts) || any(~isfinite(verts(:)))
            continue;
        end

        minX = max(1, floor(min(verts(:,1))));
        maxX = min(imgW, ceil(max(verts(:,1))));
        minY = max(1, floor(min(verts(:,2))));
        maxY = min(imgH, ceil(max(verts(:,2))));
        if maxX <= minX || maxY <= minY
            continue;
        end

        % Build local grid for the polygon bbox
        [X, Y] = meshgrid(minX:maxX, minY:maxY);

        % Polygon mask for clipping
        polyMask = poly2mask(verts(:,1) - (minX - 1), verts(:,2) - (minY - 1), maxY - minY + 1, maxX - minX + 1);
        if ~any(polyMask(:))
            continue;
        end

        % Choose line params centered near polygon centroid
        cx = mean(verts(:,1));
        cy = mean(verts(:,2));
        angle = rand() * 2 * pi;      % random orientation
        halfWidth = 1 + rand() * 2;   % ~2-3 px thick

        % Distance to a line through (cx,cy) with normal [sin, -cos]
        d = abs((X - cx) * sin(angle) - (Y - cy) * cos(angle));
        lineMask = double(d <= halfWidth);
        lineMask = imgaussfilt(lineMask, 0.8);

        % Clip to polygon region
        lineMask = lineMask .* double(polyMask);
        if ~any(lineMask(:))
            continue;
        end

        % Random intensity: slight darken or lighten
        if rand() < 0.5
            delta = -(20 + randi(20));
        else
            delta =  (20 + randi(20));
        end

        % Blend into background
        region = bg(minY:maxY, minX:maxX, :);
        numChannels = size(region, 3);
        for c = 1:numChannels
            plane = double(region(:,:,c));
            plane = plane + lineMask * double(delta);
            region(:,:,c) = clamp_uint8(plane);
        end
        bg(minY:maxY, minX:maxX, :) = region;
    end
end


%% =========================================================================
%% PHOTOMETRIC AUGMENTATION (NEW IN V3)
%% =========================================================================

function img = apply_photometric_augmentation(img, mode)
    % Apply color-safe photometric augmentation to entire scene
    % Preserves relative color relationships between concentration regions
    %
    % Inputs:
    %   img - RGB image (uint8)
    %   mode - 'subtle' (default), 'moderate', or 'extreme' (Phase 1.7)

    if nargin < 2
        mode = 'subtle';
    end

    % Convert to double in [0,1] for processing
    imgDouble = im2double(img);

    % 1. Global brightness adjustment
    if strcmp(mode, 'subtle')
        brightRange = [0.95, 1.05];  % ±5% (reduced from ±10%)
    elseif strcmp(mode, 'extreme')
        brightRange = [0.40, 0.60];  % Very low lighting (Phase 1.7)
    else
        brightRange = [0.90, 1.10];  % ±10% (reduced from ±15%)
    end
    brightFactor = brightRange(1) + rand() * diff(brightRange);
    imgDouble = imgDouble * brightFactor;

    % 2. Global contrast adjustment (around image mean)
    if strcmp(mode, 'subtle')
        contrastRange = [0.96, 1.04];  % ±4% (reduced from ±8%)
    else
        contrastRange = [0.92, 1.08];  % ±8% (reduced from ±12%)
    end
    contrastFactor = contrastRange(1) + rand() * diff(contrastRange);
    imgMean = mean(imgDouble(:));
    imgDouble = (imgDouble - imgMean) * contrastFactor + imgMean;

    % 3. White balance jitter (per-channel gain), 60% probability
    if rand() < 0.60
        numChannels = size(imgDouble, 3);
        if numChannels == 3  % White balance requires RGB
            gains = [0.92 + rand() * 0.16, 0.92 + rand() * 0.16, 0.92 + rand() * 0.16];
            for c = 1:numChannels
                imgDouble(:,:,c) = imgDouble(:,:,c) * gains(c);
            end
        end
    end

    % 4. Subtle saturation adjustment (preserve hue) - 60% of augmented samples
    if rand() < 0.6
        % Clamp before color-space conversion to avoid numeric spill
        imgDouble = min(1, max(0, imgDouble));
        imgHSV = rgb2hsv(imgDouble);
        satFactor = 0.94 + rand() * 0.12;  % [0.94, 1.06]
        imgHSV(:,:,2) = min(1, max(0, imgHSV(:,:,2) * satFactor));
        imgDouble = hsv2rgb(imgHSV);
    end

    % 5. Gamma correction (exposure simulation) - 40% of augmented samples
    %    Ensure input is within [0,1] before exponentiation to avoid
    %    negative^fraction -> complex results.
    if rand() < 0.4
        gamma = 0.92 + rand() * 0.16;  % [0.92, 1.08]
        imgDouble = min(1, max(0, imgDouble));
        imgDouble = imgDouble .^ gamma;
    end

    % Final clamp and convert back to uint8
    img = im2uint8(min(1, max(0, imgDouble)));
end

%% =========================================================================
%% TRANSFORMATION FUNCTIONS
%% =========================================================================

function viewParams = sample_viewpoint(cameraCfg)
    % Sample camera viewpoint with uniform distribution

    % Simple uniform sampling over camera range
    vx = cameraCfg.xRange(1) + rand() * diff(cameraCfg.xRange);
    vy = cameraCfg.yRange(1) + rand() * diff(cameraCfg.yRange);
    vz = cameraCfg.zRange(1) + rand() * diff(cameraCfg.zRange);

    viewParams = struct('vx', vx, 'vy', vy, 'vz', vz);
end

function tform = compute_homography(imageSize, viewParams, cameraCfg)
    imgHeight = imageSize(1);
    imgWidth = imageSize(2);
    corners = [1 1; imgWidth 1; imgWidth imgHeight; 1 imgHeight];

    yawDeg = normalize_to_angle(viewParams.vx, cameraCfg.xRange, cameraCfg.maxAngleDeg);
    pitchDeg = normalize_to_angle(viewParams.vy, cameraCfg.yRange, cameraCfg.maxAngleDeg);

    projected = project_corners(imgWidth, imgHeight, yawDeg, pitchDeg, viewParams.vz);
    coverage = cameraCfg.coverageOffcenter;
    if abs(viewParams.vx) < 1e-3 && abs(viewParams.vy) < 1e-3
        coverage = cameraCfg.coverageCenter;
    end
    aligned = fit_points_to_frame(projected, imgWidth, imgHeight, coverage);

    tform = fitgeotrans(corners, aligned, 'projective');
end

function projected = project_corners(imgWidth, imgHeight, yawDeg, pitchDeg, viewZ)
    corners = [1 1; imgWidth 1; imgWidth imgHeight; 1 imgHeight];
    cx = (imgWidth + 1) / 2;
    cy = (imgHeight + 1) / 2;
    scale = max(imgWidth, imgHeight);

    pts = [(corners(:,1) - cx) / scale, (corners(:,2) - cy) / scale, zeros(4,1)];
    yaw = deg2rad(yawDeg);
    pitch = deg2rad(pitchDeg);
    Ry = [cos(yaw) 0 sin(yaw); 0 1 0; -sin(yaw) 0 cos(yaw)];
    Rx = [1 0 0; 0 cos(pitch) -sin(pitch); 0 sin(pitch) cos(pitch)];
    R = Rx * Ry;

    rotated = (R * pts')';
    rotated(:,3) = rotated(:,3) + viewZ;

    f = viewZ;
    u = f * rotated(:,1) ./ rotated(:,3);
    v = f * rotated(:,2) ./ rotated(:,3);

    projected = [u * scale + cx, v * scale + cy];
end

function aligned = fit_points_to_frame(projected, imgWidth, imgHeight, coverage)
    MIN_DIMENSION = 1.0;  % Minimum valid dimension in pixels

    minX = min(projected(:,1));
    maxX = max(projected(:,1));
    minY = min(projected(:,2));
    maxY = max(projected(:,2));
    width = maxX - minX;
    height = maxY - minY;
    if width < MIN_DIMENSION || height < MIN_DIMENSION
        aligned = projected;
        return;
    end

    scale = coverage * min((imgWidth - 1) / width, (imgHeight - 1) / height);
    center = [(maxX + minX) / 2, (maxY + minY) / 2];
    targetCenter = [(imgWidth + 1) / 2, (imgHeight + 1) / 2];

    aligned = (projected - center) * scale + targetCenter;
end

function polygonOut = transform_polygon(vertices, tform)
    [x, y] = transformPointsForward(tform, vertices(:,1), vertices(:,2));
    polygonOut = [x, y];
end

function hashVal = stable_string_hash(strInput)
    % Deterministic uint32 hash for reproducible RNG offsets

    if nargin == 0 || isempty(strInput)
        hashVal = 0;
        return;
    end

    if isstring(strInput)
        strData = strjoin(cellstr(strInput), '#');
    elseif iscellstr(strInput)
        strData = strjoin(strInput, '#');
    else
        strData = char(strInput);
    end

    if isempty(strData)
        hashVal = 0;
        return;
    end

    chars = uint32(strData(:).');
    hash = uint32(2166136261);
    prime = uint32(16777619);

    for idx = 1:numel(chars)
        hash = bitxor(hash, chars(idx));
        hash = hash * prime;
    end

    hashVal = double(hash);
end

function ellipseOut = transform_ellipse(ellipseIn, tform)
    conic = ellipse_to_conic(ellipseIn);

    % Check if conic is degenerate (from invalid input ellipse)
    if all(conic(:) == 0)
        ellipseOut = invalid_ellipse();
        return;
    end

    H = tform.T';

    % Validate transformation matrix is not singular before inversion
    if abs(det(H)) < 1e-10
        ellipseOut = invalid_ellipse();
        return;
    end

    Hinv = inv(H);
    transformedConic = Hinv' * conic * Hinv;
    ellipseOut = conic_to_ellipse(transformedConic);
end

function ellipseImageSpace = map_ellipse_crop_to_image(ellipseCrop, cropBbox)
    % Map ellipse from crop space to image space

    validateattributes(cropBbox, {'numeric'}, {'vector','numel',4}, mfilename, 'cropBbox');

    xOffset = double(cropBbox(1) - 1);
    yOffset = double(cropBbox(2) - 1);

    ellipseImageSpace = ellipseCrop;
    ellipseImageSpace.center = double(ellipseCrop.center) + [xOffset, yOffset];
    ellipseImageSpace.semiMajor = double(ellipseCrop.semiMajor);
    ellipseImageSpace.semiMinor = double(ellipseCrop.semiMinor);
    ellipseImageSpace.rotation = double(ellipseCrop.rotation);
    ellipseImageSpace.valid = true;
end

function conic = ellipse_to_conic(ellipse)
    MIN_AXIS = 0.1;  % Minimum ellipse axis length in pixels

    xc = ellipse.center(1);
    yc = ellipse.center(2);
    a = ellipse.semiMajor;
    b = ellipse.semiMinor;
    theta = deg2rad(ellipse.rotation);

    % Validate axes are positive to prevent division by zero
    if a < MIN_AXIS || b < MIN_AXIS || ~isfinite(a) || ~isfinite(b)
        % Return degenerate conic that will be detected by conic_to_ellipse
        conic = zeros(3, 3);
        return;
    end

    c = cos(theta);
    s = sin(theta);
    R = [c -s; s c];
    D = diag([1/a^2, 1/b^2]);
    Q = R * D * R';
    center = [xc; yc];

    conic = [Q, -Q*center; -center'*Q, center'*Q*center - 1];
end

function ellipse = conic_to_ellipse(C)
    % Validate C(3,3) is not zero before normalization to prevent division by zero
    if abs(C(3,3)) < 1e-10
        ellipse = invalid_ellipse();
        return;
    end

    C = C ./ C(3,3);
    A = C(1,1);
    B = 2*C(1,2);
    Cc = C(2,2);
    D = 2*C(1,3);
    E = 2*C(2,3);
    F = C(3,3);

    denom = B^2 - 4*A*Cc;
    if denom >= 0
        ellipse = invalid_ellipse();
        return;
    end

    xc = (2*Cc*D - B*E) / denom;
    yc = (2*A*E - B*D) / denom;

    theta = 0.5 * atan2(B, A - Cc);
    cosT = cos(theta);
    sinT = sin(theta);

    A1 = A*cosT^2 + B*cosT*sinT + Cc*sinT^2;
    C1 = A*sinT^2 - B*cosT*sinT + Cc*cosT^2;

    F0 = F + D*xc + E*yc + A*xc^2 + B*xc*yc + Cc*yc^2;

    % Defensive check (should never trigger with validated inputs upstream)
    if ~all(isfinite([xc, yc, A1, C1, F0]))
        ellipse = invalid_ellipse();
        return;
    end

    % Validate ellipse exists (F0 < 0 and positive diagonal elements)
    if F0 >= 0 || A1 <= 0 || C1 <= 0
        ellipse = invalid_ellipse();
        return;
    end

    % Compute axes (arguments guaranteed positive by above checks)
    a = sqrt(-F0 / A1);
    b = sqrt(-F0 / C1);

    if a < b
        tmp = a;
        a = b;
        b = tmp;
        theta = theta + pi/2;
    end

    ellipse = struct( ...
        'center', [xc, yc], ...
        'semiMajor', a, ...
        'semiMinor', b, ...
        'rotation', rad2deg(theta), ...
        'valid', true);
end

function ellipse = invalid_ellipse()
    ellipse = struct('center', [NaN, NaN], 'semiMajor', NaN, 'semiMinor', NaN, ...
                     'rotation', NaN, 'valid', false);
end

function tform = centered_rotation_tform(imageSize, angleDeg)
    height = imageSize(1);
    width = imageSize(2);
    cx = (width + 1) / 2;
    cy = (height + 1) / 2;

    cosA = cosd(angleDeg);
    sinA = sind(angleDeg);

    translateToOrigin = [1 0 0; 0 1 0; -cx -cy 1];
    rotation = [cosA -sinA 0; sinA cosA 0; 0 0 1];
    translateBack = [1 0 0; 0 1 0; cx cy 1];

    matrix = translateToOrigin * rotation * translateBack;
    tform = affine2d(matrix);
end

function angleDeg = normalize_to_angle(value, range, maxAngle)
    if numel(range) ~= 2
        error('augmentDataset:invalidRange', ...
            'Range parameter must have exactly 2 elements. Got %d.', numel(range));
    end

    mid = mean(range);
    span = range(2) - range(1);
    if span <= 0
        angleDeg = 0;
        return;
    end
    normalized = 2 * (value - mid) / span;
    angleDeg = normalized * maxAngle;
end

function val = rand_range(range)
    val = range(1) + (range(2) - range(1)) * rand();
end

%% =========================================================================
%% COORDINATE FILE I/O
%% =========================================================================

function write_coordinates_atomic(coordPath, headerLine, formatFunc, entries)
    % Generic atomic coordinate writer with deduplication
    %
    % Inputs:
    %   coordPath - target coordinate file path
    %   headerLine - header line string
    %   formatFunc - function handle that formats one entry struct as string
    %   entries - cell array of coordinate entry structs

    tmpPath = tempname(fileparts(coordPath));
    fid = fopen(tmpPath, 'wt');
    if fid == -1
        error('augmentDataset:coordWrite', 'Cannot open temp file: %s', tmpPath);
    end

    % Write header
    fprintf(fid, '%s\n', headerLine);

    % Write entries
    for i = 1:numel(entries)
        if ~isempty(entries{i})
            fprintf(fid, '%s\n', formatFunc(entries{i}));
        end
    end

    fclose(fid);

    % Atomic move with fallback
    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        warning(msgid, 'movefile failed: %s. Trying copyfile fallback.', msg);
        [ok2, msg2, msgid2] = copyfile(tmpPath, coordPath, 'f');
        if ok2
            delete(tmpPath);
        else
            error(msgid2, 'Coordinate write failed: %s', msg2);
        end
    end
end

function write_stage2_coordinates(coords, outputDir, filename)
    % Atomically write stage 2 coordinates (deduplicated by image+concentration)
    % Format: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation

    coordFolder = outputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, filename);

    % Load existing entries (if any)
    existing = read_polygon_coordinates(coordPath);
    map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    if ~isempty(existing)
        for k = 1:numel(existing)
            e = existing(k);
            key = sprintf('%s|%d', char(e.image), e.concentration);
            map(key) = e;
        end
    end

    % Merge/override with new rows
    for i = 1:numel(coords)
        c = coords{i};
        key = sprintf('%s|%d', char(c.image), c.concentration);
        map(key) = c;
    end

    % Stage 2 format function
    function str = formatStage2(e)
        verts = round(e.vertices);
        rotation = 0;
        if isfield(e, 'rotation') && ~isempty(e.rotation)
            rotation = e.rotation;
        end
        str = sprintf('%s %d %d %d %d %d %d %d %d %d %.2f', ...
            e.image, e.concentration, ...
            verts(1,1), verts(1,2), verts(2,1), verts(2,2), ...
            verts(3,1), verts(3,2), verts(4,1), verts(4,2), rotation);
    end

    headerLine = 'image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation';

    % Convert map to cell array
    keysArr = map.keys;
    entries = cell(numel(keysArr), 1);
    for i = 1:numel(keysArr)
        entries{i} = map(keysArr{i});
    end

    write_coordinates_atomic(coordPath, headerLine, @formatStage2, entries);
end

function write_stage3_coordinates(coords, outputDir, filename)
    % Atomically write stage 3 coordinates (dedup by image+concentration+replicate)
    % Format: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle

    % Skip if no coordinates to write
    if isempty(coords)
        return;
    end

    coordFolder = outputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, filename);

    % Load existing entries (if any)
    existing = read_ellipse_coordinates(coordPath);
    map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    if ~isempty(existing)
        for k = 1:numel(existing)
            e = existing(k);
            key = sprintf('%s|%d|%d', char(e.image), e.concentration, e.replicate);
            map(key) = e;
        end
    end

    % Merge/override with new rows
    for i = 1:numel(coords)
        c = coords{i};
        key = sprintf('%s|%d|%d', char(c.image), c.concentration, c.replicate);
        map(key) = c;
    end

    % Stage 3 format function
    function str = formatStage3(e)
        str = sprintf('%s %d %d %.2f %.2f %.4f %.4f %.2f', ...
            e.image, e.concentration, e.replicate, ...
            e.center(1), e.center(2), e.semiMajor, e.semiMinor, e.rotation);
    end

    headerLine = 'image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle';

    % Convert map to cell array
    keysArr = map.keys;
    entries = cell(numel(keysArr), 1);
    for i = 1:numel(keysArr)
        entries{i} = map(keysArr{i});
    end

    write_coordinates_atomic(coordPath, headerLine, @formatStage3, entries);
end

function lines = read_coordinate_file_lines(coordPath)
    % Read non-empty lines from coordinate file, skipping header
    % Returns cell array of trimmed lines, or empty cell array if file doesn't exist
    lines = {};

    if ~isfile(coordPath)
        return;
    end

    fid = fopen(coordPath, 'rt');
    if fid == -1
        warning('augmentDataset:coordReadFail', 'Cannot open %s for reading', coordPath);
        return;
    end
    cleaner = onCleanup(@() fclose(fid));

    % Skip header if present (header starts with literal word "image")
    headerLine = fgetl(fid);
    if ~ischar(headerLine)
        fseek(fid, 0, 'bof');
    else
        tokens = strsplit(strtrim(headerLine));
        if isempty(tokens) || ~strcmp(tokens{1}, 'image')
            fseek(fid, 0, 'bof');
        end
    end

    % Read all non-empty lines
    lines = cell(1000, 1);
    lineCount = 0;
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        trimmed = strtrim(line);
        if ~isempty(trimmed)
            lineCount = lineCount + 1;
            if lineCount > length(lines)
                lines = [lines; cell(1000, 1)]; %#ok<AGROW> % Intentional chunked growth (1000-line batches)
            end
            lines{lineCount} = trimmed;
        end
    end
    lines = lines(1:lineCount);
end

function entries = read_polygon_coordinates(coordPath)
    % Read polygon coordinates from stage 2 (2_micropads)
    % Format: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation
    lines = read_coordinate_file_lines(coordPath);

    % Pre-allocate with efficient estimate
    maxEntries = max(100, numel(lines) * 2);
    entries = struct('image', {}, 'concentration', {}, 'vertices', {}, 'rotation', {});
    entries(maxEntries).image = '';
    count = 0;
    skippedCount = 0;

    for i = 1:numel(lines)
        parts = strsplit(lines{i});
        if numel(parts) < 10
            skippedCount = skippedCount + 1;
            continue;
        end

        imgName = parts{1};
        concentration = str2double(parts{2});
        coords = str2double(parts(3:10));

        % Read rotation if available (11th field), default to 0 if missing
        if numel(parts) >= 11
            rotation = str2double(parts{11});
            if isnan(rotation)
                rotation = 0;
            end
        else
            rotation = 0;
        end

        if any(isnan([concentration, coords]))
            skippedCount = skippedCount + 1;
            continue;
        end

        vertices = reshape(coords, [2, 4])';

        count = count + 1;
        entries(count) = struct('image', imgName, ...
                                'concentration', concentration, ...
                                'vertices', vertices, ...
                                'rotation', rotation);
    end

    if skippedCount > 0
        warning('augmentDataset:invalidCoords', ...
                'Skipped %d invalid coordinate entries in %s', skippedCount, coordPath);
    end

    entries = entries(1:count);
end

function entries = read_ellipse_coordinates(coordPath)
    % Read ellipse coordinates from stage 3
    lines = read_coordinate_file_lines(coordPath);

    % Pre-allocate with generous estimate
    maxEntries = max(100, numel(lines) * 2);
    entries = struct('image', {}, 'concentration', {}, 'replicate', {}, ...
                     'center', {}, 'semiMajor', {}, 'semiMinor', {}, 'rotation', {});
    entries(maxEntries).image = '';
    count = 0;
    skippedCount = 0;

    for i = 1:numel(lines)
        parts = strsplit(lines{i});
        if numel(parts) < 8
            skippedCount = skippedCount + 1;
            continue;
        end

        imgName = parts{1};
        nums = str2double(parts(2:8));

        if any(isnan(nums))
            skippedCount = skippedCount + 1;
            continue;
        end

        count = count + 1;
        entries(count) = struct('image', imgName, ...
                                'concentration', nums(1), ...
                                'replicate', nums(2), ...
                                'center', nums(3:4), ...
                                'semiMajor', nums(5), ...
                                'semiMinor', nums(6), ...
                                'rotation', nums(7));
    end

    if skippedCount > 0
        fprintf('  Note: Skipped %d invalid ellipse coordinate entries in %s (missing or invalid values)\n', ...
                skippedCount, coordPath);
    end

    entries = entries(1:count);
end

%% =========================================================================
%% GROUPING AND LOOKUP
%% =========================================================================

function ellipseMap = group_ellipses_by_parent(ellipseEntries, hasEllipses)
    % Group ellipses by parent image and concentration
    ellipseMap = containers.Map('KeyType', 'char', 'ValueType', 'any');

    if ~hasEllipses
        return;
    end

    for i = 1:numel(ellipseEntries)
        e = ellipseEntries(i);
        % Extract base image name
        imgName = e.image;
        tokens = regexp(imgName, '(.+)_con_\d+', 'tokens');
        if ~isempty(tokens)
            baseName = tokens{1}{1};
        else
            [~, baseName, ~] = fileparts(imgName);
        end

        key = sprintf('%s#%d', baseName, e.concentration);

        if ~isKey(ellipseMap, key)
            ellipseMap(key) = e;
        else
            buf = ellipseMap(key);
            ellipseMap(key) = [buf, e];
        end
    end
end

function paperGroups = group_polygons_by_image(polygonEntries)
    % Group polygons by source image name
    paperGroups = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:numel(polygonEntries)
        p = polygonEntries(i);
        [~, imgBase, ~] = fileparts(p.image);

        if ~isKey(paperGroups, imgBase)
            paperGroups(imgBase) = p;
        else
            buf = paperGroups(imgBase);
            paperGroups(imgBase) = [buf, p];
        end
    end
end

%% =========================================================================
%% VALIDATION AND UTILITIES
%% =========================================================================

function angle = normalizeAngle(angle)
    % Normalize angle to range [-180, 180] degrees
    angle = mod(angle + 180, 360) - 180;
end

function valid = is_valid_polygon(vertices, minArea)
    % Check if polygon is valid (non-degenerate)
    area = polyarea(vertices(:,1), vertices(:,2));
    valid = area > minArea;
end

function positions = place_polygons_nonoverlapping(polygonBboxes, bgWidth, bgHeight, margin, minSpacing, maxRetries)
    % Place polygons randomly with collision avoidance using spatial grid acceleration

    numPolygons = numel(polygonBboxes);
    positions = cell(numPolygons, 1);
    placedBboxes = zeros(numPolygons, 4);

    % Initialize spatial grid for O(1) collision detection
    gridCellSize = minSpacing;
    gridWidth = ceil(bgWidth / gridCellSize);
    gridHeight = ceil(bgHeight / gridCellSize);
    grid = cell(gridHeight, gridWidth);

    for i = 1:numPolygons
        bbox = polygonBboxes{i};
        placed = false;
        lastCandidate = [];

        for attempt = 1:maxRetries
            [x, y] = random_top_left(bbox, margin, bgWidth, bgHeight);

            if ~isfinite(x) || ~isfinite(y)
                lastCandidate = [x, y];
                continue;
            end

            candidateBbox = [x, y, x + bbox.width, y + bbox.height];

            % Check only grid cells in bbox neighborhood
            cellMinX = max(1, floor((candidateBbox(1) - minSpacing) / gridCellSize) + 1);
            cellMaxX = min(gridWidth, ceil((candidateBbox(3) + minSpacing) / gridCellSize));
            cellMinY = max(1, floor((candidateBbox(2) - minSpacing) / gridCellSize) + 1);
            cellMaxY = min(gridHeight, ceil((candidateBbox(4) + minSpacing) / gridCellSize));

            hasOverlap = false;
            for cy = cellMinY:cellMaxY
                for cx = cellMinX:cellMaxX
                    cellPolygons = grid{cy, cx};
                    for k = 1:numel(cellPolygons)
                        j = cellPolygons(k);
                        if bboxes_overlap(candidateBbox, placedBboxes(j, :), minSpacing)
                            hasOverlap = true;
                            break;
                        end
                    end
                    if hasOverlap, break; end
                end
                if hasOverlap, break; end
            end

            if ~hasOverlap
                positions{i} = struct('x', x, 'y', y);
                placedBboxes(i, :) = candidateBbox;
                placed = true;

                % Add to grid
                cellMinX = max(1, floor(candidateBbox(1) / gridCellSize) + 1);
                cellMaxX = min(gridWidth, ceil(candidateBbox(3) / gridCellSize));
                cellMinY = max(1, floor(candidateBbox(2) / gridCellSize) + 1);
                cellMaxY = min(gridHeight, ceil(candidateBbox(4) / gridCellSize));

                for cy = cellMinY:cellMaxY
                    for cx = cellMinX:cellMaxX
                        grid{cy, cx} = [grid{cy, cx}, i];
                    end
                end

                break;
            end

            lastCandidate = [x, y];
        end

        % If all retries failed, force placement anyway
        if ~placed
            if isempty(lastCandidate) || any(~isfinite(lastCandidate))
                x = max(0, (bgWidth - bbox.width) / 2);
                y = max(0, (bgHeight - bbox.height) / 2);
            else
                x = lastCandidate(1);
                y = lastCandidate(2);
            end
            positions{i} = struct('x', x, 'y', y);
            placedBboxes(i, :) = [x, y, x + bbox.width, y + bbox.height];
        end
    end
end

function [x, y] = random_top_left(bboxStruct, margin, widthVal, heightVal)
    availX = max(0, widthVal - bboxStruct.width - 2 * margin);
    if availX > 0
        x = margin + rand() * availX;
    else
        x = max(0, (widthVal - bboxStruct.width) / 2);
    end
    x = min(x, widthVal - bboxStruct.width);

    availY = max(0, heightVal - bboxStruct.height - 2 * margin);
    if availY > 0
        y = margin + rand() * availY;
    else
        y = max(0, (heightVal - bboxStruct.height) / 2);
    end
    y = min(y, heightVal - bboxStruct.height);
end

function overlap = bboxes_overlap(bbox1, bbox2, minSpacing)
    % Check if two axis-aligned bounding boxes overlap with minimum spacing
    % bbox format: [x1, y1, x2, y2]

    % Expand bboxes by minSpacing/2 on all sides
    bbox1_expanded = [bbox1(1) - minSpacing/2, bbox1(2) - minSpacing/2, ...
                      bbox1(3) + minSpacing/2, bbox1(4) + minSpacing/2];
    bbox2_expanded = [bbox2(1) - minSpacing/2, bbox2(2) - minSpacing/2, ...
                      bbox2(3) + minSpacing/2, bbox2(4) + minSpacing/2];

    % Check for overlap
    overlap = ~(bbox1_expanded(3) < bbox2_expanded(1) || ...  % bbox1 is left of bbox2
                bbox1_expanded(1) > bbox2_expanded(3) || ...  % bbox1 is right of bbox2
                bbox1_expanded(4) < bbox2_expanded(2) || ...  % bbox1 is above bbox2
                bbox1_expanded(2) > bbox2_expanded(4));       % bbox1 is below bbox2
end

function imgPath = find_stage1_image(folder, baseName, supportedFormats)
    % Find stage 1 image by base name
    imgPath = '';
    for i = 1:numel(supportedFormats)
        candidate = fullfile(folder, [baseName supportedFormats{i}]);
        if isfile(candidate)
            imgPath = candidate;
            return;
        end
    end
end


function phoneDirs = list_phones(stage2Root)
    if ~isfolder(stage2Root)
        phoneDirs = {};
        return;
    end
    d = dir(stage2Root);
    mask = [d.isdir] & ~ismember({d.name}, {'.', '..'});
    phoneDirs = {d(mask).name};
end

function img = clamp_uint8(img)
    % Clamp image values to [0, 255] and convert to uint8
    % Input must be in [0, 255] range (not normalized [0, 1])
    img = uint8(min(255, max(0, img)));
end

function img = apply_motion_blur(img)
    % Apply slight motion blur with cached PSFs to avoid redundant kernel generation
    persistent psf_cache
    if isempty(psf_cache)
        psf_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end

    len = 4 + randi(4);            % 5-8 px
    ang = rand() * 180;            % degrees
    ang_rounded = round(ang);
    cache_key = sprintf('%d_%d', len, ang_rounded);

    if isKey(psf_cache, cache_key)
        psf = psf_cache(cache_key);
    else
        psf = fspecial('motion', len, ang_rounded);
        psf_cache(cache_key) = psf;
    end

    img = imfilter(img, psf, 'replicate');
end

function I = imread_raw(fname)
% Read image pixels in their recorded layout without applying EXIF orientation
% metadata. Any user-requested rotation is stored in coordinates.txt and applied
% downstream rather than via image metadata.

    I = imread(fname);
end

function ensure_folder(pathStr)
    if ~isempty(pathStr) && ~isfolder(pathStr)
        mkdir(pathStr);
    end
end

function projectRoot = find_project_root(inputFolder)
    % Find project root by searching up directory tree
    currentDir = pwd;
    searchDir = currentDir;
    maxLevels = 5;

    for level = 1:maxLevels
        [parentDir, ~] = fileparts(searchDir);

        if exist(fullfile(searchDir, inputFolder), 'dir')
            projectRoot = searchDir;
            return;
        end

        if strcmp(searchDir, parentDir)
            break;
        end
        searchDir = parentDir;
    end

    warning('augmentDataset:pathResolution', ...
            'Could not find input folder "%s". Using current directory.', inputFolder);
    projectRoot = currentDir;
end

%% -------------------------------------------------------------------------
function [damagedRGB, damagedAlpha, maskEditable, maskProtected, damageCorners] = apply_paper_damage(polygonRGB, polygonAlpha, quadCorners, ellipseList, damageCfg, rngState)
    % Apply paper defect augmentation pipeline to polygon patch
    %
    % INPUTS:
    %   polygonRGB - [H x W x 3] uint8 color content
    %   polygonAlpha - [H x W] logical mask
    %   quadCorners - [4 x 2] vertex coordinates
    %   ellipseList - struct array with fields: center, semiMajor, semiMinor, rotation
    %   damageCfg - cfg.damage struct
    %   rngState - RNG state for reproducibility
    %
    % OUTPUTS:
%   damagedRGB - [H x W x 3] uint8 damaged color content
%   damagedAlpha - [H x W] logical damaged mask
%   maskEditable - [H x W] logical editable mask
%   maskProtected - [H x W] logical protected mask
%   damageCorners - [4 x 2] polygon vertices after warp/shear (crop space)

    % Save/restore RNG only when determinism is requested so random draws do
    % not replay for every polygon.
    savedRngState = [];
    if ~isempty(rngState)
        savedRngState = rng();
        rng(rngState);
    end

    % Initialize outputs
    damagedRGB = polygonRGB;
    damagedAlpha = polygonAlpha;
    damageCorners = double(quadCorners);

    % Check if damage should be applied
    if rand() > damageCfg.probability
        maskEditable = false(size(polygonAlpha));
        maskProtected = false(size(polygonAlpha));
        if ~isempty(savedRngState)
            rng(savedRngState);
        end
        return;
    end

    % Sample damage profile using precomputed cumulative weights
    rVal = rand();
    profileIdx = find(rVal <= damageCfg.profileCumWeights, 1, 'first');
    if isempty(profileIdx)
        profileIdx = numel(damageCfg.profileNames);
    end
    selectedProfile = damageCfg.profileNames{profileIdx};

    % Phase 0: Prepare masks for damage operations
    [H, W, ~] = size(polygonRGB);
    maskPolygon = damagedAlpha;
    maskProtected = false(H, W);
    maskPreCuts = damagedAlpha;
    hasEllipses = ~isempty(ellipseList) && numel(ellipseList) > 0;
    originalArea = sum(maskPolygon(:));

    guardCenters = zeros(max(1, numel(ellipseList)), 2);
    guardCenterCount = 0;
    ellipseProtectionActive = false;

    maxRemovalFraction = [];
    if isfield(damageCfg, 'maxAreaRemovalFraction') && ~isempty(damageCfg.maxAreaRemovalFraction)
        maxRemovalFraction = min(max(damageCfg.maxAreaRemovalFraction, 0), 1);
    end

    if hasEllipses
        % Compute bounding box for all ellipses to minimize meshgrid size
        ellipseBBox = compute_ellipse_bbox_union(ellipseList, W, H);
        
        if ~isempty(ellipseBBox)
            % Create meshgrid only for ROI containing all ellipses
            [xx, yy] = meshgrid(ellipseBBox.xMin:ellipseBBox.xMax, ellipseBBox.yMin:ellipseBBox.yMax);

            keepFraction = 1;
            if ~isempty(maxRemovalFraction)
                keepFraction = max(0, 1 - maxRemovalFraction);
            end
            ellipseAxisScale = sqrt(keepFraction);
            
            for eIdx = 1:numel(ellipseList)
                ellipse = ellipseList(eIdx);

                if ~isfield(ellipse, 'center') || ~isfield(ellipse, 'semiMajor') || ...
                   ~isfield(ellipse, 'semiMinor') || ~isfield(ellipse, 'rotation')
                    continue;
                end

                cx = ellipse.center(1);
                cy = ellipse.center(2);
                a = ellipse.semiMajor;
                b = ellipse.semiMinor;
                theta = ellipse.rotation * pi / 180;

                if a <= 0 || b <= 0
                    continue;
                end

                if cx < 1 || cx > W || cy < 1 || cy > H
                    continue;
                end

                guardCenterCount = guardCenterCount + 1;
                guardCenters(guardCenterCount, :) = [cx, cy];

                % Compute ellipse mask in ROI coordinates
                xx_rot = (xx - cx) * cos(theta) + (yy - cy) * sin(theta);
                yy_rot = -(xx - cx) * sin(theta) + (yy - cy) * cos(theta);

                ellipseCoreMask = false(size(xx_rot));
                if keepFraction > 0 && ellipseAxisScale > 0
                    coreA = a * ellipseAxisScale;
                    coreB = b * ellipseAxisScale;
                    if coreA > 0 && coreB > 0
                        ellipseCoreMask = (xx_rot .^ 2) / (coreA ^ 2) + (yy_rot .^ 2) / (coreB ^ 2) <= 1;
                    end
                end

                % Map ROI mask back to full image coordinates
                if isempty(maskProtected) || ~any(maskProtected(:))
                    maskProtected = false(H, W);
                end
                if any(ellipseCoreMask(:))
                    maskProtected(ellipseBBox.yMin:ellipseBBox.yMax, ellipseBBox.xMin:ellipseBBox.xMax) = ...
                        maskProtected(ellipseBBox.yMin:ellipseBBox.yMax, ellipseBBox.xMin:ellipseBBox.xMax) | ellipseCoreMask;
                    ellipseProtectionActive = true;
                end
            end

            maskProtected = maskProtected & maskPolygon;
        end
    end

    if guardCenterCount > 0 && ellipseProtectionActive
        guardCenters = guardCenters(1:guardCenterCount, :);
        bridgeMask = build_guard_bridge_mask(maskPolygon, guardCenters, damageCorners);
        if ~isempty(bridgeMask)
            maskProtected = maskProtected | bridgeMask;
        end
    end

    minAllowedAreaPixels = [];
    if ~hasEllipses && ~isempty(maxRemovalFraction)
        coreGuardScale = 1 - maxRemovalFraction;
        if coreGuardScale > 0
            coreMask = build_core_guard_mask(damageCorners, H, W, coreGuardScale);
            if ~isempty(coreMask)
                maskProtected = maskProtected | (coreMask & maskPolygon);
            end
        end
        minAllowedAreaPixels = originalArea * (1 - maxRemovalFraction);
    end

    maskEditable = maskPolygon & ~maskProtected;
    hasProtectedRegions = any(maskProtected(:));

    if sum(maskEditable(:)) == 0
        maskEditable = false(size(maskProtected));
        if ~isempty(savedRngState)
            rng(savedRngState);
        end
        return;
    end

    % Phase 1: Base Warp & Shear

    % 1.1 Projective jitter (all profiles)
    imgCorners = [1, 1; W, 1; W, H; 1, H];  % TL, TR, BR, BL
    jitterScale = 0.03;
    rotJitter = 2.0 * pi / 180;

    jitteredCorners = imgCorners;
    for i = 1:4
        % Translation jitter: +/- 3% of image dimensions
        jitteredCorners(i, 1) = imgCorners(i, 1) + (rand() * 2 - 1) * jitterScale * W;
        jitteredCorners(i, 2) = imgCorners(i, 2) + (rand() * 2 - 1) * jitterScale * H;

        % Rotation jitter: +/- 2° around corner
        angle = (rand() * 2 - 1) * rotJitter;
        cx = imgCorners(i, 1);
        cy = imgCorners(i, 2);
        dx = jitteredCorners(i, 1) - cx;
        dy = jitteredCorners(i, 2) - cy;
        jitteredCorners(i, 1) = cx + dx * cos(angle) - dy * sin(angle);
        jitteredCorners(i, 2) = cy + dx * sin(angle) + dy * cos(angle);
    end

    % Clamp to image bounds
    jitteredCorners(:, 1) = max(1, min(W, jitteredCorners(:, 1)));
    jitteredCorners(:, 2) = max(1, min(H, jitteredCorners(:, 2)));

    try
        tform = fitgeotrans(imgCorners, jitteredCorners, 'projective');
        outputView = imref2d([H, W]);

        damagedRGB = imwarp(damagedRGB, tform, 'OutputView', outputView, ...
                           'InterpolationMethod', 'linear', 'FillValues', 0);

        alphaDouble = double(damagedAlpha);
        warpedAlpha = imwarp(alphaDouble, tform, 'OutputView', outputView, ...
                            'InterpolationMethod', 'linear', 'FillValues', 0);
        damagedAlphaCand = warpedAlpha > 0.5;

        if hasProtectedRegions
            maskProtectedDouble = double(maskProtected);
            warpedProtected = imwarp(maskProtectedDouble, tform, 'OutputView', outputView, ...
                                    'InterpolationMethod', 'nearest', 'FillValues', 0);
            maskProtected = warpedProtected > 0.5;
            hasProtectedRegions = any(maskProtected(:));
        end

        if hasProtectedRegions
            damagedAlpha = damagedAlphaCand | maskProtected;
            maskEditable = damagedAlpha & ~maskProtected;
        else
            damagedAlpha = damagedAlphaCand;
            maskEditable = damagedAlphaCand;
        end
        [damageCorners(:,1), damageCorners(:,2)] = transformPointsForward(tform, damageCorners(:,1), damageCorners(:,2));
        maskPreCuts = damagedAlpha;
    catch
    end

    % 1.2 Nonlinear edge bending (minimalWarp and sideCollapse only)
    % Skip for small polygons where warp overhead exceeds benefit
    minDim = min(H, W);
    applyNonlinearWarp = minDim > 200;
    
    if applyNonlinearWarp && (strcmp(selectedProfile, 'minimalWarp') || strcmp(selectedProfile, 'sideCollapse'))
        % Control points at edge midpoints
        controlPoints = [
            W/2, 1;      % Top edge midpoint
            W, H/2;      % Right edge midpoint
            W/2, H;      % Bottom edge midpoint
            1, H/2       % Left edge midpoint
        ];

        % Add corners as fixed anchor points
        movingPts = [controlPoints; [1, 1; W, 1; W, H; 1, H]];
        fixedPts = movingPts;

        % Offset edge midpoints perpendicular to edge
        ampRange = damageCfg.edgeWaveAmplitudeRange * minDim;
        for i = 1:4
            offset = ampRange(1) + rand() * (ampRange(2) - ampRange(1));
            offset = offset * (2 * (rand() > 0.5) - 1);  % Random sign

            if i == 1  % Top edge
                fixedPts(i, 2) = fixedPts(i, 2) + offset;
            elseif i == 2  % Right edge
                fixedPts(i, 1) = fixedPts(i, 1) + offset;
            elseif i == 3  % Bottom edge
                fixedPts(i, 2) = fixedPts(i, 2) + offset;
            else  % Left edge
                fixedPts(i, 1) = fixedPts(i, 1) + offset;
            end
        end

        % Clamp to valid region
        fixedPts(:, 1) = max(1, min(W, fixedPts(:, 1)));
        fixedPts(:, 2) = max(1, min(H, fixedPts(:, 2)));

        try
            tform = fitgeotrans(movingPts, fixedPts, 'lwm', 8);
            outputView = imref2d([H, W]);

            damagedRGB = imwarp(damagedRGB, tform, 'OutputView', outputView, ...
                               'InterpolationMethod', 'linear', 'FillValues', 0);

            alphaDouble = double(damagedAlpha);
            warpedAlpha = imwarp(alphaDouble, tform, 'OutputView', outputView, ...
                                'InterpolationMethod', 'linear', 'FillValues', 0);
            damagedAlphaCand = warpedAlpha > 0.5;

            if hasProtectedRegions
                maskProtectedDouble = double(maskProtected);
                warpedProtected = imwarp(maskProtectedDouble, tform, 'OutputView', outputView, ...
                                        'InterpolationMethod', 'nearest', 'FillValues', 0);
                maskProtected = warpedProtected > 0.5;
                hasProtectedRegions = any(maskProtected(:));
            end

            damagedAlpha = damagedAlphaCand;
            if hasProtectedRegions
                damagedAlpha = damagedAlpha | maskProtected;
                maskEditable = damagedAlpha & ~maskProtected;
            else
                maskEditable = damagedAlpha;
            end
            [damageCorners(:,1), damageCorners(:,2)] = transformPointsForward(tform, damageCorners(:,1), damageCorners(:,2));
            maskPreCuts = damagedAlpha;
        catch
        end
    end

    % Phase 2: Structural Cuts (Material Removal)
    if strcmp(selectedProfile, 'cornerChew') || strcmp(selectedProfile, 'sideCollapse')
        if strcmp(selectedProfile, 'cornerChew')
            operationPool = {'cornerClip', 'cornerTear', 'sideBite', 'taperedSide'};
            operationWeights = [0.35, 0.25, 0.25, 0.15];
        else
            operationPool = {'cornerClip', 'cornerTear', 'sideBite', 'taperedSide'};
            operationWeights = [0.10, 0.10, 0.50, 0.30];
        end

        numOps = randi([1, damageCfg.maxOperations]);

        prevOp = '';
        for opIdx = 1:numOps
            weights = operationWeights;
            if ~isempty(prevOp)
                weights(strcmp(operationPool, prevOp)) = 0;
            end
            if sum(weights) == 0
                weights = operationWeights;
            end
            weights = weights / sum(weights);
            cumWeights = cumsum(weights);
            opType = operationPool{find(rand() <= cumWeights, 1, 'first')};

            maskBeforeOp = maskEditable;
            switch opType
                case 'cornerClip'
                    maskEditable = apply_corner_clip(maskEditable, damageCorners, damageCfg);
                case 'cornerTear'
                    maskEditable = apply_corner_tear(maskEditable, damageCorners, damageCfg);
                case 'sideBite'
                    maskEditable = apply_side_bite(maskEditable, damageCorners, damageCfg, maskProtected, hasProtectedRegions);
                case 'taperedSide'
                    maskEditable = apply_tapered_side(maskEditable, damageCorners, damageCfg);
            end

            currentArea = sum(maskEditable(:)) + sum(maskProtected(:));
            if ~isempty(minAllowedAreaPixels) && currentArea < minAllowedAreaPixels
                maskEditable = maskBeforeOp;
                break;
            end

            prevOp = opType;
        end

        damagedAlpha = maskEditable;
        if hasProtectedRegions
            damagedAlpha = damagedAlpha | maskProtected;
        end

        for c = 1:3
            channel = damagedRGB(:,:,c);
            channel(~damagedAlpha) = 0;
            damagedRGB(:,:,c) = channel;
        end
    end

    maskEditablePreWear = maskEditable;
    damagedAlphaPreWear = damagedAlpha;
    damagedRGBPreWear = damagedRGB;

    % Phase 3: Edge Wear & Thickness Cues
    if strcmp(selectedProfile, 'cornerChew') || strcmp(selectedProfile, 'sideCollapse')
        maskEditable = apply_edge_wave_noise(maskEditable, damageCfg);

        if rand() < 0.5
            maskEditable = apply_edge_fray(maskEditable, maskProtected);
        end

        damagedAlpha = maskEditable;
        if hasProtectedRegions
            damagedAlpha = damagedAlpha | maskProtected;
        end

        damagedRGB = apply_thickness_shadows(damagedRGB, damagedAlpha, maskPreCuts);

        for c = 1:3
            channel = damagedRGB(:,:,c);
            channel(~damagedAlpha) = 0;
            damagedRGB(:,:,c) = channel;
        end
    elseif strcmp(selectedProfile, 'minimalWarp')
        damagedAlpha = maskEditable;
        if hasProtectedRegions
            damagedAlpha = damagedAlpha | maskProtected;
        end

        for c = 1:3
            channel = damagedRGB(:,:,c);
            channel(~damagedAlpha) = 0;
            damagedRGB(:,:,c) = channel;
        end
    end

    if ~isempty(minAllowedAreaPixels)
        finalArea = sum(maskEditable(:)) + sum(maskProtected(:));
        if finalArea < minAllowedAreaPixels
            maskEditable = maskEditablePreWear;
            damagedAlpha = damagedAlphaPreWear;
            damagedRGB = damagedRGBPreWear;
        end
    end

    if ~isempty(savedRngState)
        rng(savedRngState);
    end
end

function coreMask = build_core_guard_mask(quadCorners, height, width, scaleFactor)
    % Construct a scaled version of the polygon that remains untouched
    if isempty(quadCorners) || size(quadCorners, 2) ~= 2 || scaleFactor <= 0
        coreMask = [];
        return;
    end

    scaleFactor = min(scaleFactor, 1);

    centroid = mean(quadCorners, 1, 'omitnan');
    if any(isnan(centroid))
        coreMask = [];
        return;
    end

    scaledCorners = (quadCorners - centroid) * scaleFactor + centroid;
    coreMask = poly2mask(scaledCorners(:, 1), scaledCorners(:, 2), height, width);
end

function ellipseBBox = compute_ellipse_bbox_union(ellipseList, width, height)
    % Compute tight bounding box containing all ellipses for ROI-based processing
    %
    % INPUTS:
    %   ellipseList - struct array with fields: center, semiMajor, semiMinor, rotation
    %   width, height - image dimensions
    %
    % OUTPUT:
    %   ellipseBBox - struct with xMin, xMax, yMin, yMax (empty if no valid ellipses)
    
    if isempty(ellipseList)
        ellipseBBox = [];
        return;
    end
    
    % Collect all ellipse bounding boxes
    numEllipses = numel(ellipseList);
    xMins = zeros(numEllipses, 1);
    xMaxs = zeros(numEllipses, 1);
    yMins = zeros(numEllipses, 1);
    yMaxs = zeros(numEllipses, 1);
    validCount = 0;
    
    for i = 1:numEllipses
        ellipse = ellipseList(i);
        
        if ~isfield(ellipse, 'center') || ~isfield(ellipse, 'semiMajor') || ...
           ~isfield(ellipse, 'semiMinor') || ~isfield(ellipse, 'rotation')
            continue;
        end
        
        cx = ellipse.center(1);
        cy = ellipse.center(2);
        
        if cx < 1 || cx > width || cy < 1 || cy > height
            continue;
        end
        
        % Compute axis-aligned bounding box for rotated ellipse
        theta = ellipse.rotation * pi / 180;
        a = ellipse.semiMajor;
        b = ellipse.semiMinor;
        if a <= 0 || b <= 0
            continue;
        end
        dx = sqrt((a * cos(theta))^2 + (b * sin(theta))^2);
        dy = sqrt((a * sin(theta))^2 + (b * cos(theta))^2);
        
        validCount = validCount + 1;
        xMins(validCount) = cx - dx;
        xMaxs(validCount) = cx + dx;
        yMins(validCount) = cy - dy;
        yMaxs(validCount) = cy + dy;
    end
    
    if validCount == 0
        ellipseBBox = [];
        return;
    end
    
    xMins = xMins(1:validCount);
    xMaxs = xMaxs(1:validCount);
    yMins = yMins(1:validCount);
    yMaxs = yMaxs(1:validCount);
    
    ellipseBBox = struct();
    ellipseBBox.xMin = max(1, floor(min(xMins)));
    ellipseBBox.xMax = min(width, ceil(max(xMaxs)));
    ellipseBBox.yMin = max(1, floor(min(yMins)));
    ellipseBBox.yMax = min(height, ceil(max(yMaxs)));
end

function bridgeMask = build_guard_bridge_mask(maskPolygon, guardCenters, quadCorners)
    % Add protected connectors so guarded ellipses remain attached to the strip

    bridgeMask = false(size(maskPolygon));
    if isempty(guardCenters)
        return;
    end

    centroid = mean(quadCorners, 1, 'omitnan');
    if any(~isfinite(centroid))
        return;
    end

    [height, width] = size(maskPolygon);
    centroid(1) = max(1, min(width, centroid(1)));
    centroid(2) = max(1, min(height, centroid(2)));

    valid = all(isfinite(guardCenters), 2);
    guardCenters = guardCenters(valid, :);
    if isempty(guardCenters)
        return;
    end

    for idx = 1:size(guardCenters, 1)
        startPt = guardCenters(idx, :);
        delta = abs(startPt - centroid);
        numSteps = max(delta);
        numSteps = max(ceil(numSteps), 1);
        xLine = round(linspace(startPt(1), centroid(1), numSteps));
        yLine = round(linspace(startPt(2), centroid(2), numSteps));
        keep = xLine >= 1 & xLine <= width & yLine >= 1 & yLine <= height;
        xLine = xLine(keep);
        yLine = yLine(keep);
        ind = sub2ind([height, width], yLine, xLine);
        bridgeMask(ind) = true;
    end

    se = strel('disk', 2, 0);
    bridgeMask = imdilate(bridgeMask, se) & maskPolygon;
end

%% -------------------------------------------------------------------------
%% Phase 2 Helper Functions (Structural Cuts)
%% -------------------------------------------------------------------------

function edgeInfo = get_polygon_edges(quadCorners)
    % Extract edge geometry from polygon corners for edge-aware damage operations
    %
    % INPUTS:
    %   quadCorners - [4 x 2] vertex coordinates in clockwise order [TL, TR, BR, BL]
    %
    % OUTPUTS:
    %   edgeInfo - struct with fields:
    %     .corners - [4 x 2] original corners
    %     .edges - [4 x 2 x 2] array where edges(i,:,:) = [startPt; endPt]
    %     .midpoints - [4 x 2] midpoint of each edge
    %     .normals - [4 x 2] inward-pointing unit normals
    %     .lengths - [4 x 1] edge lengths
    %     .directions - [4 x 2] unit direction vectors along edges

    edgeInfo = struct();
    edgeInfo.corners = quadCorners;

    % Compute edges (clockwise order: top, right, bottom, left)
    edgeInfo.edges = zeros(4, 2, 2);
    for i = 1:4
        nextIdx = mod(i, 4) + 1;
        edgeInfo.edges(i, :, :) = [quadCorners(i, :); quadCorners(nextIdx, :)];
    end

    % Compute edge midpoints, lengths, directions
    edgeInfo.midpoints = zeros(4, 2);
    edgeInfo.lengths = zeros(4, 1);
    edgeInfo.directions = zeros(4, 2);
    edgeInfo.normals = zeros(4, 2);

    for i = 1:4
        startPt = squeeze(edgeInfo.edges(i, 1, :))';
        endPt = squeeze(edgeInfo.edges(i, 2, :))';

        % Midpoint
        edgeInfo.midpoints(i, :) = (startPt + endPt) / 2;

        % Direction vector and length
        edgeVec = endPt - startPt;
        edgeInfo.lengths(i) = norm(edgeVec);
        edgeInfo.directions(i, :) = edgeVec / edgeInfo.lengths(i);

        % Inward normal (rotate direction 90 deg clockwise: [x,y] -> [y,-x])
        edgeInfo.normals(i, :) = [edgeVec(2), -edgeVec(1)] / edgeInfo.lengths(i);
    end

    % Verify normals point inward (toward polygon centroid)
    centroid = mean(quadCorners, 1);
    for i = 1:4
        toCenter = centroid - edgeInfo.midpoints(i, :);
        if dot(edgeInfo.normals(i, :), toCenter) < 0
            % Normal points outward, flip it
            edgeInfo.normals(i, :) = -edgeInfo.normals(i, :);
        end
    end
end

%% -------------------------------------------------------------------------
function mask = apply_corner_clip(mask, quadCorners, damageCfg)
    % Remove triangular section from 1-3 corners with straight cuts
    %
    % ALGORITHM:
    %   1. Sample number of corners (1-3, weighted 30/50/20)
    %   2. For adjacent corners (70% of 2-corner clips), create trapezoid look
    %   3. Compute clip depth from reference dimension (mean edge length)
    %   4. Create triangle vertices using edge directions from polygon geometry
    %   5. Rasterize and subtract from mask
    %
    % INPUTS:
    %   mask - [H×W] logical editable mask
    %   quadCorners - [4×2] polygon vertices (TL, TR, BR, BL clockwise)
    %   damageCfg.cornerClipRange - [2×1] depth fraction range [0.06, 0.22]
    %
    % OUTPUT:
    %   mask - [H×W] logical mask with clips subtracted
    %
    % Uses actual polygon corners for geometry-aware clipping

    [H, W] = size(mask);
    edgeInfo = get_polygon_edges(quadCorners);

    % Reference dimension for depth calculation (use mean edge length)
    refDim = mean(edgeInfo.lengths);

    % Sample number of corners to clip (1-3, weighted toward 2)
    % Sample corner count without Statistics Toolbox
    cornerWeights = [0.3, 0.5, 0.2];
    rVal = rand();
    if rVal < cornerWeights(1)
        numCorners = 1;
    elseif rVal < cornerWeights(1) + cornerWeights(2)
        numCorners = 2;
    else
        numCorners = 3;
    end

    % Sample which corners (prefer adjacent for trapezoid look)
    if numCorners == 1
        selectedCorners = randi([1, 4], 1);
    elseif numCorners == 2
        % 70% chance of adjacent corners
        if rand() < 0.7
            startCorner = randi([1, 4]);
            selectedCorners = [startCorner, mod(startCorner, 4) + 1];
        else
            perm = randperm(4);
            selectedCorners = perm(1:2);
        end
    else  % 3 corners
        perm = randperm(4);
        selectedCorners = perm(1:3);
    end

    % Apply clips using actual corner geometry
    for i = 1:numel(selectedCorners)
        cornerIdx = selectedCorners(i);
        cornerPt = edgeInfo.corners(cornerIdx, :);

        % Sample clip depth (fraction of reference dimension)
        depthFrac = damageCfg.cornerClipRange(1) + ...
                    rand() * (damageCfg.cornerClipRange(2) - damageCfg.cornerClipRange(1));
        depth = depthFrac * refDim;

        % Get edge directions for this corner
        % Previous edge: from previous corner to this corner
        prevCornerIdx = mod(cornerIdx - 2, 4) + 1;
        prevDir = edgeInfo.directions(prevCornerIdx, :);  % Direction toward this corner

        % Next edge: from this corner to next corner
        nextDir = edgeInfo.directions(cornerIdx, :);  % Direction away from this corner

        % Create triangle vertices for clip
        pt1 = cornerPt - prevDir * depth;  % Point along incoming edge
        pt2 = cornerPt;                    % Corner itself
        pt3 = cornerPt + nextDir * depth;  % Point along outgoing edge

        % Rasterize triangle and subtract from mask
        triMask = poly2mask([pt1(1), pt2(1), pt3(1)], ...
                           [pt1(2), pt2(2), pt3(2)], H, W);
        mask = mask & ~triMask;
    end
end

%% -------------------------------------------------------------------------
function mask = apply_corner_tear(mask, quadCorners, damageCfg)
    % Remove irregular jagged section from one corner
    %
    % ALGORITHM:
    %   1. Select one corner randomly
    %   2. Generate 4-6 vertices along adjacent edges with perpendicular jitter
    %   3. Rasterize tear polygon using poly2mask
    %   4. Optionally apply morphological dilation for roughness (50% chance)
    %
    % INPUTS:
    %   mask - [H×W] logical editable mask
    %   quadCorners - [4×2] polygon vertices (TL, TR, BR, BL clockwise)
    %   damageCfg.cornerClipRange - [2×1] depth fraction range [0.06, 0.22]
    %
    % OUTPUT:
    %   mask - [H×W] logical mask with tear subtracted
    %
    % Uses actual polygon corners for geometry-aware tearing

    [H, W] = size(mask);
    edgeInfo = get_polygon_edges(quadCorners);

    % Reference dimension for depth calculation
    refDim = mean(edgeInfo.lengths);

    % Select one corner randomly
    cornerIdx = randi([1, 4]);
    cornerPt = edgeInfo.corners(cornerIdx, :);

    % Sample tear depth (similar range to clips)
    depthFrac = damageCfg.cornerClipRange(1) + ...
                rand() * (damageCfg.cornerClipRange(2) - damageCfg.cornerClipRange(1));
    depth = depthFrac * refDim;

    % Generate 4-6 vertices along edges with jitter
    numVerts = randi([4, 6]);
    tearVerts = zeros(numVerts, 2);

    % Get edge directions for this corner
    prevCornerIdx = mod(cornerIdx - 2, 4) + 1;
    prevDir = edgeInfo.directions(prevCornerIdx, :);  % Incoming edge direction
    nextDir = edgeInfo.directions(cornerIdx, :);      % Outgoing edge direction

    % Place vertices along edges with random jitter
    for i = 1:numVerts
        t = (i - 1) / (numVerts - 1);  % 0 to 1
        if t < 0.5
            % Along incoming edge
            dir = -prevDir;  % Direction away from corner
            offset = t * 2 * depth;
        else
            % Along outgoing edge
            dir = nextDir;   % Direction away from corner
            offset = (t - 0.5) * 2 * depth;
        end

        % Add perpendicular jitter
        perpDir = [-dir(2), dir(1)];  % 90° rotation
        jitter = (rand() - 0.5) * depth * 0.3;

        tearVerts(i, :) = cornerPt + dir * offset + perpDir * jitter;
    end

    % Rasterize tear polygon
    tearMask = poly2mask(tearVerts(:, 1), tearVerts(:, 2), H, W);

    % Add roughness via morphological operations
    if rand() < 0.5
        se = strel('disk', 2);
        tearMask = imdilate(tearMask, se);
    end

    % Subtract from mask
    mask = mask & ~tearMask;
end

%% -------------------------------------------------------------------------
function mask = apply_side_bite(mask, quadCorners, damageCfg, maskProtected, hasProtectedRegions)
    % Remove concave circular bite from one edge
    %
    % ALGORITHM:
    %   1. Select random edge and position along edge (30-70% of length)
    %   2. Compute bite radius from edge length (8-28% fraction)
    %   3. Place circle center along inward normal direction
    %   4. Rasterize circle and subtract from mask
    %   5. Optionally add second bite (40% chance) with collision check
    %
    % INPUTS:
    %   mask - [H×W] logical editable mask
    %   quadCorners - [4×2] polygon vertices (TL, TR, BR, BL clockwise)
    %   damageCfg.sideBiteRange - [2×1] depth fraction range [0.08, 0.28]
    %   maskProtected - [H×W] logical protected region mask
    %   hasProtectedRegions - logical flag
    %
    % OUTPUT:
    %   mask - [H×W] logical mask with bite(s) subtracted

    [H, W] = size(mask);
    edgeInfo = get_polygon_edges(quadCorners);

    edgeIdx = randi([1, 4]);
    edgePos = 0.3 + rand() * 0.4;
    depthFrac = damageCfg.sideBiteRange(1) + ...
                rand() * (damageCfg.sideBiteRange(2) - damageCfg.sideBiteRange(1));

    edgeLength = edgeInfo.lengths(edgeIdx);
    edgeMidpoint = edgeInfo.midpoints(edgeIdx, :);
    edgeDirection = edgeInfo.directions(edgeIdx, :);
    edgeNormal = edgeInfo.normals(edgeIdx, :);

    tParam = (edgePos - 0.5) * 2;
    biteCenterOnEdge = edgeMidpoint + tParam * edgeDirection * (edgeLength / 2);
    radius = depthFrac * edgeLength;
    circleCenter = biteCenterOnEdge + edgeNormal * radius;

    [xx, yy] = meshgrid(1:W, 1:H);
    biteMask = ((xx - circleCenter(1)).^2 + (yy - circleCenter(2)).^2) <= radius^2;

    mask = mask & ~biteMask;

    if rand() < 0.4
        offsetSign = (rand() < 0.5) * 2 - 1;
        edgePos2 = edgePos + offsetSign * (0.2 + rand() * 0.2);
        edgePos2 = max(0.1, min(0.9, edgePos2));

        tParam2 = (edgePos2 - 0.5) * 2;
        biteCenterOnEdge2 = edgeMidpoint + tParam2 * edgeDirection * (edgeLength / 2);
        
        depthFrac2 = depthFrac * (0.5 + rand() * 0.2);
        radius2 = depthFrac2 * edgeLength;
        circleCenter2 = biteCenterOnEdge2 + edgeNormal * radius2;

        centerDist = norm(circleCenter2 - circleCenter);
        minSpacing = (radius + radius2) * 0.7;
        
        if centerDist >= minSpacing
            biteMask2 = ((xx - circleCenter2(1)).^2 + (yy - circleCenter2(2)).^2) <= radius2^2;
            
            if hasProtectedRegions && any(biteMask2(:) & maskProtected(:))
                return;
            end
            
            mask = mask & ~biteMask2;
        end
    end
end

%% -------------------------------------------------------------------------
function mask = apply_tapered_side(mask, quadCorners, damageCfg)
    % Create diagonal cut making one side shorter
    %
    % ALGORITHM:
    %   1. Select random edge
    %   2. Compute taper strength (10-30% of edge length)
    %   3. Create gradient mask: removal increases linearly from edge start (0%) to end (max%)
    %   4. Use signed distance from edge line to create smooth taper
    %
    % INPUTS:
    %   mask - [H×W] logical editable mask
    %   quadCorners - [4×2] polygon vertices (TL, TR, BR, BL clockwise)
    %   damageCfg.taperStrengthRange - [2×1] taper fraction range [0.10, 0.30]
    %
    % OUTPUT:
    %   mask - [H×W] logical mask with tapered edge
    %
    % Uses actual polygon edges for geometry-aware tapering

    [H, W] = size(mask);
    edgeInfo = get_polygon_edges(quadCorners);

    % Choose edge randomly
    edgeIdx = randi([1, 4]);

    % Sample taper strength (fraction of edge length to remove at max)
    taperStrength = damageCfg.taperStrengthRange(1) + ...
                    rand() * (damageCfg.taperStrengthRange(2) - damageCfg.taperStrengthRange(1));

    % Get edge geometry
    edgeStart = squeeze(edgeInfo.edges(edgeIdx, 1, :))';  % Start point
    edgeEnd = squeeze(edgeInfo.edges(edgeIdx, 2, :))';    % End point
    edgeNormal = edgeInfo.normals(edgeIdx, :);            % Inward normal
    edgeLength = edgeInfo.lengths(edgeIdx);

    % Create coordinate grid
    [xx, yy] = meshgrid(1:W, 1:H);

    % Compute signed distance from each pixel to the edge line
    % Edge parameterized as: P(t) = edgeStart + t * (edgeEnd - edgeStart), t ∈ [0,1]
    edgeVec = edgeEnd - edgeStart;

    % For each pixel, find projection onto edge line
    % Project vector from edgeStart to pixel onto edge direction
    px = xx(:) - edgeStart(1);
    py = yy(:) - edgeStart(2);
    t = (px * edgeVec(1) + py * edgeVec(2)) / (edgeVec(1)^2 + edgeVec(2)^2);
    t = reshape(t, H, W);
    t = max(0, min(1, t));  % Clamp to [0,1]

    % Compute taper threshold that varies linearly along edge
    % At edge start (t=0): no removal, at edge end (t=1): maximum removal
    taperDepth = taperStrength * edgeLength * t;

    % Compute signed distance to edge (positive = inward, negative = outward)
    % Distance = dot product of (pixel - pointOnEdge) with inward normal
    pointOnEdgeX = edgeStart(1) + t * edgeVec(1);
    pointOnEdgeY = edgeStart(2) + t * edgeVec(2);
    signedDist = (xx - pointOnEdgeX) * edgeNormal(1) + (yy - pointOnEdgeY) * edgeNormal(2);

    % Keep pixels where signed distance > -taperDepth
    % (pixels removed if they're beyond the tapered boundary)
    gradientMask = signedDist > -taperDepth;

    % Apply taper
    mask = mask & gradientMask;
end

%% -------------------------------------------------------------------------
%% Phase 3 Helper Functions (Edge Wear & Thickness)
%% -------------------------------------------------------------------------

function mask = apply_edge_wave_noise(mask, damageCfg)
    % Add sinusoidal wave pattern to edges to prevent straight lines
    %
    % ALGORITHM:
    %   1. Compute approximate signed distance transform
    %   2. Generate sinusoidal wave pattern + Gaussian noise
    %   3. Modulate distance field and rethreshold to create wavy edges
    %
    % PERFORMANCE: Skips processing for small masks (<150px) where wave effect is negligible

    [H, W] = size(mask);
    minDim = min(H, W);
    originalMask = mask;
    
    % Skip for small polygons where wave effect is negligible
    if minDim < 150
        return;
    end

    % Extract perimeter
    perim = bwperim(mask);
    if ~any(perim(:))
        return;
    end

    % Compute approximate signed distance (faster than full bwdist for small distances)
    maxWaveDist = damageCfg.edgeWaveAmplitudeRange(2) * minDim * 2;
    signedDist = fast_signed_distance(mask, ceil(maxWaveDist));

    % Sample wave frequency from config
    freq = damageCfg.edgeWaveFrequencyRange(1) + ...
           rand() * (damageCfg.edgeWaveFrequencyRange(2) - damageCfg.edgeWaveFrequencyRange(1));

    % Sample wave amplitude from config
    amp = damageCfg.edgeWaveAmplitudeRange(1) + ...
          rand() * (damageCfg.edgeWaveAmplitudeRange(2) - damageCfg.edgeWaveAmplitudeRange(1));
    amp = amp * minDim;

    % Create coordinate grid for wave pattern
    [xx, yy] = meshgrid(1:W, 1:H);

    % Compute angle from image center for sinusoidal modulation
    cx = W / 2;
    cy = H / 2;
    theta = atan2(yy - cy, xx - cx);

    % Sinusoidal wave + low-frequency Gaussian noise
    wavePattern = amp * sin(freq * theta);

    % Add low-pass filtered Gaussian noise
    noiseAmp = amp * 0.3;
    noise = randn(H, W) * noiseAmp;
    noise = imgaussfilt(noise, 3);

    % Combined modulation
    modulation = wavePattern + noise;

    % Apply wave modulation to signed distance
    modulatedDist = signedDist + modulation;

    % New mask: pixels with non-negative signed distance remain, but never add material
    mask = (modulatedDist >= 0) & originalMask;
end

function signedDist = fast_signed_distance(mask, maxDist)
    % Fast approximate signed distance transform for small distances
    %
    % ALGORITHM:
    %   Uses morphological erosion/dilation instead of Euclidean distance.
    %   Much faster for maxDist < 20 pixels (typical for edge wave noise).
    %
    % INPUTS:
    %   mask - [H×W] logical binary mask
    %   maxDist - maximum distance to compute (pixels)
    %
    % OUTPUT:
    %   signedDist - [H×W] approximate signed distance (positive inside, negative outside)
    
    if maxDist > 20
        % Fall back to accurate method for large distances
        D_out = bwdist(~mask);
        D_in = bwdist(mask);
        signedDist = D_out - D_in;
        return;
    end
    
    % Approximate using iterative erosion/dilation
    [H, W] = size(mask);
    signedDist = zeros(H, W);
    
    % Compute distance inside mask (positive)
    se = strel('disk', 1, 0);
    tempMask = mask;
    for d = 1:maxDist
        tempMask = imerode(tempMask, se);
        if ~any(tempMask(:))
            break;
        end
        signedDist = signedDist + double(tempMask);
    end
    
    % Compute distance outside mask (negative)
    tempMask = ~mask;
    for d = 1:maxDist
        tempMask = imerode(tempMask, se);
        if ~any(tempMask(:))
            break;
        end
        signedDist = signedDist - double(tempMask);
    end
end

%% -------------------------------------------------------------------------
function mask = apply_edge_fray(mask, maskProtected)
    % Add micro-chipping along edges simulating paper fiber fraying
    %
    % ALGORITHM:
    %   1. Extract perimeter pixels
    %   2. Select 3-N fray points with minimum spacing (10px)
    %   3. Create small circular blobs (radius 1-3px) at fray points
    %   4. Subtract from mask while respecting protected regions
    %
    % INPUTS:
    %   mask - [H×W] logical editable mask
    %   maskProtected - [H×W] logical protected region mask
    %
    % OUTPUT:
    %   mask - [H×W] logical mask with fray damage subtracted

    [H, W] = size(mask);

    perim = bwperim(mask);
    [py, px] = find(perim);

    if isempty(px)
        return;
    end

    numPerimPixels = numel(px);
    numFrayPoints = max(3, round(numPerimPixels / 100 * rand() * 3));

    minSpacing = 10;
    selectedPoints = zeros(numFrayPoints, 2);
    pointCount = 0;
    attempts = 0;
    maxAttempts = numFrayPoints * 10;

    while pointCount < numFrayPoints && attempts < maxAttempts
        idx = randi(numPerimPixels);
        candidate = [px(idx), py(idx)];

        if pointCount == 0
            pointCount = 1;
            selectedPoints(pointCount, :) = candidate;
        else
            dists = sqrt(sum((selectedPoints(1:pointCount, :) - candidate).^2, 2));
            if all(dists >= minSpacing)
                pointCount = pointCount + 1;
                selectedPoints(pointCount, :) = candidate;
            end
        end
        attempts = attempts + 1;
    end
    selectedPoints = selectedPoints(1:pointCount, :);

    frayMask = false(H, W);
    [xx, yy] = meshgrid(1:W, 1:H);
    for i = 1:size(selectedPoints, 1)
        radius = randi([1, 3]);

        distFromPoint = sqrt((xx - selectedPoints(i, 1)).^2 + (yy - selectedPoints(i, 2)).^2);
        blob = distFromPoint <= radius;

        blob = blob & ~maskProtected;
        frayMask = frayMask | blob;
    end

    mask = mask & ~frayMask;
end

%% -------------------------------------------------------------------------
function img = apply_thickness_shadows(img, damagedMask, originalMask)
    % Darken edges where material was removed to create depth cues
    %
    % ALGORITHM:
    %   1. Compute removed region (original minus damaged)
    %   2. Calculate distance from removed boundary
    %   3. Create shadow ramp (15% darkening at boundary, fading over 8px)
    %   4. Apply multiplicative darkening to all channels
    %
    % INPUTS:
    %   img - [H×W×3] uint8 RGB image
    %   damagedMask - [H×W] logical mask after damage
    %   originalMask - [H×W] logical mask before damage
    %
    % OUTPUT:
    %   img - [H×W×3] uint8 RGB image with thickness shadows

    % Compute removed region (original minus damaged)
    removedRegion = originalMask & ~damagedMask;

    if ~any(removedRegion(:))
        return;  % No material removed, skip shadowing
    end

    % Compute distance from removed region
    distFromRemoved = bwdist(removedRegion);

    % Create shadow ramp (darken pixels near removed regions)
    shadowWidth = 8;  % pixels
    shadowRamp = max(0, 1 - distFromRemoved / shadowWidth);
    shadowRamp = shadowRamp .* double(damagedMask);  % Only shadow remaining material

    % Darken by 15% at removed boundary, fading over shadowWidth pixels
    shadowStrength = 0.15;
    darkening = 1 - shadowStrength * shadowRamp;

    % Apply to all channels
    for c = 1:3
        channel = double(img(:,:,c));
        channel = channel .* darkening;
        img(:,:,c) = uint8(channel);
    end
end

