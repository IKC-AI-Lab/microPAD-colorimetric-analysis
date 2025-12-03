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

    %% Load utility modules
    geomTform = geometry_transform();
    imageIO = image_io();
    pathUtils = path_utils();
    augSynth = augmentation_synthesis();
    coordIO = coordinate_io();  % Authoritative source for coordinate I/O

    % Build configuration
    cfg = struct();
    cfg.geomTform = geomTform;
    cfg.pathUtils = pathUtils;
    cfg.imageIO = imageIO;
    cfg.augSynth = augSynth;
    cfg.coordIO = coordIO;
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
    projectRoot = pathUtils.findProjectRoot(DEFAULT_INPUT_STAGE1);
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

    phoneList = pathUtils.listSubfolders(cfg.paths.stage1Input);
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
    cfg.pathUtils.ensureFolder(stage1PhoneOut);
    cfg.pathUtils.ensureFolder(stage2PhoneOut);
    % Only create stage3 output folder if we have ellipse data
    if hasEllipses
        cfg.pathUtils.ensureFolder(stage3PhoneOut);
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

        % Load image once (using loadImageRaw to handle EXIF orientation)
        stage1Img = cfg.imageIO.loadImageRaw(imgPath);

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
        cfg.pathUtils.ensureFolder(concDir);
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
                cfg.pathUtils.ensureFolder(ellipseDir);
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
        viewParams = cfg.geomTform.homog.sampleViewpoint(extremeCamera);
    else
        % Normal camera viewpoint
        viewParams = cfg.geomTform.homog.sampleViewpoint(cfg.camera);
    end
    tformPersp = cfg.geomTform.homog.computeHomography(size(stage1Img), viewParams, cfg.camera);
    rotAngle = cfg.geomTform.homog.randRange(cfg.rotationRange);
    tformRot = cfg.geomTform.homog.centeredRotationTform(size(stage1Img), rotAngle);

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
        augVertices = cfg.geomTform.homog.transformPolygon(origVertices, tformPersp);
        augVertices = cfg.geomTform.homog.transformPolygon(augVertices, tformRot);

        % Apply independent rotation per polygon if enabled
        if cfg.independentRotation
            independentRotAngle = cfg.geomTform.homog.randRange(cfg.rotationRange);
            tformIndepRot = cfg.geomTform.homog.centeredRotationTform(size(stage1Img), independentRotAngle);
        else
            independentRotAngle = 0;
            tformIndepRot = affine2d(eye(3));
        end
        augVertices = cfg.geomTform.homog.transformPolygon(augVertices, tformIndepRot);

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
        augPolygonImg = cfg.geomTform.homog.transformPolygon_content(polygonContent, ...
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
            ellipseAugList = cfg.geomTform.homog.transformRegionEllipses( ...
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
        cfg.pathUtils.ensureFolder(concDirOut);

        polygonFileName = sprintf('%s_%s%d%s', sceneName, cfg.concPrefix, concentration, imgExt);
        polygonOutPath = fullfile(concDirOut, polygonFileName);
        imwrite(augPolygonImg, polygonOutPath);

        % Record stage 2 coordinates (polygon in scene)
        % Compute total applied rotation and update saved rotation field
        %
        % ROTATION SEMANTICS:
        %   The rotation field in Stage 2 coordinates is a UI-only alignment hint
        %   (see cut_micropads.m documentation). When augmenting:
        %   - originalRotation: The source image's UI hint (from 2_micropads)
        %   - totalAppliedRotation: Sum of scene rotation + any independent polygon rotation
        %   - augmentedRotation: Adjusted hint for the augmented image
        %
        %   Formula: augmentedRotation = originalRotation - totalAppliedRotation
        %   This ensures that if a user loads the augmented image in the UI and applies
        %   the saved rotation, the visual alignment is preserved.
        totalAppliedRotation = rotAngle + region.independentRotAngle;
        augmentedRotation = cfg.geomTform.normalizeAngle(region.originalRotation - totalAppliedRotation);

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
                cfg.pathUtils.ensureFolder(ellipseConcDir);
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
    %
    % Delegates to mask_utils.cropWithPolygonMask for consistent masking
    % across all scripts. See mask_utils.m for authoritative implementation.
    %
    % OUTPUTS:
    %   content - Cropped and masked image (bounding box size)
    %   bbox    - [minX, minY, width, height] of cropped region
    %
    % See also: mask_utils.cropWithPolygonMask

    persistent masks
    if isempty(masks)
        masks = mask_utils();
    end

    [imgH, imgW, numChannels] = size(img);

    if isempty(vertices) || any(~isfinite(vertices(:)))
        content = zeros(0, 0, numChannels, 'like', img);
        bbox = [1, 1, 0, 0];
        return;
    end

    % Compute bounding box for return value
    minX = max(1, floor(min(vertices(:,1))));
    maxX = min(imgW, ceil(max(vertices(:,1))));
    minY = max(1, floor(min(vertices(:,2))));
    maxY = min(imgH, ceil(max(vertices(:,2))));

    if maxX < minX || maxY < minY
        content = zeros(0, 0, numChannels, 'like', img);
        bbox = [1, 1, 0, 0];
        return;
    end

    % Delegate masking to mask_utils
    [content, ~] = masks.cropWithPolygonMask(img, vertices);

    if isempty(content)
        content = zeros(0, 0, numChannels, 'like', img);
        bbox = [1, 1, 0, 0];
        return;
    end

    bboxWidth = maxX - minX + 1;
    bboxHeight = maxY - minY + 1;
    bbox = [minX, minY, bboxWidth, bboxHeight];
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
    % Delegating wrapper to augmentation_synthesis helper
    % Build placement functions struct for the helper
    placementFuncs = struct();
    placementFuncs.randomTopLeft = @random_top_left;
    placementFuncs.bboxesOverlap = @bboxes_overlap;
    placementFuncs.compositeToBackground = @composite_to_background;

    [bg, placedCount] = cfg.augSynth.distractors.addPolygon(bg, regions, polygonBboxes, occupiedBboxes, cfg, placementFuncs);
end

function [patchImg, isValid] = crop_ellipse_patch(polygonImg, ellipse)
    % Crop elliptical patch from polygon image
    %
    % Delegates ellipse mask creation to mask_utils.createEllipseMask for
    % consistent masking across all scripts.
    %
    % See also: mask_utils.createEllipseMask

    persistent masks
    if isempty(masks)
        masks = mask_utils();
    end

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

    % Create ellipse mask using mask_utils (center relative to patch)
    cx = ellipse.center(1) - x1 + 1;
    cy = ellipse.center(2) - y1 + 1;
    mask = masks.createEllipseMask([h, w], cx, cy, ellipse.semiMajor, ellipse.semiMinor, ellipse.rotation);

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
    %
    % Delegates to mask_utils.computeEllipseBoundingBox for consistent
    % calculation across all scripts. Uses Inf for image dimensions to
    % get unbounded bbox (caller handles clamping).
    %
    % See also: mask_utils.computeEllipseBoundingBox

    persistent masks
    if isempty(masks)
        masks = mask_utils();
    end

    [x1, y1, x2, y2] = masks.computeEllipseBoundingBox(...
        ellipse.center(1), ellipse.center(2), ...
        ellipse.semiMajor, ellipse.semiMinor, ...
        ellipse.rotation, Inf, Inf);

    bbox = [x1, y1, x2, y2];
end

%% =========================================================================
%% BACKGROUND GENERATION (delegates to augmentation_synthesis helper)
%% =========================================================================

function bg = generate_realistic_lab_surface(width, height, textureCfg, artifactCfg)
    %% Generate realistic lab surface backgrounds (delegates to helper)
    % Generates background using augmentation_synthesis, then adds sparse artifacts.
    augSynth = augmentation_synthesis();
    bg = augSynth.bg.generate(width, height, textureCfg);
    bg = augSynth.artifacts.addSparse(bg, width, height, artifactCfg);
end

function bg = add_sparse_artifacts(bg, width, height, artifactCfg)
    %% Add sparse artifacts (delegates to augmentation_synthesis helper)
    augSynth = augmentation_synthesis();
    bg = augSynth.artifacts.addSparse(bg, width, height, artifactCfg);
end

function mask = generate_polygon_mask(verticesPix, targetSize)
    %% Generate polygon mask (delegates to augmentation_synthesis helper)
    augSynth = augmentation_synthesis();
    mask = augSynth.artifacts.generatePolygonMask(verticesPix, targetSize);
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
            region(:,:,c) = uint8(min(255, max(0, plane)));
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
%% UTILITY FUNCTIONS
%% =========================================================================

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

%% =========================================================================
%% COORDINATE FILE I/O (delegated to coordinate_io.m)
%% =========================================================================
% These functions delegate parsing to coordinate_io.m (authoritative source)
% and adapt field names to match augment_dataset's internal conventions.

function write_stage2_coordinates(coords, outputDir, filename)
    % Atomically write stage 2 coordinates (deduplicated by image+concentration)
    % Format: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation (11 columns)
    %
    % Delegates to coordinate_io for atomic write with standard header.

    coordIO = coordinate_io();

    coordFolder = outputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, filename);

    % Load existing entries using coordinate_io (with field mapping)
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

    % Convert map to names/nums arrays for coordinate_io.atomicWriteCoordinates
    keysArr = map.keys;
    numEntries = numel(keysArr);
    names = cell(numEntries, 1);
    nums = zeros(numEntries, 10);  % concentration + 8 coords + rotation

    for i = 1:numEntries
        e = map(keysArr{i});
        names{i} = char(e.image);
        verts = round(e.vertices);
        rotation = 0;
        if isfield(e, 'rotation') && ~isempty(e.rotation)
            rotation = e.rotation;
        end
        nums(i, :) = [e.concentration, ...
                      verts(1,1), verts(1,2), verts(2,1), verts(2,2), ...
                      verts(3,1), verts(3,2), verts(4,1), verts(4,2), rotation];
    end

    % Use coordinate_io's header and format constants for atomic write
    header = coordIO.POLYGON_HEADER;
    writeFmt = coordIO.POLYGON_WRITE_FMT;

    coordIO.atomicWriteCoordinates(coordPath, header, names, nums, writeFmt, coordFolder);
end

function write_stage3_coordinates(coords, outputDir, filename)
    % Atomically write stage 3 coordinates (dedup by image+concentration+replicate)
    % Format: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
    %
    % Delegates to coordinate_io for atomic write with standard header.

    % Skip if no coordinates to write
    if isempty(coords)
        return;
    end

    coordIO = coordinate_io();

    coordFolder = outputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, filename);

    % Load existing entries using coordinate_io (with field mapping)
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

    % Convert map to names/nums arrays for coordinate_io.atomicWriteCoordinates
    keysArr = map.keys;
    numEntries = numel(keysArr);
    names = cell(numEntries, 1);
    nums = zeros(numEntries, 7);  % concentration, replicate, x, y, semiMajor, semiMinor, rotation

    for i = 1:numEntries
        e = map(keysArr{i});
        names{i} = char(e.image);
        nums(i, :) = [e.concentration, e.replicate, ...
                      e.center(1), e.center(2), e.semiMajor, e.semiMinor, e.rotation];
    end

    % Use coordinate_io's header and format constants for atomic write
    header = coordIO.ELLIPSE_HEADER;
    writeFmt = coordIO.ELLIPSE_WRITE_FMT;

    coordIO.atomicWriteCoordinates(coordPath, header, names, nums, writeFmt, coordFolder);
end

function entries = read_polygon_coordinates(coordPath)
    % Read polygon coordinates from stage 2 (2_micropads)
    % Delegates parsing to coordinate_io.parsePolygonCoordinateFile and maps field names.
    %
    % Returns struct array with fields: .image, .concentration, .vertices, .rotation
    % (Maps from coordinate_io's .imageName to .image for backward compatibility)

    entries = struct('image', {}, 'concentration', {}, 'vertices', {}, 'rotation', {});

    if ~isfile(coordPath)
        return;
    end

    coordIO = coordinate_io();
    rawEntries = coordIO.parsePolygonCoordinateFile(coordPath);

    if isempty(rawEntries)
        return;
    end

    % Map field names from coordinate_io format to augment_dataset format
    numEntries = numel(rawEntries);
    entries = struct('image', cell(1, numEntries), ...
                     'concentration', cell(1, numEntries), ...
                     'vertices', cell(1, numEntries), ...
                     'rotation', cell(1, numEntries));

    for i = 1:numEntries
        entries(i).image = rawEntries(i).imageName;  % Map imageName -> image
        entries(i).concentration = rawEntries(i).concentration;
        entries(i).vertices = rawEntries(i).vertices;
        entries(i).rotation = rawEntries(i).rotation;
    end
end

function entries = read_ellipse_coordinates(coordPath)
    % Read ellipse coordinates from stage 3 (3_elliptical_regions)
    % Delegates parsing to coordinate_io.parseEllipseCoordinateFile and maps field names.
    %
    % Returns struct array with fields: .image, .concentration, .replicate,
    %                                   .center, .semiMajor, .semiMinor, .rotation
    % (Maps from coordinate_io field names for backward compatibility)

    entries = struct('image', {}, 'concentration', {}, 'replicate', {}, ...
                     'center', {}, 'semiMajor', {}, 'semiMinor', {}, 'rotation', {});

    if ~isfile(coordPath)
        return;
    end

    coordIO = coordinate_io();
    rawEntries = coordIO.parseEllipseCoordinateFile(coordPath);

    if isempty(rawEntries)
        return;
    end

    % Map field names from coordinate_io format to augment_dataset format
    numEntries = numel(rawEntries);
    entries = struct('image', cell(1, numEntries), ...
                     'concentration', cell(1, numEntries), ...
                     'replicate', cell(1, numEntries), ...
                     'center', cell(1, numEntries), ...
                     'semiMajor', cell(1, numEntries), ...
                     'semiMinor', cell(1, numEntries), ...
                     'rotation', cell(1, numEntries));

    for i = 1:numEntries
        entries(i).image = rawEntries(i).imageName;  % Map imageName -> image
        entries(i).concentration = rawEntries(i).concentration;
        entries(i).replicate = rawEntries(i).replicate;
        entries(i).center = [rawEntries(i).x, rawEntries(i).y];  % Combine x,y -> center
        entries(i).semiMajor = rawEntries(i).semiMajorAxis;  % Map semiMajorAxis -> semiMajor
        entries(i).semiMinor = rawEntries(i).semiMinorAxis;  % Map semiMinorAxis -> semiMinor
        entries(i).rotation = rawEntries(i).rotationAngle;   % Map rotationAngle -> rotation
    end
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

