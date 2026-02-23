function augment_dataset(varargin)
    %% microPAD Colorimetric Analysis — Dataset Augmentation Tool
    %% Generates synthetic training datasets from microPAD paper images for quadrilateral detection
    %% Author: Veysel Y. Yilmaz
    %
    % PURPOSE: YOLO KEYPOINT DETECTION TRAINING ONLY
    %   This augmentation pipeline generates synthetic data for training YOLO keypoint
    %   detection models to locate test zone quadrilaterals. It is NOT intended for:
    %   - Colorimetric feature extraction (use extract_features.m with real data)
    %   - Training concentration prediction models
    %   The augmentations prioritize detector robustness over color fidelity.
    %
    % FEATURES:
    % - Procedural textured backgrounds (uniform, speckled, laminate, skin)
    % - Perspective and rotation transformations
    % - Random spatial placement with uniform distribution
    % - Independent rotation per concentration region
    % - Collision detection to prevent overlap
    % - Optional photometric augmentation, white-balance jitter, and blur
    % - Optional ROI noise augmentation (camera sensor, screen capture, old photo, JPEG)
    % - Optional thin occlusions over quads (hair/strap-like) for robustness
    % - Variable-density distractor artifacts (1-20 per image, unconstrained placement)
    % - Quadrilateral-shaped distractor generation for detection robustness
    %
    % Generates synthetic training data by applying geometric and photometric
    % transformations to microPAD paper images and their labeled concentration regions.
    %
    % PIPELINE:
    % 1. Copy real captures from 1_dataset/ into augmented_1_dataset/ (passthrough)
    % 2. Load quadrilateral coordinates from 2_micropads/
    % 3. Load ellipse coordinates from 3_elliptical_regions/ (optional)
    % 4. Generate N synthetic augmentations per paper (augIdx = 1..N)
    % 5. Write outputs to augmented_* directories
    %
    % TRANSFORMATION ORDER (applied to each concentration region):
    %   a) Shared perspective transformation (same for all regions from one paper)
    %   b) Shared rotation (same for all regions from one paper)
    %   c) Independent rotation (unique per region)
    %   d) Random spatial translation (uniformly distributed within margins)
    %   e) Composite onto procedural background
    %
    % OUTPUT STRUCTURE:
    %   augmented_1_dataset/[phone]/           - Real copies + synthetic scenes
    %   augmented_2_micropads/[phone]/con_*/   - Quadrilateral crops + coordinates.txt
    %   augmented_3_elliptical_regions/[phone]/con_*/ - Elliptical patches + coordinates.txt
    %
    % IMPORTANT: If you change backgroundWidth/backgroundHeight parameters mid-session,
    % run 'clear functions' to reset the internal texture cache.
    %
    % Parameters (Name-Value):
    % - 'numAugmentations' (positive integer): synthetic versions per paper
    %   Note: Real captures are always copied; synthetic scenes are labelled *_aug_XXX
    % - 'rngSeed' (numeric, optional): for reproducibility
    % - 'phones' (cellstr/string array): subset of phones to process
    % - 'backgroundWidth' (positive integer): synthetic background width override
    % - 'backgroundHeight' (positive integer): synthetic background height override
    % - 'scenePrefix' (char/string): Optional prefix for synthetic filenames.
    %     Only applied when explicitly provided. Default: no prefix.
    % - 'photometricAugmentation' (logical): enable color/lighting variation
    % - 'blurProbability' (0-1): fraction of samples with Gaussian blur
    % - 'motionBlurProbability' (0-1): fraction of samples with motion blur
    % - 'occlusionProbability' (0-1): per-quad probability of thin occlusions
    %     Note: Applied independently to each quad. With 7 quads at 0.12 each,
    %     ~57% of images will have at least one occluded quad.
    % - 'independentRotation' (logical): enable per-quad rotation
    % - 'extremeCasesProbability' (0-1): fraction using extreme viewpoints
    % - 'enableDistractorQuads' (logical): add synthetic look-alike distractors
    % - 'distractorMultiplier' (numeric): scale factor for distractor count
    % - 'distractorMaxCount' (integer): maximum distractors per scene
    % - 'paperDamageProbability' (0-1): fraction of quads with physical defects
    % - 'damageSeed' (numeric, optional): RNG seed for reproducible damage patterns
    % - 'damageProfileWeights' (DEPRECATED, ignored): damage profiles no longer used
    % - 'maxAreaRemovalFraction' (0-1): max removable fraction (per ellipse or micropad fallback)
    % - 'applyROINoiseAugmentation' (logical): enable realistic sensor noise on ROIs
    % - 'roiNoiseProfileWeights' (struct, optional): custom noise profile weights
    %     Fields: camera, screen, old_photo, jpeg (weights must sum to 1)
    %
    % EDGE FEATHERING:
    % - 'edgeFeatherWidth' (0-20, default 4): erosion amount in pixels for edge softening
    %     Set to 0 to disable feathering (sharp edges). Higher values create wider
    %     soft transitions at ROI boundaries for more natural blending.
    % - 'edgeFeatherSigma' (positive, default auto): Gaussian blur sigma
    %     Auto-computed as featherWidth/2 if not specified. Controls softness of falloff.
    %
    % IMAGE SIZE VARIATION:
    % - 'imageSizeMinWidth' (integer): minimum synthetic image width
    % - 'imageSizeMaxWidth' (integer): maximum synthetic image width
    % - 'imageSizeMinHeight' (integer): minimum synthetic image height
    % - 'imageSizeMaxHeight' (integer): maximum synthetic image height
    %     Note: Each augmentation samples a random size within these bounds.
    %     Use backgroundWidth/backgroundHeight to override with fixed dimensions.
    %
    % ROI SIZE CONTROL (dimension-based):
    % - 'roiSizeMin' (0-1): ROI's largest side >= this fraction of image's smallest side
    % - 'roiSizeMax' (0-1): ROI's largest side <= this fraction of image's smallest side
    % - 'roiFitMargin' (>=0): margin in pixels between ROI and image edge
    % - 'roiCountSensitivity' (0-1): how much object count reduces max size
    %     Note: Uses log-uniform sampling for perceptually even size distribution.
    %     More objects → smaller effective max (models real-world photography).
    %     Example: roiSizeMin=0.08, roiSizeMax=0.80 means ROI spans 8-80% of image.
    %     With 7 objects and sensitivity=0.12, effective max becomes ~46%.
    %
    % OBJECT COUNT BALANCING:
    % - 'balanceObjectCount' (logical): distribute augmentations across 1..N objects
    %     When enabled, generates images with varying numbers of objects (1 to numQuads),
    %     creating balanced training data for detection models. When disabled, all
    %     augmentations use all available quads from each paper.
    %
    % PAPER DAMAGE AUGMENTATION:
    %   Simulates realistic paper defects from storage and handling.
    %   Simplified to corner clips only, applied to outer zone of quad.
    %
    %   Protected regions (never damaged):
    %     - Ellipse regions when ellipse coordinates are available (primary mechanism)
    %     - Fallback: core guard zone sized by maxAreaRemovalFraction
    %       This creates ~36% area protection when no ellipses exist
    %   Note: Stains/occlusions use coreAreaFraction (60% area) from CORE_PROTECTION
    %
    % Examples:
    %   augment_dataset('numAugmentations', 5, 'rngSeed', 42)
    %   augment_dataset('phones', {'iphone_11'}, 'photometricAugmentation', false)
    %   augment_dataset('paperDamageProbability', 0.7, 'damageSeed', 42)
    %   augment_dataset('maxAreaRemovalFraction', 0.3)
    %   augment_dataset('applyROINoiseAugmentation', true, ...
    %                   'roiNoiseProfileWeights', struct('camera', 0.5, 'screen', 0.3, 'old_photo', 0.1, 'jpeg', 0.1))
    %
    %   % Variable image sizes (800-2000 width, 600-1500 height)
    %   augment_dataset('imageSizeMinWidth', 800, 'imageSizeMaxWidth', 2000, ...
    %                   'imageSizeMinHeight', 600, 'imageSizeMaxHeight', 1500)
    %
    %   % ROI size range (10-60% of image dimension)
    %   augment_dataset('roiSizeMin', 0.10, 'roiSizeMax', 0.60)
    %
    %   % Balanced object count distribution (images with 1..N objects)
    %   augment_dataset('numAugmentations', 10, 'balanceObjectCount', true)
    %
    %   % Disable object count balancing (all images use all quads)
    %   augment_dataset('balanceObjectCount', false)

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
    MIN_VALID_QUAD_AREA = 100;  % square pixels

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
        'poolMaxMemoryMB', 512, ...
        'whiteRGBVariation', 2, ...
        'whiteNoiseStrength', 3, ...
        'ambientGradientCount', [2, 4], ...
        'ambientGradientRadiusPercent', [0.30, 0.60], ...
        'ambientGradientStrength', [0.02, 0.08], ...
        'dropShadowOffsetRange', [2, 8], ...
        'dropShadowBlurRange', [8, 15], ...
        'dropShadowDarknessRange', [0.80, 0.90], ...  % Default (used for bgType 5)
        'shadowProbabilityByBgType', [0.50, 0.45, 0.30, 0.40, 0.85], ...  % Per background type
        'shadowDarknessByBgType', {{ ...
            [0.88, 0.95], ...   % Type 1: Uniform (medium gray) - subtle
            [0.90, 0.96], ...   % Type 2: Speckled (gray) - very subtle
            [0.94, 0.98], ...   % Type 3: Laminate (can be dark) - minimal but present
            [0.90, 0.97], ...   % Type 4: Skin tone - subtle
            [0.80, 0.90] ...    % Type 5: White - most visible (original behavior)
        }}, ...
        'specularHighlightProb', 0.15, ...  % Probability of specular highlight per quad
        'specularHighlightIntensityRange', [1.05, 1.25], ...  % Multiplicative intensity range
        'specularHighlightRadiusRange', [0.05, 0.15], ...  % Fraction of quad diagonal
        'specularHighlightBlurRange', [0.3, 0.6]);

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

    % Quad placement parameters
    PLACEMENT = struct( ...
        'margin', 50, ...  % pixels from edge
        'minSpacing', 30, ...  % pixels between regions
        'maxOverlapRetries', 5);

    % Distractor quadrilateral parameters (synthetic look-alikes)
    DISTRACTOR_QUADS = struct( ...
        'enabled', true, ...
        'minCount', 1, ...
        'maxCount', 10, ...
        'sizeScaleRange', [0.5, 1.5], ...  % fraction of original quad size
        'maxPlacementAttempts', 30, ...
        'brightnessOffsetRange', [-20, 20], ...  % uint8 intensity units
        'contrastScaleRange', [0.9, 1.15], ...
        'noiseStd', 6, ...  % uint8 intensity units
        'typeWeights', [1, 1, 1], ...
        'outlineWidthRange', [1.5, 4.0], ...  % pixels
        'textureGainRange', [0.06, 0.18], ...  % normalized modulation strength
        'textureSurfaceTypes', [1, 2, 3, 4]);        % Background texture primitives to reuse

    % Paper damage augmentation parameters
    % NOTE: Simplified to corner clips only after Phase 3 refactor.
    % Removed unused params: profileWeights, sideBiteRange, taperStrengthRange,
    % edgeWaveAmplitudeRange, edgeWaveFrequencyRange, maxOperations
    PAPER_DAMAGE = struct( ...
        'probability', 0.20, ...  % Fraction of quads with damage
        'cornerClipRange', [0.10, 0.35], ...  % Fraction of edge length (corner clip depth)
        'maxAreaRemovalFraction', 0.40);  % Fraction allowed to be removed per ellipse (or micropad fallback)

    % ROI noise augmentation parameters
    ROI_NOISE = struct( ...
        'enabled', true, ...
        'profileWeights', struct( ...
            'camera', 0.35, ...
            'screen', 0.25, ...
            'old_photo', 0.20, ...
            'jpeg', 0.20));

    % Core protection configuration (used by stains and occlusions)
    % Note: Paper damage uses PAPER_DAMAGE.maxAreaRemovalFraction instead.
    % maxOuterDamageFraction/minOuterDamageFraction are reference values only -
    % corner clips naturally produce ~10-35% outer zone damage without enforcement.
    CORE_PROTECTION = struct( ...
        'coreAreaFraction', 0.60, ...
        'maxOuterDamageFraction', 0.40, ...  % Reference: max acceptable
        'minOuterDamageFraction', 0.02, ...  % Reference: min visible
        'thinOcclusionMaxWidth', 3, ...
        'thinOcclusionMinWidth', 1, ...
        'thickOcclusionMinGap', 2, ...
        'thickOcclusionMaxWidth', 15, ...
        'thinOcclusionProbability', 0.40, ...
        'thinOcclusionMaxLength', 0.5, ...
        'thinOcclusionMaxCount', 2, ...
        'thinOcclusionMaxContrast', 30, ...
        'thinOcclusionMaxCoverage', 0.08, ...
        'outerStainProbability', 0.15, ...
        'outerStainOpacityRange', [0.30, 0.80], ...
        'coreStainProbability', 0.05, ...
        'coreStainMaxOpacity', 0.15, ...
        'coreStainMaxCoverage', 0.05, ...
        'maxOuterStainCoverage', 0.40, ...
        'stainSemiMajorRange', [10, 50], ...  % pixels
        'stainSemiMinorRange', [5, 25], ...   % pixels
        'enablePerQuadShadows', false, ...
        'enableValidation', false, ...
        'occlusionTypeWeights', [0.60, 0.25, 0.15]);

    %% =====================================================================
    %% INPUT PARSING
    %% =====================================================================
    parser = inputParser();
    parser.FunctionName = mfilename;

    addParameter(parser, 'numAugmentations', 20, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>=',1}));
    addParameter(parser, 'rngSeed', [], @(n) isempty(n) || isnumeric(n));
    addParameter(parser, 'phones', {}, @(c) iscellstr(c) || isstring(c));
    addParameter(parser, 'backgroundWidth', 4000, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>',0}));
    addParameter(parser, 'backgroundHeight', 3000, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>',0}));
    addParameter(parser, 'scenePrefix', 'synthetic', @(s) validateattributes(s, {'char','string'}, {'scalartext'}));
    addParameter(parser, 'photometricAugmentation', true, @islogical);
    addParameter(parser, 'blurProbability', 0.20, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'motionBlurProbability', 0.15, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'occlusionProbability', 0.12, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'independentRotation', true, @islogical);
    addParameter(parser, 'extremeCasesProbability', 0.08, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    addParameter(parser, 'enableDistractorQuads', true, @islogical);
    addParameter(parser, 'distractorMultiplier', 0.6, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0}));
    addParameter(parser, 'distractorMaxCount', 6, @(x) validateattributes(x, {'numeric'}, {'scalar','integer','>=',0}));
    addParameter(parser, 'paperDamageProbability', [], @(n) isempty(n) || (isnumeric(n) && isscalar(n) && n >= 0 && n <= 1));
    addParameter(parser, 'damageSeed', [], @(n) isempty(n) || (isnumeric(n) && isscalar(n) && isfinite(n)));
    addParameter(parser, 'damageProfileWeights', [], @(s) isempty(s) || isstruct(s));  % DEPRECATED: profiles no longer used after simplification
    addParameter(parser, 'maxAreaRemovalFraction', [], @(n) isempty(n) || (isnumeric(n) && isscalar(n) && n >= 0 && n <= 1));
    addParameter(parser, 'applyROINoiseAugmentation', true, @(x) validateattributes(x, {'logical'}, {'scalar'}));
    addParameter(parser, 'roiNoiseProfileWeights', [], @(s) isempty(s) || isstruct(s));

    % Image size bounds (variable synthetic image size)
    addParameter(parser, 'imageSizeMinWidth', 1280, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>=',320}));
    addParameter(parser, 'imageSizeMaxWidth', 5000, @(n) validateattributes(n, {'numeric'}, {'scalar','integer'}));
    addParameter(parser, 'imageSizeMinHeight', 1280, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>=',240}));
    addParameter(parser, 'imageSizeMaxHeight', 5000, @(n) validateattributes(n, {'numeric'}, {'scalar','integer'}));

    % ROI size control (dimension-based)
    addParameter(parser, 'roiSizeMin', 0.08, @(n) validateattributes(n, {'numeric'}, {'scalar','>',0,'<=',1}));
    addParameter(parser, 'roiSizeMax', 0.80, @(n) validateattributes(n, {'numeric'}, {'scalar','>',0,'<=',1}));
    addParameter(parser, 'roiFitMargin', 20, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0}));
    addParameter(parser, 'roiCountSensitivity', 0.12, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));

    % Object count balancing
    addParameter(parser, 'balanceObjectCount', true, @islogical);

    % Edge feathering for natural ROI blending
    addParameter(parser, 'edgeFeatherWidth', 4, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>=',0,'<=',20}));
    addParameter(parser, 'edgeFeatherSigma', [], @(n) isempty(n) || (isnumeric(n) && isscalar(n) && n > 0));
    addParameter(parser, 'edgeFeatherProbability', 0.5, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));

    % Stale labels prevention
    addParameter(parser, 'clearOutputOnRerun', true, @islogical);

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

    %% Add helper_scripts to path (contains geometry_transform and other utilities)
    scriptDir = fileparts(mfilename('fullpath'));
    helperDir = fullfile(scriptDir, 'helper_scripts');
    if exist(helperDir, 'dir')
        addpath(helperDir);
    end

    %% Load utility modules
    geomTform = geometry_transform();
    imageIO = image_io();
    pathUtils = path_utils();
    augSynth = augmentation_synthesis();
    coordIO = coordinate_io();  % Authoritative source for coordinate I/O
    roiNoise = roi_noise();  % ROI noise augmentation
    occlusionUtils = occlusion_utils();  % Occlusion generation

    % Build configuration
    cfg = struct();
    cfg.geomTform = geomTform;
    cfg.pathUtils = pathUtils;
    cfg.imageIO = imageIO;
    cfg.augSynth = augSynth;
    cfg.coordIO = coordIO;
    cfg.roiNoise = roiNoise;
    cfg.occlusionUtils = occlusionUtils;
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
    cfg.minValidQuadArea = MIN_VALID_QUAD_AREA;
    cfg.texture = TEXTURE;
    cfg.artifacts = ARTIFACTS;
    cfg.placement = PLACEMENT;
    cfg.extremeCasesProbability = opts.extremeCasesProbability;
    cfg.distractors = DISTRACTOR_QUADS;
    cfg.distractors.enabled = cfg.distractors.enabled && opts.enableDistractorQuads;
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
        warning('augmentDataset:deprecatedParameter', ...
                'damageProfileWeights is deprecated and has no effect. Paper damage now uses corner clips only.');
    end

    % Validate range bounds
    if cfg.damage.cornerClipRange(2) <= cfg.damage.cornerClipRange(1)
        error('augmentDataset:invalidRange', 'cornerClipRange upper bound must exceed lower bound');
    end
    cfg.damage.rngSeed = opts.damageSeed;

    % ROI noise augmentation configuration
    cfg.roiNoiseConfig = ROI_NOISE;
    cfg.roiNoiseConfig.enabled = cfg.roiNoiseConfig.enabled && opts.applyROINoiseAugmentation;
    if ~isempty(opts.roiNoiseProfileWeights)
        cfg.roiNoiseConfig.profileWeights = opts.roiNoiseProfileWeights;
    end

    % Core protection configuration
    % Add sceneSpace function handles from augmentation_synthesis module to avoid duplication
    cfg.coreProtection = CORE_PROTECTION;
    cfg.coreProtection.getLocalScale = augSynth.sceneSpace.getLocalScale;
    cfg.coreProtection.computeCoreDiagonal = augSynth.sceneSpace.computeCoreDiagonal;

    % Image size bounds configuration (variable synthetic image size)
    cfg.imageSizeBounds = struct( ...
        'minWidth', opts.imageSizeMinWidth, ...
        'maxWidth', opts.imageSizeMaxWidth, ...
        'minHeight', opts.imageSizeMinHeight, ...
        'maxHeight', opts.imageSizeMaxHeight);

    % Validate image size bounds
    if cfg.imageSizeBounds.maxWidth < cfg.imageSizeBounds.minWidth
        error('augmentDataset:invalidRange', 'imageSizeMaxWidth must be >= imageSizeMinWidth');
    end
    if cfg.imageSizeBounds.maxHeight < cfg.imageSizeBounds.minHeight
        error('augmentDataset:invalidRange', 'imageSizeMaxHeight must be >= imageSizeMinHeight');
    end

    % ROI size configuration (dimension-based scaling)
    cfg.roiSize = struct( ...
        'minFrac', opts.roiSizeMin, ...
        'maxFrac', opts.roiSizeMax, ...
        'fitMargin', opts.roiFitMargin, ...
        'countSensitivity', opts.roiCountSensitivity, ...
        'retries', 10);

    % Validate ROI size bounds
    if cfg.roiSize.maxFrac < cfg.roiSize.minFrac
        error('augmentDataset:invalidRange', 'roiSizeMax must be >= roiSizeMin');
    end

    % Validate fitMargin against minimum image size from bounds
    minImageDim = min(cfg.imageSizeBounds.minWidth, cfg.imageSizeBounds.minHeight);
    if 2 * cfg.roiSize.fitMargin >= minImageDim
        error('augmentDataset:fitMarginExceedsBounds', ...
            'roiFitMargin (%d) too large for minimum image size (%d). Require 2*margin < min dimension.', ...
            cfg.roiSize.fitMargin, minImageDim);
    end

    % Also validate fitMargin against explicit background dimension overrides
    if cfg.backgroundOverride.useWidth && 2 * cfg.roiSize.fitMargin >= cfg.backgroundOverride.width
        error('augmentDataset:fitMarginExceedsWidth', ...
            'roiFitMargin (%d) too large for backgroundWidth (%d). Require 2*margin < width.', ...
            cfg.roiSize.fitMargin, cfg.backgroundOverride.width);
    end
    if cfg.backgroundOverride.useHeight && 2 * cfg.roiSize.fitMargin >= cfg.backgroundOverride.height
        error('augmentDataset:fitMarginExceedsHeight', ...
            'roiFitMargin (%d) too large for backgroundHeight (%d). Require 2*margin < height.', ...
            cfg.roiSize.fitMargin, cfg.backgroundOverride.height);
    end

    % Object count balancing configuration
    cfg.balanceObjectCount = opts.balanceObjectCount;

    % Edge feathering configuration (probabilistic: 50% soft edges, 50% hard edges by default)
    cfg.edgeFeather = struct('width', opts.edgeFeatherWidth, 'sigma', opts.edgeFeatherSigma, ...
                             'probability', opts.edgeFeatherProbability);

    % Stale labels prevention configuration
    cfg.clearOutputOnRerun = opts.clearOutputOnRerun;

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
        fprintf('Image size range: %d-%d x %d-%d px\n', ...
            cfg.imageSizeBounds.minWidth, cfg.imageSizeBounds.maxWidth, ...
            cfg.imageSizeBounds.minHeight, cfg.imageSizeBounds.maxHeight);
    end
    fprintf('ROI size range: %.0f%%-%.0f%% of image (margin=%dpx, count sensitivity=%.2f)\n', ...
        cfg.roiSize.minFrac * 100, cfg.roiSize.maxFrac * 100, ...
        cfg.roiSize.fitMargin, cfg.roiSize.countSensitivity);
    fprintf('Object count balancing: %s\n', char(string(cfg.balanceObjectCount)));
    if cfg.useScenePrefix
        fprintf('Scene prefix: %s\n', cfg.scenePrefix);
    else
        fprintf('Scene prefix: (none)\n');
    end
    fprintf('Backgrounds: 5 types (uniform, speckled, laminate, skin, white+shadows)\n');
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

    % Load quadrilateral coordinates from stage 2 (required)
    if ~isfile(stage2PhoneCoords)
        warning('augmentDataset:noQuadCoords', 'No quad coordinates for %s. Skipping.', phoneName);
        return;
    end

    quadEntries = read_quad_coordinates(stage2PhoneCoords);
    if isempty(quadEntries)
        warning('augmentDataset:emptyQuads', 'No valid quad entries for %s', phoneName);
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
        fprintf('  [INFO] No ellipse coordinates. Will process quads only.\n');
    end

    % Build lookup structures
    ellipseMap = group_ellipses_by_parent(ellipseEntries, hasEllipses);
    paperGroups = group_quads_by_image(quadEntries);

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

    % Clear output folders to prevent stale labels from previous runs
    if cfg.clearOutputOnRerun
        clear_folder_contents(stage1PhoneOut);
        clear_folder_contents(stage2PhoneOut);
        if hasEllipses
            clear_folder_contents(stage3PhoneOut);
        end
    end

    stage2CoordPath = fullfile(stage2PhoneOut, cfg.files.coordinates);
    % Start fresh (don't load existing) when clearing outputs
    stage2Map = init_stage2_map(stage2CoordPath, cfg.clearOutputOnRerun);
    stage3Map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    if hasEllipses
        stage3CoordPath = fullfile(stage3PhoneOut, cfg.files.coordinates);
        % Start fresh (don't load existing) when clearing outputs
        stage3Map = init_stage3_map(stage3CoordPath, cfg.clearOutputOnRerun);
    end

    % Get unique paper names
    paperNames = keys(paperGroups);
    fprintf('  Total papers: %d\n', numel(paperNames));
    fprintf('  Total quads: %d\n', numel(quadEntries));
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

        % Get all quads from this paper
        quads = paperGroups(paperBase);
        quadCache = build_quad_cache(stage1Img, quads);

        % Emit passthrough sample (augIdx = 0) with original geometry
        emit_passthrough_sample(paperBase, stage1Img, quads, quadCache, ellipseMap, ...
                                 hasEllipses, stage1PhoneOut, stage2PhoneOut, ...
                                 stage3PhoneOut, cfg, stage2Map, stage3Map);

        % Generate synthetic augmentations only
        if cfg.numAugmentations < 1
            continue;
        end

        % Compute augmentation schedule for balanced object count distribution
        schedule = compute_augmentation_schedule(numel(quads), cfg.numAugmentations, cfg.balanceObjectCount);
        globalAugIdx = 0;

        for schedIdx = 1:numel(schedule)
            k = schedule(schedIdx).k;  % Number of objects to use
            scheduleCount = schedule(schedIdx).count;  % Number of augmentations with k objects

            for localIdx = 1:scheduleCount
                globalAugIdx = globalAugIdx + 1;

                % Select random subset of quads if balancing is enabled
                if cfg.balanceObjectCount && k < numel(quads)
                    selectedIndices = randperm(numel(quads), k);  % No Statistics Toolbox needed
                    selectedQuads = quads(selectedIndices);
                    selectedCache = quadCache(selectedIndices);
                    selectedEllipseMap = filter_ellipse_map(ellipseMap, selectedQuads, paperBase);
                else
                    selectedQuads = quads;
                    selectedCache = quadCache;
                    selectedEllipseMap = ellipseMap;
                end

                augment_single_paper(paperBase, imgExt, stage1Img, selectedQuads, selectedEllipseMap, ...
                                     hasEllipses, globalAugIdx, stage1PhoneOut, stage2PhoneOut, ...
                                     stage3PhoneOut, paperIdx, phoneName, cfg, stage2Map, ...
                                     stage3Map, selectedCache);
            end
        end
    end

    % Paper damage summary logging
    if cfg.damage.probability > 0
        fprintf('  Paper damage applied per quad with %.0f%% probability\n', cfg.damage.probability * 100);
    end

    write_stage2_map(stage2Map, stage2PhoneOut, cfg.files.coordinates);
    if hasEllipses && stage3Map.Count > 0
        write_stage3_map(stage3Map, stage3PhoneOut, cfg.files.coordinates);
    end

end

%% -------------------------------------------------------------------------
function quadCache = build_quad_cache(stage1Img, quads)
    quadCache = struct('quadImg', cell(numel(quads), 1), ...
                       'contentBbox', cell(numel(quads), 1));

    for idx = 1:numel(quads)
        origVertices = double(quads(idx).vertices);
        [quadImg, contentBbox] = extract_quad_masked(stage1Img, origVertices);
        quadCache(idx).quadImg = quadImg;
        quadCache(idx).contentBbox = contentBbox;
    end
end

%% -------------------------------------------------------------------------
function emit_passthrough_sample(paperBase, stage1Img, quads, quadCache, ellipseMap, ...
                                 hasEllipses, stage1PhoneOut, stage2PhoneOut, ...
                                 stage3PhoneOut, cfg, stage2Map, stage3Map)
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

    stage2Coords = cell(numel(quads), 1);
    % Pre-allocate for 3 ellipses per quad (experimental design)
    maxEllipsesPerQuad = 3;
    stage3Coords = cell(max(1, numel(quads) * maxEllipsesPerQuad), 1);
    quadCount = 0;
    s2Count = 0;
    s3Count = 0;

    for idx = 1:numel(quads)
        quad = quads(idx);
        origVertices = double(quad.vertices);

        if ~is_valid_quad(origVertices, cfg.minValidQuadArea)
            warning('augmentDataset:passthroughInvalidQuad', ...
                    '  ! Quad %s con %d invalid for passthrough. Skipping.', ...
                    paperBase, quad.concentration);
            continue;
        end

        quadImg = quadCache(idx).quadImg;
        if isempty(quadImg)
            warning('augmentDataset:passthroughEmptyCrop', ...
                    '  ! Quad %s con %d produced empty crop.', ...
                    paperBase, quad.concentration);
            continue;
        end

        concDir = fullfile(stage2PhoneOut, sprintf('%s%d', cfg.concPrefix, quad.concentration));
        cfg.pathUtils.ensureFolder(concDir);
        quadFileName = sprintf('%s_%s%d%s', sceneName, cfg.concPrefix, quad.concentration, imgExt);
        quadOutPath = fullfile(concDir, quadFileName);
        imwrite(quadImg, quadOutPath);

        quadCount = quadCount + 1;

        s2Count = s2Count + 1;
        stage2Coords{s2Count} = struct( ...
            'image', quadFileName, ...
            'concentration', quad.concentration, ...
            'vertices', origVertices, ...
            'rotation', quad.rotation);

        ellipseKey = sprintf('%s#%d', paperBase, quad.concentration);
        if hasEllipses && isKey(ellipseMap, ellipseKey)
            ellipseList = ellipseMap(ellipseKey);
            % Get contentBbox for image-to-crop coordinate conversion
            contentBbox = quadCache(idx).contentBbox;  % [minX, minY, width, height]
            for eIdx = 1:numel(ellipseList)
                ellipseIn = ellipseList(eIdx);
                % Convert ellipse center from image-space to crop-space
                % Stage-3 coordinates are stored in image-space (per coordinate_io.m)
                cropSpaceCenter = ellipseIn.center - [contentBbox(1)-1, contentBbox(2)-1];
                ellipseGeom = struct( ...
                    'center', cropSpaceCenter, ...
                    'semiMajor', ellipseIn.semiMajor, ...
                    'semiMinor', ellipseIn.semiMinor, ...
                    'rotation', ellipseIn.rotation);

                [patchImg, patchValid] = crop_ellipse_patch(quadImg, ellipseGeom);
                if ~patchValid || isempty(patchImg)
                    warning('augmentDataset:passthroughPatchInvalid', ...
                            '  ! Ellipse %s con %d rep %d invalid for passthrough.', ...
                            paperBase, quad.concentration, ellipseIn.replicate);
                    continue;
                end

                ellipseDir = fullfile(stage3PhoneOut, sprintf('%s%d', cfg.concPrefix, quad.concentration));
                cfg.pathUtils.ensureFolder(ellipseDir);
                patchFileName = sprintf('%s_%s%d_rep%d%s', sceneName, cfg.concPrefix, ...
                                        quad.concentration, ellipseIn.replicate, imgExt);
                patchOutPath = fullfile(ellipseDir, patchFileName);
                imwrite(patchImg, patchOutPath);

                s3Count = s3Count + 1;
                if s3Count > numel(stage3Coords)
                    stage3Coords = [stage3Coords; cell(s3Count, 1)]; %#ok<AGROW> % Rare case: more ellipses than expected
                end
                stage3Coords{s3Count} = struct( ...
                    'image', patchFileName, ...
                    'concentration', quad.concentration, ...
                    'replicate', ellipseIn.replicate, ...
                    'center', ellipseGeom.center, ...
                    'semiMajor', ellipseGeom.semiMajor, ...
                    'semiMinor', ellipseGeom.semiMinor, ...
                    'rotation', ellipseGeom.rotation);
            end
        end
    end

    if quadCount == 0
        warning('augmentDataset:passthroughNoQuads', ...
                '  ! No valid quads for passthrough %s. Removing scene.', sceneName);
        if exist(sceneOutPath, 'file') == 2
            delete(sceneOutPath);
        end
        return;
    end

    stage2Coords = stage2Coords(1:s2Count);
    stage3Coords = stage3Coords(1:s3Count);

    update_stage2_map(stage2Map, stage2Coords);
    if s3Count > 0
        update_stage3_map(stage3Map, stage3Coords);
    end

    fprintf('     Passthrough: %s (%d quads, %d ellipses)\n', ...
            sceneFileName, numel(stage2Coords), numel(stage3Coords));
end

%% -------------------------------------------------------------------------
function augment_single_paper(paperBase, imgExt, stage1Img, quads, ellipseMap, ...
                               hasEllipses, augIdx, stage1PhoneOut, stage2PhoneOut, ...
                               stage3PhoneOut, paperIdx, phoneName, cfg, stage2Map, ...
                               stage3Map, quadCache)
    % Generate one augmented version of a paper with all its concentration regions

    % Sample transformation (same for all regions in this augmentation)
    if rand() < cfg.extremeCasesProbability
        % Extreme camera viewpoint (more challenging than normal 60° max)
        extremeCamera = cfg.camera;
        extremeCamera.maxAngleDeg = 70;
        extremeCamera.zRange = [0.8, 2.5];  % Closer range for more perspective distortion
        viewParams = cfg.geomTform.homog.sampleViewpoint(extremeCamera);
        activeCameraCfg = extremeCamera;
    else
        % Normal camera viewpoint
        viewParams = cfg.geomTform.homog.sampleViewpoint(cfg.camera);
        activeCameraCfg = cfg.camera;
    end
    tformPersp = cfg.geomTform.homog.computeHomography(size(stage1Img), viewParams, activeCameraCfg);
    rotAngle = cfg.geomTform.homog.randRange(cfg.rotationRange);
    tformRot = cfg.geomTform.homog.centeredRotationTform(size(stage1Img), rotAngle);

    % Pre-allocate coordinate accumulators
    % Stage 2: one entry per quad (upper bound = validCount after validation)
    % Stage 3: 3 ellipses per quad (experimental design)
    maxQuads = numel(quads);
    maxEllipsesPerQuad = 3;
    stage2Coords = cell(maxQuads, 1);
    stage3Coords = cell(maxQuads * maxEllipsesPerQuad, 1);
    s2Count = 0;
    s3Count = 0;

    % Cache deterministic hashes for reproducible per-quad damage sampling
    phoneHash = stable_string_hash(phoneName);
    paperHash = stable_string_hash(paperBase);

    % Temporary storage for transformed quad crops and their properties
    transformedRegions = cell(numel(quads), 1);
    validCount = 0;

    % Transform all quads and extract crops
    for quadIdx = 1:numel(quads)
        quadEntry = quads(quadIdx);
        concentration = quadEntry.concentration;
        origVertices = quadEntry.vertices;

        % Apply shared perspective transformation
        augVertices = cfg.geomTform.homog.transformQuad(origVertices, tformPersp);
        augVertices = cfg.geomTform.homog.transformQuad(augVertices, tformRot);

        % Apply independent rotation per quad if enabled
        if cfg.independentRotation
            independentRotAngle = cfg.geomTform.homog.randRange(cfg.rotationRange);
            tformIndepRot = cfg.geomTform.homog.centeredRotationTform(size(stage1Img), independentRotAngle);
        else
            independentRotAngle = 0;
            tformIndepRot = affine2d(eye(3));
        end
        augVertices = cfg.geomTform.homog.transformQuad(augVertices, tformIndepRot);

        % Validate transformed quad
        if ~is_valid_quad(augVertices, cfg.minValidQuadArea)
            warning('augmentDataset:degenerateQuad', ...
                    '  ! Quad %s con %d degenerate after transform. Skipping.', ...
                    paperBase, concentration);
            continue;
        end

        % Extract quad content with masking
        quadContent = quadCache(quadIdx).quadImg;
        contentBbox = quadCache(quadIdx).contentBbox;

        % Transform extracted content to match augmented shape
        % Returns both RGB (pre-multiplied at edges) and continuous alpha for compositing
        [augQuadImg, augAlpha] = cfg.geomTform.homog.transformQuadContent(quadContent, ...
                                                  origVertices, augVertices, contentBbox);

        % Note: Edge feathering is applied AFTER scaling (see scaling loop below)
        % to ensure consistent feather width regardless of scale factor.

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

        % Apply paper damage augmentation per quad using configured probability
        if cfg.damage.probability > 0
            % Convert continuous alpha to binary for damage operations
            quadAlphaBinary = augAlpha > 0.5;

            % Apply damage (with deterministic RNG if seed provided)
            damageRng = [];
            if ~isempty(cfg.damage.rngSeed)
                % Create deterministic seed combining global seed + paper + concentration + augmentation
                % Multipliers ensure no collisions: 10000 papers × 100 concentrations × 100 augmentations
                damageRng = cfg.damage.rngSeed + phoneHash + paperHash + ...
                            paperIdx * 10000 + concentration * 100 + augIdx;
            end

            [damagedRGB, damagedAlpha, ~, ~, ~] = cfg.augSynth.damage.apply( ...
                augQuadImg, quadAlphaBinary, augVerticesCrop, ellipseAugList, cfg.damage, damageRng);

            % Check if damage was actually applied by comparing masks
            % Note: damageCorners is never updated by apply_paper_damage, so we detect
            % damage via mask differences instead
            damageWasApplied = ~isequal(damagedAlpha, quadAlphaBinary);

            % Replace augQuadImg (vertices unchanged since damage doesn't modify corners)
            augQuadImg = damagedRGB;

            % Only use binary alpha if damage was applied (changes shape)
            % Otherwise preserve continuous alpha for smooth edge compositing
            if damageWasApplied
                % Un-premultiply edge pixels that survive damage but had fractional alpha.
                % These pixels have RGB = original * continuous_alpha, need to restore original
                % so they composite correctly with the new binary alpha (prevents dark edges).
                %
                % Use ratio clamping (max 10x) rather than just alpha threshold to prevent
                % extreme amplification that could cause bright rims from quantization noise.
                % Alpha threshold 0.01 skips near-invisible pixels; ratio cap handles the rest.
                MAX_UNPREMULTIPLY_RATIO = 10.0;
                survivingEdges = damagedAlpha & (augAlpha < 0.99) & (augAlpha > 0.01);
                if any(survivingEdges(:))
                    % Compute capped ratio: min(1/alpha, MAX_RATIO)
                    ratio = ones(size(augAlpha), 'double');
                    ratio(survivingEdges) = min(MAX_UNPREMULTIPLY_RATIO, ...
                        1.0 ./ double(augAlpha(survivingEdges)));
                    for c = 1:3
                        channel = double(augQuadImg(:,:,c));
                        channel(survivingEdges) = channel(survivingEdges) .* ratio(survivingEdges);
                        augQuadImg(:,:,c) = uint8(min(255, max(0, channel)));
                    end
                end
                augAlpha = single(damagedAlpha);
                % Note: Edge feathering is applied AFTER scaling to ensure
                % consistent feather width regardless of scale factor.
            end
        end

        % Store transformed region for later composition
        validCount = validCount + 1;
        transformedRegions{validCount} = struct( ...
            'concentration', concentration, ...
            'augVertices', augVertices, ...
            'augQuadImg', augQuadImg, ...
            'augAlpha', augAlpha, ...  % Continuous alpha for pre-multiplied compositing
            'contentBbox', contentBbox, ...
            'origVertices', origVertices, ...
            'independentRotAngle', independentRotAngle, ...
            'originalRotation', quadEntry.rotation, ...
            'augEllipses', ellipseAugList);
    end

    % Trim to valid regions
    transformedRegions = transformedRegions(1:validCount);

    if validCount == 0
        warning('augmentDataset:noValidRegions', '  ! No valid regions for %s aug %d', paperBase, augIdx);
        return;
    end

    % Compute individual quad bounding boxes for random placement
    quadBboxes = cell(validCount, 1);
    for i = 1:validCount
        verts = transformedRegions{i}.augVertices;
        quadBboxes{i} = struct( ...
            'minX', min(verts(:,1)), ...
            'maxX', max(verts(:,1)), ...
            'minY', min(verts(:,2)), ...
            'maxY', max(verts(:,2)), ...
            'width', max(verts(:,1)) - min(verts(:,1)), ...
            'height', max(verts(:,2)) - min(verts(:,2)));
    end

    % Sample random background dimensions within configured bounds
    bgWidth = round(cfg.imageSizeBounds.minWidth + rand() * (cfg.imageSizeBounds.maxWidth - cfg.imageSizeBounds.minWidth));
    bgHeight = round(cfg.imageSizeBounds.minHeight + rand() * (cfg.imageSizeBounds.maxHeight - cfg.imageSizeBounds.minHeight));

    % Allow explicit override for backward compatibility
    if cfg.backgroundOverride.useWidth
        bgWidth = cfg.backgroundOverride.width;
    end
    if cfg.backgroundOverride.useHeight
        bgHeight = cfg.backgroundOverride.height;
    end

    % Sample edge feathering decision once per augmentation (probabilistic: soft vs hard edges)
    % When feathering is disabled for this augmentation, use 0 width (hard edges)
    useFeathering = rand() < cfg.edgeFeather.probability;
    effectiveFeatherWidth = useFeathering * cfg.edgeFeather.width;

    % Store original (pre-scale) copies for retry logic
    originalRegions = transformedRegions;
    originalBboxes = quadBboxes;

    % Apply randomized object scale with fit constraints
    for i = 1:validCount
        region = originalRegions{i};
        bbox = originalBboxes{i};

        % Sample scale factor using dimension-based constraints
        scaleFactor = sample_roi_scale(cfg.roiSize, bgWidth, bgHeight, bbox.width, bbox.height, validCount);

        % Scale quad content if scale factor is valid and not 1.0
        if scaleFactor > 0 && abs(scaleFactor - 1.0) > 1e-6
            % Convert vertices to relative coordinates for scaling
            relativeVerts = region.augVertices - [bbox.minX, bbox.minY];

            % Scale image, vertices, ellipses, and alpha
            [scaledImg, scaledVerts, scaledEllipses, scaledAlpha] = scale_quad_content( ...
                region.augQuadImg, relativeVerts, region.augEllipses, scaleFactor, region.augAlpha);

            % Update transformed region with scaled content
            transformedRegions{i}.augQuadImg = scaledImg;
            transformedRegions{i}.augVertices = scaledVerts + [bbox.minX, bbox.minY] * scaleFactor;
            transformedRegions{i}.augEllipses = scaledEllipses;
            transformedRegions{i}.augAlpha = scaledAlpha;
        end

        % Apply edge feathering AFTER scaling for consistent feather width
        % regardless of scale factor. This ensures e.g., 3px feather is always
        % 3px in the final output, not 1.5px at 0.5x scale or 6px at 2x scale.
        % Uses effectiveFeatherWidth (sampled probabilistically per-augmentation).
        if effectiveFeatherWidth > 0
            [featheredImg, featheredAlpha] = cfg.imageIO.featherQuadEdges( ...
                transformedRegions{i}.augQuadImg, transformedRegions{i}.augAlpha, ...
                effectiveFeatherWidth, cfg.edgeFeather.sigma);
            transformedRegions{i}.augQuadImg = featheredImg;
            transformedRegions{i}.augAlpha = featheredAlpha;
        end

        % Recompute bounding box after scaling (feathering doesn't change bounds)
        verts = transformedRegions{i}.augVertices;
        quadBboxes{i} = struct( ...
            'minX', min(verts(:,1)), ...
            'maxX', max(verts(:,1)), ...
            'minY', min(verts(:,2)), ...
            'maxY', max(verts(:,2)), ...
            'width', max(verts(:,1)) - min(verts(:,1)), ...
            'height', max(verts(:,2)) - min(verts(:,2)));
    end

    % Place quads at random non-overlapping positions with retry for scale adjustment
    placementSuccess = false;
    maxPlacementAttempts = cfg.roiSize.retries;

    for placementAttempt = 1:maxPlacementAttempts
        randomPositions = place_quads_nonoverlapping(quadBboxes, ...
                                                         bgWidth, bgHeight, ...
                                                         cfg.placement.margin, ...
                                                         cfg.placement.minSpacing, ...
                                                         cfg.placement.maxOverlapRetries);

        % Verify all positions are valid (within bounds with margin for quad size)
        allValid = true;
        for i = 1:validCount
            pos = randomPositions{i};
            bbox = quadBboxes{i};
            if isempty(pos) || pos.x < 0 || pos.y < 0 || ...
               (pos.x + bbox.width > bgWidth) || (pos.y + bbox.height > bgHeight)
                allValid = false;
                break;
            end
        end

        if allValid
            placementSuccess = true;
            break;
        end

        % If placement failed and more attempts remain, resample scales from ORIGINAL state
        if placementAttempt < maxPlacementAttempts
            for i = 1:validCount
                % Use ORIGINAL unscaled region and bbox to avoid compounding
                region = originalRegions{i};
                origBbox = originalBboxes{i};

                % Resample with a smaller scale factor (bias toward fitting)
                % Reduce by 10% per retry attempt to progressively shrink
                retryReduction = 0.9 ^ placementAttempt;
                scaleFactor = sample_roi_scale(cfg.roiSize, bgWidth, bgHeight, ...
                                               origBbox.width, origBbox.height, validCount) * retryReduction;

                if scaleFactor > 0 && abs(scaleFactor - 1.0) > 1e-6
                    relativeVerts = region.augVertices - [origBbox.minX, origBbox.minY];
                    [scaledImg, scaledVerts, scaledEllipses, scaledAlpha] = scale_quad_content( ...
                        region.augQuadImg, relativeVerts, region.augEllipses, scaleFactor, region.augAlpha);

                    transformedRegions{i}.augQuadImg = scaledImg;
                    transformedRegions{i}.augVertices = scaledVerts + [origBbox.minX, origBbox.minY] * scaleFactor;
                    transformedRegions{i}.augEllipses = scaledEllipses;
                    transformedRegions{i}.augAlpha = scaledAlpha;
                else
                    % Reset to original unscaled, unfeathered content to prevent
                    % double feathering from previous iteration
                    transformedRegions{i} = originalRegions{i};
                end

                % Apply edge feathering AFTER scaling (consistent with main loop)
                % Uses effectiveFeatherWidth (sampled probabilistically per-augmentation).
                if effectiveFeatherWidth > 0
                    [featheredImg, featheredAlpha] = cfg.imageIO.featherQuadEdges( ...
                        transformedRegions{i}.augQuadImg, transformedRegions{i}.augAlpha, ...
                        effectiveFeatherWidth, cfg.edgeFeather.sigma);
                    transformedRegions{i}.augQuadImg = featheredImg;
                    transformedRegions{i}.augAlpha = featheredAlpha;
                end

                % Update bounding box
                verts = transformedRegions{i}.augVertices;
                quadBboxes{i} = struct( ...
                    'minX', min(verts(:,1)), ...
                    'maxX', max(verts(:,1)), ...
                    'minY', min(verts(:,2)), ...
                    'maxY', max(verts(:,2)), ...
                    'width', max(verts(:,1)) - min(verts(:,1)), ...
                    'height', max(verts(:,2)) - min(verts(:,2)));
            end
        end
    end

    if ~placementSuccess
        warning('augmentDataset:placementFailed', ...
                '  ! Placement failed for %s aug %d after %d attempts. Skipping this augmentation.', ...
                paperBase, augIdx, maxPlacementAttempts);
        return;  % Skip this augmentation to avoid coordinate/visual mismatch
    end

    % Generate realistic background with final size
    [background, bgType] = generate_realistic_lab_surface(bgWidth, bgHeight, cfg.texture, cfg.artifacts);

    % Sample shared light direction once per scene (consistent across all quads)
    lightAngle = rand() * 360;

    % Determine if shadows should be applied for this scene based on bgType probability
    shadowProbByType = cfg.texture.shadowProbabilityByBgType;
    applyShadowsThisScene = rand() < shadowProbByType(bgType);

    % Get darkness range for this background type
    darknessByType = cfg.texture.shadowDarknessByBgType;
    shadowDarknessRange = darknessByType{bgType};

    % Composite each region onto background and save outputs
    if cfg.useScenePrefix
        baseSceneId = sprintf('%s_%s', cfg.scenePrefix, paperBase);
    else
        baseSceneId = paperBase;
    end
    sceneName = sprintf('%s_aug_%03d', baseSceneId, augIdx);
    sceneQuads = cell(validCount, 1);
    occupiedBboxes = zeros(validCount, 4);
    quadIdx = 0;

    % Collect quad vertices for ROI noise augmentation
    sceneQuadVertices = cell(validCount, 1);

    % Store crop metadata for saving AFTER scene augmentations
    % This ensures crops include all augmentations (distractors, stains, ROI noise, photometric, blur)
    cropMetadata = cell(validCount, 1);

    for i = 1:validCount
        region = transformedRegions{i};
        concentration = region.concentration;
        augVertices = region.augVertices;
        augQuadImg = region.augQuadImg;
        augAlpha = region.augAlpha;  % Continuous alpha for pre-multiplied compositing

        % Get random position for this quad
        pos = randomPositions{i};

        % Compute offset to move quad from its current position to random position
        % Random position specifies the top-left corner of the bounding box
        currentMinX = quadBboxes{i}.minX;
        currentMinY = quadBboxes{i}.minY;
        offsetX = pos.x - currentMinX;
        offsetY = pos.y - currentMinY;

        % Translate vertices to background coordinates
        sceneVertices = augVertices + [offsetX, offsetY];

        % Apply drop shadow with adaptive darkness based on background type
        if applyShadowsThisScene
            % Create shadow params with bgType-specific darkness range
            shadowParams = struct( ...
                'dropShadowOffsetRange', cfg.texture.dropShadowOffsetRange, ...
                'dropShadowBlurRange', cfg.texture.dropShadowBlurRange, ...
                'dropShadowDarknessRange', shadowDarknessRange);

            % Generate and apply drop shadow
            imgSize = [size(background, 1), size(background, 2)];
            shadowMask = cfg.augSynth.shadows.generateDropShadow( ...
                sceneVertices, lightAngle, imgSize, shadowParams);

            % Compute shadow region bounding box (account for shadow offset and blur)
            % Use max offset + 3-sigma Gaussian tail for proper coverage
            shadowExpand = cfg.texture.dropShadowOffsetRange(2) + 3 * cfg.texture.dropShadowBlurRange(2);
            minX = max(1, floor(min(sceneVertices(:,1))) - shadowExpand);
            maxX = min(size(background, 2), ceil(max(sceneVertices(:,1))) + shadowExpand);
            minY = max(1, floor(min(sceneVertices(:,2))) - shadowExpand);
            maxY = min(size(background, 1), ceil(max(sceneVertices(:,2))) + shadowExpand);

            % Apply shadow mask to background region (per-channel multiply)
            if maxX >= minX && maxY >= minY
                bgRegion = background(minY:maxY, minX:maxX, :);
                shadowRegion = shadowMask(minY:maxY, minX:maxX);
                for c = 1:3
                    bgRegion(:,:,c) = uint8(double(bgRegion(:,:,c)) .* shadowRegion);
                end
                background(minY:maxY, minX:maxX, :) = bgRegion;
            end
        end

        % Composite onto background using pre-multiplied alpha
        background = composite_to_background(background, augQuadImg, sceneVertices, augAlpha);

        % Collect quad vertices for ROI noise augmentation
        sceneQuadVertices{i} = sceneVertices;

        % Paper lies flat on surface; no shadows needed
        minSceneX = min(sceneVertices(:,1));
        minSceneY = min(sceneVertices(:,2));
        maxSceneX = max(sceneVertices(:,1));
        maxSceneY = max(sceneVertices(:,2));
        occupiedBboxes(i, :) = [minSceneX, minSceneY, maxSceneX, maxSceneY];

        % Prepare quad output path (saving deferred until after scene augmentations)
        concDirOut = fullfile(stage2PhoneOut, sprintf('%s%d', cfg.concPrefix, concentration));
        cfg.pathUtils.ensureFolder(concDirOut);

        quadFileName = sprintf('%s_%s%d%s', sceneName, cfg.concPrefix, concentration, imgExt);
        quadOutPath = fullfile(concDirOut, quadFileName);

        % Store crop metadata for saving AFTER all scene augmentations
        % This ensures stage 2/3 crops include distractors, stains, ROI noise, photometric, blur
        cropMetadata{i} = struct( ...
            'quadOutPath', quadOutPath, ...
            'concentration', concentration, ...
            'ellipseCropList', region.augEllipses, ...
            'stage3PhoneOut', stage3PhoneOut);

        % Record stage 2 coordinates (quad in scene)
        % Compute total applied rotation and update saved rotation field
        %
        % ROTATION SEMANTICS:
        %   The rotation field in Stage 2 coordinates is a UI-only alignment hint
        %   (see cut_micropads.m documentation). When augmenting:
        %   - originalRotation: The source image's UI hint (from 2_micropads)
        %   - totalAppliedRotation: Sum of scene rotation + any independent quad rotation
        %   - augmentedRotation: Adjusted hint for the augmented image
        %
        %   Formula: augmentedRotation = originalRotation - totalAppliedRotation
        %   This ensures that if a user loads the augmented image in the UI and applies
        %   the saved rotation, the visual alignment is preserved.
        totalAppliedRotation = rotAngle + region.independentRotAngle;
        augmentedRotation = cfg.geomTform.normalizeAngle(region.originalRotation - totalAppliedRotation);

        s2Count = s2Count + 1;
        stage2Coords{s2Count} = struct( ...
            'image', quadFileName, ...
            'concentration', concentration, ...
            'vertices', sceneVertices, ...
            'rotation', augmentedRotation);

        % Track quad in scene space for optional occlusions
        quadIdx = quadIdx + 1;
        sceneQuads{quadIdx} = sceneVertices;

        % Record stage 3 coordinates (ellipse in quad-crop space)
        % NOTE: These coordinates are in QUAD-CROP space (not scene space) and are keyed
        % by patch filenames. They are NOT intended for use with extract_features.m, which
        % expects scene base names and scene-space coordinates. Augmented Stage 3 data is
        % preserved for potential future use but is currently not consumed by downstream tools.
        % Saving deferred until after scene augmentations
        ellipseCropList = region.augEllipses;
        if hasEllipses && ~isempty(ellipseCropList)
            for eIdx = 1:numel(ellipseCropList)
                ellipseCrop = ellipseCropList(eIdx);
                replicateId = ellipseCrop.replicate;

                % Compute patch filename (must match the filename used when saving at lines 1496-1497)
                patchFileName = sprintf('%s_%s%d_rep%d%s', sceneName, cfg.concPrefix, concentration, replicateId, imgExt);

                % Record stage 3 coordinates (ellipse in quad-crop space)
                s3Count = s3Count + 1;
                if s3Count > numel(stage3Coords)
                    stage3Coords = [stage3Coords; cell(s3Count, 1)]; %#ok<AGROW> % Rare case: more ellipses than expected
                end
                stage3Coords{s3Count} = struct( ...
                    'image', patchFileName, ...
                    'concentration', concentration, ...
                    'replicate', replicateId, ...
                    'center', ellipseCrop.center, ...
                    'semiMajor', ellipseCrop.semiMajor, ...
                    'semiMinor', ellipseCrop.semiMinor, ...
                    'rotation', ellipseCrop.rotation);
            end
        end

    end

    % Trim sceneQuads to actual size
    sceneQuads = sceneQuads(1:quadIdx);

    additionalDistractors = 0;
    if cfg.distractors.enabled && cfg.distractors.multiplier > 0
        [background, additionalDistractors] = add_quad_distractors(background, transformedRegions, quadBboxes, occupiedBboxes, cfg);
    end

    % Apply per-quad augmentations: paper stains and occlusions
    if ~isempty(sceneQuads)
        for qIdx = 1:numel(sceneQuads)
            verts = sceneQuads{qIdx};
            if isempty(verts) || any(~isfinite(verts(:)))
                continue;
            end

            % Create masks for this quad
            [imgH, imgW, ~] = size(background);
            quadMask = poly2mask(verts(:,1), verts(:,2), imgH, imgW);

            % Compute core mask
            coreScale = getCoreScale(cfg.coreProtection.coreAreaFraction);
            coreVertices = shrink_quad_toward_centroid(verts, coreScale);
            coreMask = poly2mask(coreVertices(:,1), coreVertices(:,2), imgH, imgW);
            outerMask = quadMask & ~coreMask;

            % Apply paper stains (post-composite, per-quad)
            background = cfg.augSynth.stains.apply(background, quadMask, coreMask, cfg.coreProtection);

            % Apply occlusions (if enabled)
            if cfg.occlusionProbability > 0 && rand() < cfg.occlusionProbability
                % Use identity transform (acceptable simplification - extreme viewpoints
                % are a small fraction of samples, so local scale is close to 1.
                % Full Jacobian-based conversion would add complexity for minimal benefit.)
                quadTransform = eye(3);

                % Apply occlusions (augRecord output unused - kept for debugging)
                [background, ~] = cfg.occlusionUtils.addQuadOcclusions(background, quadMask, coreMask, coreVertices, ...
                                                       outerMask, quadTransform, cfg.coreProtection, struct());
            end

            % Apply specular highlights (simulates phone camera reflections on laminated/glossy surfaces)
            if rand() < cfg.texture.specularHighlightProb
                background = cfg.augSynth.specular.apply(background, verts, cfg.texture);
            end
        end
    end

    % NOTE: Global projective jitter removed - applying geometric transforms after
    % coordinate recording causes label/image mismatch. Other augmentations
    % (rotation, perspective, viewpoint variation) provide sufficient geometric diversity.

    % Apply ROI noise augmentation (before photometric augmentation)
    % Track if JPEG was applied to prevent double compression in photometric step
    jpegAlreadyApplied = false;
    if cfg.roiNoiseConfig.enabled
        % Create ROI mask from all quad vertices
        [imgHeight, imgWidth, ~] = size(background);
        roiMask = cfg.roiNoise.createMaskFromQuads(sceneQuadVertices, imgHeight, imgWidth);

        % Only apply noise if ROI mask has pixels
        if any(roiMask(:))
            % Select noise profile
            profileName = cfg.roiNoise.selectProfile(cfg.roiNoiseConfig.profileWeights);

            % Track JPEG selection to prevent double compression (case-insensitive)
            if strcmpi(profileName, 'jpeg')
                jpegAlreadyApplied = true;
            end

            % Generate reproducible seed from current RNG state
            roiNoiseSeed = randi(2^31 - 1);

            % Apply noise to ROIs with soft edge blending (uses effectiveFeatherWidth: 0 for hard edges, cfg value for soft)
            background = cfg.roiNoise.applyToROIs(background, roiMask, profileName, roiNoiseSeed, effectiveFeatherWidth);
        end
    end

    % Apply photometric augmentation and non-overlapping blur before saving
    if cfg.photometricAugmentation
        % Phase 1.7: Extreme photometric conditions (low lighting)
        if rand() < cfg.extremeCasesProbability
            background = apply_photometric_augmentation(background, 'extreme', jpegAlreadyApplied);
        else
            background = apply_photometric_augmentation(background, 'subtle', jpegAlreadyApplied);
        end
    end

    % Ensure at most one blur type is applied to avoid double-softening
    blurApplied = false;
    if cfg.motionBlurProbability > 0 && rand() < cfg.motionBlurProbability
        background = cfg.imageIO.applyMotionBlur(background);
        blurApplied = true;
    end
    if ~blurApplied && cfg.blurProbability > 0 && rand() < cfg.blurProbability
        blurSigma = 0.5 + rand() * 1.5;  % [0.5, 2.0] pixels - realistic defocus/motion
        background = imgaussfilt(background, blurSigma);
    end

    % Save synthetic scene (stage 1 output)
    sceneFileName = sprintf('%s%s', sceneName, '.png');
    sceneOutPath = fullfile(stage1PhoneOut, sceneFileName);
    imwrite(background, sceneOutPath);

    % === Extract and save augmented crops from final scene ===
    % Now that all scene augmentations are applied (distractors, stains, occlusions,
    % ROI noise, photometric, blur), extract quad regions and ellipse patches.
    % This ensures stage 2/3 crops match the augmented scene pixels.
    for i = 1:validCount
        meta = cropMetadata{i};
        sceneVerts = sceneQuadVertices{i};

        % Extract quad region from final augmented background
        augmentedQuadCrop = extract_quad_from_scene(background, sceneVerts);
        if isempty(augmentedQuadCrop)
            warning('augmentDataset:extractFailed', ...
                    '  ! Failed to extract quad %d from scene. Skipping crop save.', i);
            continue;
        end

        % Save quad crop (stage 2 output)
        imwrite(augmentedQuadCrop, meta.quadOutPath);

        % Extract and save ellipse patches (stage 3 output)
        if hasEllipses && ~isempty(meta.ellipseCropList)
            for eIdx = 1:numel(meta.ellipseCropList)
                ellipseCrop = meta.ellipseCropList(eIdx);
                replicateId = ellipseCrop.replicate;

                % Extract ellipse patch from augmented quad crop
                [patchImg, patchValid] = crop_ellipse_patch(augmentedQuadCrop, ellipseCrop);
                if ~patchValid
                    warning('augmentDataset:patchInvalid', ...
                            '  ! Ellipse patch %s con %d rep %d invalid after augmentation. Skipping.', ...
                            paperBase, meta.concentration, replicateId);
                    continue;
                end

                % Save ellipse patch (stage 3 output)
                ellipseConcDir = fullfile(meta.stage3PhoneOut, sprintf('%s%d', cfg.concPrefix, meta.concentration));
                cfg.pathUtils.ensureFolder(ellipseConcDir);
                patchFileName = sprintf('%s_%s%d_rep%d%s', sceneName, cfg.concPrefix, ...
                                        meta.concentration, replicateId, imgExt);
                patchOutPath = fullfile(ellipseConcDir, patchFileName);
                imwrite(patchImg, patchOutPath);
            end
        end
    end

    % Trim coordinate arrays to actual size
    stage2Coords = stage2Coords(1:s2Count);
    stage3Coords = stage3Coords(1:s3Count);

    % Write coordinates
    update_stage2_map(stage2Map, stage2Coords);
    if s3Count > 0
        update_stage3_map(stage3Map, stage3Coords);
    end

    fprintf('     Generated: %s (%d quads, %d ellipses, %d distractors)\n', ...
            sceneFileName, numel(stage2Coords), numel(stage3Coords), additionalDistractors);
end

%% =========================================================================
%% CORE PROTECTION HELPERS
%% =========================================================================

function coreScale = getCoreScale(coreAreaFraction)
    % Compute coreScale from coreAreaFraction (SINGLE SOURCE OF TRUTH)
    % coreScale is the linear scaling factor such that coreScale^2 = coreAreaFraction
    coreScale = sqrt(coreAreaFraction);
end

function scaledVertices = shrink_quad_toward_centroid(vertices, scaleFactor)
    % Shrink quad vertices toward centroid by scaleFactor
    % scaleFactor = 0.775 for 60% area retention
    centroid = mean(vertices, 1);
    scaledVertices = (vertices - centroid) * scaleFactor + centroid;
end

function quadCrop = extract_quad_from_scene(sceneImg, sceneVerts)
    % Extract quadrilateral region from augmented scene image
    %
    % This extracts the bounding box region that was composited onto the scene.
    % Uses the same coordinate calculation as composite_to_background to ensure
    % pixel-perfect alignment between scene and crop.
    %
    % INPUTS:
    %   sceneImg   - Final augmented scene image (H x W x 3)
    %   sceneVerts - 4x2 quad vertices in scene coordinates
    %
    % OUTPUTS:
    %   quadCrop   - Extracted region from scene (may include background pixels
    %                in corners outside the quad shape)

    [imgH, imgW, ~] = size(sceneImg);

    % Use same bounds calculation as composite_to_background for consistency
    minX = max(1, floor(min(sceneVerts(:,1))));
    maxX = min(imgW, ceil(max(sceneVerts(:,1))));
    minY = max(1, floor(min(sceneVerts(:,2))));
    maxY = min(imgH, ceil(max(sceneVerts(:,2))));

    % Guard: degenerate region
    if maxX < minX || maxY < minY
        quadCrop = [];
        return;
    end

    % Extract bounding box region from scene
    quadCrop = sceneImg(minY:maxY, minX:maxX, :);
end

% NOTE: Paper stain functions (apply_paper_stains, generateIrregularStain,
% shrinkMaskToCoverage, sampleRealisticStainColor, blendStain) have been moved to
% augmentation_synthesis.m. Access via cfg.augSynth.stains.apply() and related functions.

%% =========================================================================
%% CORE PROCESSING FUNCTIONS
%% =========================================================================

function [content, bbox] = extract_quad_masked(img, vertices)
    % Extract quadrilateral region with masking to avoid black pixels.
    %
    % Delegates to mask_utils.cropWithQuadMask for consistent masking
    % across all scripts. See mask_utils.m for authoritative implementation.
    %
    % OUTPUTS:
    %   content - Cropped and masked image (bounding box size)
    %   bbox    - [minX, minY, width, height] of cropped region
    %
    % See also: mask_utils.cropWithQuadMask

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
    [content, ~] = masks.cropWithQuadMask(img, vertices);

    if isempty(content)
        content = zeros(0, 0, numChannels, 'like', img);
        bbox = [1, 1, 0, 0];
        return;
    end

    bboxWidth = maxX - minX + 1;
    bboxHeight = maxY - minY + 1;
    bbox = [minX, minY, bboxWidth, bboxHeight];
end

function bg = composite_to_background(bg, quadImg, sceneVerts, quadAlpha)
    % Composite transformed quad onto background using pre-multiplied alpha blending
    %
    % When quadAlpha is provided (continuous 0-1), uses proper pre-multiplied
    % alpha compositing: result = foreground + background * (1 - alpha)
    % This correctly handles edge pixels where RGB is blended with black fill.
    %
    % When quadAlpha is not provided (backward compatibility for distractors),
    % falls back to binary mask from color values.

    hasProvidedAlpha = nargin >= 4 && ~isempty(quadAlpha);

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

    % Resize quad to target size only when necessary
    [patchH, patchW, ~] = size(quadImg);
    needsResize = patchH ~= targetHeight || patchW ~= targetWidth;

    % Handle alpha mask and choose appropriate interpolation method
    if hasProvidedAlpha
        % With alpha: use bilinear for both RGB and alpha to maintain pre-multiplied
        % relationship. Bilinear with black fill naturally preserves pre-multiplied
        % state: RGB blends toward black at edges in proportion to alpha blend.
        if needsResize
            resized = imresize(quadImg, [targetHeight, targetWidth], 'bilinear');
            resizedAlpha = imresize(quadAlpha, [targetHeight, targetWidth], 'bilinear');
        else
            resized = quadImg;
            resizedAlpha = quadAlpha;
        end
        % Convert to double and clamp to [0,1] for robustness
        alpha = max(0, min(1, double(resizedAlpha)));
    else
        % Without alpha (distractors): use nearest to avoid dark halos from
        % bilinear blending with black, since mask is derived from color values
        if needsResize
            resized = imresize(quadImg, [targetHeight, targetWidth], 'nearest');
        else
            resized = quadImg;
        end

        % Backward compatibility: derive binary mask from color values (for distractors)
        vertsTarget = sceneVerts - [minX - 1, minY - 1];
        targetMask = poly2mask(vertsTarget(:,1), vertsTarget(:,2), targetHeight, targetWidth);

        if ~any(targetMask(:))
            return;
        end

        % Use the synthesized patch mask to support hollow distractors
        patchMask = any(resized > 0, 3);
        effectiveMask = targetMask & patchMask;
        if ~any(effectiveMask(:))
            return;
        end
        alpha = double(effectiveMask);
    end

    % Guard: no alpha coverage
    if ~any(alpha(:) > 0)
        return;
    end

    % Composite per-channel using pre-multiplied alpha formula:
    % result = foreground_premultiplied + background * (1 - alpha)
    %
    % The transformed quad RGB is effectively pre-multiplied because imwarp
    % with FillValues=0 blends edge pixels with black, producing: RGB * alpha
    % This formula correctly reconstructs the original colors at edges.
    bgRegion = bg(minY:maxY, minX:maxX, :);

    numChannels = size(bgRegion, 3);
    for c = 1:numChannels
        R = double(bgRegion(:,:,c));
        F = double(resized(:,:,c));
        % Pre-multiplied compositing: F already contains F_original * alpha at edges
        % Explicit min(255, ...) for robustness against edge cases
        bgRegion(:,:,c) = uint8(min(255, F + R .* (1 - alpha)));
    end

    bg(minY:maxY, minX:maxX, :) = bgRegion;
end

function [bg, placedCount] = add_quad_distractors(bg, regions, quadBboxes, occupiedBboxes, cfg)
    % Delegating wrapper to augmentation_synthesis helper
    % Build placement functions struct for the helper
    placementFuncs = struct();
    placementFuncs.randomTopLeft = @random_top_left;
    placementFuncs.bboxesOverlap = @bboxes_overlap;
    placementFuncs.compositeToBackground = @composite_to_background;

    [bg, placedCount] = cfg.augSynth.distractors.addQuad(bg, regions, quadBboxes, occupiedBboxes, cfg, placementFuncs);
end

function [patchImg, isValid] = crop_ellipse_patch(quadImg, ellipse)
    % Crop elliptical patch from quad image
    %
    % Delegates ellipse mask creation to mask_utils.createEllipseMask for
    % consistent masking across all scripts.
    %
    % See also: mask_utils.createEllipseMask

    persistent masks
    if isempty(masks)
        masks = mask_utils();
    end

    [imgHeight, imgWidth, ~] = size(quadImg);
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

    patchImg = quadImg(y1:y2, x1:x2, :);
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

function [bg, bgType] = generate_realistic_lab_surface(width, height, textureCfg, artifactCfg)
    %% Generate realistic lab surface backgrounds (delegates to helper)
    % Generates background using augmentation_synthesis, then adds sparse artifacts.
    augSynth = augmentation_synthesis();
    [bg, bgType] = augSynth.bg.generate(width, height, textureCfg);
    bg = augSynth.artifacts.addSparse(bg, width, height, artifactCfg);
end

% NOTE: Occlusion functions (add_quad_occlusions, sampleOcclusionWidth, drawLine,
% applyOcclusionColor, generateOcclusionColor, generateBlobOcclusion,
% generateFingerOcclusion, enforceContrastLimit, clipToMaxLength) have been moved to
% occlusion_utils.m. Access via cfg.occlusionUtils.addQuadOcclusions().

% NOTE: getLocalScale and computeCoreDiagonal functions are in augmentation_synthesis.m
% (sceneSpace module). They are accessed via function handles in cfg.coreProtection.

% NOTE: Specular highlight function (apply_specular_highlight) has been moved to
% augmentation_synthesis.m. Access via cfg.augSynth.specular.apply().

%% =========================================================================
%% PHOTOMETRIC AUGMENTATION (NEW IN V3)
%% =========================================================================

function img = apply_photometric_augmentation(img, mode, jpegAlreadyApplied)
    % Apply lighting-realistic photometric augmentation to entire scene
    % Simulates real-world lighting/white-balance variation for detector robustness
    %
    % NOTE: Brightness, contrast, white balance, and saturation adjustments have been
    % removed to avoid redundancy with YOLO's HSV augmentation (hsv_h=0.015, hsv_s=0.7,
    % hsv_v=0.4). Only unique augmentations are kept: gamma, color temperature, sensor
    % noise, and JPEG artifacts.
    %
    % The 'extreme' mode still applies brightness reduction for very low lighting simulation.
    %
    % Inputs:
    %   img - RGB image (uint8)
    %   mode - 'subtle' (default), 'moderate', or 'extreme' (Phase 1.7)
    %   jpegAlreadyApplied - (optional) If true, skip JPEG to prevent double compression

    % Configuration constants (centralized for easy tuning)
    PHOTOMETRIC = struct( ...
        'gammaProb', 0.40, ...         % Gamma correction probability
        'gammaRange', [0.92, 1.08], ...% Gamma exponent range
        'colorTempProb', 0.35, ...     % Color temperature variation probability
        'warmthRange', [-0.10, 0.10], ...% Warmth shift range
        'noiseProb', 0.30, ...         % Sensor noise probability
        'noiseSigmaRange', [3, 11], ...% Noise sigma range (uint8 scale)
        'jpegProb', 0.25, ...          % JPEG compression probability
        'jpegQualityRange', [60, 95]); % JPEG quality range

    if nargin < 2
        mode = 'subtle';
    end
    if nargin < 3
        jpegAlreadyApplied = false;
    end

    % Store original before photometric (for clamping)
    imgOrig = im2double(img);

    % Convert to double in [0,1] for processing
    imgDouble = imgOrig;

    % 1. Extreme brightness reduction (only for 'extreme' mode - Phase 1.7)
    % This simulates very low lighting conditions that YOLO's HSV augmentation doesn't cover.
    if strcmp(mode, 'extreme')
        brightRange = [0.40, 0.60];  % Very low lighting
        brightFactor = brightRange(1) + rand() * diff(brightRange);
        imgDouble = imgDouble * brightFactor;
    end

    % 2. Gamma correction (exposure simulation) - unique to MATLAB
    if rand() < PHOTOMETRIC.gammaProb
        gamma = PHOTOMETRIC.gammaRange(1) + rand() * diff(PHOTOMETRIC.gammaRange);
        imgDouble = min(1, max(0, imgDouble));
        imgDouble = imgDouble .^ gamma;
    end

    % 3. Color temperature variation - unique to MATLAB
    if rand() < PHOTOMETRIC.colorTempProb
        warmth = PHOTOMETRIC.warmthRange(1) + rand() * diff(PHOTOMETRIC.warmthRange);
        imgDouble(:,:,1) = imgDouble(:,:,1) * (1 + warmth);       % Red channel
        imgDouble(:,:,3) = imgDouble(:,:,3) * (1 - warmth * 0.6); % Blue channel (less affected)
    end

    % Clamp cumulative color shifts to prevent excessive color drift
    MAX_CUMULATIVE_COLOR_SHIFT = 0.40;
    for ch = 1:3
        imgDouble(:,:,ch) = clampToMaxShift(imgOrig(:,:,ch)*255, imgDouble(:,:,ch)*255, MAX_CUMULATIVE_COLOR_SHIFT) / 255;
    end

    % 4. Apply sensor noise (AFTER color clamping, BEFORE final conversion) - unique to MATLAB
    if rand() < PHOTOMETRIC.noiseProb
        sigmaRange = PHOTOMETRIC.noiseSigmaRange;
        noiseSigma = (sigmaRange(1) + rand() * diff(sigmaRange)) / 255;
        imgDouble = imgDouble + randn(size(imgDouble)) * noiseSigma;
    end

    % 5. JPEG compression artifacts (FINAL step) - unique to MATLAB
    % Skip if JPEG was already applied by ROI noise to prevent double compression
    if ~jpegAlreadyApplied && rand() < PHOTOMETRIC.jpegProb
        qualityRange = PHOTOMETRIC.jpegQualityRange;
        quality = qualityRange(1) + randi(diff(qualityRange));
        img = uint8(min(255, max(0, imgDouble * 255)));  % Convert first

        tempFile = [tempname '.jpg'];
        try
            imwrite(img, tempFile, 'Quality', quality);
            img = imread(tempFile);
            delete(tempFile);
        catch ME
            % Cleanup temp file on error
            if exist(tempFile, 'file')
                delete(tempFile);
            end
            warning('augmentDataset:jpegFailed', 'JPEG compression failed: %s', ME.message);
        end
        return;  % Already converted to uint8
    end

    % Final clamp and convert back to uint8
    img = im2uint8(min(1, max(0, imgDouble)));
end

function clampedValues = clampToMaxShift(origChannel, adjChannel, maxShift)
    % Clamp adjusted channel to be within maxShift of original
    % Returns CLAMPED VALUES (not deltas) - assign directly to output

    DARK_THRESHOLD = 30;
    absMaxDelta = maxShift * 255;

    clampedValues = adjChannel;

    % Bright pixels: relative clamping
    brightMask = origChannel > DARK_THRESHOLD;
    if any(brightMask(:))
        origVals = origChannel(brightMask);
        origVals(origVals == 0) = 1;  % Defensive: prevent division by zero
        relDelta = (adjChannel(brightMask) - origChannel(brightMask)) ./ origVals;
        clampedRelDelta = max(-maxShift, min(maxShift, relDelta));
        clampedValues(brightMask) = origChannel(brightMask) .* (1 + clampedRelDelta);
    end

    % Dark pixels: absolute clamping
    darkMask = ~brightMask;
    if any(darkMask(:))
        absDelta = adjChannel(darkMask) - origChannel(darkMask);
        clampedAbsDelta = max(-absMaxDelta, min(absMaxDelta, absDelta));
        clampedValues(darkMask) = origChannel(darkMask) + clampedAbsDelta;
    end

    clampedValues = max(0, min(255, clampedValues));
end

%% =========================================================================
%% UTILITY FUNCTIONS
%% =========================================================================

function clear_folder_contents(folderPath)
    % Clear all contents of a folder (images and subfolders) for clean reruns
    %
    % Deletes all files and subdirectories within folderPath but preserves
    % the folder itself. Used to prevent stale labels from previous runs.
    % Warns on permission errors but continues with remaining files.
    %
    % Input:
    %   folderPath - Path to folder to clear

    if ~isfolder(folderPath)
        return;
    end

    % Delete all files in the folder
    files = dir(folderPath);
    for i = 1:numel(files)
        if files(i).isdir
            if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
                continue;
            end
            % Recursively delete subfolders (e.g., con_* directories)
            try
                rmdir(fullfile(folderPath, files(i).name), 's');
            catch ME
                warning('augmentDataset:clearFolderFailed', ...
                        'Failed to delete folder %s: %s', files(i).name, ME.message);
            end
        else
            try
                delete(fullfile(folderPath, files(i).name));
            catch ME
                warning('augmentDataset:clearFileFailed', ...
                        'Failed to delete file %s: %s', files(i).name, ME.message);
            end
        end
    end
end

function schedule = compute_augmentation_schedule(numQuads, numAugmentations, balanceEnabled)
    % Distributes numAugmentations across k=1..N object counts for balanced training
    %
    % When balanceEnabled is true, generates a schedule that produces augmented
    % images with varying numbers of objects (1 to numQuads), distributing
    % augmentations evenly across all object counts.
    %
    % Inputs:
    %   numQuads - Total number of quads available from this paper
    %   numAugmentations - Total number of augmentations to generate
    %   balanceEnabled - If true, distribute across object counts; if false, use all quads
    %
    % Output:
    %   schedule - Struct array with fields:
    %              .k - Number of objects to use
    %              .count - Number of augmentations to generate with k objects

    if ~balanceEnabled || numQuads <= 1
        % No balancing: use all quads for every augmentation
        schedule = struct('k', numQuads, 'count', numAugmentations);
        return;
    end

    % Distribute augmentations evenly across k=1..numQuads
    baseCount = floor(numAugmentations / numQuads);
    remainder = mod(numAugmentations, numQuads);

    schedule = struct('k', {}, 'count', {});
    for k = 1:numQuads
        % Distribute remainder to lower k values first (more variety)
        count = baseCount + (k <= remainder);
        if count > 0
            schedule(end+1) = struct('k', k, 'count', count); %#ok<AGROW>
        end
    end
end

function filtered = filter_ellipse_map(ellipseMap, selectedQuads, paperBase)
    % Filter ellipse map to include only entries for selected quads
    %
    % Inputs:
    %   ellipseMap - containers.Map with keys like 'paperBase#concentration'
    %   selectedQuads - Struct array of selected quads with .concentration field
    %   paperBase - Base name of the paper image
    %
    % Output:
    %   filtered - containers.Map containing only entries for selected quads

    filtered = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:numel(selectedQuads)
        key = sprintf('%s#%d', paperBase, selectedQuads(i).concentration);
        if ellipseMap.isKey(key)
            filtered(key) = ellipseMap(key);
        end
    end
end

function scaleFactor = sample_roi_scale(sizeCfg, bgWidth, bgHeight, bboxWidth, bboxHeight, objectCount)
    % Compute scale factor using dimension-based constraints
    %
    % Constraint: ROI's largest side is [minFrac, maxFrac] of image's smallest side
    % Object count scaling: More objects -> smaller max size (realistic photography)
    %
    % Inputs:
    %   sizeCfg - ROI size configuration struct with minFrac, maxFrac, fitMargin, countSensitivity
    %   bgWidth, bgHeight - Background image dimensions
    %   bboxWidth, bboxHeight - ROI bounding box dimensions (before scaling)
    %   objectCount - Number of objects being placed in this image

    % Guard against division by zero (degenerate bounding box)
    if bboxWidth <= 0 || bboxHeight <= 0
        scaleFactor = 1.0;
        return;
    end

    imgMinDim = min(bgWidth, bgHeight);
    roiMaxDim = max(bboxWidth, bboxHeight);

    % Apply object count scaling to maxFrac
    % More objects -> smaller effective max (models zooming out to fit all)
    % Floor at minFrac to preserve documented minimum size guarantee
    countScaling = 1 / (1 + sizeCfg.countSensitivity * (objectCount - 1));
    effectiveMaxFrac = max(sizeCfg.maxFrac * countScaling, sizeCfg.minFrac);

    % Dimension-based scale bounds
    minScaleDim = (sizeCfg.minFrac * imgMinDim) / roiMaxDim;
    maxScaleDim = (effectiveMaxFrac * imgMinDim) / roiMaxDim;

    % Fit constraint (ensure ROI fits within image with margin)
    maxScaleW = (bgWidth - 2 * sizeCfg.fitMargin) / bboxWidth;
    maxScaleH = (bgHeight - 2 * sizeCfg.fitMargin) / bboxHeight;
    maxScaleFit = min(maxScaleW, maxScaleH);

    % Final valid range
    effectiveMin = minScaleDim;
    effectiveMax = min(maxScaleDim, maxScaleFit);

    % Handle edge case where constraints conflict
    if effectiveMax < effectiveMin
        % Fit constraint is tighter than size constraint; use fit if positive
        scaleFactor = max(effectiveMax, 0.01);
        return;
    end

    % Log-uniform sampling for perceptually even size distribution
    logMin = log(effectiveMin);
    logMax = log(effectiveMax);
    scaleFactor = exp(logMin + rand() * (logMax - logMin));

    % Ensure valid result
    if ~isfinite(scaleFactor) || scaleFactor <= 0
        scaleFactor = effectiveMin;
    end
end

function [scaledImg, scaledVerts, scaledEllipses, scaledAlpha] = scale_quad_content(quadImg, vertices, ellipseList, scaleFactor, quadAlpha)
    % Scale quad content (image, vertices, ellipses, alpha) by scaleFactor
    %
    % Inputs:
    %   quadImg - The quad image (masked crop)
    %   vertices - 4x2 array of quad vertices (relative to bbox origin)
    %   ellipseList - Array of ellipse structs with .center, .semiMajor, .semiMinor
    %   scaleFactor - Scale factor to apply
    %   quadAlpha - (optional) Alpha mask for compositing (single, 0-1 range)
    %
    % Outputs:
    %   scaledImg - Scaled image
    %   scaledVerts - Scaled vertices
    %   scaledEllipses - Scaled ellipse annotations
    %   scaledAlpha - Scaled alpha mask (if provided)

    hasAlpha = nargin >= 5 && ~isempty(quadAlpha);

    if scaleFactor == 1.0 || abs(scaleFactor - 1.0) < 1e-6
        scaledImg = quadImg;
        scaledVerts = vertices;
        scaledEllipses = ellipseList;
        if hasAlpha
            scaledAlpha = quadAlpha;
        else
            scaledAlpha = [];
        end
        return;
    end

    [h, w, ~] = size(quadImg);
    newH = max(1, round(h * scaleFactor));
    newW = max(1, round(w * scaleFactor));

    % Use bilinear for RGB to maintain pre-multiplied relationship with alpha.
    % Mathematical basis: bilinear interpolation naturally preserves pre-multiplied
    % state because blending RGB with black (zero padding at boundaries) produces
    % RGB_new = RGB * blend_factor, while alpha blends to alpha_new = alpha *
    % blend_factor, maintaining RGB_new = original * alpha_new.
    scaledImg = imresize(quadImg, [newH, newW], 'bilinear');

    % Scale alpha with bilinear interpolation to maintain smooth edge transitions
    if hasAlpha
        scaledAlpha = imresize(quadAlpha, [newH, newW], 'bilinear');
        % Clamp to [0,1] and cast to single for type consistency
        scaledAlpha = single(max(0, min(1, scaledAlpha)));
    else
        scaledAlpha = [];
    end

    % Scale vertices
    scaledVerts = vertices * scaleFactor;

    % Scale ellipse annotations proportionally
    scaledEllipses = ellipseList;
    for eIdx = 1:numel(scaledEllipses)
        scaledEllipses(eIdx).center = scaledEllipses(eIdx).center * scaleFactor;
        scaledEllipses(eIdx).semiMajor = scaledEllipses(eIdx).semiMajor * scaleFactor;
        scaledEllipses(eIdx).semiMinor = scaledEllipses(eIdx).semiMinor * scaleFactor;
    end
end

function hashVal = stable_string_hash(strInput)
    % Deterministic uint32 hash for reproducible RNG offsets.
    % Uses FNV-1a algorithm. Note: As with any 32-bit hash, collisions are
    % possible for different inputs - this is acceptable for seeding purposes
    % where occasional collision causes different (but still valid) results.

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

function stage2Map = init_stage2_map(coordPath, startFresh)
    % Initialize stage 2 coordinate map
    %
    % Inputs:
    %   coordPath - Path to existing coordinates.txt (may not exist)
    %   startFresh - If true, return empty map (ignore existing coordinates)
    %                If false, load and merge with existing coordinates
    %                Typically passed cfg.clearOutputOnRerun from caller
    %
    % When clearOutputOnRerun=true, we pass startFresh=true which creates an
    % empty map (output folders are cleared separately). When clearOutputOnRerun=false,
    % we pass startFresh=false which loads existing coordinates for merging.
    stage2Map = containers.Map('KeyType', 'char', 'ValueType', 'any');

    if nargin < 2
        startFresh = true;  % Default to fresh start
    end
    if startFresh
        return;
    end

    existing = read_quad_coordinates(coordPath);
    if ~isempty(existing)
        for k = 1:numel(existing)
            e = existing(k);
            key = sprintf('%s|%d', char(e.image), e.concentration);
            stage2Map(key) = e;
        end
    end
end

function stage3Map = init_stage3_map(coordPath, startFresh)
    % Initialize stage 3 coordinate map
    %
    % Inputs:
    %   coordPath - Path to existing coordinates.txt (may not exist)
    %   startFresh - If true, return empty map (ignore existing coordinates)
    %                If false, load and merge with existing coordinates
    %                Typically passed cfg.clearOutputOnRerun from caller
    %
    % See init_stage2_map for detailed semantics.
    stage3Map = containers.Map('KeyType', 'char', 'ValueType', 'any');

    if nargin < 2
        startFresh = true;  % Default to fresh start
    end
    if startFresh
        return;
    end

    existing = read_ellipse_coordinates(coordPath);
    if ~isempty(existing)
        for k = 1:numel(existing)
            e = existing(k);
            key = sprintf('%s|%d|%d', char(e.image), e.concentration, e.replicate);
            stage3Map(key) = e;
        end
    end
end

function update_stage2_map(stage2Map, coords)
    if isempty(coords)
        return;
    end
    % Strip extension from image name to match init_stage2_map key format
    % (coordinate_io.read_quad_coordinates strips extensions when reading)
    coordIO = coordinate_io();
    for i = 1:numel(coords)
        c = coords{i};
        strippedImage = coordIO.strip_image_extension(char(c.image));
        key = sprintf('%s|%d', strippedImage, c.concentration);
        if stage2Map.isKey(key)
            warning('augmentDataset:duplicateKey', 'Overwriting stage2 key: %s', key);
        end
        stage2Map(key) = c;
    end
end

function update_stage3_map(stage3Map, coords)
    if isempty(coords)
        return;
    end
    % Strip extension from image name to match init_stage3_map key format
    % (coordinate_io.read_ellipse_coordinates strips extensions when reading)
    coordIO = coordinate_io();
    for i = 1:numel(coords)
        c = coords{i};
        strippedImage = coordIO.strip_image_extension(char(c.image));
        key = sprintf('%s|%d|%d', strippedImage, c.concentration, c.replicate);
        if stage3Map.isKey(key)
            warning('augmentDataset:duplicateKey', 'Overwriting stage3 key: %s', key);
        end
        stage3Map(key) = c;
    end
end

function write_stage2_map(stage2Map, outputDir, filename)
    coordIO = coordinate_io();

    coordFolder = outputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, filename);

    keysArr = stage2Map.keys;
    numEntries = numel(keysArr);
    names = cell(numEntries, 1);
    nums = zeros(numEntries, 10);  % concentration + 8 coords + rotation

    for i = 1:numEntries
        e = stage2Map(keysArr{i});
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

    header = coordIO.QUAD_HEADER;
    writeFmt = coordIO.QUAD_WRITE_FMT;

    coordIO.atomicWriteCoordinates(coordPath, header, names, nums, writeFmt, coordFolder);
end

function write_stage3_map(stage3Map, outputDir, filename)
    if stage3Map.Count == 0
        return;
    end

    coordIO = coordinate_io();

    coordFolder = outputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, filename);

    keysArr = stage3Map.keys;
    numEntries = numel(keysArr);
    names = cell(numEntries, 1);
    nums = zeros(numEntries, 7);  % concentration, replicate, x, y, semiMajor, semiMinor, rotation

    for i = 1:numEntries
        e = stage3Map(keysArr{i});
        names{i} = char(e.image);
        nums(i, :) = [e.concentration, e.replicate, ...
                      e.center(1), e.center(2), e.semiMajor, e.semiMinor, e.rotation];
    end

    header = coordIO.ELLIPSE_HEADER;
    writeFmt = coordIO.ELLIPSE_WRITE_FMT;

    coordIO.atomicWriteCoordinates(coordPath, header, names, nums, writeFmt, coordFolder);
end

function entries = read_quad_coordinates(coordPath)
    % Read quadrilateral coordinates from stage 2 (2_micropads)
    % Delegates parsing to coordinate_io.parseQuadCoordinateFile and maps field names.
    %
    % Returns struct array with fields: .image, .concentration, .vertices, .rotation
    % (Maps from coordinate_io's .imageName to .image for backward compatibility)

    entries = struct('image', {}, 'concentration', {}, 'vertices', {}, 'rotation', {});

    if ~isfile(coordPath)
        return;
    end

    coordIO = coordinate_io();
    rawEntries = coordIO.parseQuadCoordinateFile(coordPath);

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

        % Validate ellipse axis constraint: semiMajor >= semiMinor
        major = rawEntries(i).semiMajorAxis;
        minor = rawEntries(i).semiMinorAxis;
        rot = rawEntries(i).rotationAngle;
        if minor > major
            % Swap axes and rotate 90 degrees to maintain ellipse shape
            entries(i).semiMajor = minor;
            entries(i).semiMinor = major;
            entries(i).rotation = rot + 90;
        else
            entries(i).semiMajor = major;
            entries(i).semiMinor = minor;
            entries(i).rotation = rot;
        end
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

function paperGroups = group_quads_by_image(quadEntries)
    % Group quads by source image name
    paperGroups = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:numel(quadEntries)
        p = quadEntries(i);
        % p.image is already a base name; avoid stripping dot-suffixes (e.g., Roboflow hashes)
        imgBase = char(p.image);

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

function valid = is_valid_quad(vertices, minArea)
    % Check if quadrilateral is valid (non-degenerate)
    area = polyarea(vertices(:,1), vertices(:,2));
    valid = area > minArea;
end

function positions = place_quads_nonoverlapping(quadBboxes, bgWidth, bgHeight, margin, minSpacing, maxRetries)
    % Place quads randomly with collision avoidance using spatial grid acceleration

    numQuads = numel(quadBboxes);
    positions = cell(numQuads, 1);
    placedBboxes = zeros(numQuads, 4);

    % Initialize spatial grid for O(1) collision detection
    gridCellSize = minSpacing;
    gridWidth = ceil(bgWidth / gridCellSize);
    gridHeight = ceil(bgHeight / gridCellSize);
    grid = cell(gridHeight, gridWidth);

    for i = 1:numQuads
        bbox = quadBboxes{i};
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
                    cellQuads = grid{cy, cx};
                    for k = 1:numel(cellQuads)
                        j = cellQuads(k);
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

        % If all retries failed, force placement anyway (may cause overlap)
        if ~placed
            if isempty(lastCandidate) || any(~isfinite(lastCandidate))
                x = max(0, (bgWidth - bbox.width) / 2);
                y = max(0, (bgHeight - bbox.height) / 2);
                warning('augmentDataset:forcedPlacement', ...
                        'Quad %d placement failed after %d retries. Forcing center placement (may overlap).', ...
                        i, maxRetries);
            else
                x = lastCandidate(1);
                y = lastCandidate(2);
                warning('augmentDataset:forcedPlacement', ...
                        'Quad %d placement failed after %d retries. Using last candidate position (may overlap).', ...
                        i, maxRetries);
            end
            positions{i} = struct('x', x, 'y', y);
            placedBboxes(i, :) = [x, y, x + bbox.width, y + bbox.height];
        end
    end
end

function [x, y] = random_top_left(bboxStruct, margin, widthVal, heightVal)
    % Generate random placement position for quad bounding box
    % Returns INTEGER coordinates to ensure pixel-perfect alignment between:
    % - composite_to_background target region sizing
    % - extract_quad_from_scene extraction region sizing
    % This preserves ellipse coordinate validity in the extracted crop.
    availX = max(0, widthVal - bboxStruct.width - 2 * margin);
    if availX > 0
        x = margin + rand() * availX;
    else
        x = max(0, (widthVal - bboxStruct.width) / 2);
    end
    x = max(0, min(x, widthVal - bboxStruct.width));
    x = round(x);  % Integer position for pixel-perfect alignment

    availY = max(0, heightVal - bboxStruct.height - 2 * margin);
    if availY > 0
        y = margin + rand() * availY;
    else
        y = max(0, (heightVal - bboxStruct.height) / 2);
    end
    y = max(0, min(y, heightVal - bboxStruct.height));
    y = round(y);  % Integer position for pixel-perfect alignment
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

% NOTE: Motion blur and edge feathering functions (apply_motion_blur, feather_quad_edges)
% have been moved to image_io.m. Access via cfg.imageIO.applyMotionBlur() and
% cfg.imageIO.featherQuadEdges().

