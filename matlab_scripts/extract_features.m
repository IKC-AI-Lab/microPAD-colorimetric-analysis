function extract_features(varargin)
    %% microPAD Colorimetric Analysis — Feature Extraction Tool
    %% Process parent polygon images with elliptical patch coordinates and export features
    %% Author: Veysel Y. Yilmaz
    %
    % Inputs (Name-Value pairs):
    % - 'chemical' char|string: Chemical name used in output naming.
    % - 'preset'   char|string: One of {'minimal','robust','full','custom'}.
    % - 'caching'  logical: Enable internal color-space caching.
    % - 'useDialog' logical: Show selection dialog for feature groups.
    % - 'features' struct: Feature group selection (required when preset='custom').
    % - 'paperTempK' double: Fallback paper color temperature (Kelvin).
    % - 'includeLabelInExcel' logical: Include Label column in export.
    % - 'trainTestSplit' logical: Export additional train_/test_ splits (default true).
    % - 'testSize' double: Fraction of rows for test split (0-1, default 0.3).
    % - 'splitGroupColumn' char|string: Column name for grouping during train/test split (default 'PhoneType' for phone-based grouping; use 'ImageName' for image-level splits).
    % - 'randomSeed' double: Random seed for reproducible train/test splits (default: random).
    %
    % Output:
    % - Writes a features file to 4_extract_features with columns defined by the registry.
    %   No return value.
    %
    % Workflow:
    % - Primary mode: Original images in <ORIGINAL_IMAGES_FOLDER>/phone/ (flat structure)
    %   Default: 1_dataset - images in 1_dataset/phone/*.jpg
    % - Legacy mode: Polygon crops in <ORIGINAL_IMAGES_FOLDER>/phone/con_*/
    %   Use 'augmented_2_micropads' for augmented data with con_* subfolders
    % - Uses elliptical patch coordinates from <COORDINATES_FOLDER>/phone/coordinates.txt
    %   Default: 3_elliptical_regions (coordinates relative to original image frame)
    % - Extracts features per patch and aggregates results by concentration replicate
    %
    % ROTATION SEMANTICS:
    %   This script reads ellipse coordinates from 3_elliptical_regions/coordinates.txt
    %   (8-column format). The rotationAngle field represents the ellipse's geometric
    %   orientation in the original (unrotated) image reference frame. The micropad UI
    %   rotation column from 2_micropads/coordinates.txt is NOT used by this script.
    %
    % Usage:
    %   extract_features()                    % Show dialog
    %   extract_features('preset', 'minimal') % Use 'minimal' feature group set
    %   extract_features('preset', 'robust')  % Use 'robust' feature group set
    %   extract_features('chemical', 'glucose') % Different chemical
    %   extract_features('useDialog', false)  % Skip dialog
    %   extract_features('preset','custom','features', S) % Use custom feature selection (no dialog)
    %   extract_features('paperTempK', 6000)             % Override fallback paper color temperature (Kelvin)
    %   extract_features('trainTestSplit', true, 'testSize', 0.25) % Save train_/test_ Excel files

%% ========================================================================
    %% EXPERIMENT CONFIGURATION CONSTANTS
    %% ========================================================================
    %
    % All experimental constants are centralized here for easy modification.
    % Change these values to adapt the script for different experimental setups
    % without hunting through the code.
    %
    
    % === DATASET AND FOLDER STRUCTURE ===
    % Primary mode: Process original images directly from 1_dataset
    % with ellipse coordinates relative to original (unrotated) image frame.
    % For augmented data, use 'augmented_2_micropads' and 'augmented_3_elliptical_regions'.
    ORIGINAL_IMAGES_FOLDER = '1_dataset';           % Original images (flat structure)
    COORDINATES_FOLDER = '3_elliptical_regions';    % Ellipse coordinate data
    OUTPUT_FOLDER = '4_extract_features';           % Output features folder

    % === NAMING CONVENTIONS ===
    CONC_FOLDER_PREFIX = 'con_';                             % Concentration folder prefix (e.g., con_0)
    COORDINATE_FILENAME = 'coordinates.txt';                 % Coordinate file name

    % === EXPERIMENT PARAMETERS ===
    DEFAULT_CHEMICAL = 'lactate';
    DEFAULT_PAPER_TEMP_K = 5500;                        % Fallback paper color temperature (Kelvin)
    MIN_PATCH_PIXELS = 50;                              % Minimum valid pixels required in patch mask
    % Note: Concentration/replicate counts are inferred from folders; no fixed constants needed.

    % === FEATURE EXTRACTION THRESHOLDS ===
    COLOR_RATIO_RELATIVE_THRESHOLD = 1e-2;              % Fraction of peak channel intensity considered unstable
    COLOR_RATIO_MIN_ABSOLUTE_THRESHOLD = 1e-3;          % Absolute guard for already-normalized inputs
    ENTROPY_HISTOGRAM_BINS = 256;                       % Standard 8-bit image histogram bins
    MAX_LAB_SHIFT = 100;                                % Maximum reasonable a*/b* shift vs. paper white point

    % === COLOR TEMPERATURE ESTIMATION (McCamy's CCT Approximation) ===
    % Reference: Wyszecki & Stiles (2000), Color Science: Concepts and Methods
    MCCAMY_X_REF = 0.3320;                              % CIE D65 x chromaticity reference
    MCCAMY_Y_REF = 0.1858;                              % CIE D65 y chromaticity reference
    MCCAMY_COEFF_3 = 437;                               % Cubic coefficient
    MCCAMY_COEFF_2 = 3601;                              % Quadratic coefficient
    MCCAMY_COEFF_1 = 6861;                              % Linear coefficient
    MCCAMY_COEFF_0 = 5517;                              % Constant term
    CCT_MIN_KELVIN = 2500;                              % Minimum plausible color temperature
    CCT_MAX_KELVIN = 10000;                             % Maximum plausible color temperature

    % === R/B RATIO FALLBACK THRESHOLDS (for CCT estimation) ===
    RB_RATIO_TUNGSTEN_THRESHOLD = 1.4;                  % R/B > 1.4: tungsten (~3200K)
    RB_RATIO_MIXED_THRESHOLD = 1.1;                     % R/B > 1.1: mixed (~4000K)
    CCT_TUNGSTEN_K = 3200;                              % Tungsten light color temperature
    CCT_MIXED_K = 4000;                                 % Mixed light color temperature
    CCT_DAYLIGHT_K = 6500;                              % Daylight color temperature

    % === BATCH SIZE THRESHOLDS ===
    SMALL_DATASET_THRESHOLD = 10;                       % Files below this: process all at once
    MEDIUM_DATASET_THRESHOLD = 50;                      % Files below this: use 3-batch split
    LARGE_DATASET_THRESHOLD = 500;                      % Files above this: scale down batch size
    MEDIUM_DATASET_BATCH_DIVISOR = 3;                   % Number of batches for medium datasets

    % === PERFORMANCE SETTINGS ===
    BASE_BATCH_SIZE = 10;                               % Reduced batch size (fewer images to process)
    MEMORY_THRESHOLD = 0.8;                             % Memory usage threshold (0.1-0.95)
    
    % === OUTPUT FORMATTING ===
    OUTPUT_DECIMALS = 6;                                % Decimal places in output
    OUTPUT_EXTENSION = '.xlsx';                         % Output file extension (.xlsx/.csv)
    
    % === DEBUG / VISUALIZATION ===
    DEBUG_VISUALIZE_MASKS = false;                       % Show patch mask overlay during processing
    DEBUG_SAMPLE_PROBABILITY = 0.05;                     % Fraction of images to visualize
    DEBUG_SAVE_TO_DISK = true;                           % Persist debug artifacts to disk
    
    %% ========================================================================

    % All helper functions are now inlined as nested functions for standalone execution

    p = inputParser;
    addParameter(p, 'chemical', DEFAULT_CHEMICAL, @(x) validateattributes(x, {'char', 'string'}, {'nonempty'}));
    addParameter(p, 'preset', 'robust', @(x) (ischar(x) || isstring(x)));
    addParameter(p, 'caching', true, @(x) validateattributes(x, {'logical'}, {'scalar'}));
    addParameter(p, 'useDialog', isempty(varargin), @(x) validateattributes(x, {'logical'}, {'scalar'}));
    addParameter(p, 'features', [], @(x) isempty(x) || (isstruct(x) && isscalar(x)));
    addParameter(p, 'paperTempK', DEFAULT_PAPER_TEMP_K, @(x) validateattributes(x, {'double','single','numeric'}, {'scalar','finite','positive'}));
    addParameter(p, 'includeLabelInExcel', true, @(x) (islogical(x) && isscalar(x)));
    addParameter(p, 'trainTestSplit', true, @(x) (islogical(x) && isscalar(x)));
    addParameter(p, 'testSize', 0.3, @(x) validateattributes(x, {'double','single'}, {'scalar','>',0,'<',1}));
    addParameter(p, 'splitGroupColumn', 'PhoneType', @(x) (ischar(x) && ~isempty(x)) || (isstring(x) && isscalar(x) && all(strlength(x) > 0)));
    addParameter(p, 'randomSeed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 0));
    parse(p, varargin{:});
    
    CHEMICAL_NAME = char(p.Results.chemical);
    feature_preset = char(p.Results.preset);
    enable_caching = p.Results.caching;
    use_dialog = p.Results.useDialog;
    features_input = p.Results.features;
    train_test_split = logical(p.Results.trainTestSplit);
    test_size = double(p.Results.testSize);
    include_label_excel = logical(p.Results.includeLabelInExcel);
    split_group_column = char(p.Results.splitGroupColumn);
    random_seed = p.Results.randomSeed;

    feature_preset = validatestring(feature_preset, {'minimal','robust','full','custom'});

    if use_dialog
        fprintf('Opening feature group selection dialog...\n');
        [dialogConfig, userCanceled] = showFeatureSelectionDialog(feature_preset, CHEMICAL_NAME);
        if userCanceled
            fprintf('Feature extraction canceled by user.\n');
            return;
        end
        
        CHEMICAL_NAME = dialogConfig.chemical;
        feature_preset = dialogConfig.preset;
        custom_features = dialogConfig.features;
        fprintf('Dialog selection: %s preset with %d feature groups\n', feature_preset, ...
                sum(structfun(@(x) x, custom_features)));
    else
        if strcmp(feature_preset, 'custom')
            if isempty(features_input)
                error('extract_features:customPresetMissingFeatures', ...
                      'When preset is ''custom'', provide a features struct via ''features'' or set useDialog=true.');
            end
            custom_features = features_input;
        else
            if ~isempty(features_input)
                warning('extract_features:featuresIgnored', ...
                        'Ignoring ''features'' because preset is ''%s''. Use preset ''custom'' to apply custom selection.', feature_preset);
            end
            custom_features = [];
        end
    end

    %% Add helper_scripts to path and load utility modules
    scriptDir = fileparts(mfilename('fullpath'));
    helperDir = fullfile(scriptDir, 'helper_scripts');
    if exist(helperDir, 'dir')
        addpath(helperDir);
    end

    % Load shared utility modules for coordinate I/O, image loading, and mask creation
    coordIO = coordinate_io();
    imageIO = image_io();
    maskUtils = mask_utils();
    pathUtils = path_utils();
    colorAnalysis = color_analysis();
    featPipe = feature_pipeline();

    cfg = createConfiguration(ORIGINAL_IMAGES_FOLDER, COORDINATES_FOLDER, OUTPUT_FOLDER, ...
                                        CHEMICAL_NAME, feature_preset, enable_caching, custom_features, ...
                                        MIN_PATCH_PIXELS, ...
                                        BASE_BATCH_SIZE, MEMORY_THRESHOLD, OUTPUT_DECIMALS, OUTPUT_EXTENSION, ...
                                        include_label_excel, train_test_split, test_size, split_group_column, random_seed, ...
                                        DEBUG_VISUALIZE_MASKS, DEBUG_SAMPLE_PROBABILITY, DEBUG_SAVE_TO_DISK, ...
                                        CONC_FOLDER_PREFIX, COORDINATE_FILENAME, p.Results.paperTempK, ...
                                        COLOR_RATIO_RELATIVE_THRESHOLD, COLOR_RATIO_MIN_ABSOLUTE_THRESHOLD, ...
                                        ENTROPY_HISTOGRAM_BINS, SMALL_DATASET_THRESHOLD, MEDIUM_DATASET_THRESHOLD, ...
                                        LARGE_DATASET_THRESHOLD, MEDIUM_DATASET_BATCH_DIVISOR, MAX_LAB_SHIFT, ...
                                        MCCAMY_X_REF, MCCAMY_Y_REF, MCCAMY_COEFF_3, MCCAMY_COEFF_2, MCCAMY_COEFF_1, MCCAMY_COEFF_0, ...
                                        CCT_MIN_KELVIN, CCT_MAX_KELVIN, RB_RATIO_TUNGSTEN_THRESHOLD, RB_RATIO_MIXED_THRESHOLD, ...
                                        CCT_TUNGSTEN_K, CCT_MIXED_K, CCT_DAYLIGHT_K);

    % Store utility module handles in configuration for use throughout processing
    cfg.modules = struct();
    cfg.modules.coordIO = coordIO;
    cfg.modules.imageIO = imageIO;
    cfg.modules.maskUtils = maskUtils;
    cfg.modules.pathUtils = pathUtils;
    cfg.modules.colorAnalysis = colorAnalysis;
    cfg.modules.featPipe = featPipe;

    displayConfigurationSummary(cfg, feature_preset);
    
    try
        processAllFolders(cfg);
        fprintf('>> Feature extraction completed successfully!\n');
        displayFeatureSummary(cfg);
        
    catch ME
        handleError(ME, cfg);
    end
end

function registry = getFeatureRegistry()
    %% Get centralized feature registry (delegates to feature_pipeline helper)
    featPipe = feature_pipeline();
    registry = featPipe.registry.get();
end

function cfg = createConfiguration(originalImagesFolder, coordinatesFolder, outputFolder, ...
                                           chemicalName, featurePreset, enableCaching, customFeatures, ...
                                           minPatchPixels, ...
                                           baseBatchSize, memoryThreshold, outputDecimals, outputExtension, ...
                                           includeLabelInExcel, trainTestSplit, testSize, splitGroupColumn, randomSeed, ...
                                           debugVisualizeMasks, debugSampleProbability, debugSaveToDisk, ...
                                           concFolderPrefix, coordinateFileName, defaultPaperTempK, ...
                                           colorRatioRelativeThreshold, colorRatioMinAbsoluteThreshold, ...
                                           entropyHistogramBins, smallDatasetThreshold, mediumDatasetThreshold, ...
                                           largeDatasetThreshold, mediumDatasetBatchDivisor, maxLabShift, ...
                                           mccamyXRef, mccamyYRef, mccamyCoeff3, mccamyCoeff2, mccamyCoeff1, mccamyCoeff0, ...
                                           cctMinKelvin, cctMaxKelvin, rbRatioTungstenThreshold, rbRatioMixedThreshold, ...
                                           cctTungstenK, cctMixedK, cctDaylightK)
    %% Create configuration structure
    
    % Validate inputs
    validateattributes(originalImagesFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'originalImagesFolder');
    validateattributes(coordinatesFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'coordinatesFolder');
    validateattributes(outputFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'outputFolder');
    validateattributes(chemicalName, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'chemicalName');
    validateattributes(debugVisualizeMasks, {'logical','numeric'}, {'scalar'}, 'createConfiguration', 'debugVisualizeMasks');
    validateattributes(debugSampleProbability, {'double','single'}, {'scalar','>=',0,'<=',1}, 'createConfiguration', 'debugSampleProbability');
    validateattributes(debugSaveToDisk, {'logical','numeric'}, {'scalar'}, 'createConfiguration', 'debugSaveToDisk');
    
    debugVisualizeMasks = logical(debugVisualizeMasks);
    debugSaveToDisk = logical(debugSaveToDisk);
    debugSampleProbability = double(debugSampleProbability);
    
    % Core processing constants
    cfg.chemicalName = char(chemicalName);
    cfg.enableCaching = enableCaching;
    cfg.featurePreset = char(featurePreset);
    
    % Feature group configuration
    if ~isempty(customFeatures)
        cfg.features = completeFeatureSelection(customFeatures);
    else
        cfg.features = getFeatureConfiguration(featurePreset);
    end
    
    % Path/naming configuration
    cfg = addPathConfiguration(cfg, originalImagesFolder, coordinatesFolder, outputFolder);
    if ~isempty(concFolderPrefix)
        cfg.concFolderPrefix = char(concFolderPrefix);
    else
        cfg.concFolderPrefix = 'con_';
    end
    if ~isempty(coordinateFileName)
        cfg.coordinateFileName = char(coordinateFileName);
    else
        cfg.coordinateFileName = 'coordinates.txt';
    end
    
    % Output settings
    cfg.output = createOutputConfiguration(chemicalName, outputDecimals, outputExtension, includeLabelInExcel, trainTestSplit, testSize, splitGroupColumn, randomSeed);
    
    % microPAD processing parameters (masking configuration)
    cfg.upad = struct('minPatchPixels', minPatchPixels, ...
                      'patchMaskMarginFactor', 0.20, ...           % 20% halo around patch ellipses
                      'minHaloPx', 2, ...                           % absolute minimum halo in pixels
                      'cleanupMinAreaFraction', 0.001, ...           % bwareaopen fraction
                      'minPaperPixelsAbsolute', 50, ...              % relaxed absolute min
                      'minPaperFraction', 0.0002, ...               % relaxed fractional min
                      'blackFillCutoff', 5);                         % ignore <= this intensity when estimating threshold (uint8 scale)
    
    % Performance settings
    cfg.performance = struct('cacheColorSpaces', enableCaching, ...
                             'adaptiveBatchSize', true, 'baseBatchSize', baseBatchSize, ...
                             'clearTempOnError', true, 'memoryThreshold', memoryThreshold);
    
    % Default/fallback constants
    if isempty(defaultPaperTempK)
        cfg.defaults = struct('paperTempK', 5500);
    else
        cfg.defaults = struct('paperTempK', defaultPaperTempK);
    end

    % Feature extraction thresholds
    cfg.thresholds = struct('colorRatioRelative', colorRatioRelativeThreshold, ...
                            'colorRatioMinAbsolute', colorRatioMinAbsoluteThreshold, ...
                            'entropyHistogramBins', entropyHistogramBins, ...
                            'smallDataset', smallDatasetThreshold, ...
                            'mediumDataset', mediumDatasetThreshold, ...
                            'largeDataset', largeDatasetThreshold, ...
                            'mediumDatasetBatchDivisor', mediumDatasetBatchDivisor, ...
                            'maxLabShift', maxLabShift);

    % Color temperature estimation constants (McCamy's CCT approximation)
    cfg.colorTemp = struct('mccamyXRef', mccamyXRef, ...
                           'mccamyYRef', mccamyYRef, ...
                           'mccamyCoeff3', mccamyCoeff3, ...
                           'mccamyCoeff2', mccamyCoeff2, ...
                           'mccamyCoeff1', mccamyCoeff1, ...
                           'mccamyCoeff0', mccamyCoeff0, ...
                           'cctMinKelvin', cctMinKelvin, ...
                           'cctMaxKelvin', cctMaxKelvin, ...
                           'rbRatioTungstenThreshold', rbRatioTungstenThreshold, ...
                           'rbRatioMixedThreshold', rbRatioMixedThreshold, ...
                           'cctTungstenK', cctTungstenK, ...
                           'cctMixedK', cctMixedK, ...
                           'cctDaylightK', cctDaylightK);

    % Debug controls (disabled by default)
    cfg.debug = struct('visualizeMasks', debugVisualizeMasks, ...
                       'sampleProb', debugSampleProbability, ...
                       'saveToDisk', debugSaveToDisk);
    
    % Initialize tNValue (will be set during processing)
    cfg.tNValue = 't0';  % Default fallback
end

function featureConfig = getFeatureConfiguration(presetName)
    %% Get feature configuration for preset (delegates to feature_pipeline helper)
    featPipe = feature_pipeline();
    featureConfig = featPipe.registry.getConfiguration(presetName);
end

function out = completeFeatureSelection(in)
    %% Complete partial feature selection (delegates to feature_pipeline helper)
    featPipe = feature_pipeline();
    out = featPipe.registry.completeSelection(in);
end

function featureStruct = extractFeaturesFromCoordinates(originalImage, labOriginal, patch, paperStats, imageName, phoneName, cfg, imageColorCache)
    %% Extract features from patch area defined by coordinates

    % Input validation
    validateattributes(originalImage, {'uint8'}, {'3d', 'size', [NaN, NaN, 3]}, mfilename, 'originalImage');
    validateattributes(patch, {'struct'}, {'scalar'}, mfilename, 'patch');
    requiredFields = {'xCenter', 'yCenter', 'semiMajorAxis', 'semiMinorAxis', 'rotationAngle', 'concentration', 'patchID', 'replicate'};
    if ~all(isfield(patch, requiredFields))
        error('extract_features:invalidPatch', 'patch struct missing required fields');
    end

    if nargin < 8 || isempty(imageColorCache)
        imageColorCache = struct();
    end

    % Extract maskUtils from cfg modules (required for local functions)
    maskUtils = cfg.modules.maskUtils;

    featureStruct = [];

    try
        [height, width, ~] = size(originalImage);
        patchMask = maskUtils.createEllipseMask([height, width], patch.xCenter, patch.yCenter, ...
                                                 patch.semiMajorAxis, patch.semiMinorAxis, patch.rotationAngle);
        % Calculate bounding box from mask
        rows = find(any(patchMask, 2));
        cols = find(any(patchMask, 1));
        if ~isempty(rows) && ~isempty(cols)
            bbox = [rows(1), rows(end), cols(1), cols(end)];
        else
            bbox = [1, height, 1, width];
        end

        if sum(patchMask(:)) < cfg.upad.minPatchPixels
            warning('extract_features:patchPixels', 'Insufficient patch pixels (%d) for %s patch %d', sum(patchMask(:)), imageName, patch.patchID);
            return;
        end

        % Initialize feature structure with metadata (phone and image context)
        featureStruct = struct( ...
            'PhoneType', phoneName, ...
            'ImageName', char(imageName), ...
            'Concentration', patch.concentration, ...
            'PatchID', patch.patchID, ...
            'Replicate', patch.replicate);

        rowRange = bbox(1):bbox(2);
        colRange = bbox(3):bbox(4);

        localMask = patchMask(rowRange, colRange);
        patchRegion = originalImage(rowRange, colRange, :);
        if any(localMask(:))
            patchRegion(~repmat(localMask, [1, 1, size(patchRegion, 3)])) = 0;
        else
            patchRegion(:) = 0;
        end

        % CENTRALIZED COLOR SPACE MANAGEMENT
        colorData = precomputeAllColorSpaces(patchRegion, localMask, cfg.features, cfg.performance.cacheColorSpaces, imageColorCache, rowRange, colRange);

        % FEATURE EXTRACTION PIPELINE (reuse existing functions)
        featureStruct = extractAllFeatures(featureStruct, patchRegion, colorData, cfg);

        % Add coordinate-specific normalization features (reuse precomputed Lab)
        featureStruct = addCoordinateBasedNormalizationFeatures(featureStruct, originalImage, patchMask, paperStats, cfg, labOriginal);

        % Add concentration label
        featureStruct.Label = patch.concentration;

    catch ME
        warning('extract_features:coordinateFeatureExtraction', 'Error extracting features from coordinates for %s patch %d: %s', ...
               imageName, patch.patchID, ME.message);
    end
end


function colorData = precomputeAllColorSpaces(image, mask, featureConfig, enableCaching, globalCache, rowRange, colRange)
    %% Color space conversions and caching with optional shared image cache

    if nargin < 2 || isempty(mask)
        mask = createValidPixelMask(image);
    end
    if nargin < 5 || isempty(globalCache)
        globalCache = struct();
    end
    if nargin < 6 || isempty(rowRange)
        rowRange = 1:size(image, 1);
    end
    if nargin < 7 || isempty(colRange)
        colRange = 1:size(image, 2);
    end

    colorData = struct('mask', mask, 'image', image);

    try
        if ~enableCaching
            return;
        end

        requirements = determineColorSpaceRequirements(featureConfig);
        doubleImg = [];

        if requirements.needsRGB
            if isfield(globalCache, 'rgb') && ~isempty(globalCache.rgb)
                colorData.rgb = globalCache.rgb(rowRange, colRange, :);
            else
                colorData.rgb = im2double(image);
            end
            doubleImg = colorData.rgb;
        end

        if requirements.needsHSV
            if isfield(globalCache, 'hsv') && ~isempty(globalCache.hsv)
                colorData.hsv = globalCache.hsv(rowRange, colRange, :);
            else
                if isempty(doubleImg)
                    doubleImg = im2double(image);
                end
                colorData.hsv = rgb2hsv(doubleImg);
            end
        end

        if requirements.needsLab
            if isfield(globalCache, 'lab') && ~isempty(globalCache.lab)
                colorData.lab = globalCache.lab(rowRange, colRange, :);
            else
                if isempty(doubleImg)
                    doubleImg = im2double(image);
                end
                colorData.lab = rgb2lab(doubleImg);
            end
        end

        if requirements.needsGray
            if isfield(globalCache, 'gray') && ~isempty(globalCache.gray)
                colorData.gray = globalCache.gray(rowRange, colRange);
            else
                if isempty(doubleImg)
                    doubleImg = im2double(image);
                end
                colorData.gray = rgb2gray(doubleImg);
            end
        end

    catch ME
        warning('extract_features:colorSpaceCache', 'Error pre-computing color spaces: %s', ME.message);
        colorData = struct('mask', mask, 'image', image);
    end
end

function requirements = determineColorSpaceRequirements(featureConfig)
    %% Map enabled features to required color spaces

    requirements = struct();
    requirements.needsRGB = isConfigEnabled(featureConfig, 'RGB') || isConfigEnabled(featureConfig, 'ColorUniformity');
    requirements.needsHSV = isConfigEnabled(featureConfig, 'HSV') || isConfigEnabled(featureConfig, 'ColorUniformity') || ...
                          isConfigEnabled(featureConfig, 'ConcentrationMetrics') || isConfigEnabled(featureConfig, 'RobustColorStats') || ...
                          isConfigEnabled(featureConfig, 'RadialProfile');
    requirements.needsLab = isConfigEnabled(featureConfig, 'Lab') || isConfigEnabled(featureConfig, 'ColorUniformity') || ...
                          isConfigEnabled(featureConfig, 'ColorGradients') || isConfigEnabled(featureConfig, 'SpatialDistribution') || ...
                          isConfigEnabled(featureConfig, 'ConcentrationMetrics') || isConfigEnabled(featureConfig, 'IlluminantInvariant') || ...
                          isConfigEnabled(featureConfig, 'RobustColorStats') || isConfigEnabled(featureConfig, 'RadialProfile');
    requirements.needsGray = isConfigEnabled(featureConfig, 'GLCM') || isConfigEnabled(featureConfig, 'Entropy') || ...
                          isConfigEnabled(featureConfig, 'ColorGradients') || isConfigEnabled(featureConfig, 'FrequencyEnergy');
end

function tf = isConfigEnabled(featureConfig, fieldName)
    tf = isfield(featureConfig, fieldName) && logical(featureConfig.(fieldName));
end

function imageCache = buildImageColorCache(originalImage, featureConfig)
    %% Build color space cache for image

    imageCache = struct();
    doubleImg = im2double(originalImage);

    % Lab is always required for normalization features downstream
    imageCache.lab = rgb2lab(doubleImg);

    requirements = determineColorSpaceRequirements(featureConfig);

    if requirements.needsRGB
        imageCache.rgb = doubleImg;
    end

    if requirements.needsHSV
        imageCache.hsv = rgb2hsv(doubleImg);
    end

    if requirements.needsGray
        imageCache.gray = rgb2gray(doubleImg);
    end
end

function featureStruct = extractAllFeatures(featureStruct, image, colorData, cfg)
    %% Extract all enabled features

    % Get feature registry
    registry = getFeatureRegistry();
    
    % Extract basic color space features using registry
    for i = 1:length(registry.basicFeatureGroups)
        featureName = registry.basicFeatureGroups{i};
        if cfg.features.(featureName)
            features = struct();  % Initialize to empty struct for safety
            switch featureName
                case 'RGB'
                    features = safeExtract(@() extractBasicRGBFeatures(image, colorData), ...
                                         @() getDefaultFeatures(featureName), featureName);
                case 'HSV'
                    features = safeExtract(@() extractBasicHSVFeatures(image, colorData), ...
                                         @() getDefaultFeatures(featureName), featureName);
                case 'Lab'
                    features = safeExtract(@() extractBasicLabFeatures(image, colorData), ...
                                         @() getDefaultFeatures(featureName), featureName);
                case 'Skewness'
                    features = safeExtract(@() extractSkewnessFeatures(image, colorData), ...
                                         @() getDefaultFeatures(featureName), featureName);
                case 'Kurtosis'
                    features = safeExtract(@() extractKurtosisFeatures(image, colorData), ...
                                         @() getDefaultFeatures(featureName), featureName);
                case 'GLCM'
                    features = safeExtract(@() extractGLCMFeatures(image, colorData), ...
                                         @() getDefaultFeatures(featureName), featureName);
                case 'Entropy'
                    features = safeExtract(@() extractEntropyFeatures(image, colorData, cfg), ...
                                         @() getDefaultFeatures(featureName), featureName);
            end
            featureStruct = mergeStructs(featureStruct, features);
        end
    end
    
    % Extract extended features using registry - dynamic function mapping
    if isfield(colorData, 'mask') && ~isempty(colorData.mask)
        patchMask = colorData.mask;
    else
        patchMask = createValidPixelMask(image);
    end

    advancedFeatureFunctions = containers.Map(...
        {'ColorRatios', 'Chromaticity', 'IlluminantInvariant', ...
         'ColorUniformity', 'RobustColorStats', 'ColorGradients', ...
         'SpatialDistribution', 'RadialProfile', 'ConcentrationMetrics', ...
         'LogarithmicColorTransforms', 'FrequencyEnergy'}, ...
        {@() extractColorRatioFeatures(image, patchMask, cfg), ...
         @() extractChromaticityFeatures(image, patchMask), ...
         @() extractIlluminantInvariantFeatures(image, patchMask), ...
         @() extractColorUniformityFeatures(image, colorData), ...
         @() extractRobustColorStats(image, colorData), ...
         @() extractColorGradientFeatures(image, colorData), ...
         @() extractSpatialDistributionFeatures(image, colorData), ...
         @() extractRadialProfileFeatures(image, colorData), ...
         @() extractConcentrationMetrics(image, colorData), ...
         @() extractLogarithmicColorTransformFeatures(image, patchMask), ...
         @() extractFrequencyEnergyFeatures(image, colorData)});
    
    for i = 1:length(registry.advancedFeatureGroups)
        featureName = registry.advancedFeatureGroups{i};
        if isKey(advancedFeatureFunctions, featureName)
            extractFunc = advancedFeatureFunctions(featureName);
        else
            continue; % Skip unknown advanced features
        end
        
        if cfg.features.(featureName)
            features = safeExtract(extractFunc, @() getDefaultFeatures(featureName), featureName);
            featureStruct = mergeStructs(featureStruct, features);
        end
    end
end

function result = safeExtract(extractFunc, defaultFunc, errorContext)
    %% Error handling wrapper for feature extraction
    try
        result = extractFunc();
    catch ME
        warning('extract_features:safeExtract', 'Context: %s | Error: %s', errorContext, ME.message);
        result = defaultFunc();
    end
end

%% Statistical calculations
function stats = calculateChannelStats(channelData, mask, statTypes)
    %% Calculate channel statistics
    
    stats = struct();
    
    if ~any(mask(:))
        % Return zeros for empty mask
        for i = 1:length(statTypes)
            stats.(statTypes{i}) = 0;
        end
        return;
    end
    
    pixels = channelData(mask);
    
    % Vectorized computation - calculate all requested statistics at once
    if any(strcmp(statTypes, 'mean'))
        stats.mean = mean(pixels);
    end
    if any(strcmp(statTypes, 'std'))
        stats.std = std(pixels);
    end
    if any(strcmp(statTypes, 'skewness'))
        stats.skew = skewness(pixels);
    end
    if any(strcmp(statTypes, 'kurtosis'))
        stats.kurt = kurtosis(pixels);
    end
    if any(strcmp(statTypes, 'min'))
        stats.min = min(pixels);
    end
    if any(strcmp(statTypes, 'max'))
        stats.max = max(pixels);
    end
end

function [hsvImg, labImg] = getColorSpacesFromCache(image, colorData)
    %% Get HSV and Lab color spaces from cache or compute if needed

    if isfield(colorData, 'hsv')
        hsvImg = colorData.hsv;
    else
        hsvImg = rgb2hsv(im2double(image));
    end

    if isfield(colorData, 'lab')
        labImg = colorData.lab;
    else
        labImg = rgb2lab(image);
    end
end

%% Basic feature extraction functions
function features = extractBasicRGBFeatures(image, colorData)
    %% Extract RGB channel means
    
    if isfield(colorData, 'rgb')
        img = colorData.rgb;
    else
        img = im2double(image);
    end
    
    mask = colorData.mask;
    features = struct();
    
    if any(mask(:))
        channels = {'R', 'G', 'B'};
        for i = 1:3
            channelStats = calculateChannelStats(img(:, :, i), mask, {'mean'});
            features.(channels{i}) = channelStats.mean;
        end
    else
        features = struct('R', 0, 'G', 0, 'B', 0);
    end
end

function features = extractBasicHSVFeatures(image, colorData)
    %% Extract HSV features using centralized color data
    
    if isfield(colorData, 'hsv')
        hsvImg = colorData.hsv;
    else
        hsvImg = rgb2hsv(image);
    end
    
    mask = colorData.mask;
    features = struct('H', 0, 'S', 0, 'V', 0);
    
    if any(mask(:))
        channels = {'H', 'S', 'V'};
        scale_factors = [360, 100, 100];  % Degrees, percentage, percentage
        
        for i = 1:3
            channelStats = calculateChannelStats(hsvImg(:, :, i), mask, {'mean'});
            features.(channels{i}) = channelStats.mean * scale_factors(i);
        end
    end
end

function features = extractBasicLabFeatures(image, colorData)
    %% Extract Lab features using centralized color data
    
    if isfield(colorData, 'lab')
        labImg = colorData.lab;
    else
        labImg = rgb2lab(image);
    end
    
    mask = colorData.mask;
    features = struct('L', 0, 'a', 0, 'b', 0);
    
    if any(mask(:))
        channels = {'L', 'a', 'b'};
        for i = 1:3
            channelStats = calculateChannelStats(labImg(:, :, i), mask, {'mean'});
            features.(channels{i}) = channelStats.mean;
        end
    end
end

function features = extractSkewnessFeatures(image, colorData)
    %% Extract skewness statistics across color channels
    
    features = struct();
    % Input validation
    validateattributes(image, {'uint8','double'}, {'3d','size',[NaN,NaN,3]}, 'extractSkewnessFeatures', 'image');
    if ~isstruct(colorData) || ~isfield(colorData, 'mask')
        error('extract_features:invalidColorData', 'extractSkewnessFeatures requires colorData with a logical mask');
    end
    mask = colorData.mask;
    
    if ~any(mask(:))
        features = getDefaultFeatures('Skewness');
        return;
    end

    % Get color spaces from cache or compute
    [hsvImg, labImg] = getColorSpacesFromCache(image, colorData);

    % Calculate skewness for each channel
    channels = { ...
        {'R', double(image(:,:,1))}, {'G', double(image(:,:,2))}, {'B', double(image(:,:,3))}; ...
        {'H', hsvImg(:,:,1) * 360}, {'S', hsvImg(:,:,2) * 100}, {'V', hsvImg(:,:,3) * 100}; ...
        {'L', labImg(:,:,1)}, {'a', labImg(:,:,2)}, {'b', labImg(:,:,3)} ...
    };
    
    for i = 1:length(channels)
        channelName = [channels{i}{1} '_skew'];
        channelData = channels{i}{2};
        stats = calculateChannelStats(channelData, mask, {'skewness'});
        features.(channelName) = stats.skew;
    end
end

function features = extractKurtosisFeatures(image, colorData)
    %% Extract kurtosis across color channels

    features = struct();
    % Input validation
    validateattributes(image, {'uint8','double'}, {'3d','size',[NaN,NaN,3]}, 'extractKurtosisFeatures', 'image');
    if ~isstruct(colorData) || ~isfield(colorData, 'mask')
        error('extract_features:invalidColorData', 'extractKurtosisFeatures requires colorData with a logical mask');
    end
    mask = colorData.mask;

    if ~any(mask(:))
        features = getDefaultFeatures('Kurtosis');
        return;
    end

    % Get color spaces from cache or compute
    [hsvImg, labImg] = getColorSpacesFromCache(image, colorData);

    % Calculate kurtosis for each channel
    channels = { ...
        {'R', double(image(:,:,1))}, {'G', double(image(:,:,2))}, {'B', double(image(:,:,3))}; ...
        {'H', hsvImg(:,:,1) * 360}, {'S', hsvImg(:,:,2) * 100}, {'V', hsvImg(:,:,3) * 100}; ...
        {'L', labImg(:,:,1)}, {'a', labImg(:,:,2)}, {'b', labImg(:,:,3)} ...
    };
    
    for i = 1:length(channels)
        channelName = [channels{i}{1} '_kurto'];
        channelData = channels{i}{2};
        stats = calculateChannelStats(channelData, mask, {'kurtosis'});
        features.(channelName) = stats.kurt;
    end
end

function features = extractGLCMFeatures(image, colorData)
    %% Extract GLCM texture features
    
    features = getDefaultFeatures('GLCM');
    % Input validation
    validateattributes(image, {'uint8','double'}, {'3d','size',[NaN,NaN,3]}, 'extractGLCMFeatures', 'image');
    if ~isstruct(colorData) || ~isfield(colorData, 'mask')
        error('extract_features:invalidColorData', 'extractGLCMFeatures requires colorData with a logical mask');
    end
    
    try
        mask = logical(colorData.mask);
        if ~any(mask(:)) || nnz(mask) < 2
            return;
        end
        
        if isstruct(colorData) && isfield(colorData, 'gray')
            grayImg = im2uint8(colorData.gray);
        else
            grayImg = rgb2gray(im2uint8(image));
        end
        
        % Crop to patch bounding box to avoid background pixels in analysis
        [rows, cols] = find(mask);
        if isempty(rows)
            return;
        end
        rMin = min(rows); rMax = max(rows);
        cMin = min(cols); cMax = max(cols);
        grayCrop = grayImg(rMin:rMax, cMin:cMax);
        maskCrop = mask(rMin:rMax, cMin:cMax);
        
        glcmStack = buildMaskedGLCM(grayCrop, maskCrop, 16);
        if isempty(glcmStack)
            return;
        end
        
        stats = aggregateGLCMStats(glcmStack);
        features.stripe_contrast = stats.Contrast;
        features.stripe_correlation = stats.Correlation;
        features.stripe_energy = stats.Energy;
        features.stripe_homogeneity = stats.Homogeneity;
        
    catch ME
        warning('extract_features:glcmFeatures', 'Error calculating GLCM features: %s', ME.message);
        features = getDefaultFeatures('GLCM');
    end
    
    function glcmStack = buildMaskedGLCM(patchGray, patchMask, numLevels)
        glcmStack = [];
        maskedValues = double(patchGray(patchMask));
        if isempty(maskedValues) || numel(maskedValues) < 2
            return;
        end
        
        offsets = [0 1; -1 1; -1 0; -1 -1];
        glcmStack = zeros(numLevels, numLevels, size(offsets, 1));
        
        minVal = min(maskedValues);
        maxVal = max(maskedValues);
        if maxVal == minVal
            levelIdx = 1 + round((double(minVal) / 255) * (numLevels - 1));
            levelIdx = max(1, min(numLevels, levelIdx));
            glcmStack(levelIdx, levelIdx, :) = 1;
            return;
        end
        
        scaled = (double(patchGray) - minVal) / (maxVal - minVal);
        scaled = max(0, min(1, scaled));
        quantImage = uint8(floor(scaled * (numLevels - 1)));
        
        [nRows, nCols] = size(patchMask);
        for k = 1:size(offsets, 1)
            dRow = offsets(k, 1);
            dCol = offsets(k, 2);
            [rows1, cols1, rows2, cols2] = getAlignedRanges(nRows, nCols, dRow, dCol);
            if isempty(rows1) || isempty(cols1)
                continue;
            end
            
            mask1 = patchMask(rows1, cols1);
            mask2 = patchMask(rows2, cols2);
            validPairs = mask1 & mask2;
            if ~any(validPairs(:))
                continue;
            end
            
            vals1 = quantImage(rows1, cols1);
            vals2 = quantImage(rows2, cols2);
            vals1 = double(vals1(validPairs)) + 1;
            vals2 = double(vals2(validPairs)) + 1;
            
            counts = accumarray([vals1(:), vals2(:)], 1, [numLevels, numLevels]);
            total = sum(counts(:));
            if total > 0
                glcmStack(:, :, k) = counts / total;
            end
        end
        
        if ~any(glcmStack(:))
            glcmStack = [];
        end
    end
    
    function stats = aggregateGLCMStats(glcmStack)
        stats = struct('Contrast', 0, 'Correlation', 0, 'Energy', 0, 'Homogeneity', 0);
        if isempty(glcmStack)
            return;
        end
        
        levels = 0:(size(glcmStack, 1) - 1);
        [I, J] = meshgrid(levels, levels);
        validCount = 0;
        
        for idx = 1:size(glcmStack, 3)
            P = glcmStack(:, :, idx);
            if ~any(P(:))
                continue;
            end
            validCount = validCount + 1;
            
            px = sum(P, 2);
            py = sum(P, 1);
            mu_x = sum(levels' .* px);
            mu_y = sum(levels .* py);
            sigma_x = sqrt(sum(((levels' - mu_x) .^ 2) .* px));
            sigma_y = sqrt(sum(((levels  - mu_y) .^ 2) .* py));
            
            contrast = sum(sum(((I - J) .^ 2) .* P));
            energy = sum(sum(P .^ 2));
            homogeneity = sum(sum(P ./ (1 + abs(I - J))));
            
            if sigma_x > 0 && sigma_y > 0
                correlation = sum(sum(((I - mu_x) .* (J - mu_y) .* P))) / (sigma_x * sigma_y);
            else
                correlation = 0;
            end
            
            stats.Contrast = stats.Contrast + contrast;
            stats.Energy = stats.Energy + energy;
            stats.Homogeneity = stats.Homogeneity + homogeneity;
            stats.Correlation = stats.Correlation + correlation;
        end
        
        if validCount > 0
            stats.Contrast = stats.Contrast / validCount;
            stats.Energy = stats.Energy / validCount;
            stats.Homogeneity = stats.Homogeneity / validCount;
            stats.Correlation = stats.Correlation / validCount;
        end
    end
    
    function [rows1, cols1, rows2, cols2] = getAlignedRanges(nRows, nCols, dRow, dCol)
        if abs(dRow) >= nRows || abs(dCol) >= nCols
            rows1 = []; cols1 = []; rows2 = []; cols2 = [];
            return;
        end
        
        if dRow >= 0
            rows1 = 1:(nRows - dRow);
            rows2 = (1 + dRow):nRows;
        else
            rows1 = (1 - dRow):nRows;
            rows2 = 1:(nRows + dRow);
        end
        
        if dCol >= 0
            cols1 = 1:(nCols - dCol);
            cols2 = (1 + dCol):nCols;
        else
            cols1 = (1 - dCol):nCols;
            cols2 = 1:(nCols + dCol);
        end
    end
end
function features = extractEntropyFeatures(image, colorData, cfg)
    %% Extract entropy from grayscale patch

    features = struct('entropyValue', 0);
    % Input validation
    validateattributes(image, {'uint8','double'}, {'3d','size',[NaN,NaN,3]}, 'extractEntropyFeatures', 'image');
    if ~isstruct(colorData) || ~isfield(colorData, 'mask')
        error('extract_features:invalidColorData', 'extractEntropyFeatures requires colorData with a logical mask');
    end

    % Get histogram bins from configuration
    if nargin >= 3 && isstruct(cfg) && isfield(cfg, 'thresholds')
        numBins = cfg.thresholds.entropyHistogramBins;
    else
        numBins = 256;
    end

    try
        mask = logical(colorData.mask);
        if ~any(mask(:))
            return;
        end

        if isstruct(colorData) && isfield(colorData, 'gray')
            grayImg = im2double(colorData.gray);
        else
            grayImg = im2double(rgb2gray(image));
        end

        maskedValues = grayImg(mask);
        if isempty(maskedValues)
            return;
        end

        features.entropyValue = computeMaskedEntropy(maskedValues, numBins);

    catch ME
        warning('extract_features:entropyFeatures', 'Error calculating entropy: %s', ME.message);
        features.entropyValue = 0;
    end

    function value = computeMaskedEntropy(vals, nBins)
        vals = double(vals(:));
        if isempty(vals)
            value = 0;
            return;
        end

        minVal = min(vals);
        maxVal = max(vals);
        if maxVal <= minVal
            value = 0;
            return;
        end
        edges = linspace(minVal, maxVal, nBins + 1);
        counts = histcounts(vals, edges);
        total = sum(counts);
        if total == 0
            value = 0;
            return;
        end
        
        probs = counts / total;
        probs = probs(probs > 0);
        value = -sum(probs .* log2(probs));
    end
end

%% VALIDATION UTILITIES


%% CONSOLIDATED DEFAULT FEATURES FUNCTION
function defaults = getDefaultFeatures(featureType)
    %% Return default feature values for specified feature type
    % Uses persistent cache for repeated calls

    persistent defaultMap;

    if isempty(defaultMap)
        defaultMap = containers.Map();

        % RGB, HSV, Lab defaults
        defaultMap('RGB') = struct('R', 0, 'G', 0, 'B', 0);
        defaultMap('HSV') = struct('H', 0, 'S', 0, 'V', 0);
        defaultMap('Lab') = struct('L', 0, 'a', 0, 'b', 0);

        % Statistical defaults
        defaultMap('Skewness') = struct('R_skew', 0, 'G_skew', 0, 'B_skew', 0, ...
                                       'H_skew', 0, 'S_skew', 0, 'V_skew', 0, ...
                                       'L_skew', 0, 'a_skew', 0, 'b_skew', 0);
        defaultMap('Kurtosis') = struct('R_kurto', 0, 'G_kurto', 0, 'B_kurto', 0, ...
                                       'H_kurto', 0, 'S_kurto', 0, 'V_kurto', 0, ...
                                       'L_kurto', 0, 'a_kurto', 0, 'b_kurto', 0);

        % GLCM defaults
        defaultMap('GLCM') = struct('stripe_correlation', 0, 'stripe_contrast', 0, ...
                                   'stripe_energy', 0, 'stripe_homogeneity', 0);

        % Ratio/chromaticity defaults
        defaultMap('ColorRatios') = struct('RG_ratio', 1.0, 'RB_ratio', 1.0, 'GB_ratio', 1.0);
        defaultMap('Chromaticity') = struct('r_chromaticity', 0.33, 'g_chromaticity', 0.33, ...
                                            'chroma_magnitude', 0.57, 'dominant_chroma', 0.33, ...
                                            'chromaticity_std', 0);

        % Extended feature defaults
        defaultMap('Background') = struct('paper_R', 0, 'paper_G', 0, 'paper_B', 0, ...
                                          'paper_L', 0, 'paper_a', 0, 'paper_b', 0, ...
                                          'paper_tempK', 0);
        defaultMap('ColorUniformity') = struct('RGB_CV_R', 0, 'RGB_CV_G', 0, 'RGB_CV_B', 0, ...
                                              'Saturation_uniformity', 0, 'Value_uniformity', 0, ...
                                              'L_uniformity', 0, 'chroma_uniformity', 0);
        defaultMap('RobustColorStats') = struct('L_median', 0, 'L_iqr', 0, 'a_median', 0, 'a_iqr', 0, ...
                                               'b_median', 0, 'b_iqr', 0, 'S_median', 0, 'S_iqr', 0, ...
                                               'V_median', 0, 'V_iqr', 0);

        defaultMap('ColorGradients') = struct('L_gradient_mean', 0, 'L_gradient_std', 0, 'L_gradient_max', 0, ...
                                             'a_gradient_mean', 0, 'a_gradient_std', 0, 'a_gradient_max', 0, ...
                                             'b_gradient_mean', 0, 'b_gradient_std', 0, 'b_gradient_max', 0, ...
                                             'edge_density', 0);

        defaultMap('SpatialDistribution') = struct('L_spatial_std', 0, 'a_spatial_std', 0, 'b_spatial_std', 0, ...
                                                  'radial_L_gradient', 0, 'spatial_uniformity', 0);

        defaultMap('RadialProfile') = struct('radial_L_inner', 0, 'radial_L_outer', 0, ...
                                            'radial_L_ratio', 0, 'radial_chroma_slope', 0, ...
                                            'radial_saturation_slope', 0);

        defaultMap('ConcentrationMetrics') = struct('saturation_range', 0, ...
                                                   'chroma_intensity', 0, 'chroma_max', 0, ...
                                                   'Lab_L_range', 0, 'Lab_a_range', 0, 'Lab_b_range', 0);

        defaultMap('PaperNormalization') = struct('R_paper_ratio', 1.0, 'G_paper_ratio', 1.0, 'B_paper_ratio', 1.0);

        defaultMap('PaperNormalizationExtras') = struct('R_norm', 0, 'G_norm', 0, 'B_norm', 0, ...
                                                       'R_reflectance', 0, 'G_reflectance', 0, 'B_reflectance', 0, ...
                                                       'R_chromatic_adapted', 0, 'G_chromatic_adapted', 0, 'B_chromatic_adapted', 0, ...
                                                       'L_norm', 0, 'a_norm', 0, 'b_norm', 0);

        defaultMap('EnhancedNormalization') = struct('L_corrected_mean', 0, 'L_corrected_median', 0, ...
                                                    'a_corrected_mean', 0, 'a_corrected_median', 0, ...
                                                    'b_corrected_mean', 0, 'b_corrected_median', 0, ...
                                                    'delta_E_from_paper', 0, 'delta_E_median', 0);

        defaultMap('LogarithmicColorTransforms') = struct('log_RG', 0, 'log_GB', 0, 'log_RB', 0, ...
                                                         'log_RGB_magnitude', 0, 'log_RGB_angle', 0);

        defaultMap('FrequencyEnergy') = struct('fft_low_energy', 0, 'fft_high_energy', 0, ...
                                              'fft_band_contrast', 0);

        defaultMap('AdvancedColorAnalysis') = struct('absorption_estimate', 0, 'hue_shift_from_paper', 0, ...
                                                    'chroma_difference', 0);
    end
    
    if isKey(defaultMap, featureType)
        defaults = defaultMap(featureType);
    else
        warning('extract_features:unknownFeatureType', 'Unknown feature type: %s', featureType);
        defaults = struct();
    end
end


%% EXTENDED FEATURE FUNCTIONS
% Extended feature functions with consistent structure and error handling

function features = extractColorUniformityFeatures(image, colorData)
    %% Extract color uniformity statistics
    
    features = getDefaultFeatures('ColorUniformity');
    mask = colorData.mask;
    
    if sum(mask(:)) <= 10
        return;
    end
    
    try
        % Use cached color spaces if available
        if isfield(colorData, 'rgb')
            img = colorData.rgb;
        else
            img = im2double(image);
        end
        
        if isfield(colorData, 'hsv')
            hsvImg = colorData.hsv;
        else
            hsvImg = rgb2hsv(image);
        end
        
        if isfield(colorData, 'lab')
            labImg = colorData.lab;
        else
            labImg = rgb2lab(image);
        end
        
        % RGB uniformity (coefficient of variation)
        rgb_channels = {'R', 'G', 'B'};
        for c = 1:3
            stats = calculateChannelStats(img(:, :, c), mask, {'mean', 'std'});
            if stats.mean > 0
                features.(['RGB_CV_' rgb_channels{c}]) = stats.std / stats.mean;
            end
        end
        
        % HSV uniformity
        S_stats = calculateChannelStats(hsvImg(:, :, 2), mask, {'std'});
        V_stats = calculateChannelStats(hsvImg(:, :, 3), mask, {'std'});
        features.Saturation_uniformity = 1 / (1 + S_stats.std);
        features.Value_uniformity = 1 / (1 + V_stats.std);
        
        % Lab uniformity
        L_stats = calculateChannelStats(labImg(:, :, 1), mask, {'std'});
        a_pixels = labImg(:, :, 2);
        b_pixels = labImg(:, :, 3);
        a_vals = a_pixels(mask);
        b_vals = b_pixels(mask);
        chroma_pixels = sqrt(a_vals.^2 + b_vals.^2);
        
        features.L_uniformity = 1 / (1 + L_stats.std);
        features.chroma_uniformity = 1 / (1 + std(chroma_pixels));
        
        
    catch ME
        warning('extract_features:colorUniformity', 'Error extracting color uniformity: %s', ME.message);
    end
end

function features = extractRobustColorStats(image, colorData)
    %% Extract median and IQR color statistics
    
    features = getDefaultFeatures('RobustColorStats');
    mask = colorData.mask;
    
    if sum(mask(:)) <= 10
        return;
    end
    
    if isfield(colorData, 'lab')
        labImg = colorData.lab;
    else
        labImg = rgb2lab(image);
    end
    
    if isfield(colorData, 'hsv')
        hsvImg = colorData.hsv;
    else
        hsvImg = rgb2hsv(image);
    end
    
    computeIQR = @(vals) prctile(vals, 75) - prctile(vals, 25);
    
    L_pixels = labImg(:, :, 1); L_vals = L_pixels(mask);
    a_pixels = labImg(:, :, 2); a_vals = a_pixels(mask);
    b_pixels = labImg(:, :, 3); b_vals = b_pixels(mask);
    S_vals = hsvImg(:, :, 2); S_vals = S_vals(mask) * 100;
    V_vals = hsvImg(:, :, 3); V_vals = V_vals(mask) * 100;
    
    if ~isempty(L_vals)
        features.L_median = median(L_vals);
        features.L_iqr = computeIQR(L_vals);
    end
    if ~isempty(a_vals)
        features.a_median = median(a_vals);
        features.a_iqr = computeIQR(a_vals);
    end
    if ~isempty(b_vals)
        features.b_median = median(b_vals);
        features.b_iqr = computeIQR(b_vals);
    end
    if ~isempty(S_vals)
        features.S_median = median(S_vals);
        features.S_iqr = computeIQR(S_vals);
    end
    if ~isempty(V_vals)
        features.V_median = median(V_vals);
        features.V_iqr = computeIQR(V_vals);
    end
end

function features = extractRadialProfileFeatures(image, colorData)
    %% Extract radial profile features
    
    features = getDefaultFeatures('RadialProfile');
    mask = colorData.mask;
    
    if sum(mask(:)) <= 50
        return;
    end
    
    if isfield(colorData, 'lab')
        labImg = colorData.lab;
    else
        labImg = rgb2lab(image);
    end
    if isfield(colorData, 'hsv')
        hsvImg = colorData.hsv;
    else
        hsvImg = rgb2hsv(image);
    end
    
    [height, width] = size(mask);
    [X, Y] = meshgrid(1:width, 1:height);
    totalPixels = sum(mask(:));
    if totalPixels == 0
        return;
    end
    cx = sum(X(mask)) / totalPixels;
    cy = sum(Y(mask)) / totalPixels;
    distances = sqrt((X - cx).^2 + (Y - cy).^2);
    maxDist = max(distances(mask));
    if maxDist <= 0
        return;
    end
    
    innerMask = mask & (distances <= maxDist/3);
    outerMask = mask & (distances >= (2*maxDist/3));
    if ~any(innerMask(:)) || ~any(outerMask(:))
        return;
    end
    
    L_channel = labImg(:, :, 1);
    a_channel = labImg(:, :, 2);
    b_channel = labImg(:, :, 3);
    chroma = sqrt(a_channel.^2 + b_channel.^2);
    saturation = hsvImg(:, :, 2) * 100;
    
    innerL = mean(L_channel(innerMask));
    outerL = mean(L_channel(outerMask));
    features.radial_L_inner = innerL;
    features.radial_L_outer = outerL;
    if outerL > 0
        features.radial_L_ratio = innerL / outerL;
    else
        features.radial_L_ratio = 0;
    end
    features.radial_chroma_slope = mean(chroma(innerMask)) - mean(chroma(outerMask));
    features.radial_saturation_slope = mean(saturation(innerMask)) - mean(saturation(outerMask));
end

function features = extractFrequencyEnergyFeatures(image, colorData)
    %% Extract FFT-based frequency features
    
    features = getDefaultFeatures('FrequencyEnergy');
    mask = colorData.mask;
    if sum(mask(:)) <= 50
        return;
    end
    
    if isfield(colorData, 'gray')
        grayImg = colorData.gray;
    else
        grayImg = rgb2gray(im2double(image));
    end
    
    grayImg(~mask) = 0;
    patchPixels = grayImg(mask);
    if isempty(patchPixels) || all(abs(patchPixels - patchPixels(1)) < eps)
        return;
    end
    
    centred = grayImg;
    centred(mask) = centred(mask) - mean(patchPixels);
    spectrum = abs(fftshift(fft2(centred)));
    if ~any(isfinite(spectrum(:)))
        return;
    end
    
    [height, width] = size(spectrum);
    [X, Y] = meshgrid(1:width, 1:height);
    cx = (width + 1) / 2;
    cy = (height + 1) / 2;
    radii = sqrt((X - cx).^2 + (Y - cy).^2);
    maxRadius = max(radii(:));
    if maxRadius <= 0
        return;
    end
    
    lowMask = radii <= 0.15 * maxRadius;
    highMask = radii >= 0.35 * maxRadius;
    if any(lowMask(:))
        features.fft_low_energy = mean(spectrum(lowMask));
    end
    if any(highMask(:))
        features.fft_high_energy = mean(spectrum(highMask));
    end
    features.fft_band_contrast = features.fft_low_energy - features.fft_high_energy;
end

function features = extractConcentrationMetrics(image, colorData)
    %% Extract concentration-specific metrics
    
    features = getDefaultFeatures('ConcentrationMetrics');
    mask = colorData.mask;
    
    if sum(mask(:)) <= 10
        return;
    end
    
    try
        if isfield(colorData, 'hsv')
            hsvImg = colorData.hsv;
        else
            hsvImg = rgb2hsv(image);
        end
        
        if isfield(colorData, 'lab')
            labImg = colorData.lab;
        else
            labImg = rgb2lab(image);
        end
        
        saturation_pixels = hsvImg(:, :, 2);
        saturation_pixels = saturation_pixels(mask);
        L_pixels = labImg(:, :, 1); L_pixels = L_pixels(mask);
        a_pixels = labImg(:, :, 2); a_pixels = a_pixels(mask);
        b_pixels = labImg(:, :, 3); b_pixels = b_pixels(mask);
        
        if ~isempty(saturation_pixels)
            p90 = prctile(saturation_pixels, 90);
            p10 = prctile(saturation_pixels, 10);
            features.saturation_range = p90 - p10;
        end
        
        if ~isempty(a_pixels) && ~isempty(b_pixels)
            chroma_pixels = sqrt(a_pixels.^2 + b_pixels.^2);
            features.chroma_intensity = mean(chroma_pixels);
            features.chroma_max = max(chroma_pixels);
        end
        
        if ~isempty(L_pixels)
            features.Lab_L_range = max(L_pixels) - min(L_pixels);
        end
        if ~isempty(a_pixels)
            features.Lab_a_range = max(a_pixels) - min(a_pixels);
        end
        if ~isempty(b_pixels)
            features.Lab_b_range = max(b_pixels) - min(b_pixels);
        end
    catch ME
        warning('extract_features:concentrationMetrics', 'Error extracting concentration metrics: %s', ME.message);
    end
end

%% ADDITIONAL FEATURE FUNCTIONS

function features = extractColorRatioFeatures(croppedCircularImage, mask, cfg)
    %% Extract illumination-invariant RGB channel ratios

    % Neutral defaults favour a unit response when patch data is unusable
    features = getDefaultFeatures('ColorRatios');

    [rows, cols, ~] = size(croppedCircularImage);
    if nargin < 2 || isempty(mask)
        mask = createValidPixelMask(croppedCircularImage);
    else
        validateattributes(mask, {'logical'}, {'2d','size',[rows, cols]}, 'extractColorRatioFeatures', 'mask');
    end
    mask = logical(mask);

    % Get thresholds from configuration
    if nargin >= 3 && isstruct(cfg) && isfield(cfg, 'thresholds')
        RELATIVE_THRESHOLD = cfg.thresholds.colorRatioRelative;
        MIN_ABSOLUTE_THRESHOLD = cfg.thresholds.colorRatioMinAbsolute;
    else
        RELATIVE_THRESHOLD = 1e-2;
        MIN_ABSOLUTE_THRESHOLD = 1e-3;
    end

    % Pre-allocate message arrays
    maxMessages = 10;
    fallbackMessages = cell(maxMessages, 1);
    lowChannels = cell(maxMessages, 1);
    nonFiniteRatios = cell(maxMessages, 1);
    fallbackCount = 0;
    lowChannelCount = 0;
    nonFiniteCount = 0;
    channelThreshold = NaN;

    try
        img = double(croppedCircularImage);

        if ~any(mask(:))
            fallbackCount = fallbackCount + 1;
            fallbackMessages{fallbackCount} = 'No valid pixels in patch mask.';
        else
            R_channel = img(:, :, 1);
            G_channel = img(:, :, 2);
            B_channel = img(:, :, 3);

            R_pixels = R_channel(mask);
            G_pixels = G_channel(mask);
            B_pixels = B_channel(mask);

            maxValues = [max(R_pixels), ...
                         max(G_pixels), ...
                         max(B_pixels)];
            maxValues = maxValues(isfinite(maxValues));
            if isempty(maxValues)
                channelMax = NaN;
            else
                channelMax = max(maxValues);
            end

            if ~isfinite(channelMax) || channelMax <= 0
                fallbackCount = fallbackCount + 1;
                fallbackMessages{fallbackCount} = 'Combined channel intensity near zero.';
            else
                channelThreshold = max(channelMax * RELATIVE_THRESHOLD, MIN_ABSOLUTE_THRESHOLD);
                totalThreshold = 3 * channelThreshold;

                R_mean = mean(R_pixels);
                G_mean = mean(G_pixels);
                B_mean = mean(B_pixels);
                totalMean = R_mean + G_mean + B_mean;

                if totalMean <= totalThreshold
                    fallbackCount = fallbackCount + 1;
                    fallbackMessages{fallbackCount} = 'Combined channel intensity below stability threshold.';
                else
                    if G_mean > channelThreshold
                        rg_ratio = R_mean / G_mean;
                        if isfinite(rg_ratio)
                            features.RG_ratio = rg_ratio;
                        else
                            nonFiniteCount = nonFiniteCount + 1;
                            nonFiniteRatios{nonFiniteCount} = 'RG';
                        end
                    else
                        lowChannelCount = lowChannelCount + 1;
                        lowChannels{lowChannelCount} = 'G';
                    end

                    if B_mean > channelThreshold
                        rb_ratio = R_mean / B_mean;
                        if isfinite(rb_ratio)
                            features.RB_ratio = rb_ratio;
                        else
                            nonFiniteCount = nonFiniteCount + 1;
                            nonFiniteRatios{nonFiniteCount} = 'RB';
                        end

                        gb_ratio = G_mean / B_mean;
                        if isfinite(gb_ratio)
                            features.GB_ratio = gb_ratio;
                        else
                            nonFiniteCount = nonFiniteCount + 1;
                            nonFiniteRatios{nonFiniteCount} = 'GB';
                        end
                    else
                        lowChannelCount = lowChannelCount + 1;
                        lowChannels{lowChannelCount} = 'B';
                    end
                end
            end
        end
    catch ME
        warning('extract_features:colorRatios', 'Error extracting color ratios: %s', ME.message);
    end

    % Trim arrays to actual count
    lowChannels = lowChannels(1:lowChannelCount);
    nonFiniteRatios = nonFiniteRatios(1:nonFiniteCount);
    fallbackMessages = fallbackMessages(1:fallbackCount);

    if lowChannelCount > 0
        uniqueChannels = unique(lowChannels, 'stable');
        if isnan(channelThreshold)
            fallbackCount = fallbackCount + 1;
            fallbackMessages{fallbackCount} = sprintf('Low mean intensity for channel(s) %s.', strjoin(uniqueChannels, ', '));
        else
            fallbackCount = fallbackCount + 1;
            fallbackMessages{fallbackCount} = sprintf('Low mean intensity for channel(s) %s (threshold %.3g).', strjoin(uniqueChannels, ', '), channelThreshold);
        end
    end

    if nonFiniteCount > 0
        uniqueRatios = unique(nonFiniteRatios, 'stable');
        fallbackCount = fallbackCount + 1;
        fallbackMessages{fallbackCount} = sprintf('Non-finite ratio(s) detected for %s.', strjoin(uniqueRatios, ', '));
    end

    % Trim final messages array
    fallbackMessages = fallbackMessages(1:fallbackCount);

    if fallbackCount > 0
        combinedMessages = unique(fallbackMessages, 'stable');
        warning('extract_features:colorRatiosFallback', strjoin(combinedMessages, ' '));
    end
end


function features = extractChromaticityFeatures(croppedCircularImage, mask)
    %% Extract normalized RGB chromaticity coordinates

    % Default to a neutral chromaticity triplet (r=g=b=1/3) for invalid patches.
    features = struct('r_chromaticity', 0.33, 'g_chromaticity', 0.33, ...
                      'chroma_magnitude', 0.57, 'dominant_chroma', 0.33, ...
                      'chromaticity_std', 0);

    [rows, cols, ~] = size(croppedCircularImage);
    if nargin < 2 || isempty(mask)
        mask = createValidPixelMask(croppedCircularImage);
    else
        validateattributes(mask, {'logical'}, {'2d','size',[rows, cols]}, 'extractChromaticityFeatures', 'mask');
    end
    mask = logical(mask);

    try
        if sum(mask(:)) == 0
            return;
        end

        % Calculate chromaticity coordinates
        img = double(croppedCircularImage);
        R_channel = img(:,:,1);
        G_channel = img(:,:,2);
        B_channel = img(:,:,3);
        R_mean = mean(R_channel(mask));
        G_mean = mean(G_channel(mask));
        B_mean = mean(B_channel(mask));

        sum_RGB = R_mean + G_mean + B_mean;
        if sum_RGB == 0
            return;
        end

        r_c = R_mean / sum_RGB;
        g_c = G_mean / sum_RGB;
        b_c = B_mean / sum_RGB;
        features.r_chromaticity = r_c;
        features.g_chromaticity = g_c;
        % Do not output b_chromaticity (redundant), use it internally only

        % Additional chromaticity features
        features.chroma_magnitude = sqrt(r_c^2 + g_c^2 + b_c^2);
        features.dominant_chroma = max([r_c, g_c, b_c]);

        % Chromaticity standard deviation across patch
        R_pixels = R_channel(mask);
        G_pixels = G_channel(mask);
        B_pixels = B_channel(mask);
        sum_pixels = R_pixels + G_pixels + B_pixels;
        valid_pixels = sum_pixels > 0;

        if sum(valid_pixels) > 1
            r_chrom_pixels = R_pixels(valid_pixels) ./ sum_pixels(valid_pixels);
            g_chrom_pixels = G_pixels(valid_pixels) ./ sum_pixels(valid_pixels);
            b_chrom_pixels = B_pixels(valid_pixels) ./ sum_pixels(valid_pixels);

            % Calculate standard deviation of chromaticity across patch pixels
            chrom_variations = [std(r_chrom_pixels), std(g_chrom_pixels), std(b_chrom_pixels)];
            features.chromaticity_std = mean(chrom_variations);
        end

    catch ME
        warning('extract_features:chromaticity', 'Error extracting chromaticity: %s', ME.message);
    end
end

function features = extractIlluminantInvariantFeatures(croppedCircularImage, mask)
    %% Extract illuminant-invariant features using color space transformations
    
    features = struct();
    
    [rows, cols, ~] = size(croppedCircularImage);
    if nargin < 2 || isempty(mask)
        mask = createValidPixelMask(croppedCircularImage);
    else
        validateattributes(mask, {'logical'}, {'2d','size',[rows, cols]}, 'extractIlluminantInvariantFeatures', 'mask');
    end
    mask = logical(mask);
    
    try
        if sum(mask(:)) > 0
            % Lab color space (more perceptually uniform)
            labImg = rgb2lab(croppedCircularImage);
            
            % Use only a* and b* (chromaticity, lighting-independent)
            a_channel = labImg(:,:,2);
            b_channel = labImg(:,:,3);
            a_mean = mean(a_channel(mask));
            b_mean = mean(b_channel(mask));
            features.ab_magnitude = sqrt(a_mean^2 + b_mean^2);
            features.ab_angle = atan2(b_mean, a_mean);
            
            % HSV features (hue is illumination-invariant)
            hsvImg = rgb2hsv(croppedCircularImage);
            H_channel = hsvImg(:,:,1);
            S_channel = hsvImg(:,:,2);
            hue_values = H_channel(mask);
            features.hue_circular_mean = atan2(mean(sin(hue_values * 2 * pi)), ...
                                              mean(cos(hue_values * 2 * pi))) / (2 * pi);
            if features.hue_circular_mean < 0
                features.hue_circular_mean = features.hue_circular_mean + 1;
            end
            
            features.saturation_mean = mean(S_channel(mask));
            
        else
            features.ab_magnitude = 0; features.ab_angle = 0;
            features.hue_circular_mean = 0; features.saturation_mean = 0;
        end
        
    catch ME
        warning('extract_features:illuminantInvariant', 'Error extracting illuminant-invariant features: %s', ME.message);
        features.ab_magnitude = 0; features.ab_angle = 0;
        features.hue_circular_mean = 0; features.saturation_mean = 0;
    end
end

%% Gradient and spatial feature implementations
function features = extractColorGradientFeatures(croppedCircularImage, colorData)
    %% Color gradient features
    
    % Input validation
    validateattributes(croppedCircularImage, {'uint8', 'double'}, {'3d', 'nonempty'}, 'extractColorGradientFeatures', 'croppedCircularImage');
    
    % Initialize default features
    features = getDefaultFeatures('ColorGradients');
    
    try
        % Create mask for non-background pixels
        mask = colorData.mask;
        
        if sum(mask(:)) > 100  % Need sufficient pixels for gradient analysis
            % Convert to Lab for perceptually uniform gradients
            if isfield(colorData, 'lab')
                labImg = colorData.lab;
            else
                labImg = rgb2lab(croppedCircularImage);
            end
            
            % Vectorized gradient computation for all channels at once
            [Gx, Gy] = gradient(labImg);
            gradient_magnitude = sqrt(Gx.^2 + Gy.^2);
            
            % Extract gradient statistics for all channels
            channel_names = {'L', 'a', 'b'};
            for c = 1:3
                channel_gradients = gradient_magnitude(:, :, c);
                masked_gradients = channel_gradients(mask);
                
                if ~isempty(masked_gradients)
                    % Extract gradient statistics
                    features.([channel_names{c} '_gradient_mean']) = mean(masked_gradients);
                    features.([channel_names{c} '_gradient_std']) = std(masked_gradients);
                    features.([channel_names{c} '_gradient_max']) = max(masked_gradients);
                end
            end
            
            % Edge density (higher concentration might have fewer edges)
            if isfield(colorData, 'gray')
                grayImg = colorData.gray;
            else
                grayImg = rgb2gray(im2double(croppedCircularImage));
            end
            edges = edge(grayImg, 'canny');
            features.edge_density = sum(edges(mask)) / sum(mask(:));
        end
        
    catch ME
        warning('extract_features:gradientFeatures', 'Error extracting gradient features: %s', ME.message);
    end
end

function features = extractSpatialDistributionFeatures(croppedCircularImage, colorData)
    %% Extract spatial color distribution features
    
    features = getDefaultFeatures('SpatialDistribution');
    
    try
        % Create mask for non-background pixels
        mask = colorData.mask;
        
        if sum(mask(:)) > 50
            [height, width, ~] = size(croppedCircularImage);
            
            % Divide patch into quadrants and analyze color distribution
            mid_h = round(height/2);
            mid_w = round(width/2);
            
            quadrants = {[1, mid_h, 1, mid_w], [1, mid_h, mid_w+1, width], ...
                        [mid_h+1, height, 1, mid_w], [mid_h+1, height, mid_w+1, width]};
            
            % Extract mean Lab values for each quadrant
            if isfield(colorData, 'lab')
                labImg = colorData.lab;
            else
                labImg = rgb2lab(croppedCircularImage);
            end
            
            quadrant_L = zeros(1, 4);
            quadrant_a = zeros(1, 4);
            quadrant_b = zeros(1, 4);
            
            for q = 1:4
                r_range = quadrants{q}(1):quadrants{q}(2);
                c_range = quadrants{q}(3):quadrants{q}(4);
                
                quad_mask = mask(r_range, c_range);
                if sum(quad_mask(:)) > 0
                    L_quad = labImg(r_range, c_range, 1);
                    a_quad = labImg(r_range, c_range, 2);
                    b_quad = labImg(r_range, c_range, 3);
                    
                    quadrant_L(q) = mean(L_quad(quad_mask));
                    quadrant_a(q) = mean(a_quad(quad_mask));
                    quadrant_b(q) = mean(b_quad(quad_mask));
                end
            end
            
            % Spatial uniformity metrics
            features.L_spatial_std = std(quadrant_L);
            features.a_spatial_std = std(quadrant_a);
            features.b_spatial_std = std(quadrant_b);
            features.spatial_uniformity = 1 / (1 + features.L_spatial_std + features.a_spatial_std + features.b_spatial_std);
            
            % Radial distribution analysis
            [Y, X] = ndgrid(1:height, 1:width);
            validCount = sum(mask(:));
            if validCount <= 0
                return;
            end
            cx = sum(X(mask)) / validCount;
            cy = sum(Y(mask)) / validCount;
            distances = sqrt((X - cx).^2 + (Y - cy).^2);
            max_dist = max(distances(mask));

            % Divide into concentric rings centred on patch centroid
            if max_dist > 0
                inner_mask = mask & (distances <= max_dist/3);
                outer_mask = mask & (distances >= (2*max_dist/3));

                if sum(inner_mask(:)) > 0 && sum(outer_mask(:)) > 0
                    L_channel = labImg(:,:,1);
                    inner_L = mean(L_channel(inner_mask));
                    outer_L = mean(L_channel(outer_mask));
                    features.radial_L_gradient = abs(inner_L - outer_L);
                else
                    features.radial_L_gradient = 0;
                end
            else
                features.radial_L_gradient = 0;
            end
        end
        
    catch ME
        warning('extract_features:spatialFeatures', 'Error extracting spatial features: %s', ME.message);
    end
end

function features = extractLogarithmicColorTransformFeatures(croppedCircularImage, mask)
    %% Extract logarithmic color transformation features
    
    features = struct('log_RG', 0, 'log_GB', 0, 'log_RB', 0, 'log_RGB_magnitude', 0, 'log_RGB_angle', 0);
    
    [rows, cols, ~] = size(croppedCircularImage);
    if nargin < 2 || isempty(mask)
        mask = createValidPixelMask(croppedCircularImage);
    else
        validateattributes(mask, {'logical'}, {'2d','size',[rows, cols]}, 'extractLogarithmicColorTransformFeatures', 'mask');
    end
    mask = logical(mask);
    
    try
        if sum(mask(:)) > 0
            img = double(croppedCircularImage);
            R_channel = img(:,:,1);
            G_channel = img(:,:,2);
            B_channel = img(:,:,3);
            R_mean = mean(R_channel(mask));
            G_mean = mean(G_channel(mask));
            B_mean = mean(B_channel(mask));
            
            % Avoid log of zero by adding small epsilon
            epsilon = 1e-6;
            R_safe = max(R_mean, epsilon);
            G_safe = max(G_mean, epsilon);
            B_safe = max(B_mean, epsilon);
            
            % Logarithmic ratios
            features.log_RG = log(R_safe / G_safe);
            features.log_GB = log(G_safe / B_safe);
            features.log_RB = log(R_safe / B_safe);
            
            % RGB vector in log space
            log_R = log(R_safe);
            log_G = log(G_safe);
            log_B = log(B_safe);
            
            features.log_RGB_magnitude = sqrt(log_R^2 + log_G^2 + log_B^2);
            features.log_RGB_angle = atan2(sqrt(log_G^2 + log_B^2), log_R);
        end
        
    catch ME
        warning('extract_features:logarithmicTransforms', 'Error extracting logarithmic transforms: %s', ME.message);
    end
end

%

function features = extractAdvancedColorAnalysisFeatures(croppedCircularImage, paperStats, mask)
    %% Extract advanced color analysis features.
    % Relies on extractChromaticityFeatures for normalized RGB descriptors to avoid duplication.

    features = struct('absorption_estimate', 0, 'hue_shift_from_paper', 0, ...
                     'chroma_difference', 0);

    [rows, cols, ~] = size(croppedCircularImage);
    if nargin < 3 || isempty(mask)
        mask = createValidPixelMask(croppedCircularImage);
    else
        validateattributes(mask, {'logical'}, {'2d','size',[rows, cols]}, 'extractAdvancedColorAnalysisFeatures', 'mask');
    end
    mask = logical(mask);

    try
        if ~any(mask(:))
            return;
        end

        img = double(croppedCircularImage);
        R_pixels = img(:, :, 1); R_pixels = R_pixels(mask);
        G_pixels = img(:, :, 2); G_pixels = G_pixels(mask);
        B_pixels = img(:, :, 3); B_pixels = B_pixels(mask);

        R_mean = mean(R_pixels);
        G_mean = mean(G_pixels);
        B_mean = mean(B_pixels);

        if paperStats.isValid
            paper_intensity = mean(paperStats.paperRGB);
            patch_intensity = (R_mean + G_mean + B_mean) / 3;
            if paper_intensity > 1 && patch_intensity > 0
                transmittance = patch_intensity / paper_intensity;
                % Clamp transmittance to [1e-6, 1.0] for physical validity
                % Upper bound prevents negative absorption from specular reflections
                % Lower bound prevents log(0) singularity
                transmittance_clamped = max(min(transmittance, 1.0), 1e-6);
                features.absorption_estimate = -log(transmittance_clamped);
            else
                features.absorption_estimate = 0;
            end
        end

        hsvImg = rgb2hsv(croppedCircularImage);
        H_pixels = hsvImg(:, :, 1); H_pixels = H_pixels(mask);
        % Use circular mean for hue (angular quantity)
        patch_hue_circular = atan2(mean(sin(H_pixels * 2 * pi)), ...
                                   mean(cos(H_pixels * 2 * pi))) / (2 * pi);
        if patch_hue_circular < 0
            patch_hue_circular = patch_hue_circular + 1;
        end
        patch_hue = patch_hue_circular * 360;

        if paperStats.isValid
            paper_rgb_norm = paperStats.paperRGB / 255;
            paper_hsv = rgb2hsv(reshape(paper_rgb_norm, [1 1 3]));
            paper_hue = squeeze(paper_hsv(1,1,1)) * 360;
            hue_diff = abs(patch_hue - paper_hue);
            if hue_diff > 180
                hue_diff = 360 - hue_diff;
            end
            features.hue_shift_from_paper = hue_diff;
        end

        labImg = rgb2lab(croppedCircularImage);
        a_pixels = labImg(:, :, 2); a_pixels = a_pixels(mask);
        b_pixels = labImg(:, :, 3); b_pixels = b_pixels(mask);
        patch_chroma = sqrt(mean(a_pixels)^2 + mean(b_pixels)^2);

        if paperStats.isValid && isfield(paperStats, 'paperLab')
            paper_chroma = sqrt(paperStats.paperLab(2)^2 + paperStats.paperLab(3)^2);
            features.chroma_difference = abs(patch_chroma - paper_chroma);
        else
            features.chroma_difference = patch_chroma;
        end
    catch ME
        warning('extract_features:advancedColorAnalysis', 'Error extracting advanced color analysis: %s', ME.message);
    end
end

%% UTILITY FUNCTIONS

function validImage = loadAndValidateImage(imageName)
    % Load image without EXIF rotation (rotation handled via coordinates.txt)
    validImage = [];

    try
        img = imread(imageName);
        
        if isempty(img)
            warning('extract_features:imageLoad', 'Could not load image: %s', imageName);
            return;
        end
        
        % Normalize to uint8 RGB (0..255) for consistent downstream expectations
        % Handle class and dynamic range carefully to avoid unintended clipping
        switch class(img)
            case 'uint8'
                Iu8 = img;
            case 'uint16'
                % Scale to 8-bit using im2uint8
                Iu8 = im2uint8(img);
            case {'double','single'}
                maxVal = max(img(:));
                minVal = min(img(:));
                if maxVal <= 1.01 && minVal >= -0.01
                    % Assume [0,1] range
                    Iu8 = im2uint8(img);
                elseif maxVal <= 255.5 && minVal >= -0.5
                    % Assume [0,255] range
                    Iu8 = uint8(min(max(round(img), 0), 255));
                else
                    % Unknown range: scale via normalization
                    Iu8 = im2uint8(mat2gray(img));
                end
            otherwise
                % Fallback for other integer classes
                try
                    Iu8 = im2uint8(img); 
                catch
                    Iu8 = uint8(min(max(double(img),0),255));
                end
        end
        
        % Ensure exactly 3 channels
        if size(Iu8, 3) == 1
            Iu8 = repmat(Iu8, [1, 1, 3]);
        elseif size(Iu8, 3) > 3
            Iu8 = Iu8(:, :, 1:3);
        end
        
        validImage = Iu8;
        validateattributes(validImage, {'uint8', 'double'}, {'3d', 'size', [NaN, NaN, 3]}, 'loadAndValidateImage', 'validImage');
        
    catch ME
        warning('extract_features:imageLoad', 'Error loading image %s: %s', imageName, ME.message);
    end
end

function mask = createValidPixelMask(image)
    validateattributes(image, {'uint8', 'double'}, {'3d'}, 'createValidPixelMask', 'image');
    mask = any(im2double(image) > 0, 3);
end

function mergedStruct = mergeStructs(struct1, struct2)
    validateattributes(struct1, {'struct'}, {'scalar'}, 'mergeStructs', 'struct1');
    validateattributes(struct2, {'struct'}, {'scalar'}, 'mergeStructs', 'struct2');
    
    mergedStruct = struct1;
    fields = fieldnames(struct2);
    
    for i = 1:length(fields)
        mergedStruct.(fields{i}) = struct2.(fields{i});
    end
end


%% Helper Functions for Processing Pipeline

function displayConfigurationSummary(cfg, preset)

    fprintf('\n=== microPAD Feature Extraction Configuration ===\n');
    fprintf('Chemical: %s\n', cfg.chemicalName);
    fprintf('Feature Preset: %s\n', preset);
    registry = getFeatureRegistry();
    if isfield(registry, 'presetDescriptions') && isfield(registry.presetDescriptions, preset)
        fprintf('Preset Summary: %s\n', registry.presetDescriptions.(preset));
    elseif strcmpi(preset, 'custom')
        fprintf('Preset Summary: Custom selection defined at runtime.\n');
    end
    fprintf('Caching Enabled: %s\n', char(string(cfg.performance.cacheColorSpaces)));
    if isfield(cfg, 'output') && isfield(cfg.output, 'trainTestSplit') && cfg.output.trainTestSplit
        fprintf('Train/Test Split: enabled (test fraction = %.2f)\n', cfg.output.testSize);
        if isfield(cfg.output, 'splitGroupColumn')
            fprintf('Grouping Column: %s\n', cfg.output.splitGroupColumn);
        end
    else
        fprintf('Train/Test Split: disabled\n');
    end
    
    % Count enabled feature groups
    featureFields = fieldnames(cfg.features);
    enabledCount = sum(structfun(@(x) x, cfg.features));
    fprintf('Enabled Feature Groups: %d/%d\n', enabledCount, length(featureFields));
    
    if enabledCount < length(featureFields)
        fprintf('Disabled Feature Groups: ');
        disabledFeatures = featureFields(~structfun(@(x) x, cfg.features));
        fprintf('%s ', disabledFeatures{:});
        fprintf('\n');
    end
    
    fprintf('\nCoordinate-Based Workflow:\n');
    fprintf('- Processes original images rather than separate patch images\n');
    fprintf('- Uses same-image paper background for normalization\n');
    fprintf('- Single pass extracts all patch features from each image\n');
    fprintf('- Maintains consistent lighting context between patches and background\n\n');
end

function displayFeatureSummary(cfg)
    fprintf('\n=== Feature Extraction Summary ===\n');
    
    outputFile = sprintf('%s_%s_%s_features.xlsx', cfg.featurePreset, cfg.chemicalName, cfg.tNValue);
    outputPath = fullfile(cfg.outputPath, outputFile);
    
    if exist(outputPath, 'file')
        try
            data = readtable(outputPath);
            fprintf('Output File: %s\n', outputFile);
            fprintf('Total Records: %d\n', height(data));
            fprintf('Total Features: %d\n', width(data));
            
            registry = getFeatureRegistry();
            presentBackground = intersect(registry.backgroundFeatureCols, data.Properties.VariableNames);
            presentPatch = intersect(registry.patchFeatureCols, data.Properties.VariableNames);
            presentNormalized = intersect(registry.normalizedFeatureCols, data.Properties.VariableNames);
            fprintf('Background features: %d present\n', length(presentBackground));
            fprintf('Patch features: %d present\n', length(presentPatch));
            fprintf('Normalized features: %d present\n', length(presentNormalized));
            
        catch ME
            warning('extract_features:outputAnalysis', 'Could not analyze output file: %s', ME.message);
        end
    end
end

function outputConfig = createOutputConfiguration(chemicalName, outputDecimals, outputExtension, includeLabelInExcel, trainTestSplit, testSize, splitGroupColumn, randomSeed)
    %% Create output configuration (delegates to feature_pipeline helper)
    featPipe = feature_pipeline();
    outputConfig = featPipe.output.createConfig(chemicalName, outputDecimals, outputExtension, includeLabelInExcel, trainTestSplit, testSize, splitGroupColumn, randomSeed);
end

function cfg = addPathConfiguration(cfg, originalImagesFolder, coordinatesFolder, outputFolder)
    %% Configure paths for coordinate-based processing

    projectRoot = cfg.modules.pathUtils.findProjectRoot(originalImagesFolder);
    cfg.projectRoot = projectRoot;
    cfg.originalImagesPath = fullfile(projectRoot, originalImagesFolder);
    cfg.coordinatesPath = fullfile(projectRoot, coordinatesFolder);
    cfg.outputPath = fullfile(projectRoot, outputFolder);
    cfg.originalImagesFolder = originalImagesFolder;
    cfg.coordinatesFolder = coordinatesFolder;
    cfg.outputFolder = outputFolder;
end

function processAllFolders(cfg)
    %% Process all folders using coordinate-based approach

    validatePaths(cfg);
    phoneList = cfg.modules.pathUtils.listSubfolders(cfg.originalImagesPath);
    
    if isempty(phoneList)
        error('extract_features:noPhoneFolders', 'No phone folders found in: %s', cfg.originalImagesPath);
    end
    
    fprintf('\n>> Dataset Structure: %d phones | tN=%s | %s preset\n', length(phoneList), cfg.tNValue, cfg.featurePreset);
    fprintf('>> Phones: %s\n', strjoin(phoneList, ' | '));
    
    % Determine tN value from coordinate files
    tNValue = determineTNValueFromCoordinates(cfg.coordinatesPath, phoneList);
    cfg.tNValue = tNValue;
    fprintf('\n=== Processing %d phones with coordinate-based extraction ===\n', length(phoneList));

    % Process all phones (paths are absolute, no directory change needed)
    for i = 1:length(phoneList)
        processPhone(phoneList{i}, cfg, i, length(phoneList));
    end

    generateConsolidatedExcelFile(cfg);
end

function tNValue = determineTNValueFromCoordinates(coordinatesPath, phoneList)
    %% Determine tN value based on phone naming or default to t0

    tNValue = 't0';

    if nargin < 2 || isempty(phoneList)
        return;
    end

    for phoneIdx = 1:length(phoneList)
        phoneName = phoneList{phoneIdx};
        match = regexp(phoneName, 't\d+', 'match', 'once');
        if ~isempty(match)
            tNValue = match;
            return;
        end

        phoneCoordPath = fullfile(coordinatesPath, phoneName);
        if isfile(fullfile(phoneCoordPath, 'coordinates.txt'))
            tokens = regexp(phoneCoordPath, 't\d+', 'match');
            if ~isempty(tokens)
                tNValue = tokens{1};
                return;
            end
        end
    end
end

function processPhone(phoneName, cfg, phoneIdx, totalPhones)
    %% Process phone by iterating concentration folders in flattened structure

    fprintf('[%d/%d] Processing Phone: %s\n', phoneIdx, totalPhones, phoneName);

    originalImagePath = fullfile(cfg.originalImagesPath, phoneName);
    coordinatePath = fullfile(cfg.coordinatesPath, phoneName);

    if ~exist(originalImagePath, 'dir')
        warning('extract_features:missingOriginal', 'Original images folder missing: %s', originalImagePath);
        return;
    end

    if ~exist(coordinatePath, 'dir')
        warning('extract_features:missingCoords', 'Coordinates folder missing: %s', coordinatePath);
        return;
    end

    rootCoordFile = fullfile(coordinatePath, cfg.coordinateFileName);
    if ~isfile(rootCoordFile)
        warning('extract_features:missingPhoneCoordinates', 'Phone-level coordinates file missing: %s', rootCoordFile);
        return;
    end

    phoneCoordinates = parseCoordinatesFile(rootCoordFile);
    if ~phoneCoordinates.isValid
        warning('extract_features:invalidPhoneCoordinates', 'No valid coordinates found in %s', rootCoordFile);
        return;
    end

    cfg.modules.pathUtils.executeInFolder(originalImagePath, @() processConcentrationsInPhone(phoneName, cfg, phoneCoordinates));
end

function processConcentrationsInPhone(phoneName, cfg, phoneCoordinates)
    %% Iterate through all concentration folders for the phone using phone-level coordinates
    % Supports two modes:
    %   1. Flat structure (1_dataset): Images in phone folder, concentrations in coordinates
    %   2. Legacy structure (2_micropads): Images in con_* subfolders
    if nargin < 3 || ~isstruct(phoneCoordinates) || ~isfield(phoneCoordinates, 'isValid') || ~phoneCoordinates.isValid
        warning('extract_features:invalidPhoneCoordinates', 'Phone-level coordinates missing or invalid for %s', phoneName);
        return;
    end

    pathUtils = cfg.modules.pathUtils;
    subFolders = pathUtils.listSubfolders('.');
    prefix = cfg.concFolderPrefix;

    if ~isempty(subFolders)
        isConc = strncmpi(subFolders, prefix, numel(prefix));
        concFolders = subFolders(isConc);
    else
        concFolders = {};
    end

    % Check if we're in flat mode (no con_* folders) or legacy mode (with con_* folders)
    if isempty(concFolders)
        % Flat structure mode: process all images from current folder
        % Concentrations are determined from coordinate file
        fprintf('  Mode: Flat structure (images in phone folder)\n');
        processPhoneFlatStructure(phoneName, cfg, phoneCoordinates);
    else
        % Legacy mode: process each con_* subfolder
        totalConcs = numel(concFolders);
        fprintf('  Mode: Legacy structure (%d concentration folders)\n', totalConcs);
        fprintf('  Concentrations: %s\n', strjoin(concFolders, ' | '));

        allFeatureData = cell(totalConcs, 1);
        featureCount = 0;

        for idx = 1:totalConcs
            concName = concFolders{idx};
            fprintf('  [%d/%d] Folder: %s -> ', idx, totalConcs, concName);

            concFeatures = processConcentrationFolder(phoneName, concName, cfg, phoneCoordinates);

            if ~isempty(concFeatures)
                featureCount = featureCount + 1;
                allFeatureData{featureCount} = concFeatures;
                numFeatures = numel(fieldnames(concFeatures(1)));
                fprintf('OK (%d patches, %d features each)\n', numel(concFeatures), numFeatures);
            else
                fprintf('SKIP (no features extracted)\n');
            end
        end

        if featureCount > 0
            allFeatureData = vertcat(allFeatureData{1:featureCount});
            storePhoneFeatureData(phoneName, allFeatureData, cfg);
        end
    end
end

function processPhoneFlatStructure(phoneName, cfg, phoneCoordinates)
    %% Process all images from flat phone folder structure (1_dataset mode)
    % Images are in the phone folder directly, concentrations come from coordinates

    imageIO = cfg.modules.imageIO;
    imageFiles = imageIO.listImageFiles('.');
    if isempty(imageFiles)
        fprintf('  No images found in flat folder for phone: %s\n', phoneName);
        return;
    end

    fprintf('  Processing %d images from flat structure\n', numel(imageFiles));

    allFeatures = cell(numel(imageFiles), 1);
    validCount = 0;
    totalPatches = 0;

    for imgIdx = 1:numel(imageFiles)
        imgName = imageFiles{imgIdx};

        if phoneCoordinates.images.isKey(imgName)
            patches = phoneCoordinates.images(imgName);
            imageFeatures = processOriginalImageWithCoordinates(imgName, patches, phoneName, cfg);

            if ~isempty(imageFeatures)
                validCount = validCount + 1;
                allFeatures{validCount} = imageFeatures;
                totalPatches = totalPatches + numel(patches);
            end
        end
    end

    fprintf('  Processed %d images, %d patches total\n', validCount, totalPatches);

    if validCount > 0
        allFeatureData = vertcat(allFeatures{1:validCount});
        storePhoneFeatureData(phoneName, allFeatureData, cfg);
    end
end

function featureData = processConcentrationFolder(phoneName, concName, cfg, phoneCoordinates)
    %% Process a single concentration folder and extract features for all patches
    if nargin < 4 || ~isstruct(phoneCoordinates) || ~isfield(phoneCoordinates, 'isValid') || ~phoneCoordinates.isValid
        warning('extract_features:invalidPhoneCoordinates', 'Phone-level coordinates missing or invalid for %s', phoneName);
        featureData = [];
        return;
    end

    imageIO = cfg.modules.imageIO;
    concIndex = parseConcentrationFolderIndex(concName, cfg.concFolderPrefix);
    if isnan(concIndex)
        warning('extract_features:invalidConcentrationFolder', 'Unable to parse concentration index from folder: %s', concName);
        featureData = [];
        return;
    end
    
    coordinateData = filterCoordinateDataByConcentration(phoneCoordinates, concIndex);
    if ~coordinateData.isValid
        fprintf('  Skipping %s (no matching coordinates)\n', concName);
        featureData = [];
        return;
    end

    featureData = [];
    cfg.modules.pathUtils.executeInFolder(concName, @collectFeatures);

    function collectFeatures()
        imageFiles = imageIO.listImageFiles('.');
        if isempty(imageFiles)
            fprintf('No images found in %s/%s\n', phoneName, concName);
            return;
        end
        
        fprintf(' processing %d images | ', numel(imageFiles));
        allFeatures = cell(numel(imageFiles), 1);
        validCount = 0;
        processedPatches = 0;
        
        for imgIdx = 1:numel(imageFiles)
            imgName = imageFiles{imgIdx};
            if coordinateData.images.isKey(imgName)
                patches = coordinateData.images(imgName);
                imageFeatures = processOriginalImageWithCoordinates(imgName, patches, phoneName, cfg);
                if ~isempty(imageFeatures)
                    validCount = validCount + 1;
                    allFeatures{validCount} = imageFeatures;
                    processedPatches = processedPatches + numel(patches);
                end
            end
        end
        
        fprintf('%d patches processed successfully\n', processedPatches);
        
        if validCount > 0
            featureData = vertcat(allFeatures{1:validCount});
        end
    end
end

function concIndex = parseConcentrationFolderIndex(folderName, prefix)
    %% Extract numeric concentration index from folder name
    if nargin < 2 || isempty(prefix)
        prefix = '';
    end

    if ~isempty(prefix) && strncmpi(folderName, prefix, numel(prefix))
        numericPart = folderName(numel(prefix)+1:end);
    else
        numericPart = folderName;
    end

    concIndex = str2double(numericPart);
    if isnan(concIndex)
        token = regexp(numericPart, '-?\d+', 'match', 'once');
        if isempty(token)
            token = regexp(folderName, '-?\d+', 'match', 'once');
        end
        if ~isempty(token)
            concIndex = str2double(token);
        end
    end
end

function filteredData = filterCoordinateDataByConcentration(coordinateData, targetConcentration)
    %% Filter phone-level coordinates down to a single concentration index
    filteredData = struct('images', containers.Map('KeyType', 'char', 'ValueType', 'any'), 'isValid', false);

    if nargin < 2 || isempty(targetConcentration) || isnan(targetConcentration)
        return;
    end
    if ~isstruct(coordinateData) || ~isfield(coordinateData, 'images') || ~isa(coordinateData.images, 'containers.Map')
        return;
    end

    keys = coordinateData.images.keys;
    totalEntries = 0;
    for k = 1:numel(keys)
        key = keys{k};
        patches = coordinateData.images(key);
        if isempty(patches)
            continue;
        end
        mask = arrayfun(@(p) isfield(p, 'concentration') && ~isempty(p.concentration) && p.concentration == targetConcentration, patches);
        if any(mask)
            filteredPatches = patches(mask);
            filteredData.images(key) = filteredPatches;
            totalEntries = totalEntries + numel(filteredPatches);
        end
    end

    if totalEntries > 0
        filteredData.isValid = true;
    end
end

function allPatchFeatures = processOriginalImageWithCoordinates(imageName, patches, phoneName, cfg)
    %% Process original image with coordinate data to extract features from all patches

    % Extract maskUtils from cfg modules (required for local functions)
    maskUtils = cfg.modules.maskUtils;

    allPatchFeatures = [];

    try
        % Load original image
        originalImage = loadAndValidateImage(imageName);
        if isempty(originalImage)
            return;
        end
        
        %
        
        % Extract paper background mask (excluding all patch areas)
        paperBackgroundMask = extractPaperBackgroundMaskWithPatches(originalImage, patches, cfg, maskUtils);
        if isempty(paperBackgroundMask)
            warning('extract_features:paperStatsMissing', 'Could not extract paper background for %s', imageName);
            return;
        end
        % Optional lightweight visualization for debugging (random subset)
        debugVisualizeMaskIfEnabled(originalImage, paperBackgroundMask, imageName, cfg);
        
        % Extract paper background statistics
        paperStats = extractPaperBackgroundStatsFromMask(originalImage, paperBackgroundMask, cfg);
        if ~paperStats.isValid
            warning('extract_features:paperStatsInvalid', 'Invalid paper statistics for %s', imageName);
            return;
        end

        % Build a per-image color cache so expensive conversions are shared across patches
        if cfg.performance.cacheColorSpaces
            imageColorCache = buildImageColorCache(originalImage, cfg.features);
            labOriginal = imageColorCache.lab;
        else
            labOriginal = rgb2lab(originalImage);
            imageColorCache = struct('lab', labOriginal);
        end

        % Process each patch using coordinates
        patchFeatures = cell(length(patches), 1);
        validPatchCount = 0;

        for i = 1:length(patches)
            try
                patchRec = patches(i);
                patchFeature = extractFeaturesFromCoordinates(originalImage, labOriginal, patchRec, paperStats, ...
                                                            imageName, phoneName, cfg, imageColorCache);
                if ~isempty(patchFeature)
                    validPatchCount = validPatchCount + 1;
                    patchFeatures{validPatchCount} = patchFeature;
                end
            catch patchME
                warning('extract_features:patchExtractionFailed', ...
                       'Failed to extract features from patch %d/%d in %s: %s', ...
                       i, length(patches), imageName, patchME.message);
            end
        end
        
        if validPatchCount > 0
            allPatchFeatures = vertcat(patchFeatures{1:validPatchCount});
        end
        
    catch ME
        warning('extract_features:originalImageProcessing', 'Error processing %s: %s', imageName, ME.message);
    end
end

function storePhoneFeatureData(phoneName, featureData, cfg)
    %% Store phone feature data (delegates to feature_pipeline helper)
    featPipe = feature_pipeline();
    featPipe.output.storePhoneData(phoneName, featureData, cfg);
end

function debugVisualizeMaskIfEnabled(originalImage, mask, imageName, cfg)
    %% Debug visualization (delegates to feature_pipeline helper)
    featPipe = feature_pipeline();
    featPipe.debug.visualizeMask(originalImage, mask, imageName, cfg);
end

function generateConsolidatedExcelFile(cfg)
    %% Generate consolidated Excel file (delegates to feature_pipeline helper)
    featPipe = feature_pipeline();
    pathUtils = path_utils();

    % Build external dependencies struct for helper
    externalDeps = struct();
    externalDeps.listSubfolders = @pathUtils.listSubfolders;
    externalDeps.getFeatureRegistry = @getFeatureRegistry;
    externalDeps.calculateBatchSize = @calculateBatchSize;
    externalDeps.checkMemoryPressure = @checkMemoryPressure;

    featPipe.output.generateExcel(cfg, externalDeps);
end

function handleError(ME, cfg)
    if strcmp(ME.message, 'User stopped execution')
        fprintf('!! Script stopped by user\n');
    else
        fprintf('!! Error: %s\n', ME.message);
        
        % Clean up temporary data on error if enabled
        try
            if exist('cfg','var') && isstruct(cfg) && isfield(cfg, 'performance') && isfield(cfg.performance, 'clearTempOnError') && cfg.performance.clearTempOnError
                tempDir = fullfile(cfg.projectRoot, 'temp_feature_data');
                if exist(tempDir, 'dir')
                    rmdir(tempDir, 's');
                    fprintf('Cleaned up temporary data due to error.\n');
                end
            end
        catch
            % Cleanup failed or cfg not available, continue with error
        end
        
        rethrow(ME);
    end
end

function validatePaths(cfg)

    if ~exist(cfg.originalImagesPath, 'dir')
        error('extract_features:missingOriginalDir', 'Original images directory not found: %s\nPlease run cut_micropads() first.', cfg.originalImagesPath);
    end

    if ~exist(cfg.coordinatesPath, 'dir')
        error('extract_features:missingCoordinatesDir', 'Coordinates directory not found: %s\nPlease run cut_elliptical_regions() first.', cfg.coordinatesPath);
    end

    pathUtils = cfg.modules.pathUtils;
    imageIO = cfg.modules.imageIO;
    phoneList = pathUtils.listSubfolders(cfg.originalImagesPath);
    if isempty(phoneList)
        error('extract_features:noPhoneInOriginal', 'No phone folders found in original images. Expected structure: %s/phone/', cfg.originalImagesPath);
    end

    coordPhoneList = pathUtils.listSubfolders(cfg.coordinatesPath);
    if isempty(coordPhoneList)
        error('extract_features:noPhoneInCoordinates', 'No phone folders found in coordinates. Expected structure: %s/phone/%s', cfg.coordinatesPath, cfg.coordinateFileName);
    end

    hasValidStructure = false;
    phonesMissingCoords = cell(length(phoneList), 1);
    missingCount = 0;

    for i = 1:length(phoneList)
        phoneName = phoneList{i};
        originalPhonePath = fullfile(cfg.originalImagesPath, phoneName);

        % Check for con_N subdirectories (the actual structure: phone/con_N/images)
        if exist(originalPhonePath, 'dir')
            conFolders = pathUtils.listSubfolders(originalPhonePath);
            if ~isempty(conFolders)
                % Verify at least one con_N folder contains images
                for j = 1:length(conFolders)
                    conPath = fullfile(originalPhonePath, conFolders{j});
                    if ~isempty(imageIO.listImageFiles(conPath))
                        hasValidStructure = true;
                        break;
                    end
                end
            end
        end

        if ismember(phoneName, coordPhoneList)
            coordFile = fullfile(cfg.coordinatesPath, phoneName, cfg.coordinateFileName);
            if ~isfile(coordFile)
                missingCount = missingCount + 1;
                phonesMissingCoords{missingCount} = phoneName;
            end
        else
            missingCount = missingCount + 1;
            phonesMissingCoords{missingCount} = phoneName;
        end
    end

    % Trim to actual count
    phonesMissingCoords = phonesMissingCoords(1:missingCount);

    if ~hasValidStructure
        error('extract_features:invalidOriginalStructure', 'Invalid original images folder structure. Expected images organized as phone/con_N/<image files>.');
    end

    if missingCount > 0
        missingList = strjoin(unique(phonesMissingCoords), ', ');
        error('extract_features:noCoordinateFilesFound', 'Missing phone-level coordinates.txt for: %s', missingList);
    end

    if ~exist(cfg.outputPath, 'dir')
        mkdir(cfg.outputPath);
        fprintf('Created output directory: %s\n', cfg.outputPath);
    end

    fprintf('Original images path validated: %s\n', cfg.originalImagesPath);
    fprintf('Coordinates path validated: %s\n', cfg.coordinatesPath);
    fprintf('Output directory validated: %s\n', cfg.outputPath);
    fprintf('Found %d phone folder(s) with valid structure\n', length(phoneList));
    fprintf('Coordinate-based processing ready!\n');
end


%% MEMORY MANAGEMENT HELPER FUNCTIONS

function batchSizeOut = calculateBatchSize(numFiles, cfg)
    %% Calculate batch size based on dataset size

    baseBatchSize = cfg.performance.baseBatchSize;
    smallThreshold = cfg.thresholds.smallDataset;
    mediumThreshold = cfg.thresholds.mediumDataset;
    largeThreshold = cfg.thresholds.largeDataset;
    mediumDivisor = cfg.thresholds.mediumDatasetBatchDivisor;

    if numFiles <= smallThreshold
        batchSizeOut = numFiles;
    elseif numFiles <= mediumThreshold
        batchSizeOut = max(5, round(numFiles / mediumDivisor));
    else
        scaleFactor = min(1.0, largeThreshold / numFiles);
        batchSizeOut = max(10, round(baseBatchSize * scaleFactor));
    end
end

function isUnderPressure = checkMemoryPressure(threshold)
    %% Check if system is under memory pressure
    
    isUnderPressure = false;
    
    try
        memInfo = memory;
        memUsageRatio = memInfo.MemUsedMATLAB / memInfo.MemAvailableAllArrays;
        
        if memUsageRatio > threshold
            isUnderPressure = true;
            warning('extract_features:memoryPressure', 'Memory usage: %.1f%% (threshold: %.1f%%)', ...
                   memUsageRatio * 100, threshold * 100);
        end
    catch
        % memory() function not available on all systems
        % Fall back to conservative approach
        isUnderPressure = false;
    end
end

%% UI COMPONENTS - SEPARATED FROM CORE FUNCTIONALITY
% Note: UI components are placed at the bottom for code organization

function [selectedConfig, userCanceled] = showFeatureSelectionDialog(defaultPreset, chemicalName)
    %% Feature Selection Dialog
    % Dialog UI for selecting feature groups and presets
    
    persistent featureData;
    
    % Initialize feature data (static configuration)
    if isempty(featureData)
        featureData = initializeFeatureData();
    end
    
    % Initialize output
    selectedConfig = [];
    userCanceled = true;
    
    try
        % Create main dialog
        dialog = createDialogUI(chemicalName);
        
        % Initialize with default preset
        setPreset(dialog, defaultPreset);
        
        % Wait for user action
        uiwait(dialog.figure);
        
        % Get results
        if isvalid(dialog.figure)
            [selectedConfig, userCanceled] = getDialogResults(dialog);
            delete(dialog.figure);
        end
        
    catch ME
        warning('extract_features:dialogError', 'Dialog error: %s', ME.message);
    end
    
    %% Nested functions used by dialog
    
    function data = initializeFeatureData()
        % Use centralized feature registry for dialog
        registry = getFeatureRegistry();
        
        data.names = registry.featureNames;
        data.displayNames = registry.displayNames;
        data.matrix = registry.presetMatrix;
        data.presetNames = registry.presetNames;
        data.presetLabels = registry.presetLabels;
        data.tiers = registry.tiers;
        data.tierLegend = registry.tierLegend;
        data.presetDescriptions = registry.presetDescriptions;
        % Custom-only groups: not included in minimal/robust/full presets
        if size(data.matrix,2) >= 3
            data.customOnlyMask = all(data.matrix(:,1:3) == 0, 2);
        else
            data.customOnlyMask = false(numel(data.names),1);
        end
    end
    
    function dlg = createDialogUI(chemical)
        % Create dialog with consistent margins and equal spacing
        
        % Consistent margin system - fundamental UI principle
        margin = 12; % Single consistent margin value for all elements
        innerMargin = margin; % Same margin for container inner elements
        
        % Base sizing parameters
        baseElementHeight = 22;
        baseFontSize = 11;
        titleFontSize = 13; % Larger font for titles
        statusFontSize = 12; % Larger font for status text
        pixelsPerChar = 7;
        checkboxPadding = 25;
        panelTitleHeight = 30; % Slightly increased for larger title font
        
        % Calculate content dimensions
        numFeatures = length(featureData.names);
        numPresets = length(featureData.presetLabels);
        cols = 4;
        rows = ceil(numFeatures / cols);
        
        % Dynamic width calculation based on text content
        if isempty(featureData.displayNames)
            maxFeatureNameLength = 1;
        else
            maxFeatureNameLength = max(cellfun(@numel, featureData.displayNames));
        end
        if isempty(featureData.presetLabels)
            maxPresetLabelLength = 1;
        else
            maxPresetLabelLength = max(cellfun(@numel, featureData.presetLabels));
        end
        
        % Calculate required space for elements with extra margin for truncation prevention
        checkboxMinWidth = maxFeatureNameLength * pixelsPerChar + checkboxPadding + margin;
        radioButtonMinWidth = maxPresetLabelLength * pixelsPerChar + checkboxPadding + margin;
        
        % Calculate container dimensions with consistent spacing and inner margins
        presetContainerWidth = numPresets * radioButtonMinWidth + 2 * innerMargin;
        featureContainerWidth = cols * checkboxMinWidth + 2 * innerMargin;
        
        % Dialog width based on widest container with consistent margins
        contentWidth = max(presetContainerWidth, featureContainerWidth);
        dialogWidth = contentWidth + 2 * margin;
        
        % Calculate heights with consistent spacing and proper inner margins
        titleHeight = baseElementHeight * 1.5;
        presetPanelContentHeight = baseElementHeight + 2 * innerMargin; % Top and bottom inner margins
        presetPanelHeight = presetPanelContentHeight + panelTitleHeight;
        
        featurePanelContentHeight = rows * baseElementHeight + (rows - 1) * margin + 2 * innerMargin; % Top and bottom inner margins
        featurePanelHeight = featurePanelContentHeight + panelTitleHeight;
        
        presetInfoHeight = baseElementHeight * 2.4;
        statusHeight = baseElementHeight * 0.9;
        hintHeight = baseElementHeight * 1.2;
        buttonHeight = baseElementHeight * 1.4;
        
        % Total dialog height with consistent spacing
        dialogHeight = titleHeight + presetPanelHeight + presetInfoHeight + featurePanelHeight + statusHeight + hintHeight + buttonHeight + 8 * margin;
        
        % Center dialog on screen
        screenSize = get(0, 'ScreenSize');
        xPos = (screenSize(3) - dialogWidth) / 2;
        yPos = (screenSize(4) - dialogHeight) / 2;
        
        dlg.figure = uifigure('Name', sprintf('microPAD Colorimetric Analysis — %s', upper(chemical)), ...
                             'Position', [xPos, yPos, dialogWidth, dialogHeight], 'Resize', 'off', ...
                             'WindowStyle', 'modal');
        % Ensure UI is always cleaned up, even on errors
        dlg.cleanup = onCleanup(@() cleanupDialogHandle(dlg));
        
        % Layout elements from top to bottom with consistent spacing
        currentY = dialogHeight - margin;
        
        % 1. Title - centered horizontally with consistent margins
        currentY = currentY - titleHeight;
        uilabel(dlg.figure, 'Position', [margin, currentY, contentWidth, titleHeight], ...
                'Text', sprintf('Feature Selection - %s', chemical), ...
                'FontSize', baseFontSize * 1.4, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        
        % 2. Preset panel with consistent margins
        currentY = currentY - margin - presetPanelHeight;
        presetPanel = uipanel(dlg.figure, 'Title', '', ...
                             'Position', [margin, currentY, contentWidth, presetPanelHeight], ...
                             'BorderType', 'line', 'BorderWidth', 1);
        
        % Panel title with consistent inner margins and left alignment
        uilabel(presetPanel, 'Position', [innerMargin, presetPanelHeight - panelTitleHeight, contentWidth - 2*innerMargin, panelTitleHeight], ...
                'Text', 'Presets', 'FontWeight', 'bold', 'FontSize', titleFontSize, ...
                'VerticalAlignment', 'center', 'HorizontalAlignment', 'left');
        
        % Button group with consistent inner margins
        buttonGroupY = innerMargin;
        buttonGroupHeight = baseElementHeight;
        dlg.presetGroup = uibuttongroup(presetPanel, 'BorderType', 'none', ...
                                       'Position', [innerMargin, buttonGroupY, contentWidth - 2*innerMargin, buttonGroupHeight]);
        
        % Left-aligned radio buttons with margin to prevent left truncation
        buttonLeftMargin = margin / 2; % Small margin from left edge to prevent truncation
        availableWidth = contentWidth - 2 * innerMargin - 2 * buttonLeftMargin; % Account for left/right margins
        buttonSpacing = availableWidth / numPresets;
        
        for i = 1:numPresets
            buttonX = buttonLeftMargin + (i - 1) * buttonSpacing; % Start with margin from left edge
            buttonWidth = min(radioButtonMinWidth, buttonSpacing - buttonLeftMargin); % Ensure no overlap
            dlg.presetButtons(i) = uiradiobutton(dlg.presetGroup, ...
                                               'Text', featureData.presetLabels{i}, ...
                                               'Position', [buttonX, 0, buttonWidth, buttonGroupHeight]);
        end
        
        % 2.5. Preset guidance (descriptions beneath radio buttons)
        currentY = currentY - margin - presetInfoHeight;
        minimalDesc = char(string(featureData.presetDescriptions.minimal));
        robustDesc = char(string(featureData.presetDescriptions.robust));
        fullDesc = char(string(featureData.presetDescriptions.full));
        presetInfoText = sprintf(['Minimal - %s\n', ...
                                         'Robust - %s\n', ...
                                         'Full - %s\n', ...
                                         'Custom - Manual selection using checkboxes.'], ...
                                        minimalDesc, robustDesc, fullDesc);
        uilabel(dlg.figure, 'Position', [margin, currentY, contentWidth, presetInfoHeight], ...
                'Text', presetInfoText, 'FontSize', baseFontSize-0.5, 'HorizontalAlignment', 'left', ...
                'FontColor', [0.35, 0.35, 0.35]);

        % 3. Feature panel with consistent margins
        currentY = currentY - margin - featurePanelHeight;
        featurePanel = uipanel(dlg.figure, 'Title', '', ...
                              'Position', [margin, currentY, contentWidth, featurePanelHeight], ...
                              'BorderType', 'line', 'BorderWidth', 1);
        
        % Panel title with consistent inner margins and left alignment
        uilabel(featurePanel, 'Position', [innerMargin, featurePanelHeight - panelTitleHeight, contentWidth - 2*innerMargin, panelTitleHeight], ...
                'Text', 'Feature Groups ([tag] tier, * custom-only)', 'FontWeight', 'bold', 'FontSize', titleFontSize, ...
                'VerticalAlignment', 'center', 'HorizontalAlignment', 'left');
        
        dlg.checkboxes = gobjects(numFeatures, 1);
        
        % Left-aligned grid with consistent inner margins and proper spacing
        availableFeatureWidth = contentWidth - 2 * innerMargin;
        availableFeatureHeight = featurePanelContentHeight - 2 * innerMargin;
        
        colSpacing = availableFeatureWidth / cols;
        rowSpacing = availableFeatureHeight / rows;
        
        for i = 1:numFeatures
            row = ceil(i / cols);
            col = mod(i - 1, cols) + 1;
            
            % Position checkbox with consistent inner margins and proper width
            x = innerMargin + (col - 1) * colSpacing;
            y = featurePanelHeight - panelTitleHeight - innerMargin - row * rowSpacing + (rowSpacing - baseElementHeight) / 2;
            checkboxWidth = min(checkboxMinWidth, colSpacing - margin/2); % Prevent truncation
            
            % Determine label and tooltip
            groupName = featureData.names{i};
            displayName = featureData.displayNames{i};
            isCustomOnly = logical(featureData.customOnlyMask(i));
            if isCustomOnly
                labelText = [displayName ' *'];
            else
                labelText = displayName;
            end
            dlg.checkboxes(i) = uicheckbox(featurePanel, ...
                                          'Text', labelText, ...
                                          'Position', [x, y, checkboxWidth, baseElementHeight]);
            % Add informative tooltips where helpful
            try
                tierKey = char(string(featureData.tiers{i}));
                maxTooltipParts = 6;
                tooltipParts = cell(maxTooltipParts, 1);
                tooltipCount = 0;

                if isfield(featureData, 'tierLegend') && isfield(featureData.tierLegend, tierKey)
                    legendEntry = featureData.tierLegend.(tierKey);
                    if isfield(legendEntry, 'description')
                        tooltipCount = tooltipCount + 1;

                        tooltipParts{tooltipCount} = char(string(legendEntry.description));
                    end
                end

                switch groupName
                    case 'PaperNormalization'
                        tooltipCount = tooltipCount + 1;

                        tooltipParts{tooltipCount} = 'Outputs per-channel ratios relative to paper background.';
                    case 'PaperNormalizationExtras'
                        tooltipCount = tooltipCount + 1;

                        tooltipParts{tooltipCount} = 'Adds normalized intensities, reflectance, and chromatic-adapted channels (custom-only).';
                end

                if isCustomOnly
                    tooltipCount = tooltipCount + 1;

                    tooltipParts{tooltipCount} = 'Custom-only group: enable via preset "Custom".';
                end

                if tooltipCount > 0
                    dlg.checkboxes(i).Tooltip = strjoin(tooltipParts(1:tooltipCount), ' ');
                end
            catch
                % Tooltip may not be supported in older MATLAB; ignore
            end
        end
        
        % 4. Status with consistent margins and left alignment
        currentY = currentY - margin - statusHeight;
        dlg.statusLabel = uilabel(dlg.figure, 'Position', [margin, currentY, contentWidth, statusHeight], ...
                                 'Text', '', 'FontColor', [0.2, 0.2, 0.6], 'FontSize', statusFontSize, ...
                                 'HorizontalAlignment', 'left');
        
        % 4.5. Hint label explaining preset behavior and tier legend
        currentY = currentY - margin - hintHeight;
        legendStruct = featureData.tierLegend;
        legendText = sprintf('* Legend: [%s] Minimal | [%s] Robust | [%s] Full-only | [%s] Custom-only.', ...
            char(string(legendStruct.mustHave.tag)), char(string(legendStruct.bestScore.tag)), ...
            char(string(legendStruct.experimental.tag)), char(string(legendStruct.customOnly.tag)));
        hintText = sprintf('%s\n%s\n%s', ...
            '* Toggling any checkbox switches preset to "Custom" to apply your selection.', ...
            '* Groups marked with * are custom-only (excluded from Minimal/Robust/Full).', ...
            legendText);
        dlg.hintLabel = uilabel(dlg.figure, 'Position', [margin, currentY, contentWidth, hintHeight], ...
                                'Text', hintText, 'FontColor', [0.35, 0.35, 0.35], 'FontSize', baseFontSize-1, ...
                                'HorizontalAlignment', 'left');
        
        % 5. Buttons with consistent spacing
        currentY = currentY - margin - buttonHeight;
        buttonWidth = contentWidth / 6;
        buttonSpacing = margin;
        
        dlg.extractBtn = uibutton(dlg.figure, 'Text', 'Extract', ...
                                 'Position', [dialogWidth - 2*margin - 2*buttonWidth - buttonSpacing, currentY, buttonWidth, buttonHeight], ...
                                 'BackgroundColor', [0.2, 0.7, 0.2], 'FontColor', 'white');
        dlg.cancelBtn = uibutton(dlg.figure, 'Text', 'Cancel', ...
                                'Position', [dialogWidth - margin - buttonWidth, currentY, buttonWidth, buttonHeight], ...
                                'BackgroundColor', [0.7, 0.2, 0.2], 'FontColor', 'white');
        
        % Set callbacks
        dlg.presetGroup.SelectionChangedFcn = @(~,~) updateFromPreset(dlg);
        for i = 1:numFeatures
            dlg.checkboxes(i).ValueChangedFcn = @(~,~) updateFromCheckbox(dlg);
        end
        dlg.extractBtn.ButtonPushedFcn = @(~,~) extractAction(dlg);
        dlg.cancelBtn.ButtonPushedFcn = @(~,~) cancelAction(dlg);
    end
    
    function setPreset(dlg, presetName)
        % Set initial preset selection
        idx = find(strcmp(featureData.presetNames, presetName), 1);
        if ~isempty(idx)
            dlg.presetGroup.SelectedObject = dlg.presetButtons(idx);
        else
            dlg.presetGroup.SelectedObject = dlg.presetButtons(4); % Custom
        end
        updateFromPreset(dlg);
    end
    
    function updateFromPreset(dlg)
        % Update checkboxes based on preset selection
        selectedIdx = find(dlg.presetButtons == dlg.presetGroup.SelectedObject);
        
        if selectedIdx <= 3
            % Apply preset
            for i = 1:length(featureData.names)
                dlg.checkboxes(i).Value = logical(featureData.matrix(i, selectedIdx));
            end
        end
        
        updateStatus(dlg);
    end
    
    function updateFromCheckbox(dlg)
        % Switch to custom when manually changing checkboxes
        if dlg.presetGroup.SelectedObject ~= dlg.presetButtons(4)
            dlg.presetGroup.SelectedObject = dlg.presetButtons(4);
        end
        updateStatus(dlg);
    end
    
    function updateStatus(dlg)
        % Update status display and button state
        selectedCount = sum([dlg.checkboxes.Value]);
        totalCount = length(dlg.checkboxes);
        
        selectedIdx = dlg.presetButtons == dlg.presetGroup.SelectedObject;
        presetName = featureData.presetLabels{selectedIdx};
        
        dlg.statusLabel.Text = sprintf('%s: %d/%d features selected', presetName, selectedCount, totalCount);
        dlg.extractBtn.Enable = selectedCount > 0;
    end
    
    function extractAction(dlg)
        % Handle extract button click
        dlg.figure.UserData = struct('action', 'extract');
        uiresume(dlg.figure);
    end
    
    function cancelAction(dlg)
        % Handle cancel button click
        dlg.figure.UserData = struct('action', 'cancel');
        uiresume(dlg.figure);
    end
    
    function [config, canceled] = getDialogResults(dlg)
        % Extract results from dialog
        config = [];
        canceled = true;
        
        if isfield(dlg.figure.UserData, 'action') && strcmp(dlg.figure.UserData.action, 'extract')
            selectedIdx = find(dlg.presetButtons == dlg.presetGroup.SelectedObject);
            
            if selectedIdx <= 3
                presetName = featureData.presetNames{selectedIdx};
            else
                presetName = 'custom';
            end
            
            % Build feature group configuration
            featureConfig = struct();
            for i = 1:length(featureData.names)
                featureConfig.(featureData.names{i}) = dlg.checkboxes(i).Value;
            end
            
            config = struct('preset', presetName, 'features', featureConfig, 'chemical', chemicalName);
            canceled = false;
        end
    end

    function cleanupDialogHandle(d)
        try
            if isstruct(d) && isfield(d, 'figure') && isvalid(d.figure)
                delete(d.figure);
            end
        catch
            % Ignore cleanup errors
        end
    end
end

function paperStats = extractPaperBackgroundStatsFromMask(originalImage, paperBackgroundMask, cfg)
    %% Extract paper background statistics from mask (relaxed minimum area; mean-based estimates)

    paperStats = struct('isValid', false);

    try
        if isempty(paperBackgroundMask) || ~any(paperBackgroundMask(:))
            return;
        end

        img = double(originalImage); % maintain 0-255 scale for RGB stats
        [~, ~, channels] = size(img);

        paperStats.paperMask = paperBackgroundMask;

        % Mean-based paper color estimates
        paperStats.paperRGB = zeros(1, 3);
        paperStats.paperStd = zeros(1, 3);
        for c = 1:channels
            channel = img(:, :, c);
            vals = channel(paperBackgroundMask);
            if isempty(vals)
                paperStats.paperRGB(c) = 0;
                paperStats.paperStd(c) = 0;
            else
                paperStats.paperRGB(c) = mean(vals);
                paperStats.paperStd(c) = std(vals);
            end
        end

        % Convert to Lab using normalized RGB
        paperRGB_normalized = reshape(paperStats.paperRGB / 255, [1, 1, 3]);
        paperLab = rgb2lab(paperRGB_normalized);
        paperStats.paperLab = squeeze(paperLab);

        % Keep existing illuminant cues (using color_analysis helper)
        paperStats.colorTemperature = cfg.modules.colorAnalysis.estimateColorTemperature(paperStats.paperRGB);
        paperStats.chromaticAdaptation = cfg.modules.colorAnalysis.calculateChromaticAdaptation(paperStats.paperRGB);

        paperStats.isValid = true;

    catch ME
        warning('extract_features:paperStatsExtraction', 'Error extracting paper statistics: %s', ME.message);
        paperStats.isValid = false;
    end
end

function featureStruct = addCoordinateBasedNormalizationFeatures(featureStruct, originalImage, patchMask, paperStats, cfg, labOriginal)
    %% Add normalization features using coordinate-based paper background
    
    try
        if ~paperStats.isValid
            % Add defaults only for enabled groups; trim for robust preset
            if isfield(cfg, 'features') && isfield(cfg.features, 'PaperNormalization') && cfg.features.PaperNormalization
                pm = getDefaultFeatures('PaperNormalization');
                if isfield(cfg, 'featurePreset') && strcmpi(cfg.featurePreset, 'robust')
                    % Keep only ratios in robust preset
                    pm = struct('R_paper_ratio', 1.0, 'G_paper_ratio', 1.0, 'B_paper_ratio', 1.0);
                end
                featureStruct = mergeStructs(featureStruct, pm);
            end

            if isfield(cfg, 'features') && isfield(cfg.features, 'PaperNormalizationExtras') && cfg.features.PaperNormalizationExtras
                pmExtras = getDefaultFeatures('PaperNormalizationExtras');
                featureStruct = mergeStructs(featureStruct, pmExtras);
            end

            if isfield(cfg, 'features') && isfield(cfg.features, 'EnhancedNormalization') && cfg.features.EnhancedNormalization
                en = getDefaultFeatures('EnhancedNormalization');
                featureStruct = mergeStructs(featureStruct, en);
            end

            if isfield(cfg, 'features') && isfield(cfg.features, 'Background') && cfg.features.Background
                bg = getDefaultFeatures('Background');
                featureStruct = mergeStructs(featureStruct, bg);
            end
            return;
        end

        % Extract patch pixels
        img = double(originalImage);
        includePaperExtras = isfield(cfg, 'features') && isfield(cfg.features, 'PaperNormalizationExtras') && cfg.features.PaperNormalizationExtras;
        labImg_local = []; % Defer Lab conversion until needed

        if sum(patchMask(:)) > 0
            % Paper-relative normalization features (respect preset)
            doPaperNorm = ~isfield(cfg, 'features') || ~isfield(cfg.features, 'PaperNormalization') || cfg.features.PaperNormalization;
            if doPaperNorm
                paperNormFeatures = struct();
                paperExtraFeatures = struct();
                channelNames = {'R','G','B'};
                for c = 1:3
                    channel = img(:, :, c);
                    patchPixels = channel(patchMask);
                    if isempty(patchPixels), continue; end

                    meanPatchValue = mean(patchPixels);

                    % Ratios are the core invariant metrics
                    if paperStats.paperRGB(c) > 1
                        ratioVal = meanPatchValue / paperStats.paperRGB(c);
                    else
                        ratioVal = 1.0;
                    end
                    paperNormFeatures.(sprintf('%s_paper_ratio', channelNames{c})) = ratioVal;

                    if includePaperExtras
                        paperExtraFeatures.(sprintf('%s_norm', channelNames{c})) = meanPatchValue;
                        paperExtraFeatures.(sprintf('%s_reflectance', channelNames{c})) = min(ratioVal, 1.0);
                        adaptedValue = meanPatchValue * paperStats.chromaticAdaptation(c);
                        paperExtraFeatures.(sprintf('%s_chromatic_adapted', channelNames{c})) = adaptedValue;
                    end
                end

                if includePaperExtras
                    if isempty(labImg_local)
                        if nargin < 6 || isempty(labOriginal)
                            labImg_local = rgb2lab(originalImage);
                        else
                            labImg_local = labOriginal;
                        end
                    end
                    L_pixels = labImg_local(:, :, 1);
                    a_pixels = labImg_local(:, :, 2);
                    b_pixels = labImg_local(:, :, 3);
                    paperExtraFeatures.L_norm = mean(L_pixels(patchMask));
                    paperExtraFeatures.a_norm = mean(a_pixels(patchMask));
                    paperExtraFeatures.b_norm = mean(b_pixels(patchMask));
                end

                featureStruct = mergeStructs(featureStruct, paperNormFeatures);
                if includePaperExtras && ~isempty(fieldnames(paperExtraFeatures))
                    featureStruct = mergeStructs(featureStruct, paperExtraFeatures);
                end
            end

            % Illuminant-invariant features
            enhancedFeatures = struct();
            
            if isempty(labImg_local)
                if nargin < 6 || isempty(labOriginal)
                    labImg_local = rgb2lab(originalImage);
                else
                    labImg_local = labOriginal;
                end
            end
            L_pixels = labImg_local(:, :, 1);
            a_pixels = labImg_local(:, :, 2);
            b_pixels = labImg_local(:, :, 3);
            patch_L_vals = L_pixels(patchMask);
            patch_a_vals = a_pixels(patchMask);
            patch_b_vals = b_pixels(patchMask);
            patch_L = mean(patch_L_vals);
            patch_a = mean(patch_a_vals);
            patch_b = mean(patch_b_vals);

            paper_L = paperStats.paperLab(1);
            paper_a = paperStats.paperLab(2);
            paper_b_val = paperStats.paperLab(3);

            delta_E = sqrt((patch_L - paper_L)^2 + (patch_a - paper_a)^2 + (patch_b - paper_b_val)^2);

            if paper_L > 5
                lightness_ratio = patch_L / paper_L;
                enhancedFeatures.L_corrected_mean = lightness_ratio * 100;
                if ~isempty(patch_L_vals)
                    enhancedFeatures.L_corrected_median = median(patch_L_vals) / paper_L * 100;
                end
            else
                enhancedFeatures.L_corrected_mean = patch_L;
                if ~isempty(patch_L_vals)
                    enhancedFeatures.L_corrected_median = median(patch_L_vals);
                end
            end

            % Clamp a* and b* corrections to prevent extreme outlier features
            maxLabShift = cfg.thresholds.maxLabShift;

            if ~isempty(patch_a_vals)
                a_corrected_mean_raw = patch_a - paper_a;
                a_corrected_median_raw = median(patch_a_vals) - paper_a;
                enhancedFeatures.a_corrected_mean = min(max(a_corrected_mean_raw, -maxLabShift), maxLabShift);
                enhancedFeatures.a_corrected_median = min(max(a_corrected_median_raw, -maxLabShift), maxLabShift);
            else
                enhancedFeatures.a_corrected_mean = 0;
                enhancedFeatures.a_corrected_median = 0;
            end

            if ~isempty(patch_b_vals)
                b_corrected_mean_raw = patch_b - paper_b_val;
                b_corrected_median_raw = median(patch_b_vals) - paper_b_val;
                enhancedFeatures.b_corrected_mean = min(max(b_corrected_mean_raw, -maxLabShift), maxLabShift);
                enhancedFeatures.b_corrected_median = min(max(b_corrected_median_raw, -maxLabShift), maxLabShift);
            else
                enhancedFeatures.b_corrected_mean = 0;
                enhancedFeatures.b_corrected_median = 0;
            end

            enhancedFeatures.delta_E_from_paper = delta_E;
            if ~isempty(patch_L_vals) && ~isempty(patch_a_vals) && ~isempty(patch_b_vals)
                delta_E_vals = sqrt((patch_L_vals - paper_L).^2 + (patch_a_vals - paper_a).^2 + (patch_b_vals - paper_b_val).^2);
                if ~isempty(delta_E_vals)
                    enhancedFeatures.delta_E_median = median(delta_E_vals);
                end
            end

            if (~isfield(cfg, 'features') || ~isfield(cfg.features, 'EnhancedNormalization') || cfg.features.EnhancedNormalization)
                featureStruct = mergeStructs(featureStruct, enhancedFeatures);
            end
            
            % Add relative color features that need paperStats
            try
                % Create patch region for feature extraction
                patchRegion = originalImage;
                patchRegion(~repmat(patchMask, [1, 1, 3])) = 0;
                
                %
                
                if isfield(cfg, 'features') && isfield(cfg.features, 'AdvancedColorAnalysis') && cfg.features.AdvancedColorAnalysis
                    advancedFeatures = extractAdvancedColorAnalysisFeatures(patchRegion, paperStats, patchMask);
                    featureStruct = mergeStructs(featureStruct, advancedFeatures);
                end
                
            catch relME
                warning('extract_features:relativeAdvancedFeatures', 'Error extracting relative/advanced features: %s', relME.message);
            end
        end

        % Background features (from original image/paper mask)
        try
            if isfield(cfg, 'features') && isfield(cfg.features, 'Background') && cfg.features.Background
                % Compute background features once per patch row (same across patches of same image)
                bg = struct();
                % Paper RGB means
                bg.paper_R = paperStats.paperRGB(1);
                bg.paper_G = paperStats.paperRGB(2);
                bg.paper_B = paperStats.paperRGB(3);
                % Paper Lab (convert stored paperLab if available)
                if isfield(paperStats, 'paperLab') && numel(paperStats.paperLab) >= 3
                    bg.paper_L = paperStats.paperLab(1);
                    bg.paper_a = paperStats.paperLab(2);
                    bg.paper_b = paperStats.paperLab(3);
                else
                    % Fallback: compute Lab from RGB means
                    rgbNorm = reshape(paperStats.paperRGB / 255, [1 1 3]);
                    lab = rgb2lab(rgbNorm);
                    lab = squeeze(lab);
                    bg.paper_L = lab(1);
                    bg.paper_a = lab(2);
                    bg.paper_b = lab(3);
                end
                % Additional context
                if isfield(paperStats, 'colorTemperature')
                    bg.paper_tempK = paperStats.colorTemperature;
                else
                    bg.paper_tempK = cfg.defaults.paperTempK;
                end
                

                % Merge into feature struct if enabled
                featureStruct = mergeStructs(featureStruct, bg);
            end
        catch bgME
            warning('extract_features:backgroundFeatures', 'Error assembling background features: %s', bgME.message);
        end

    catch ME
        warning('extract_features:coordinateNormalization', 'Error adding coordinate-based normalization: %s', ME.message);
    end
end

%% COORDINATE PARSING FUNCTIONS
function coordinateData = parseCoordinatesFile(coordinatesFilePath)
    %% Parse phone-level coordinates.txt file and extract patch location information
    % Expected format (8 columns, header optional):
    %   image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
    %   where rotationAngle is the ellipse's geometric orientation in degrees
    %   (-180 to 180, clockwise from horizontal major axis)

    coordinateData = struct('images', containers.Map('KeyType', 'char', 'ValueType', 'any'), ...
                            'isValid', false, ...
                            'availableConcentrations', [], ...
                            'entryCount', 0);

    try
        if ~exist(coordinatesFilePath, 'file')
            warning('extract_features:coordinatesFile', 'Coordinates file not found: %s', coordinatesFilePath);
            return;
        end

        fileID = fopen(coordinatesFilePath, 'r');
        if fileID == -1
            warning('extract_features:coordinatesFile', 'Could not open coordinates file: %s', coordinatesFilePath);
            return;
        end
        cleanupObj = onCleanup(@() fclose(fileID));

        headerLines = 0;
        while true
            position = ftell(fileID);
            line = fgetl(fileID);
            if ~ischar(line)
                return;
            end
            if ~isempty(strtrim(line))
                tokens = lower(strsplit(strtrim(line)));
                if ~isempty(tokens) && strcmp(tokens{1}, 'image')
                    headerLines = 1;
                else
                    fseek(fileID, position, 'bof');
                end
                break;
            end
        end

        frewind(fileID);
        data = textscan(fileID, '%s %f %f %f %f %f %f %f', 'Delimiter', ' ', ...
                        'MultipleDelimsAsOne', true, 'HeaderLines', headerLines);
        if isempty(data) || numel(data) < 8 || isempty(data{1})
            return;
        end

        imageNames = data{1};
        concentrations = data{2};
        replicates = data{3};
        xCenters = data{4};
        yCenters = data{5};
        semiMajorAxes = data{6};
        semiMinorAxes = data{7};
        rotationAngles = data{8};

        validRows = ~(isnan(concentrations) | isnan(replicates) | isnan(xCenters) | isnan(yCenters) | ...
                      isnan(semiMajorAxes) | isnan(semiMinorAxes) | isnan(rotationAngles));
        if ~any(validRows)
            return;
        end

        imageNames = imageNames(validRows);
        concentrations = concentrations(validRows);
        replicates = replicates(validRows);
        xCenters = xCenters(validRows);
        yCenters = yCenters(validRows);
        semiMajorAxes = semiMajorAxes(validRows);
        semiMinorAxes = semiMinorAxes(validRows);
        rotationAngles = rotationAngles(validRows);

        % Validate ellipse geometry constraints
        % 1. Check if semiMajorAxis >= semiMinorAxis (enforced in mask_utils.createEllipseMask but validate at parse time)
        invalidGeometry = semiMajorAxes < semiMinorAxes;
        if any(invalidGeometry)
            numInvalid = sum(invalidGeometry);
            warning('extract_features:invalidEllipseGeometry', ...
                   '%d patches have semiMinorAxis > semiMajorAxis. Axes will be swapped during mask creation.', ...
                   numInvalid);
        end

        % 2. Validate rotation angle is in valid range [-180, 180]
        invalidRotation = rotationAngles < -180 | rotationAngles > 180;
        if any(invalidRotation)
            numInvalidRot = sum(invalidRotation);
            warning('extract_features:invalidRotation', ...
                   '%d patches have rotation angles outside [-180, 180]. Clamping to valid range.', ...
                   numInvalidRot);
            rotationAngles = max(-180, min(180, rotationAngles));
        end

        % 3. Validate axes are positive
        invalidAxes = semiMajorAxes <= 0 | semiMinorAxes <= 0;
        if any(invalidAxes)
            numInvalidAxes = sum(invalidAxes);
            warning('extract_features:invalidEllipseAxes', ...
                   '%d patches have non-positive axis lengths. These patches will be skipped.', ...
                   numInvalidAxes);
            validRows = ~invalidAxes;
            imageNames = imageNames(validRows);
            concentrations = concentrations(validRows);
            replicates = replicates(validRows);
            xCenters = xCenters(validRows);
            yCenters = yCenters(validRows);
            semiMajorAxes = semiMajorAxes(validRows);
            semiMinorAxes = semiMinorAxes(validRows);
            rotationAngles = rotationAngles(validRows);

            if isempty(imageNames)
                return;
            end
        end

        [uniqueImages, ~, imageIndices] = unique(imageNames, 'stable');
        numUniqueImages = numel(uniqueImages);
        entryCount = numel(imageNames);

        for imgIdx = 1:numUniqueImages
            imgName = uniqueImages{imgIdx};
            patchMask = (imageIndices == imgIdx);
            numPatches = sum(patchMask);

            patchArray = struct('patchID', cell(1, numPatches), ...
                               'xCenter', cell(1, numPatches), ...
                               'yCenter', cell(1, numPatches), ...
                               'semiMajorAxis', cell(1, numPatches), ...
                               'semiMinorAxis', cell(1, numPatches), ...
                               'rotationAngle', cell(1, numPatches), ...
                               'concentration', cell(1, numPatches), ...
                               'replicate', cell(1, numPatches));

            patchIndices = find(patchMask);
            for j = 1:numPatches
                idx = patchIndices(j);
                patchArray(j).patchID = replicates(idx);
                patchArray(j).xCenter = xCenters(idx);
                patchArray(j).yCenter = yCenters(idx);
                patchArray(j).semiMajorAxis = semiMajorAxes(idx);
                patchArray(j).semiMinorAxis = semiMinorAxes(idx);
                patchArray(j).rotationAngle = rotationAngles(idx);
                patchArray(j).concentration = concentrations(idx);
                patchArray(j).replicate = replicates(idx);
            end

            coordinateData.images(imgName) = patchArray;
        end

        coordinateData.isValid = entryCount > 0;
        if coordinateData.isValid
            coordinateData.availableConcentrations = unique(concentrations);
            coordinateData.entryCount = entryCount;
        end

        fprintf('Parsed %d coordinate entries from %s\n', entryCount, coordinatesFilePath);

    catch ME
        warning('extract_features:coordinatesParsing', 'Error parsing coordinates file: %s', ME.message);
        coordinateData.isValid = false;
    end
end

function paperBackgroundMask = extractPaperBackgroundMaskWithPatches(originalImage, patches, cfg, maskUtils)
    %% Extract paper background mask by excluding all elliptical patch areas
    % Uses Otsu thresholding on the remaining pixels to isolate bright paper

    paperBackgroundMask = [];

    try
        [height, width, ~] = size(originalImage);

        % Create combined mask of all patch ellipses
        allPatchesMask = false(height, width);

        for i = 1:length(patches)
            patch = patches(i);
            patchMask = maskUtils.createEllipseMask([height, width], patch.xCenter, patch.yCenter, ...
                                                     patch.semiMajorAxis, patch.semiMinorAxis, patch.rotationAngle);

            % Add margin around patch to ensure clean separation
            if cfg.upad.patchMaskMarginFactor > 0
                marginPixels = max(cfg.upad.minHaloPx, round(patch.semiMajorAxis * cfg.upad.patchMaskMarginFactor));
                dilationSize = max(1, round(marginPixels));
                se = strel('disk', dilationSize, 0);
                patchMask = imdilate(patchMask, se);
            end

            allPatchesMask = allPatchesMask | patchMask;
        end

        % Invert to get background (everything NOT in patches)
        backgroundMask = ~allPatchesMask;

        % Validate sufficient background pixels before Otsu thresholding
        backgroundPixelCount = sum(backgroundMask(:));
        minPixelsForOtsu = max(cfg.upad.minPaperPixelsAbsolute * 2, round(numel(backgroundMask) * 0.05));

        if backgroundPixelCount < minPixelsForOtsu
            warning('extract_features:insufficientBackgroundForOtsu', ...
                   'Insufficient background pixels (%d < %d) for reliable Otsu thresholding. Patches may cover too much of image.', ...
                   backgroundPixelCount, minPixelsForOtsu);
            return;
        end

        % Convert to grayscale for thresholding (preserve uint8 scale if present)
        grayImg = rgb2gray(originalImage);

        % Estimate Otsu threshold using ONLY background pixels and guard against
        % augmentation-induced black fill dominating the histogram.
        bgPixels = grayImg(backgroundMask);

        % Exclude near-black padding from threshold estimation if present
        if ~isempty(bgPixels)
            cutoff = max(0, min(255, round(cfg.upad.blackFillCutoff)));
            bgNoFill = bgPixels(bgPixels > cutoff);
        else
            bgNoFill = bgPixels;
        end

        if isempty(bgNoFill)
            % Fall back to all background pixels
            bgNoFill = bgPixels;
        end

        % Compute histogram on ROI for robust Otsu
        h = imhist(bgNoFill, 256);
        hsum = sum(h);
        if hsum == 0
            % Degenerate case; bail out early
            return;
        end
        p = h / hsum;                             % normalized histogram
        localLevel = otsuthresh(p);               % [0,1] normalized threshold on ROI

        % Identify bright paper pixels using the ROI Otsu threshold
        brightPixelsMask = grayImg >= (localLevel * 255);

        % Combine: paper = background AND bright pixels
        paperMask = backgroundMask & brightPixelsMask;

        % Clean up small artifacts
        if sum(paperMask(:)) > cfg.upad.minPaperPixelsAbsolute
            minArea = max(cfg.upad.minPaperPixelsAbsolute, ...
                         round(numel(paperMask) * cfg.upad.cleanupMinAreaFraction));
            paperMask = bwareaopen(paperMask, minArea);
        end

        % Validate sufficient paper pixels
        paperPixelCount = sum(paperMask(:));
        minRequired = max(cfg.upad.minPaperPixelsAbsolute, ...
                         round(numel(paperMask) * cfg.upad.minPaperFraction));

        if paperPixelCount >= minRequired
            paperBackgroundMask = paperMask;
        else
            warning('extract_features:insufficientPaper', ...
                   'Insufficient paper pixels detected (%d < %d threshold)', ...
                   paperPixelCount, minRequired);
        end

    catch ME
        warning('extract_features:paperMaskExtraction', ...
               'Error extracting paper background mask: %s', ME.message);
    end
end




