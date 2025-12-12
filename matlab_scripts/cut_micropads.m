function cut_micropads(varargin)
    %% microPAD Colorimetric Analysis — Unified microPAD Processing Tool
    %% Detect and extract concentration regions from raw microPAD images
    %% Author: Veysel Y. Yilmaz
    %
    % This script combines rotation adjustment and AI-powered quad detection
    % to directly process raw microPAD images into concentration region crops.
    %
    % Pipeline stage: 1_dataset → 2_micropads
    %
    % Features:
    %   - Interactive rotation adjustment with memory
    %   - AI-powered quad detection (YOLOv11s-pose)
    %   - Manual quad editing and refinement
    %   - Saves quad coordinates with rotation angle
    %
    % Inputs (Name-Value pairs):
    % - 'numSquares': number of regions to capture per strip (default: 7)
    % - 'aspectRatio': width/height ratio of each region (default: 1.0, perfect squares)
    % - 'coverage': fraction of image width to fill (default: 0.80)
    % - 'gapPercent': gap as percent of region width, 0..1 or 0..100 (default: 0.19)
    % - 'inputFolder' | 'outputFolder': override default I/O folders
    % - 'saveCoordinates': output behavior
    % - 'useAIDetection': use YOLO for initial quad placement (default: true)
    % - 'detectionModel': path to YOLOv11 pose model (default: 'models/yolo11s-micropad-pose-1280.pt')
    % - 'minConfidence': minimum detection confidence (default: 0.6)
    % - 'inferenceSize': YOLO inference image size in pixels (default: 1280)
    % - 'pythonPath': path to Python executable (default: '' - uses MICROPAD_PYTHON env var)
    %
    % Outputs/Side effects:
    % - Writes PNG quad crops to 2_micropads/[phone]/con_*/
    % - Writes consolidated coordinates.txt at phone level (atomic, no duplicate rows per image)
    %
    % ROTATION SEMANTICS:
    %   The rotation column in coordinates.txt is a UI-only alignment hint that
    %   records how much the user rotated the image to facilitate labeling. This
    %   value is NOT applied by downstream processing (extraction, augmentation,
    %   feature extraction). All saved coordinates are in the original (unrotated)
    %   image reference frame.
    %
    % Behavior:
    % - Shows interactive UI with drawpolygon editing for every image
    % - If useAIDetection=true, attempts AI detection for initial placement
    % - If AI fails or disabled, uses default geometry (aspectRatio, coverage, gapPercent)
    % - User can manually adjust quads before saving
    % - Cuts N region crops and saves into con_0..con_(N-1) subfolders for each strip
    % - All quad coordinates written to single phone-level coordinates.txt
    %
    % Examples:
    %   cut_micropads('numSquares', 7)
    %   cut_micropads('numSquares', 7, 'useAIDetection', true)
    %   cut_micropads('useAIDetection', true, 'minConfidence', 0.7)

    %% ========================================================================
    %% EXPERIMENT CONFIGURATION CONSTANTS
    %% ========================================================================
    if mod(length(varargin), 2) ~= 0
        error('cut_micropads:invalid_args', 'Parameters must be provided as name-value pairs');
    end

    % Error handling for deprecated format parameters
    if ~isempty(varargin) && (any(strcmpi(varargin(1:2:end), 'preserveFormat')) || any(strcmpi(varargin(1:2:end), 'jpegQuality')))
        error('cut_micropads:deprecated_parameter', ...
              ['JPEG format no longer supported. Pipeline outputs PNG exclusively.\n' ...
               'Remove ''preserveFormat'' and ''jpegQuality'' parameters from function call.']);
    end

    % === DATASET AND FOLDER STRUCTURE ===
    INPUT_FOLDER = '1_dataset';
    OUTPUT_FOLDER = '2_micropads';
    OUTPUT_FOLDER_ELLIPSES = '3_elliptical_regions';

    % === OUTPUT FORMATTING ===
    SAVE_COORDINATES = true;

    % === DEFAULT GEOMETRY / SELECTION ===
    DEFAULT_NUM_SQUARES = 7;
    DEFAULT_ASPECT_RATIO = 1.0;  % width/height ratio: 1.0 = perfect squares
    DEFAULT_COVERAGE = 0.80;     % regions span 80% of image width
    DEFAULT_GAP_PERCENT = 0.19;  % 19% gap between regions

    % === ELLIPSE EDITING CONFIGURATION ===
    DEFAULT_ENABLE_ELLIPSE_EDITING = true;
    DEFAULT_ENABLE_QUAD_EDITING = false;
    REPLICATES_PER_CONCENTRATION = 3;

    % Ellipse definitions in normalized micropad coordinates [0,1]×[0,1]
    % Each row: [x, y, semiMajorAxis, semiMinorAxis, rotationAngle]
    % - x, y: center position (0,0 = top-left, 1,1 = bottom-right)
    % - semiMajorAxis, semiMinorAxis: fraction of micropad side length
    % - rotationAngle: degrees, positive = clockwise from horizontal (matches image coordinates)
    % These values define ellipse positions relative to an ideal square micropad.
    % The homography transform will adjust for perspective distortion.
    ELLIPSE_DEFAULT_RECORDS = [
        % x,    y,    semiMajor, semiMinor, rotationAngle
        0.263306,  0.434395,  0.077779,  0.071073,  178.189;  % Replicate 0 (Urea)
        0.497134,  0.342967,  0.076068,  0.069731,  176.362;  % Replicate 1 (Creatinine)
        0.721795,  0.435818,  0.074423,  0.068091,  175.217   % Replicate 2 (Lactate)
    ];

    % === ELLIPSE LAYOUT PARAMETERS (legacy, for fallback) ===
    MARGIN_TO_SPACING_RATIO = 1/3;
    VERTICAL_POSITION_RATIO = 1/3;

    % === ELLIPSE GEOMETRY DEFAULTS ===
    SEMI_MAJOR_DEFAULT_RATIO = 0.05;
    SEMI_MINOR_DEFAULT_RATIO = 0.85;
    MIN_AXIS_PERCENT = 0.005;

    % === AI DETECTION DEFAULTS ===
    DEFAULT_USE_AI_DETECTION = false;
    DEFAULT_DETECTION_MODEL = 'models/yolo11s-micropad-pose-1280.pt';
    DEFAULT_MIN_CONFIDENCE = 0.6;

    % IMPORTANT: Edit this path to match your Python installation!
    % Common locations:
    %   Windows: 'C:\Users\YourName\miniconda3\envs\YourPythonEnv\python.exe'
    %   macOS:   '/Users/YourName/miniconda3/envs/YourPythonEnv/bin/python'
    %   Linux:   '/home/YourName/miniconda3/envs/YourPythonEnv/bin/python'
    DEFAULT_PYTHON_PATH = 'C:\Users\veyse\miniconda3\envs\microPAD-python-env\python.exe';
    DEFAULT_INFERENCE_SIZE = 1280;

    % === ROTATION CONSTANTS ===
    ROTATION_ANGLE_TOLERANCE = 1e-6;  % Tolerance for detecting exact 90-degree rotations

    % === NAMING / FILE CONSTANTS ===
    COORDINATE_FILENAME = 'coordinates.txt';
    SUPPORTED_FORMATS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'};
    ALLOWED_IMAGE_EXTENSIONS = {'*.jpg','*.jpeg','*.png','*.bmp','*.tiff','*.tif'};
    CONC_FOLDER_PREFIX = 'con_';

    %% Add helper_scripts to path (contains geometry_transform and other utilities)
    scriptDir = fileparts(mfilename('fullpath'));
    helperDir = fullfile(scriptDir, 'helper_scripts');
    if exist(helperDir, 'dir')
        addpath(helperDir);
    end

    %% Load utility modules (geometry, I/O, UI, model integration)
    geomTform = geometry_transform();
    imageIO = image_io();
    pathUtils = path_utils();
    yoloUtils = yolo_integration();
    fileIOMgr = file_io_manager();
    micropadUI = micropad_ui();

    % UI configuration from shared module
    uiConfig = micropadUI.getDefaultUIConfig();

    %% Build configuration
    cfg = createConfiguration(INPUT_FOLDER, OUTPUT_FOLDER, OUTPUT_FOLDER_ELLIPSES, SAVE_COORDINATES, ...
                              DEFAULT_NUM_SQUARES, DEFAULT_ASPECT_RATIO, DEFAULT_COVERAGE, DEFAULT_GAP_PERCENT, ...
                              DEFAULT_ENABLE_ELLIPSE_EDITING, DEFAULT_ENABLE_QUAD_EDITING, REPLICATES_PER_CONCENTRATION, ...
                              MARGIN_TO_SPACING_RATIO, VERTICAL_POSITION_RATIO, MIN_AXIS_PERCENT, ...
                              SEMI_MAJOR_DEFAULT_RATIO, SEMI_MINOR_DEFAULT_RATIO, ELLIPSE_DEFAULT_RECORDS, ...
                              DEFAULT_USE_AI_DETECTION, DEFAULT_DETECTION_MODEL, DEFAULT_MIN_CONFIDENCE, DEFAULT_PYTHON_PATH, DEFAULT_INFERENCE_SIZE, ...
                              ROTATION_ANGLE_TOLERANCE, ...
                              COORDINATE_FILENAME, SUPPORTED_FORMATS, ALLOWED_IMAGE_EXTENSIONS, CONC_FOLDER_PREFIX, uiConfig, ...
                              geomTform, pathUtils, imageIO, yoloUtils, fileIOMgr, micropadUI, varargin{:});

    try
        processAllFolders(cfg);
        fprintf('>> microPAD processing completed successfully!\n');
    catch ME
        handleError(ME);
    end
end

%% -------------------------------------------------------------------------
%% Configuration
%% -------------------------------------------------------------------------

function cfg = createConfiguration(inputFolder, outputFolder, outputFolderEllipses, saveCoordinates, ...
                                   defaultNumSquares, defaultAspectRatio, defaultCoverage, defaultGapPercent, ...
                                   defaultEnableEllipseEditing, defaultEnableQuadEditing, replicatesPerConcentration, ...
                                   marginToSpacingRatio, verticalPositionRatio, minAxisPercent, ...
                                   semiMajorDefaultRatio, semiMinorDefaultRatio, ellipseDefaultRecords, ...
                                   defaultUseAI, defaultDetectionModel, defaultMinConfidence, defaultPythonPath, defaultInferenceSize, ...
                                   rotationAngleTolerance, ...
                                   coordinateFileName, supportedFormats, allowedImageExtensions, concFolderPrefix, uiConfig, ...
                                   geomTform, pathUtils, imageIO, yoloUtils, fileIOMgr, micropadUI, varargin)
    parser = inputParser;
    parser.addParameter('numSquares', defaultNumSquares, @(x) validateattributes(x, {'numeric'}, {'scalar','integer','>=',1,'<=',20}));

    validateFolder = @(s) validateattributes(s, {'char', 'string'}, {'nonempty', 'scalartext'});
    parser.addParameter('inputFolder', inputFolder, validateFolder);
    parser.addParameter('outputFolder', outputFolder, validateFolder);
    parser.addParameter('saveCoordinates', saveCoordinates, @(x) islogical(x));

    parser.addParameter('aspectRatio', defaultAspectRatio, @(x) validateattributes(x, {'numeric'}, {'scalar','>',0}));
    parser.addParameter('coverage', defaultCoverage, @(x) validateattributes(x, {'numeric'}, {'scalar','>',0,'<=',1}));
    parser.addParameter('gapPercent', defaultGapPercent, @(x) isnumeric(x) && isscalar(x) && x>=0);

    parser.addParameter('enableEllipseEditing', defaultEnableEllipseEditing, @islogical);
    parser.addParameter('enableQuadEditing', defaultEnableQuadEditing, @islogical);

    parser.addParameter('useAIDetection', defaultUseAI, @islogical);
    parser.addParameter('detectionModel', defaultDetectionModel, @(x) validateattributes(x, {'char', 'string'}, {'nonempty', 'scalartext'}));
    parser.addParameter('minConfidence', defaultMinConfidence, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));
    parser.addParameter('pythonPath', defaultPythonPath, @(x) ischar(x) || isstring(x));
    parser.addParameter('inferenceSize', defaultInferenceSize, @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>', 0}));

    parser.parse(varargin{:});

    cfg.numSquares = parser.Results.numSquares;

    if cfg.numSquares > 15
        warning('cut_micropads:many_squares', 'Large numSquares (%d) may cause UI layout issues and small regions', cfg.numSquares);
    end

    % Store model path (relative), will be resolved in addPathConfiguration
    cfg.useAIDetection = parser.Results.useAIDetection;
    cfg.detectionModelRelative = parser.Results.detectionModel;
    cfg.minConfidence = parser.Results.minConfidence;
    cfg.pythonPath = parser.Results.pythonPath;
    cfg.inferenceSize = parser.Results.inferenceSize;

    cfg = addPathConfiguration(cfg, parser.Results.inputFolder, parser.Results.outputFolder, outputFolderEllipses, pathUtils);

    cfg.output.saveCoordinates = parser.Results.saveCoordinates;
    cfg.output.supportedFormats = supportedFormats;
    cfg.allowedImageExtensions = allowedImageExtensions;

    cfg.coordinateFileName = coordinateFileName;
    cfg.concFolderPrefix = concFolderPrefix;

    % Geometry configuration
    cfg.geometry = struct();
    cfg.geometry.aspectRatio = parser.Results.aspectRatio;
    gp = parser.Results.gapPercent;
    if gp > 100
        error('cut_micropads:invalid_gap', 'gapPercent cannot exceed 100 (got %.2f)', gp);
    end
    if gp > 1
        gp = gp / 100;
    end
    cfg.geometry.gapPercentWidth = gp;
    cfg.coverage = parser.Results.coverage;

    % Ellipse editing configuration
    cfg.enableEllipseEditing = parser.Results.enableEllipseEditing;
    cfg.enableQuadEditing = parser.Results.enableQuadEditing;
    cfg.ellipse = struct();
    cfg.ellipse.replicatesPerMicropad = replicatesPerConcentration;
    cfg.ellipse.marginToSpacingRatio = marginToSpacingRatio;
    cfg.ellipse.verticalPositionRatio = verticalPositionRatio;
    cfg.ellipse.horizontalPositionRatio = 0.5;  % X-position ratio for vertical layout (centered)
    cfg.ellipse.minAxisPercent = minAxisPercent;
    cfg.ellipse.semiMajorDefaultRatio = semiMajorDefaultRatio;
    cfg.ellipse.semiMinorDefaultRatio = semiMinorDefaultRatio;
    cfg.ellipse.defaultRecords = ellipseDefaultRecords;  % Nx5 matrix [x, y, semiMajor, semiMinor, rotation]

    % Validate ellipse default records matrix dimensions
    if size(cfg.ellipse.defaultRecords, 1) ~= cfg.ellipse.replicatesPerMicropad
        error('cut_micropads:ellipse_records_mismatch', ...
              'ELLIPSE_DEFAULT_RECORDS has %d rows but REPLICATES_PER_CONCENTRATION is %d. They must match.', ...
              size(cfg.ellipse.defaultRecords, 1), cfg.ellipse.replicatesPerMicropad);
    end
    if size(cfg.ellipse.defaultRecords, 2) ~= 5
        error('cut_micropads:ellipse_records_columns', ...
              'ELLIPSE_DEFAULT_RECORDS must have 5 columns [x, y, semiMajor, semiMinor, rotation]. Got %d.', ...
              size(cfg.ellipse.defaultRecords, 2));
    end

    % Store geometry transform utilities (homography + geometry operations)
    cfg.geomTform = geomTform;

    % Store pathUtils for directory operations
    cfg.pathUtils = pathUtils;

    % Store imageIO for image loading
    cfg.imageIO = imageIO;

    % Store yoloUtils for YOLO detection
    cfg.yoloUtils = yoloUtils;

    % Store fileIOMgr for file I/O operations
    cfg.fileIOMgr = fileIOMgr;

    % Rotation configuration
    cfg.rotation.angleTolerance = rotationAngleTolerance;

    % UI configuration
    cfg.ui = uiConfig;
    cfg.dimFactor = uiConfig.dimFactor;
    cfg.micropadUI = micropadUI;
end

function cfg = addPathConfiguration(cfg, inputFolder, outputFolder, outputFolderEllipses, pathUtils)
    projectRoot = pathUtils.findProjectRoot(inputFolder);

    cfg.projectRoot = projectRoot;
    cfg.inputPath = fullfile(projectRoot, inputFolder);
    cfg.outputPath = fullfile(projectRoot, outputFolder);
    cfg.outputPathEllipses = fullfile(projectRoot, outputFolderEllipses);

    % Add helper_scripts to MATLAB path if not already present
    helperScriptsPath = fullfile(projectRoot, 'matlab_scripts', 'helper_scripts');
    if isfolder(helperScriptsPath) && ~contains(path, helperScriptsPath)
        addpath(helperScriptsPath);
    end

    % Resolve model path to absolute path
    cfg.detectionModel = fullfile(projectRoot, cfg.detectionModelRelative);

    % Resolve Python script path
    cfg.pythonScriptPath = fullfile(projectRoot, 'python_scripts', 'detect_quads.py');

    % Validate Python script and model file if AI detection enabled
    if cfg.useAIDetection
        if ~isfile(cfg.pythonScriptPath)
            warning('cut_micropads:script_missing', ...
                'AI detection enabled but Python script not found: %s\nDisabling AI detection.', cfg.pythonScriptPath);
            cfg.useAIDetection = false;
        elseif ~isfile(cfg.detectionModel)
            warning('cut_micropads:model_missing', ...
                'AI detection enabled but model not found: %s\nDisabling AI detection.', cfg.detectionModel);
            cfg.useAIDetection = false;
        end
    end

    validatePaths(cfg);
end

function validatePaths(cfg)
    if ~isfolder(cfg.inputPath)
        error('cut_micropads:missing_input', 'Input folder not found: %s', cfg.inputPath);
    end
    if ~isfolder(cfg.outputPath)
        mkdir(cfg.outputPath);
    end
end

%% -------------------------------------------------------------------------
%% Main Processing Loop
%% -------------------------------------------------------------------------

function processAllFolders(cfg)
    fprintf('\n=== Starting microPAD Processing ===\n');
    fprintf('Input: %s\n', cfg.inputPath);
    fprintf('Output: %s\n', cfg.outputPath);
    fprintf('AI Detection: %s\n', string(cfg.useAIDetection));
    if cfg.useAIDetection
        fprintf('Detection model: %s\n', cfg.detectionModel);
        fprintf('Min confidence: %.2f\n', cfg.minConfidence);
    end
    fprintf('Regions per strip: %d\n\n', cfg.numSquares);

    cfg.pathUtils.executeInFolder(cfg.inputPath, @() processPhones(cfg));
end

function processPhones(cfg)
    phoneFolders = cfg.pathUtils.listSubfolders('.');
    if isempty(phoneFolders)
        warning('cut_micropads:no_phones', 'No phone folders found in input directory');
        return;
    end

    for i = 1:numel(phoneFolders)
        processPhone(phoneFolders{i}, cfg);
    end
end

function processPhone(phoneName, cfg)
    fprintf('\n=== Processing Phone: %s ===\n', phoneName);
    cfg.pathUtils.executeInFolder(phoneName, @() processImagesInPhone(phoneName, cfg));
end

function processImagesInPhone(phoneName, cfg)
    imageList = getImageFiles('.', cfg.allowedImageExtensions);
    if isempty(imageList)
        warning('cut_micropads:no_images', 'No images found for phone folder: %s', phoneName);
        return;
    end

    fprintf('Found %d images\n', numel(imageList));

    outputDirs = createOutputDirectory(cfg.outputPath, cfg.outputPathEllipses, phoneName, cfg.numSquares, cfg.concFolderPrefix);

    % Setup Python environment once per phone if AI detection is enabled
    if cfg.useAIDetection
        cfg.yoloUtils.ensurePythonSetup(cfg.pythonPath);
    end

    persistentFig = [];
    memory = initializeMemory();

    try
        for idx = 1:numel(imageList)
            if ~isempty(persistentFig) && ~isvalid(persistentFig)
                persistentFig = [];
            end
            [success, persistentFig, memory] = processOneImage(imageList{idx}, outputDirs, cfg, persistentFig, phoneName, memory);
            if success
                fprintf('  >> Saved %d concentration regions\n', cfg.numSquares);
            else
                fprintf('  Image skipped by user\n');
            end
        end
    catch ME
        if ~isempty(persistentFig) && isvalid(persistentFig)
            close(persistentFig);
        end
        rethrow(ME);
    end

    if ~isempty(persistentFig) && isvalid(persistentFig)
        close(persistentFig);
    end

    fprintf('Completed: %s\n', phoneName);
end

function [success, fig, memory] = processOneImage(imageName, outputDirs, cfg, fig, phoneName, memory)
    success = false;

    fprintf('  -> Processing: %s\n', imageName);

    [img, isValid] = cfg.imageIO.loadImage(imageName);
    if ~isValid
        fprintf('  !! Failed to load image\n');
        return;
    end

    % Get image dimensions
    [imageHeight, imageWidth, ~] = size(img);

    % Initialize rotation (may be overridden by mode-specific logic)
    initialRotation = 0;

    % Determine operating mode based on enableQuadEditing and enableEllipseEditing
    if cfg.enableQuadEditing && cfg.enableEllipseEditing
        processingMode = 1; % Unified: quad edit -> ellipse edit -> preview
    elseif cfg.enableQuadEditing && ~cfg.enableEllipseEditing
        processingMode = 2; % Quad-only: quad edit -> preview
    elseif ~cfg.enableQuadEditing && cfg.enableEllipseEditing
        processingMode = 3; % Ellipse-only: [load/default] -> ellipse edit -> preview
    else
        processingMode = 4; % Read-only preview: [load coords] -> preview
    end

    % Mode 4: Read-only preview (both editing disabled)
    if processingMode == 4
        % Try loading quad coordinates
        quadCoordFile = fullfile(outputDirs.quadDir, cfg.coordinateFileName);
        [loadedQuads, quadsFound] = loadQuadCoordinates(quadCoordFile, imageName, cfg.numSquares);

        % Try loading ellipse coordinates
        ellipseCoordFile = fullfile(outputDirs.ellipseDir, cfg.coordinateFileName);
        [loadedEllipses, ellipsesFound] = loadEllipseCoordinates(ellipseCoordFile, imageName);

        % Error if neither found
        if ~quadsFound && ~ellipsesFound
            error('cut_micropads:no_coordinates_for_preview', ...
                ['No coordinate files found for preview mode (image: %s).\n' ...
                 'Enable at least one editing mode or ensure coordinate files exist in:\n' ...
                 '  - %s (quads)\n' ...
                 '  - %s (ellipses)'], ...
                imageName, quadCoordFile, ellipseCoordFile);
        end

        % Create figure if needed
        if isempty(fig) || ~isvalid(fig)
            fig = cfg.micropadUI.createFigure(imageName, phoneName, cfg, @keyPressHandler, @(src, evt) cleanupAndClose(src));
        end

        % Build read-only preview UI
        buildReadOnlyPreviewUI(fig, img, imageName, phoneName, cfg, loadedQuads, quadsFound, ...
                              loadedEllipses, ellipsesFound, initialRotation);

        % Wait for user to advance to next image
        [action, ~, ~] = waitForUserAction(fig);

        if strcmp(action, 'stop')
            if isvalid(fig)
                delete(fig);
            end
            error('User stopped execution');
        else
            % No save operation in preview mode - just advance to next image
            success = true;
            return;
        end
    end

    % Mode 3: Ellipse-only editing
    if processingMode == 3
        % Create figure if needed
        if isempty(fig) || ~isvalid(fig)
            fig = cfg.micropadUI.createFigure(imageName, phoneName, cfg, @keyPressHandler, @(src, evt) cleanupAndClose(src));
        end

        % Clear previous UI elements before building new UI (fixes image leaking)
        existingGuiData = get(fig, 'UserData');
        if ~isempty(existingGuiData)
            clearAllUIElements(fig, existingGuiData);
        end

        % Try loading quad coordinates for positioning context (includes rotation)
        quadCoordFile = fullfile(outputDirs.quadDir, cfg.coordinateFileName);
        [loadedQuads, quadsFound, loadedRotation] = loadQuadCoordinates(quadCoordFile, imageName, cfg.numSquares);

        orientation = 'horizontal';
        if quadsFound
            % Use loaded quads for positioning
            quadParams = loadedQuads;
            % Use the rotation from coordinates.txt to align image for easier editing
            initialRotation = loadedRotation;
            fprintf('  Mode 3: Loaded %d quad coordinates with rotation=%.1f° for ellipse positioning\n', size(quadParams, 1), initialRotation);

            % Determine strip orientation in DISPLAY space to seed ellipses correctly.
            % Loaded quads are in base (unrotated) coordinates, but ellipse defaults
            % are defined relative to the rotated display view.
            baseSize = [imageHeight, imageWidth];
            displaySize = computeDisplayImageSize(baseSize, initialRotation, cfg);
            displayQuadsForOrientation = convertBaseQuadsToDisplay(quadParams, baseSize, displaySize, initialRotation, cfg);
            [~, orientation] = sortQuadArrayByX(displayQuadsForOrientation);

            % Go directly to ellipse editing with quad overlays (image will be rotated for display)
            buildEllipseEditingUI(fig, img, imageName, phoneName, cfg, quadParams, initialRotation, memory, orientation);

        else
            % No quads - use default grid layout
            fprintf('  Mode 3: No quad coordinates - using default grid layout\n');

            % Create default ellipse positions using grid
            imageSize = [size(img, 1), size(img, 2)];
            numReplicates = cfg.ellipse.replicatesPerMicropad;
            defaultPositions = createDefaultEllipseGrid(imageSize, cfg.numSquares, numReplicates, cfg);

            % Build ellipse editing UI without quad overlays
            buildEllipseEditingUIGridMode(fig, img, imageName, phoneName, cfg, defaultPositions, initialRotation, memory);
            quadParams = []; % No quads in grid mode
        end

        % Wait for user action
        [action, ~, ~] = waitForUserAction(fig);

        if strcmp(action, 'stop') || isempty(action)
            terminateExecution(fig);
        elseif strcmp(action, 'retry') || strcmp(action, 'back')
            % Re-process same image - recurse (BACK treated as retry in Mode 3)
            [success, fig, memory] = processOneImage(imageName, outputDirs, cfg, fig, phoneName, memory);
            return;
        elseif strcmp(action, 'accept')
            % Extract ellipse data from UI
            guiData = get(fig, 'UserData');
            ellipseData = [];
            if isfield(guiData, 'ellipseData') && ~isempty(guiData.ellipseData)
                ellipseData = guiData.ellipseData;
            end

            % Transition to preview mode before saving (like Mode 1)
            clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, quadParams, initialRotation, memory, [], orientation, ellipseData);

            % Wait for preview action
            [prevAction, ~, ~] = waitForUserAction(fig);

            % Defensive check: if figure was closed/deleted, exit cleanly
            if ~isvalid(fig) || isempty(prevAction)
                return;
            end

            switch prevAction
                case 'accept'
                    % Save ONLY ellipse outputs (skip quad outputs)
                    if ~isempty(ellipseData)
                        cfg.fileIOMgr.saveEllipseData(img, imageName, quadParams, ellipseData, outputDirs.ellipseDir, cfg, cfg.fileIOMgr);
                        fprintf('  Saved %d elliptical regions\n', size(ellipseData, 1));

                        % Update memory for next image (Mode 3) - store base quads
                        memory = updateMemory(memory, quadParams, initialRotation, [imageHeight, imageWidth], ellipseData, cfg);
                    end
                    success = true;
                    return;
                case 'retry'
                    % Re-process same image - recurse
                    [success, fig, memory] = processOneImage(imageName, outputDirs, cfg, fig, phoneName, memory);
                    return;
                case {'skip', 'stop'}
                    if strcmp(prevAction, 'stop')
                        terminateExecution(fig);
                    end
                    return;
            end
        end
    end

    % Modes 1 & 2: Get initial quad positions and rotation (memory or default, NOT AI yet)
    [initialQuads, initialRotation, ~] = getInitialQuadsWithMemory(img, cfg, memory, [imageHeight, imageWidth]);

    % Memory quads are exact display coordinates - use them directly
    [initialQuads, orientation] = sortQuadArrayByX(initialQuads);

    % Display GUI immediately with memory/default quads and rotation
    [quadParams, ~, fig, rotation, ellipseData, orientation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialQuads, fig, initialRotation, memory, orientation);

    % NOTE: If AI detection is enabled, it will run asynchronously AFTER GUI is displayed

    if ~isempty(quadParams)
        cfg.fileIOMgr.saveCroppedRegions(img, imageName, quadParams, outputDirs.quadDir, cfg, rotation, cfg.fileIOMgr);
        % Update memory with base quad coordinates and rotation
        memory = updateMemory(memory, quadParams, rotation, [imageHeight, imageWidth], ellipseData, cfg, orientation);

        % Save ellipse data if ellipse editing was enabled
        if cfg.enableEllipseEditing && ~isempty(ellipseData)
            cfg.fileIOMgr.saveEllipseData(img, imageName, quadParams, ellipseData, outputDirs.ellipseDir, cfg, cfg.fileIOMgr);
        end

        success = true;
    end
end



%% -------------------------------------------------------------------------
%% Interactive UI
%% -------------------------------------------------------------------------

function [quadParams, displayQuads, fig, rotation, ellipseData, orientation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialQuads, fig, initialRotation, memory, orientation)
    % Show interactive GUI with editing, ellipse editing (optional), and preview modes
    quadParams = [];
    displayQuads = [];
    rotation = 0;
    ellipseData = [];

    % Create figure if needed
    if isempty(fig) || ~isvalid(fig)
        fig = cfg.micropadUI.createFigure(imageName, phoneName, cfg, @keyPressHandler, @(src, evt) cleanupAndClose(src));
    end

    % Use rotation from memory (or 0 if no memory)
    if nargin < 7
        initialRotation = 0;
    end

    % Initialize memory if not provided
    if nargin < 8 || isempty(memory)
        memory = initializeMemory();
    end

    % Initialize orientation if not provided
    if nargin < 9 || isempty(orientation)
        orientation = 'horizontal';
    end

    while true
        % Quad editing mode
        clearAndRebuildUI(fig, 'editing', img, imageName, phoneName, cfg, initialQuads, initialRotation, memory, [], orientation);

        [action, userQuads, userRotation] = waitForUserAction(fig);

        % Defensive check: if figure was closed/deleted, exit cleanly
        if ~isvalid(fig) || isempty(action)
            return;
        end

        switch action
            case 'skip'
                return;
            case 'stop'
                if isvalid(fig)
                    delete(fig);
                end
                error('User stopped execution');
            case 'accept'
                guiDataEditing = get(fig, 'UserData');
                baseQuads = convertDisplayQuadsToBase(guiDataEditing, userQuads, cfg);
                savedRotation = userRotation;
                savedDisplayQuads = userQuads;
                savedBaseQuads = baseQuads;

                % Retrieve updated orientation from editing mode (may have changed due to rotation/AI)
                if isfield(guiDataEditing, 'orientation') && ~isempty(guiDataEditing.orientation)
                    orientation = guiDataEditing.orientation;
                end

                % Ellipse editing mode (if enabled)
                if cfg.enableEllipseEditing
                    viewState = captureViewState(guiDataEditing);
                    clearAndRebuildUI(fig, 'ellipse_editing', img, imageName, phoneName, cfg, savedBaseQuads, savedRotation, memory, viewState, orientation);

                    [ellipseAction, ellipseCoords] = waitForUserAction(fig);

                    % Defensive check
                    if ~isvalid(fig) || isempty(ellipseAction)
                        return;
                    end

                    switch ellipseAction
                        case 'back'
                            % Return to quad editing
                            initialQuads = savedDisplayQuads;
                            initialRotation = savedRotation;
                            continue;
                        case 'skip'
                            return;
                        case 'stop'
                            terminateExecution(fig);
                        case 'accept'
                            % Store ellipse data and proceed to preview
                            savedEllipseData = ellipseCoords;
                    end
                else
                    % Skip ellipse editing, no ellipse data
                    savedEllipseData = [];
                end

                % Preview mode (pass ellipse data directly to buildPreviewUI via clearAndRebuildUI)
                clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, savedBaseQuads, savedRotation, memory, [], orientation, savedEllipseData);

                % Store rotation in guiData for preview mode (ellipse data already stored by buildPreviewUI)
                guiData = get(fig, 'UserData');
                guiData.savedRotation = savedRotation;
                set(fig, 'UserData', guiData);

                [prevAction, ~, ~] = waitForUserAction(fig);

                % Defensive check: if figure was closed/deleted, exit cleanly
                if ~isvalid(fig) || isempty(prevAction)
                    return;
                end

                switch prevAction
                    case 'accept'
                        quadParams = savedBaseQuads;
                        displayQuads = savedDisplayQuads;
                        rotation = savedRotation;
                        ellipseData = savedEllipseData;
                        return;
                    case {'skip', 'stop'}
                        if strcmp(prevAction, 'stop')
                            if isvalid(fig)
                                delete(fig);
                            end
                            error('User stopped execution');
                        end
                        return;
                    case 'retry'
                        % Use edited quads as new initial positions
                        initialQuads = savedDisplayQuads;
                        initialRotation = savedRotation;
                        continue;
                end
        end
    end
end

function clearAndRebuildUI(fig, mode, img, imageName, phoneName, cfg, quadParams, initialRotation, memory, viewState, orientation, ellipseData)
    % Modes: 'editing' (quad adjustment), 'ellipse_editing' (ellipse placement), 'preview' (final confirmation)
    % Default argument handling (ordered from lowest to highest nargin)
    if nargin < 8
        initialRotation = 0;
    end

    if nargin < 9
        memory = initializeMemory();
    end

    if nargin < 10
        viewState = [];
    end

    if nargin < 11 || isempty(orientation)
        orientation = 'horizontal';
    end

    if nargin < 12 || isempty(ellipseData)
        ellipseData = [];
    end

    guiData = get(fig, 'UserData');
    clearAllUIElements(fig, guiData);

    switch mode
        case 'editing'
            buildEditingUI(fig, img, imageName, phoneName, cfg, quadParams, initialRotation);

        case 'ellipse_editing'
            buildEllipseEditingUI(fig, img, imageName, phoneName, cfg, quadParams, initialRotation, memory, orientation);

        case 'preview'
            buildPreviewUI(fig, img, imageName, phoneName, cfg, quadParams, ellipseData);
    end

    if ~isempty(viewState)
        applyViewState(fig, viewState);
    end
end

function clearAllUIElements(fig, guiData)
    % Delete all UI elements
    allObjects = findall(fig);
    if isempty(allObjects)
        set(fig, 'UserData', []);
        return;
    end

    objTypes = get(allObjects, 'Type');
    if ~iscell(objTypes), objTypes = {objTypes}; end

    isControl = strcmp(objTypes, 'uicontrol');
    isPanel = strcmp(objTypes, 'uipanel');
    isAxes = strcmp(objTypes, 'axes');

    toDelete = allObjects(isControl | isPanel | isAxes);

    % Add quadROIs from guiData
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'quads')
        validPolys = collectValidQuads(guiData);
        if ~isempty(validPolys)
            toDelete = [toDelete; validPolys];
        end
    end

    % Add quadlabels from guiData
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'quadLabels')
        numLabels = numel(guiData.quadLabels);
        validLabels = gobjects(numLabels, 1);
        validCount = 0;
        for i = 1:numLabels
            if isvalid(guiData.quadLabels{i})
                validCount = validCount + 1;
                validLabels(validCount) = guiData.quadLabels{i};
            end
        end
        validLabels = validLabels(1:validCount);
        if ~isempty(validLabels)
            toDelete = [toDelete; validLabels];
        end
    end

    % Bulk delete
    if ~isempty(toDelete)
        validMask = arrayfun(@isvalid, toDelete);
        delete(toDelete(validMask));
    end

    % Cleanup remaining ROIs
    rois = findobj(fig, '-isa', 'images.roi.Polygon');
    if ~isempty(rois)
        validRois = rois(arrayfun(@isvalid, rois));
        if ~isempty(validRois)
            delete(validRois);
        end
    end

    % Clean up timer if still running
    if ~isempty(guiData) && isstruct(guiData)
        if isfield(guiData, 'aiTimer')
            if isfield(guiData, 'cfg') && isfield(guiData.cfg, 'micropadUI')
                guiData.cfg.micropadUI.safeStopTimer(guiData.aiTimer);
            elseif isvalid(guiData.aiTimer)
                try
                    stop(guiData.aiTimer);
                catch
                    % Ignore: timer may already be stopped/deleted
                end
                delete(guiData.aiTimer);
            end
            guiData.aiTimer = [];
        end
        if isfield(guiData, 'aiBreathingTimer')
            guiData = stopAIBreathingTimer(guiData);
            guiData.aiBreathingTimer = [];
        end
    end

    set(fig, 'UserData', []);
end

function viewState = captureViewState(guiData)
    viewState = [];
    if ~isstruct(guiData)
        return;
    end

    if isfield(guiData, 'imgAxes') && ishandle(guiData.imgAxes)
        viewState.xlim = get(guiData.imgAxes, 'XLim');
        viewState.ylim = get(guiData.imgAxes, 'YLim');
    end

    if isfield(guiData, 'zoomSlider') && ishandle(guiData.zoomSlider)
        viewState.zoomSliderValue = get(guiData.zoomSlider, 'Value');
    end

    if isfield(guiData, 'zoomValue') && ishandle(guiData.zoomValue)
        viewState.zoomLabel = get(guiData.zoomValue, 'String');
    end

    if isfield(guiData, 'zoomLevel')
        viewState.zoomLevel = guiData.zoomLevel;
    end
end

function applyViewState(fig, viewState)
    if isempty(viewState)
        return;
    end

    guiData = get(fig, 'UserData');
    if ~isstruct(guiData) || ~isfield(guiData, 'imgAxes') || ~ishandle(guiData.imgAxes)
        return;
    end

    if isfield(viewState, 'xlim')
        try
            xlim(guiData.imgAxes, viewState.xlim);
        catch
            % Ignore: axes may have been deleted during mode transition
        end
    end

    if isfield(viewState, 'ylim')
        try
            ylim(guiData.imgAxes, viewState.ylim);
        catch
            % Ignore: axes may have been deleted during mode transition
        end
    end

    if isfield(viewState, 'zoomSliderValue') && isfield(guiData, 'zoomSlider') && ishandle(guiData.zoomSlider)
        set(guiData.zoomSlider, 'Value', viewState.zoomSliderValue);
    end

    if isfield(viewState, 'zoomLabel') && isfield(guiData, 'zoomValue') && ishandle(guiData.zoomValue)
        set(guiData.zoomValue, 'String', viewState.zoomLabel);
    end

    if isfield(viewState, 'zoomLevel')
        guiData.zoomLevel = viewState.zoomLevel;
    end

    set(fig, 'UserData', guiData);
end

function polys = collectValidQuads(guiData)
    polys = [];
    if isempty(guiData) || ~isstruct(guiData) || ~isfield(guiData, 'quads')
        return;
    end
    if ~iscell(guiData.quads)
        return;
    end

    validMask = cellfun(@isvalid, guiData.quads);
    if any(validMask)
        % Clear appdata before collecting for deletion
        validPolys = guiData.quads(validMask);
        for i = 1:length(validPolys)
            if isvalid(validPolys{i})
                if isappdata(validPolys{i}, 'LastValidPosition')
                    rmappdata(validPolys{i}, 'LastValidPosition');
                end
                if isappdata(validPolys{i}, 'ListenerHandle')
                    delete(getappdata(validPolys{i}, 'ListenerHandle'));
                    rmappdata(validPolys{i}, 'ListenerHandle');
                end
                if isappdata(validPolys{i}, 'LabelUpdateListener')
                    delete(getappdata(validPolys{i}, 'LabelUpdateListener'));
                    rmappdata(validPolys{i}, 'LabelUpdateListener');
                end
            end
        end
        polys = [guiData.quads{validMask}]';
    end
end

function buildEditingUI(fig, img, imageName, phoneName, cfg, initialQuads, initialRotation)
    % Build UI for quad editing mode
    if nargin < 7
        initialRotation = 0;
    end

    set(fig, 'Name', sprintf('microPAD Processor - %s - %s', phoneName, imageName));

    guiData = struct();
    guiData.mode = 'editing';
    guiData.cfg = cfg;

    % Initialize rotation data (from memory or default to 0)
    guiData.baseImg = img;
    guiData.baseImageSize = [size(img, 1), size(img, 2)];
    guiData.currentImg = img;
    guiData.memoryRotation = initialRotation;
    guiData.adjustmentRotation = 0;
    guiData.totalRotation = initialRotation;

    % Initialize zoom state
    guiData.zoomLevel = 0;  % 0 = full image, 1 = single micropad size
    guiData.autoZoomBounds = [];

    % Title and path
    guiData.titleHandle = cfg.micropadUI.createTitle(fig, phoneName, imageName, cfg);
    guiData.pathHandle = cfg.micropadUI.createPathDisplay(fig, phoneName, imageName, cfg);

    % Image display (show image with initial rotation if any)
    if initialRotation ~= 0
        displayImg = applyRotation(img, initialRotation, cfg);
        guiData.currentImg = displayImg;
    else
        displayImg = img;
        guiData.currentImg = displayImg;
    end
    guiData.imageSize = [size(displayImg, 1), size(displayImg, 2)];
    [guiData.imgAxes, guiData.imgHandle] = cfg.micropadUI.createImageAxes(fig, displayImg, cfg);

    % Create editable quads
    guiData.quads = cfg.micropadUI.createQuads(initialQuads, cfg, @(src, evt) updateQuadLabelsCallback(src, evt));
    [guiData.quads, ~, guiData.orientation] = assignQuadLabels(guiData.quads);

    numInitialQuads = numel(guiData.quads);
    totalForColor = max(numInitialQuads, 1);
    guiData.aiBaseColors = zeros(numInitialQuads, 3);
    for idx = 1:numInitialQuads
        polyHandle = guiData.quads{idx};
        if isvalid(polyHandle)
            baseColor = cfg.micropadUI.getConcentrationColor(idx - 1, totalForColor);
            cfg.micropadUI.setQuadColor(polyHandle, baseColor, 0.25);
            guiData.aiBaseColors(idx, :) = baseColor;
        else
            guiData.aiBaseColors(idx, :) = [NaN NaN NaN];
        end
    end
    guiData.aiBreathingTimer = [];

    % Async detection state
    guiData.asyncDetection = struct();
    guiData.asyncDetection.active = false;        % Is detection running?
    guiData.asyncDetection.outputFile = '';       % Path to output file
    guiData.asyncDetection.imgPath = '';          % Path to temp image
    guiData.asyncDetection.startTime = [];        % tic() timestamp
    guiData.asyncDetection.pollingTimer = [];     % Timer handle
    guiData.asyncDetection.timeoutSeconds = 10;   % Max detection time
    guiData.asyncDetection.launchRotation = 0;    % Rotation at launch
    guiData.asyncDetection.launchImgSize = [0, 0]; % Image size at launch [width, height]
    guiData.asyncDetection.generation = 0;        % Generation counter to invalidate stale detections

    % Add concentration labels
    guiData.quadLabels = cfg.micropadUI.addQuadLabels(guiData.quads, guiData.imgAxes);

    % Rotation panel (preset buttons only)
    guiData.rotationPanel = cfg.micropadUI.createRotationButtonPanel(fig, cfg, @applyRotation_UI);

    % Run AI button sits above rotation controls for manual detection refresh
    guiData.runAIButton = cfg.micropadUI.createRunAIButton(fig, cfg, @(~,~) rerunAIDetection(fig, cfg));

    % Zoom panel
    [guiData.zoomSlider, guiData.zoomValue] = cfg.micropadUI.createZoomPanel(fig, cfg, ...
        @(src, ~) zoomSliderCallback(src, fig, cfg), ...
        @(~, ~) resetZoom(fig, cfg), ...
        @(~, ~) applyAutoZoom(fig, get(fig, 'UserData'), cfg));

    % Buttons
    guiData.cutButtonPanel = cfg.micropadUI.createEditButtonPanel(fig, cfg, @(~,~) setAction(fig, 'accept'), @(~,~) setAction(fig, 'skip'));
    guiData.stopButton = cfg.micropadUI.createStopButton(fig, cfg, @(~,~) stopExecution(fig));
    guiData.instructionText = cfg.micropadUI.createInstructions(fig, cfg, []);
    guiData.aiStatusLabel = cfg.micropadUI.createAIStatusLabel(fig, cfg);

    guiData.action = '';

    % Store guiData before auto-zoom
    set(fig, 'UserData', guiData);

    % Auto-zoom to quads after all UI is created
    guiData = get(fig, 'UserData');
    applyAutoZoom(fig, guiData, cfg);

    % Trigger deferred AI detection if enabled (after GUI fully built)
    if cfg.useAIDetection
        guiData = get(fig, 'UserData');

        % Use timer to run AI detection after GUI fully renders
        % Delay ensures UI is interactive before blocking operation starts
        t = timer('StartDelay', 0.1, ...
                  'TimerFcn', @(~,~) runDeferredAIDetection(fig, cfg), ...
                  'ExecutionMode', 'singleShot');

        % Store timer handle before starting to prevent leak on error
        guiData.aiTimer = t;
        set(fig, 'UserData', guiData);
        start(t);
    end
end

function buildPreviewUI(fig, img, imageName, phoneName, cfg, quadParams, ellipseData)
    % Build UI for preview mode
    set(fig, 'Name', sprintf('Preview - %s - %s', phoneName, imageName));

    % Handle optional ellipse data parameter
    if nargin < 7
        ellipseData = [];
    end

    guiData = struct();
    guiData.mode = 'preview';
    guiData.savedQuadParams = quadParams;
    guiData.savedEllipseData = ellipseData;

    % Preview titles occupying the top band
    numRegions = size(quadParams, 1);
    titleText = sprintf('Preview - %s', phoneName);
    guiData.titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                                   'Units', 'normalized', 'Position', cfg.ui.positions.previewTitle, ...
                                   'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                                   'ForegroundColor', cfg.ui.colors.foreground, ...
                                   'BackgroundColor', cfg.ui.colors.background, ...
                                   'HorizontalAlignment', 'center');

    metaText = sprintf('Image: %s | Regions: %d', imageName, numRegions);
    guiData.metaHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', metaText, ...
                                  'Units', 'normalized', 'Position', cfg.ui.positions.previewMeta, ...
                                  'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                                  'ForegroundColor', cfg.ui.colors.path, ...
                                  'BackgroundColor', cfg.ui.colors.background, ...
                                  'HorizontalAlignment', 'center');

    % Preview axes fill the middle band between titles and bottom controls
    [guiData.leftAxes, guiData.rightAxes, guiData.leftImgHandle, guiData.rightImgHandle] = cfg.micropadUI.createPreviewAxes(fig, img, quadParams, ellipseData, cfg);

    % Bottom controls
    guiData.stopButton = cfg.micropadUI.createStopButton(fig, cfg, @(~,~) stopExecution(fig));
    guiData.buttonPanel = cfg.micropadUI.createPreviewButtons(fig, cfg, struct( ...
        'accept', @(~,~) setAction(fig, 'accept'), ...
        'retry', @(~,~) setAction(fig, 'retry'), ...
        'skip', @(~,~) setAction(fig, 'skip')));

    guiData.action = '';
    set(fig, 'UserData', guiData);
end

function buildEllipseEditingUI(fig, img, imageName, phoneName, cfg, quadParams, rotation, memory, orientation)
    % Build UI for ellipse editing mode
    set(fig, 'Name', sprintf('Ellipse Editing - %s - %s', phoneName, imageName));

    % Initialize orientation if not provided
    if nargin < 9 || isempty(orientation)
        orientation = 'horizontal';
    end

    guiData = struct();
    guiData.mode = 'ellipse_editing';
    guiData.cfg = cfg;
    guiData.quads = quadParams;
    guiData.rotation = rotation;
    guiData.memory = memory;
    guiData.orientation = orientation;  % Store for quad ordering and ellipse positioning

    % Apply rotation to image
    if rotation ~= 0
        displayImg = applyRotation(img, rotation, cfg);
    else
        displayImg = img;
    end
    guiData.baseImg = img;
    guiData.baseImageSize = [size(img, 1), size(img, 2)];
    guiData.currentImg = displayImg;
    guiData.imageSize = [size(displayImg, 1), size(displayImg, 2)];
    guiData.displayQuads = convertBaseQuadsToDisplay(quadParams, guiData.baseImageSize, guiData.imageSize, rotation, cfg);

    % Title
    titleText = sprintf('Ellipse Editing - %s - %s', phoneName, imageName);
    guiData.titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                                   'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                                   'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                                   'ForegroundColor', cfg.ui.colors.foreground, ...
                                   'BackgroundColor', cfg.ui.colors.background, ...
                                   'HorizontalAlignment', 'center');

    pathText = sprintf('Path: %s | Image: %s', phoneName, imageName);
    guiData.pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                          'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                          'ForegroundColor', cfg.ui.colors.path, ...
                          'BackgroundColor', cfg.ui.colors.background, ...
                          'HorizontalAlignment', 'center');

    % Image display with quads
    [guiData.imgAxes, guiData.imgHandle] = cfg.micropadUI.createImageAxes(fig, displayImg, cfg);

    % Add double-click background reset callback
    set(guiData.imgAxes, 'ButtonDownFcn', @(src, evt) axesClickCallback(src, evt, fig, cfg));
    set(guiData.imgHandle, 'HitTest', 'off');  % Allow clicks to pass through to axes

    % Draw all quads (clickable for zoom navigation, but not editable)
    numConcentrations = size(quadParams, 1);
    guiData.quadHandles = cell(numConcentrations, 1);
    for i = 1:numConcentrations
        vertices = squeeze(guiData.displayQuads(i, :, :));
        quadColor = cfg.micropadUI.getConcentrationColor(i - 1, numConcentrations);
        guiData.quadHandles{i} = drawpolygon(guiData.imgAxes, 'Position', vertices, ...
                                                'Color', quadColor, 'LineWidth', 2, ...
                                                'FaceAlpha', 0.15, 'InteractionsAllowed', 'translate', ...
                                                'Tag', sprintf('quad_%d', i));
        % Add click callback for zoom
        addlistener(guiData.quadHandles{i}, 'ROIClicked', @(src, evt) zoomToQuadCallback(src, fig, i, cfg));
        % Prevent movement by resetting position when drag starts
        originalVertices = vertices;
        addlistener(guiData.quadHandles{i}, 'MovingROI', @(src, ~) set(src, 'Position', originalVertices));
    end

    % Create ALL ellipses at once (21 total for 7 quads × 3 replicates)
    numReplicates = cfg.ellipse.replicatesPerMicropad;
    totalEllipses = numConcentrations * numReplicates;
    guiData.ellipses = cell(1, totalEllipses);
    ellipseIdx = 1;

    % Check if memory has ellipse settings from previous image
    hasMemory = false;
    if isfield(guiData, 'memory') && ~isempty(guiData.memory) && ...
       isfield(guiData.memory, 'hasEllipseSettings') && guiData.memory.hasEllipseSettings
        hasMemory = true;

        % Check if quad geometry changed (need to scale)
        if isfield(guiData.memory, 'quads') && ~isempty(guiData.memory.quads)
            oldQuads = guiData.memory.quads;
            newQuads = guiData.displayQuads;

            % Scale ellipses for each concentration
            for concIdx = 1:numConcentrations
                if concIdx <= numel(guiData.memory.ellipses) && ~isempty(guiData.memory.ellipses{concIdx})
                    oldPoly = squeeze(oldQuads(concIdx, :, :));
                    newPoly = squeeze(newQuads(concIdx, :, :));
                    oldEllipses = guiData.memory.ellipses{concIdx};

                    % Scale ellipses to new quad geometry
                    scaledEllipses = cfg.geomTform.geom.scaleEllipsesForQuadChange(oldPoly, newPoly, oldEllipses, guiData.imageSize, cfg);
                    guiData.memory.ellipses{concIdx} = scaledEllipses;
                end
            end
        end
    end

    displayQuads = guiData.displayQuads;
    for concIdx = 1:numConcentrations
        quadColor = cfg.micropadUI.getConcentrationColor(concIdx - 1, numConcentrations);
        currentQuad = squeeze(displayQuads(concIdx, :, :));

        % Compute axis bounds for this quad
        ellipseBounds = cfg.geomTform.geom.computeEllipseAxisBounds(currentQuad, guiData.imageSize, cfg);

        % Check if we have memory for this concentration
        if hasMemory && concIdx <= numel(guiData.memory.ellipses) && ...
           ~isempty(guiData.memory.ellipses{concIdx})
            % Use remembered/scaled ellipse parameters
            memEllipses = guiData.memory.ellipses{concIdx};

            for repIdx = 1:numReplicates
                if repIdx <= size(memEllipses, 1)
                    % Extract from memory: [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
                    ellipseCenter = memEllipses(repIdx, 3:4);
                    ellipseSemiMajor = memEllipses(repIdx, 5);
                    ellipseSemiMinor = memEllipses(repIdx, 6);
                    ellipseRotation = memEllipses(repIdx, 7);
                else
                    % Fallback to homography-transformed defaults if memory incomplete
                    ellipseParams = cfg.geomTform.geom.transformDefaultEllipsesToQuad(currentQuad, cfg, orientation, rotation);
                    ellipseCenter = ellipseParams(repIdx, 1:2);
                    ellipseSemiMajor = ellipseParams(repIdx, 3);
                    ellipseSemiMinor = ellipseParams(repIdx, 4);
                    ellipseRotation = ellipseParams(repIdx, 5);
                end

                guiData.ellipses{ellipseIdx} = cfg.micropadUI.createEllipseROI(guiData.imgAxes, ellipseCenter, ...
                    ellipseSemiMajor, ellipseSemiMinor, ellipseRotation, quadColor, ellipseBounds, cfg);
                ellipseIdx = ellipseIdx + 1;
            end
        else
            % No memory - use homography-transformed defaults from ELLIPSE_DEFAULT_RECORDS
            ellipseParams = cfg.geomTform.geom.transformDefaultEllipsesToQuad(currentQuad, cfg, orientation, rotation);

            for repIdx = 1:numReplicates
                guiData.ellipses{ellipseIdx} = cfg.micropadUI.createEllipseROI(guiData.imgAxes, ellipseParams(repIdx, 1:2), ...
                    ellipseParams(repIdx, 3), ellipseParams(repIdx, 4), ellipseParams(repIdx, 5), quadColor, ellipseBounds, cfg);
                ellipseIdx = ellipseIdx + 1;
            end
        end
    end

    % Zoom panel (quad-specific navigation)
    guiData.zoomLevel = 0;
    guiData.autoZoomBounds = [];
    guiData.focusedQuadIndex = 0;  % 0 = none/all, 1-N = specific quad
            [guiData.prevButton, guiData.zoomIndicator, guiData.nextButton, guiData.resetButton] = cfg.micropadUI.createEllipseZoomPanel(fig, cfg, ...
                @(~,~) navigateToPrevQuad(fig, cfg), ...
                @(~,~) navigateToNextQuad(fig, cfg), ...
                @(~,~) resetZoomEllipse(fig, cfg));
    % Action buttons panel
    guiData.ellipseButtonPanel = cfg.micropadUI.createEllipseEditingButtonPanel(fig, cfg, @(~,~) setAction(fig, 'accept'), @(~,~) setAction(fig, 'back'));
    guiData.stopButton = cfg.micropadUI.createStopButton(fig, cfg, @(~,~) stopExecution(fig));

    % Instructions
    instructionText = 'Draw 21 ellipses (3 per micropad). Colors match quads. Click DONE when finished.';
    guiData.instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', instructionText, ...
                                       'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
                                       'FontSize', cfg.ui.fontSize.instruction, ...
                                       'ForegroundColor', cfg.ui.colors.foreground, ...
                                       'BackgroundColor', cfg.ui.colors.background, ...
                                       'HorizontalAlignment', 'center');

    guiData.action = '';
    set(fig, 'UserData', guiData);

    % Auto-zoom to fit all content after UI is created
    guiData = get(fig, 'UserData');
    applyAutoZoom(fig, guiData, cfg);
end

function buildEllipseEditingUIGridMode(fig, img, imageName, phoneName, cfg, ellipsePositions, rotation, memory)
    % Build ellipse editing UI for grid mode (no quad overlays)
    % This is used in Mode 3 when no quad coordinates exist

    set(fig, 'Name', sprintf('Ellipse Editing (Grid Mode) - %s - %s', phoneName, imageName));

    guiData = struct();
    guiData.mode = 'ellipse_editing_grid';
    guiData.cfg = cfg;
    guiData.quads = []; % No quads in grid mode
    guiData.displayQuads = [];
    guiData.rotation = rotation;
    guiData.memory = memory;

    % Apply rotation to image
    if rotation ~= 0
        displayImg = applyRotation(img, rotation, cfg);
    else
        displayImg = img;
    end
    guiData.baseImg = img;
    guiData.baseImageSize = [size(img, 1), size(img, 2)];
    guiData.currentImg = displayImg;
    guiData.imageSize = [size(displayImg, 1), size(displayImg, 2)];

    % Title
    titleText = sprintf('Ellipse Editing (Grid Mode) - %s - %s', phoneName, imageName);
    guiData.titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                                   'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                                   'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                                   'ForegroundColor', cfg.ui.colors.foreground, ...
                                   'BackgroundColor', cfg.ui.colors.background, ...
                                   'HorizontalAlignment', 'center');

    pathText = sprintf('Path: %s | Image: %s', phoneName, imageName);
    guiData.pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                          'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                          'ForegroundColor', cfg.ui.colors.path, ...
                          'BackgroundColor', cfg.ui.colors.background, ...
                          'HorizontalAlignment', 'center');

    % Image display (NO quad overlays in grid mode)
    [guiData.imgAxes, guiData.imgHandle] = cfg.micropadUI.createImageAxes(fig, displayImg, cfg);

    % Add double-click background reset callback
    set(guiData.imgAxes, 'ButtonDownFcn', @(src, evt) axesClickCallback(src, evt, fig, cfg));
    set(guiData.imgHandle, 'HitTest', 'off');  % Allow clicks to pass through to axes

    % Create ellipses at default grid positions
    numReplicates = cfg.ellipse.replicatesPerMicropad;
    numGroups = cfg.numSquares;
    totalEllipses = numGroups * numReplicates;
    guiData.ellipses = cell(1, totalEllipses);

    % Calculate default ellipse size based on image dimensions
    imageWidth = size(displayImg, 2);
    imageHeight = size(displayImg, 1);
    avgDim = (imageWidth + imageHeight) / 2;
    defaultSemiMajor = avgDim * cfg.ellipse.semiMajorDefaultRatio * 0.5; % Smaller for grid mode
    defaultSemiMinor = defaultSemiMajor * cfg.ellipse.semiMinorDefaultRatio;

    % Create ellipses using grid positions
    % In grid mode, use rotation angles from ELLIPSE_DEFAULT_RECORDS (no homography)
    defaultRotations = cfg.ellipse.defaultRecords(:, 5);  % Column 5 = rotation
    ellipseIdx = 1;
    gridBounds = cfg.geomTform.geom.computeEllipseAxisBounds([], guiData.imageSize, cfg);
    for groupIdx = 1:numGroups
        % Use cold-to-hot color gradient
        ellipseColor = cfg.micropadUI.getConcentrationColor(groupIdx - 1, numGroups);

        for repIdx = 1:numReplicates
            centerPos = ellipsePositions(ellipseIdx, :);

            guiData.ellipses{ellipseIdx} = cfg.micropadUI.createEllipseROI(guiData.imgAxes, centerPos, ...
                defaultSemiMajor, defaultSemiMinor, defaultRotations(repIdx), ellipseColor, gridBounds, cfg);
            ellipseIdx = ellipseIdx + 1;
        end
    end

    % Zoom panel (grid mode doesn't have quads, so no quad-specific zoom)
    guiData.zoomLevel = 0;
    guiData.autoZoomBounds = [];
    guiData.focusedQuadIndex = 0;  % Always 0 in grid mode (no quads)
    [guiData.zoomSlider, guiData.zoomValue] = cfg.micropadUI.createZoomPanel(fig, cfg, ...
        @(src, ~) zoomSliderCallback(src, fig, cfg), ...
        @(~, ~) resetZoom(fig, cfg), ...
        @(~, ~) applyAutoZoom(fig, get(fig, 'UserData'), cfg));

    % Action buttons panel
    guiData.ellipseButtonPanel = cfg.micropadUI.createEllipseEditingButtonPanel(fig, cfg, @(~,~) setAction(fig, 'accept'), @(~,~) setAction(fig, 'back'));
    guiData.stopButton = cfg.micropadUI.createStopButton(fig, cfg, @(~,~) stopExecution(fig));

    % Instructions
    instructionText = sprintf('Grid Mode: Draw %d ellipses (%d groups × %d replicates). Colors indicate groups. Click DONE when finished.', ...
        totalEllipses, numGroups, numReplicates);
    guiData.instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', instructionText, ...
                                       'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
                                       'FontSize', cfg.ui.fontSize.instruction, ...
                                       'ForegroundColor', cfg.ui.colors.foreground, ...
                                       'BackgroundColor', cfg.ui.colors.background, ...
                                       'HorizontalAlignment', 'center');

    guiData.action = '';
    set(fig, 'UserData', guiData);

    % Auto-zoom to fit all content after UI is created
    guiData = get(fig, 'UserData');
    applyAutoZoom(fig, guiData, cfg);
end

function buildReadOnlyPreviewUI(fig, img, imageName, phoneName, cfg, quadParams, hasQuads, ...
                               ellipseData, hasEllipses, rotation)
    % Build read-only preview UI for Mode 4 (both editing disabled)
    % Displays existing coordinate overlays without editing capability

    % Clear previous UI elements before building new UI
    oldGuiData = get(fig, 'UserData');
    clearAllUIElements(fig, oldGuiData);

    set(fig, 'Name', sprintf('Preview Mode - %s - %s', phoneName, imageName));

    guiData = struct();
    guiData.mode = 'preview_readonly';
    guiData.cfg = cfg;
    guiData.quads = quadParams;
    guiData.ellipses = ellipseData;
    guiData.rotation = rotation;

    % Apply rotation to image
    if rotation ~= 0
        displayImg = applyRotation(img, rotation, cfg);
    else
        displayImg = img;
    end
    guiData.baseImg = img;
    guiData.baseImageSize = [size(img, 1), size(img, 2)];
    guiData.currentImg = displayImg;
    guiData.imageSize = [size(displayImg, 1), size(displayImg, 2)];

    % Title
    titleText = sprintf('Preview Mode (Read-Only) - %s - %s', phoneName, imageName);
    guiData.titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                                   'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                                   'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                                   'ForegroundColor', cfg.ui.colors.foreground, ...
                                   'BackgroundColor', cfg.ui.colors.background, ...
                                   'HorizontalAlignment', 'center');

    pathText = sprintf('Path: %s | Image: %s', phoneName, imageName);
    guiData.pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                          'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                          'ForegroundColor', cfg.ui.colors.path, ...
                          'BackgroundColor', cfg.ui.colors.background, ...
                          'HorizontalAlignment', 'center');

    % Image display
    [guiData.imgAxes, guiData.imgHandle] = cfg.micropadUI.createImageAxes(fig, displayImg, cfg);

    displayQuadsPreview = convertBaseQuadsToDisplay(quadParams, guiData.baseImageSize, guiData.imageSize, rotation, cfg);
    displayEllipsesPreview = convertBaseEllipsesToDisplay(ellipseData, guiData.baseImageSize, guiData.imageSize, rotation, cfg);

    % Draw quad overlays if available
    if hasQuads && ~isempty(displayQuadsPreview)
        numQuads = size(displayQuadsPreview, 1);
        guiData.quadHandles = cell(numQuads, 1);
        for i = 1:numQuads
            vertices = squeeze(displayQuadsPreview(i, :, :));
            quadColor = cfg.micropadUI.getConcentrationColor(i - 1, numQuads);
            guiData.quadHandles{i} = drawpolygon(guiData.imgAxes, 'Position', vertices, ...
                                                    'Color', quadColor, 'LineWidth', 2, ...
                                                    'FaceAlpha', 0.15, 'InteractionsAllowed', 'none');
        end
    end

    % Draw ellipse overlays if available
    if hasEllipses && ~isempty(displayEllipsesPreview)
        numEllipses = size(displayEllipsesPreview, 1);
        guiData.ellipseHandles = cell(numEllipses, 1);

        for i = 1:numEllipses
            % ellipseData format: [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
            concIdx = displayEllipsesPreview(i, 1) + 1; % Convert 0-indexed to 1-indexed
            center = displayEllipsesPreview(i, 3:4);
            semiMajor = displayEllipsesPreview(i, 5);
            semiMinor = displayEllipsesPreview(i, 6);
            rotationAngle = displayEllipsesPreview(i, 7);

            % Use same color as corresponding quad
            if hasQuads
                numConcentrations = size(displayQuadsPreview, 1);
            else
                numConcentrations = max(displayEllipsesPreview(:, 1)) + 1;
            end
            ellipseColor = cfg.micropadUI.getConcentrationColor(concIdx - 1, numConcentrations);

            guiData.ellipseHandles{i} = drawellipse(guiData.imgAxes, ...
                'Center', center, ...
                'SemiAxes', [semiMajor, semiMinor], ...
                'RotationAngle', rotationAngle, ...
                'Color', ellipseColor, 'LineWidth', 2, 'FaceAlpha', 0.2, ...
                'InteractionsAllowed', 'none');
        end
    end

    % Zoom panel
    guiData.zoomLevel = 0;
    guiData.autoZoomBounds = [];
    [guiData.zoomSlider, guiData.zoomValue] = cfg.micropadUI.createZoomPanel(fig, cfg, ...
        @(src, ~) zoomSliderCallback(src, fig, cfg), ...
        @(~, ~) resetZoom(fig, cfg), ...
        @(~, ~) applyAutoZoom(fig, get(fig, 'UserData'), cfg));

    % NEXT button (replaces ACCEPT in read-only mode)
    uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'NEXT', ...
              'Units', 'normalized', 'Position', [0.80 0.02 0.15 0.06], ...
              'FontSize', cfg.ui.fontSize.button, ...
              'Callback', @(~,~) setAction(fig, 'accept'));

    % STOP button
    guiData.stopButton = cfg.micropadUI.createStopButton(fig, cfg, @(~,~) stopExecution(fig));

    % Instructions
    overlayInfo = '';
    if hasQuads && hasEllipses
        overlayInfo = sprintf('%d quads and %d ellipses', size(quadParams, 1), size(ellipseData, 1));
    elseif hasQuads
        overlayInfo = sprintf('%d quads', size(quadParams, 1));
    elseif hasEllipses
        overlayInfo = sprintf('%d ellipses', size(ellipseData, 1));
    end

    instructionText = sprintf('Preview mode (read-only) - Displaying %s from coordinate files. Press NEXT to continue.', overlayInfo);
    guiData.instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', instructionText, ...
                                       'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
                                       'FontSize', cfg.ui.fontSize.instruction, ...
                                       'ForegroundColor', cfg.ui.colors.foreground, ...
                                       'BackgroundColor', cfg.ui.colors.background, ...
                                       'HorizontalAlignment', 'center');

    guiData.action = '';
    set(fig, 'UserData', guiData);

    % Auto-zoom to fit all content after UI is created
    guiData = get(fig, 'UserData');
    applyAutoZoom(fig, guiData, cfg);
end

function showAIProgressIndicator(fig, show)
    % Toggle AI detection status indicator and quadbreathing animation
    if ~ishandle(fig) || ~strcmp(get(fig, 'Type'), 'figure')
        return;
    end

    guiData = get(fig, 'UserData');
    if isempty(guiData) || ~isstruct(guiData) || ~strcmp(guiData.mode, 'editing')
        return;
    end
    if ~isfield(guiData, 'cfg') || ~isfield(guiData.cfg, 'micropadUI')
        return;
    end

    cfg = guiData.cfg;
    ui = cfg.micropadUI;

    % Delegate label handling to shared UI helper
    ui.showAIProgressIndicator(fig, show, cfg);
    guiData = get(fig, 'UserData');

    if show
        guiData = stopAIBreathingTimer(guiData, ui);
        guiData.aiBaseColors = ui.captureQuadColors(guiData.quads);

        if ~isempty(guiData.aiBaseColors)
            guiData.aiBreathingStart = tic;
            guiData.aiBreathingFrequency = 0.8;            % Hz (slow breathing cadence)
            guiData.aiBreathingMixRange = [0.12, 0.36];    % Blend-to-white range (min..max)
            guiData.aiBreathingDimFactor = 0.22;           % Max dim amount during exhale (22%)
            guiData.aiBreathingTimer = timer(...
                'Name', 'microPAD-AI-breathing', ...
                'Period', 1/45, ...                        % ~22 ms (~45 FPS)
                'ExecutionMode', 'fixedRate', ...
                'BusyMode', 'queue', ...
                'TasksToExecute', Inf, ...
                'TimerFcn', @(~,~) animateQuadBreathing(fig));
            start(guiData.aiBreathingTimer);
        end

        drawnow limitrate;
    else
        guiData = stopAIBreathingTimer(guiData, ui);
        guiData.aiBaseColors = ui.captureQuadColors(guiData.quads);
    end

    set(fig, 'UserData', guiData);
end

function guiData = stopAIBreathingTimer(guiData, ui)
    if nargin < 2 || isempty(ui)
        ui = [];
        if isfield(guiData, 'cfg') && isfield(guiData.cfg, 'micropadUI')
            ui = guiData.cfg.micropadUI;
        end
    end

    if isfield(guiData, 'aiBreathingTimer') && ~isempty(guiData.aiBreathingTimer)
        if ~isempty(ui)
            ui.safeStopTimer(guiData.aiBreathingTimer);
        elseif isvalid(guiData.aiBreathingTimer)
            try
                stop(guiData.aiBreathingTimer);
            catch
                % Ignore: timer may already be stopped/deleted
            end
            delete(guiData.aiBreathingTimer);
        end
    end
    guiData.aiBreathingTimer = [];
    guiData.aiBreathingStart = [];
    if isfield(guiData, 'aiBreathingFrequency')
        guiData.aiBreathingFrequency = [];
    end
    if isfield(guiData, 'aiBreathingMixRange')
        guiData.aiBreathingMixRange = [];
    end
    if isfield(guiData, 'aiBreathingDimFactor')
        guiData.aiBreathingDimFactor = [];
    end
end

function animateQuadBreathing(fig)
    if ~ishandle(fig) || ~strcmp(get(fig, 'Type'), 'figure')
        return;
    end

    guiData = get(fig, 'UserData');
    if isempty(guiData) || ~isstruct(guiData)
        return;
    end
    if ~isfield(guiData, 'quads') || ~iscell(guiData.quads)
        return;
    end
    if ~isfield(guiData, 'aiBaseColors') || isempty(guiData.aiBaseColors)
        return;
    end
    if ~isfield(guiData, 'aiBreathingStart') || isempty(guiData.aiBreathingStart)
        return;
    end
    if ~isfield(guiData, 'cfg') || ~isfield(guiData.cfg, 'micropadUI')
        return;
    end
    ui = guiData.cfg.micropadUI;
    defaultsUpdated = false;
    if ~isfield(guiData, 'aiBreathingFrequency') || isempty(guiData.aiBreathingFrequency)
        guiData.aiBreathingFrequency = 0.8;
        defaultsUpdated = true;
    end
    if ~isfield(guiData, 'aiBreathingMixRange') || numel(guiData.aiBreathingMixRange) ~= 2
        guiData.aiBreathingMixRange = [0.12, 0.36];
        defaultsUpdated = true;
    end
    if ~isfield(guiData, 'aiBreathingDimFactor') || isempty(guiData.aiBreathingDimFactor)
        guiData.aiBreathingDimFactor = 0.22;
        defaultsUpdated = true;
    end
    if defaultsUpdated
        set(fig, 'UserData', guiData);
    end

    elapsed = toc(guiData.aiBreathingStart);
    phase = 2 * pi * guiData.aiBreathingFrequency * elapsed;
    wave = sin(phase);
    inhale = max(wave, 0);    % 0..1 when lightening toward white
    exhale = max(-wave, 0);   % 0..1 when dimming toward base color

    mixRange = guiData.aiBreathingMixRange;
    brightenMix = mixRange(1) + (mixRange(2) - mixRange(1)) * inhale;
    dimScale = 1 - guiData.aiBreathingDimFactor * exhale;

    numQuads = min(size(guiData.aiBaseColors, 1), numel(guiData.quads));
    for idx = 1:numQuads
        poly = guiData.quads{idx};
        baseColor = guiData.aiBaseColors(idx, :);
        if isvalid(poly) && all(isfinite(baseColor))
            whitened = baseColor * (1 - brightenMix) + brightenMix;
            newColor = min(max(whitened * dimScale, 0), 1);
            ui.setQuadColor(poly, newColor, []);
        end
    end

    drawnow limitrate;
end

function guiData = applyDetectedQuads(guiData, newQuads, cfg, ~)
    % Synchronize drawpolygon handles with detection output preserving UI ordering

    if isempty(newQuads)
        return;
    end

    % Ensure quads are ordered left-to-right, bottom-to-top in UI space
    [newQuads, newOrientation] = sortQuadArrayByX(newQuads);
    guiData.orientation = newOrientation;  % Update orientation based on new quadlayout
    targetCount = size(newQuads, 1);

    if targetCount == 0
        return;
    end

    % Determine whether we can reuse existing quadhandles
    hasQuads = isfield(guiData, 'quads') && iscell(guiData.quads) && ~isempty(guiData.quads);
    validMask = hasQuads;
    if hasQuads
        validMask = cellfun(@isvalid, guiData.quads);
    end
    reuseQuads = hasQuads && all(validMask) && numel(guiData.quads) == targetCount;

    if reuseQuads
        labelHandles = [];
        if isfield(guiData, 'quadLabels')
            labelHandles = guiData.quadLabels;
        end
        updateQuadPositions(guiData.quads, newQuads, labelHandles, cfg.micropadUI);
    else
        % Clean up existing quads if present
        if hasQuads
            for idx = 1:numel(guiData.quads)
                if isvalid(guiData.quads{idx})
                    delete(guiData.quads{idx});
                end
            end
        end

        guiData.quads = cfg.micropadUI.createQuads(newQuads, cfg, @(src, evt) updateQuadLabelsCallback(src, evt));
    end

    % Reorder quads to enforce gradient ordering
    [guiData.quads, order, newOrientation] = assignQuadLabels(guiData.quads);
    guiData.orientation = newOrientation;  % Update orientation based on current layout

    % Synchronize labels
    hasLabels = isfield(guiData, 'quadLabels') && iscell(guiData.quadLabels);
    reuseLabels = false;
    if hasLabels
        labelValidMask = cellfun(@isvalid, guiData.quadLabels);
        reuseLabels = all(labelValidMask) && numel(guiData.quadLabels) == targetCount;
    end

    if ~reuseLabels
        if hasLabels
            for idx = 1:numel(guiData.quadLabels)
                if isvalid(guiData.quadLabels{idx})
                    delete(guiData.quadLabels{idx});
                end
            end
        end
        guiData.quadLabels = cfg.micropadUI.addQuadLabels(guiData.quads, guiData.imgAxes);
    elseif ~isempty(order)
        guiData.quadLabels = guiData.quadLabels(order);
    end

    % Apply consistent cold-to-hot gradient and refresh label strings
    numQuads = numel(guiData.quads);
    totalForColor = max(numQuads, 1);
    for idx = 1:numQuads
        polyHandle = guiData.quads{idx};
        if isvalid(polyHandle)
            gradColor = cfg.micropadUI.getConcentrationColor(idx - 1, totalForColor);
            cfg.micropadUI.setQuadColor(polyHandle, gradColor, 0.25);

        end
    end

    if ~isempty(guiData.quadLabels)
        for idx = 1:min(numQuads, numel(guiData.quadLabels))
            labelHandle = guiData.quadLabels{idx};
            if isvalid(labelHandle)
                set(labelHandle, 'String', sprintf('con_%d', idx - 1));
            end
        end
        cfg.micropadUI.updateQuadLabels(guiData.quads, guiData.quadLabels);
    end

    guiData.aiBaseColors = cfg.micropadUI.captureQuadColors(guiData.quads);
end

function applyRotation_UI(angle, fig, cfg)
    % Apply preset rotation angle
    guiData = get(fig, 'UserData');
    if ~strcmp(guiData.mode, 'editing')
        return;
    end

    % Cancel pending detection before rotation
    if isfield(guiData, 'asyncDetection') && guiData.asyncDetection.active
        fprintf('  Rotation requested - canceling in-flight detection...\n');
        cancelActiveDetection(fig, guiData, cfg);
        guiData = get(fig, 'UserData');  % Refresh after cleanup
    end

    % Increment generation counter to invalidate ALL pending detections
    if isfield(guiData, 'asyncDetection')
        guiData.asyncDetection.generation = guiData.asyncDetection.generation + 1;
    end

    % Update rotation state (quick buttons are absolute presets)
    guiData.adjustmentRotation = angle;
    guiData.memoryRotation = angle;
    guiData.totalRotation = angle;

    % Apply rotation to image
    guiData.currentImg = applyRotation(guiData.baseImg, guiData.totalRotation, cfg);

    % Store current image dimensions before rotation
    currentHeight = guiData.imageSize(1);
    currentWidth = guiData.imageSize(2);

    % Update image dimensions
    newHeight = size(guiData.currentImg, 1);
    newWidth = size(guiData.currentImg, 2);
    guiData.imageSize = [newHeight, newWidth];

    % Convert quad positions to normalized coordinates [0, 1] before image update
    numQuads = 0;
    quadNormalized = {};
    if isfield(guiData, 'quads') && iscell(guiData.quads)
        numQuads = length(guiData.quads);
        quadNormalized = cell(numQuads, 1);
        for i = 1:numQuads
            if isvalid(guiData.quads{i})
                posData = guiData.quads{i}.Position;  % [N x 2] array of vertices
                % Convert to normalized axes coordinates [0, 1]
                quadNormalized{i} = [(posData(:, 1) - 1) / currentWidth, (posData(:, 2) - 1) / currentHeight];
            end
        end
    end

    % Update image data and spatial extent (preserves all axes children)
    set(guiData.imgHandle, 'CData', guiData.currentImg, ...
                            'XData', [1, newWidth], ...
                            'YData', [1, newHeight]);

    % Snap axes to new image bounds
    axis(guiData.imgAxes, 'image');

    % Update quad positions to maintain screen-space locations
    for i = 1:numQuads
        if isvalid(guiData.quads{i})
            % Convert normalized coordinates back to new data coordinates
            newPos = [1 + quadNormalized{i}(:, 1) * newWidth, 1 + quadNormalized{i}(:, 2) * newHeight];
            guiData.quads{i}.Position = newPos;
        end
    end

    % Reorder quads to maintain concentration ordering after rotation
    [guiData.quads, order, newOrientation] = assignQuadLabels(guiData.quads);
    guiData.orientation = newOrientation;  % Update orientation based on new layout

    hasLabels = isfield(guiData, 'quadLabels') && iscell(guiData.quadLabels);
    if hasLabels && ~isempty(order) && numel(guiData.quadLabels) >= numel(order)
        guiData.quadLabels = guiData.quadLabels(order);
    end

    numQuads = numel(guiData.quads);
    totalForColor = max(numQuads, 1);
    for idx = 1:numQuads
        polyHandle = guiData.quads{idx};
        if isvalid(polyHandle)
            gradColor = cfg.micropadUI.getConcentrationColor(idx - 1, totalForColor);
            cfg.micropadUI.setQuadColor(polyHandle, gradColor, 0.25);

        end

        if hasLabels && idx <= numel(guiData.quadLabels)
            labelHandle = guiData.quadLabels{idx};
            if isvalid(labelHandle)
                set(labelHandle, 'String', sprintf('con_%d', idx - 1));
            end
        end
    end

    if hasLabels
        cfg.micropadUI.updateQuadLabels(guiData.quads, guiData.quadLabels);
    end

    guiData.aiBaseColors = cfg.micropadUI.captureQuadColors(guiData.quads);
    guiData.autoZoomBounds = [];

    % Save guiData
    set(fig, 'UserData', guiData);

    % Auto-zoom to fit all content after rotation
    guiData = get(fig, 'UserData');
    applyAutoZoom(fig, guiData, cfg);

    % Re-trigger detection after rotation completes
    if cfg.useAIDetection
        fprintf('  Relaunching detection with new rotation...\n');
        runDeferredAIDetection(fig, cfg);
    end
end

function zoomSliderCallback(slider, fig, cfg)
    % Handle zoom slider changes
    guiData = get(fig, 'UserData');
    validModes = {'editing', 'preview_readonly'};
    if ~ismember(guiData.mode, validModes)
        return;
    end

    zoomLevel = get(slider, 'Value');
    guiData.zoomLevel = zoomLevel;

    % Update zoom value display
    set(guiData.zoomValue, 'String', sprintf('%d%%', round(zoomLevel * 100)));

    % Apply zoom to axes
    applyZoomToAxes(guiData, cfg);

    set(fig, 'UserData', guiData);
end

function resetZoom(fig, cfg)
    % Reset zoom to full image view
    guiData = get(fig, 'UserData');
    validModes = {'editing', 'ellipse_editing', 'ellipse_editing_grid', 'preview_readonly'};
    if ~ismember(guiData.mode, validModes)
        return;
    end

    guiData.zoomLevel = 0;

    % Handle different UI controls for different modes
    % 'editing', 'ellipse_editing_grid', and 'preview_readonly' use slider-based zoom
    sliderModes = {'editing', 'ellipse_editing_grid', 'preview_readonly'};
    if ismember(guiData.mode, sliderModes)
        if isfield(guiData, 'zoomSlider') && ishandle(guiData.zoomSlider)
            set(guiData.zoomSlider, 'Value', 0);
        end
        if isfield(guiData, 'zoomValue') && ishandle(guiData.zoomValue)
            set(guiData.zoomValue, 'String', '0%');
        end
    end

    % Clear quadfocus for ellipse editing modes
    if strcmp(guiData.mode, 'ellipse_editing') || strcmp(guiData.mode, 'ellipse_editing_grid')
        if isfield(guiData, 'focusedQuadIndex')
            guiData.focusedQuadIndex = 0;
            updateQuadHighlight(fig, guiData, cfg);
            updateQuadIndicator(fig, guiData);
        end
    end

    % Apply zoom to axes
    applyZoomToAxes(guiData, cfg);

    set(fig, 'UserData', guiData);
end

function applyAutoZoom(fig, guiData, cfg)
    % Auto-zoom to fit all quads

    if isempty(guiData) || ~isstruct(guiData)
        return;
    end

    validModes = {'editing', 'ellipse_editing', 'ellipse_editing_grid', 'preview_readonly'};
    if ~isfield(guiData, 'mode') || ~ismember(guiData.mode, validModes)
        return;
    end

    % Calculate bounding box based on available handles
    if strcmp(guiData.mode, 'ellipse_editing')
        % Use quadHandles for ellipse editing mode (has quads)
        [xmin, xmax, ymin, ymax] = calculateQuadHandlesBounds(guiData);
    elseif strcmp(guiData.mode, 'ellipse_editing_grid')
        % Use ellipse handles for grid mode (no quads, only ellipses)
        [xmin, xmax, ymin, ymax] = calculateEllipseBounds(guiData);
    elseif strcmp(guiData.mode, 'preview_readonly')
        % Use quadhandles if available, otherwise ellipse handles
        if isfield(guiData, 'quadHandles') && ~isempty(guiData.quadHandles)
            [xmin, xmax, ymin, ymax] = calculateQuadHandlesBounds(guiData);
        elseif isfield(guiData, 'ellipseHandles') && ~isempty(guiData.ellipseHandles)
            [xmin, xmax, ymin, ymax] = calculateEllipseBounds(guiData);
        else
            return;
        end
    else
        % Use quads for editing mode
        [xmin, xmax, ymin, ymax] = calculateQuadBounds(guiData);
    end

    if isempty(xmin)
        return;  % No valid quads
    end

    % Store auto-zoom bounds in guiData
    guiData.autoZoomBounds = [xmin, xmax, ymin, ymax];

    % Set zoom to auto (maximum zoom level = 1)
    guiData.zoomLevel = 1;
    if isfield(guiData, 'zoomSlider') && ishandle(guiData.zoomSlider)
        set(guiData.zoomSlider, 'Value', 1);
    end
    if isfield(guiData, 'zoomValue') && ishandle(guiData.zoomValue)
        set(guiData.zoomValue, 'String', '100%');
    end

    % Apply zoom to axes
    applyZoomToAxes(guiData, cfg);

    set(fig, 'UserData', guiData);
end

function applyZoomToAxes(guiData, cfg)
    % Apply current zoom level to image axes
    % zoomLevel: 0 = full image, 1 = auto-zoom to quads

    imgHeight = guiData.imageSize(1);
    imgWidth = guiData.imageSize(2);

    if guiData.zoomLevel == 0
        % Full image view
        xlim(guiData.imgAxes, [0.5, imgWidth + 0.5]);
        ylim(guiData.imgAxes, [0.5, imgHeight + 0.5]);
    else
        % Calculate target bounds based on zoom level
        if isfield(guiData, 'autoZoomBounds') && ~isempty(guiData.autoZoomBounds)
            autoZoomBounds = guiData.autoZoomBounds;
        else
            % Calculate bounds from quads if they exist
            xmin = [];

            % For preview_readonly mode, use quadHandles or ellipseHandles
            if isfield(guiData, 'mode') && strcmp(guiData.mode, 'preview_readonly')
                if isfield(guiData, 'quadHandles') && ~isempty(guiData.quadHandles)
                    [xmin, xmax, ymin, ymax] = calculateQuadHandlesBounds(guiData);
                elseif isfield(guiData, 'ellipseHandles') && ~isempty(guiData.ellipseHandles)
                    [xmin, xmax, ymin, ymax] = calculateEllipseBounds(guiData);
                end
            else
                [xmin, xmax, ymin, ymax] = calculateQuadBounds(guiData);
            end

            if ~isempty(xmin)
                % Use actual quadbounds
                autoZoomBounds = [xmin, xmax, ymin, ymax];
            else
                % No quads yet - use center estimate
                [autoZoomBounds] = estimateSingleMicropadBounds(guiData, cfg);
            end
            guiData.autoZoomBounds = autoZoomBounds;
        end

        % Interpolate between full image and auto-zoom bounds
        fullBounds = [0.5, imgWidth + 0.5, 0.5, imgHeight + 0.5];
        targetBounds = autoZoomBounds;

        % Linear interpolation
        t = guiData.zoomLevel;
        xmin = fullBounds(1) * (1-t) + targetBounds(1) * t;
        xmax = fullBounds(2) * (1-t) + targetBounds(2) * t;
        ymin = fullBounds(3) * (1-t) + targetBounds(3) * t;
        ymax = fullBounds(4) * (1-t) + targetBounds(4) * t;

        xlim(guiData.imgAxes, [xmin, xmax]);
        ylim(guiData.imgAxes, [ymin, ymax]);
    end
end

function [xmin, xmax, ymin, ymax] = calculateQuadBounds(guiData)
    % Calculate bounding box containing all quads
    xmin = inf;
    xmax = -inf;
    ymin = inf;
    ymax = -inf;

    if ~isfield(guiData, 'quads') || isempty(guiData.quads)
        xmin = [];
        return;
    end

    for i = 1:numel(guiData.quads)
        if isvalid(guiData.quads{i})
            pos = guiData.quads{i}.Position;
            xmin = min(xmin, min(pos(:, 1)));
            xmax = max(xmax, max(pos(:, 1)));
            ymin = min(ymin, min(pos(:, 2)));
            ymax = max(ymax, max(pos(:, 2)));
        end
    end

    if isinf(xmin)
        xmin = [];
        return;
    end

    % Add margin (10% of bounds size)
    xmargin = (xmax - xmin) * 0.1;
    ymargin = (ymax - ymin) * 0.1;

    xmin = max(0.5, xmin - xmargin);
    xmax = min(guiData.imageSize(2) + 0.5, xmax + xmargin);
    ymin = max(0.5, ymin - ymargin);
    ymax = min(guiData.imageSize(1) + 0.5, ymax + ymargin);
end

function bounds = estimateSingleMicropadBounds(guiData, cfg)
    % Estimate bounds for a single micropad size when no quads available
    imgHeight = guiData.imageSize(1);
    imgWidth = guiData.imageSize(2);

    % Use coverage parameter to estimate micropad strip width
    stripWidth = imgWidth * cfg.coverage;
    stripHeight = stripWidth / cfg.geometry.aspectRatio;

    % Center on image
    centerX = imgWidth / 2;
    centerY = imgHeight / 2;

    xmin = max(0.5, centerX - stripWidth / 2);
    xmax = min(imgWidth + 0.5, centerX + stripWidth / 2);
    ymin = max(0.5, centerY - stripHeight / 2);
    ymax = min(imgHeight + 0.5, centerY + stripHeight / 2);

    bounds = [xmin, xmax, ymin, ymax];
end

%% -------------------------------------------------------------------------
%% Ellipse Editing Zoom Functions
%% -------------------------------------------------------------------------



function zoomToQuad(fig, quadIndex, cfg)
    % Zoom to a specific quad with 5% margin
    guiData = get(fig, 'UserData');

    if ~isfield(guiData, 'quadHandles') || isempty(guiData.quadHandles)
        return;
    end

    numQuads = numel(guiData.quadHandles);
    if quadIndex < 1 || quadIndex > numQuads
        return;
    end

    % Get vertices from quadhandle
    if ~isvalid(guiData.quadHandles{quadIndex})
        return;
    end

    vertices = guiData.quadHandles{quadIndex}.Position;

    % Calculate bounding box with 5% margin
    xmin = min(vertices(:, 1));
    xmax = max(vertices(:, 1));
    ymin = min(vertices(:, 2));
    ymax = max(vertices(:, 2));

    xmargin = (xmax - xmin) * 0.05;
    ymargin = (ymax - ymin) * 0.05;

    xmin = max(0.5, xmin - xmargin);
    xmax = min(guiData.imageSize(2) + 0.5, xmax + xmargin);
    ymin = max(0.5, ymin - ymargin);
    ymax = min(guiData.imageSize(1) + 0.5, ymax + ymargin);

    % Apply zoom to axes
    xlim(guiData.imgAxes, [xmin, xmax]);
    ylim(guiData.imgAxes, [ymin, ymax]);

    % Update focused index
    guiData.focusedQuadIndex = quadIndex;
    set(fig, 'UserData', guiData);

    % Update highlight and indicator
    updateQuadHighlight(fig, guiData, cfg);
    updateQuadIndicator(fig, guiData);
end

function updateQuadHighlight(~, guiData, cfg)
    % Update quadvisual highlighting based on focused index
    if ~isfield(guiData, 'quadHandles') || isempty(guiData.quadHandles)
        return;
    end

    numQuads = numel(guiData.quadHandles);
    focusedIdx = guiData.focusedQuadIndex;

    for i = 1:numQuads
        if ~isvalid(guiData.quadHandles{i})
            continue;
        end

        baseColor = cfg.micropadUI.getConcentrationColor(i - 1, numQuads);

        if focusedIdx == i
            % Focused quad: thicker line, original color
            guiData.quadHandles{i}.LineWidth = 3;
            guiData.quadHandles{i}.Color = baseColor;
        elseif focusedIdx > 0
            % Non-focused quad when a quad is focused: normal line, dimmed color
            guiData.quadHandles{i}.LineWidth = 2;
            dimmedColor = baseColor * cfg.dimFactor + [1 1 1] * (1 - cfg.dimFactor);
            guiData.quadHandles{i}.Color = dimmedColor;
        else
            % No focus: normal line, original color
            guiData.quadHandles{i}.LineWidth = 2;
            guiData.quadHandles{i}.Color = baseColor;
        end
    end
end

function updateQuadIndicator(~, guiData)
    % Update indicator text showing current quadfocus
    if ~isfield(guiData, 'zoomIndicator') || ~ishandle(guiData.zoomIndicator)
        return;
    end

    if ~isfield(guiData, 'focusedQuadIndex')
        return;
    end

    % Grid mode has no quadHandles - nothing to indicate
    if ~isfield(guiData, 'quadHandles') || isempty(guiData.quadHandles)
        return;
    end

    focusedIdx = guiData.focusedQuadIndex;

    if focusedIdx == 0
        set(guiData.zoomIndicator, 'String', 'All');
    else
        numQuads = numel(guiData.quadHandles);
        set(guiData.zoomIndicator, 'String', sprintf('Quad %d/%d', focusedIdx, numQuads));
    end
end

function navigateToPrevQuad(fig, cfg)
    % Navigate to previous quad(wrap from 1 to last)
    guiData = get(fig, 'UserData');

    if ~isfield(guiData, 'quadHandles') || isempty(guiData.quadHandles)
        return;
    end

    numQuads = numel(guiData.quadHandles);
    currentIdx = guiData.focusedQuadIndex;

    if currentIdx <= 1
        % Wrap to last quad
        newIdx = numQuads;
    else
        newIdx = currentIdx - 1;
    end

    zoomToQuad(fig, newIdx, cfg);
end

function navigateToNextQuad(fig, cfg)
    % Navigate to next quad(wrap from last to 1)
    guiData = get(fig, 'UserData');

    if ~isfield(guiData, 'quadHandles') || isempty(guiData.quadHandles)
        return;
    end

    numQuads = numel(guiData.quadHandles);
    currentIdx = guiData.focusedQuadIndex;

    if currentIdx >= numQuads || currentIdx == 0
        % Wrap to first quad
        newIdx = 1;
    else
        newIdx = currentIdx + 1;
    end

    zoomToQuad(fig, newIdx, cfg);
end

function resetZoomEllipse(fig, cfg)
    % Reset zoom to fit all content for ellipse editing modes
    % - ellipse_editing mode: fits all quads
    % - ellipse_editing_grid mode: fits all ellipses
    guiData = get(fig, 'UserData');

    % Clear quadfocus (if applicable)
    if isfield(guiData, 'focusedQuadIndex')
        guiData.focusedQuadIndex = 0;
        set(fig, 'UserData', guiData);
        updateQuadHighlight(fig, guiData, cfg);
        updateQuadIndicator(fig, guiData);
    end

    % Apply auto-zoom to fit all content
    applyAutoZoom(fig, guiData, cfg);
end

function zoomToQuadCallback(~, fig, quadIndex, cfg)
    % Callback when quad is clicked
    zoomToQuad(fig, quadIndex, cfg);
end

function axesClickCallback(~, ~, fig, cfg)
    % Handle double-click on axes background to reset zoom
    if strcmp(get(fig, 'SelectionType'), 'open')  % Double-click
        guiData = get(fig, 'UserData');
        if strcmp(guiData.mode, 'ellipse_editing') || strcmp(guiData.mode, 'ellipse_editing_grid')
            resetZoomEllipse(fig, cfg);  % Auto-zoom to fit all content (quads or ellipses)
        else
            resetZoom(fig, cfg);  % Reset to full image for other modes
        end
    end
end

function [xmin, xmax, ymin, ymax] = calculateQuadHandlesBounds(guiData)
    % Calculate bounding box containing all quadhandles (for ellipse editing mode)
    xmin = inf;
    xmax = -inf;
    ymin = inf;
    ymax = -inf;

    if ~isfield(guiData, 'quadHandles') || isempty(guiData.quadHandles)
        xmin = [];
        return;
    end

    for i = 1:numel(guiData.quadHandles)
        if isvalid(guiData.quadHandles{i})
            pos = guiData.quadHandles{i}.Position;
            xmin = min(xmin, min(pos(:, 1)));
            xmax = max(xmax, max(pos(:, 1)));
            ymin = min(ymin, min(pos(:, 2)));
            ymax = max(ymax, max(pos(:, 2)));
        end
    end

    if isinf(xmin)
        xmin = [];
        return;
    end

    % Add margin (10% of bounds size)
    xmargin = (xmax - xmin) * 0.1;
    ymargin = (ymax - ymin) * 0.1;

    xmin = max(0.5, xmin - xmargin);
    xmax = min(guiData.imageSize(2) + 0.5, xmax + xmargin);
    ymin = max(0.5, ymin - ymargin);
    ymax = min(guiData.imageSize(1) + 0.5, ymax + ymargin);
end

function [xmin, xmax, ymin, ymax] = calculateEllipseBounds(guiData)
    % Calculate bounding box containing all ellipses (for grid mode)
    xmin = inf;
    xmax = -inf;
    ymin = inf;
    ymax = -inf;

    if ~isfield(guiData, 'ellipses') || isempty(guiData.ellipses)
        xmin = [];
        return;
    end

    for i = 1:numel(guiData.ellipses)
        if isvalid(guiData.ellipses{i})
            center = guiData.ellipses{i}.Center;
            semiAxes = guiData.ellipses{i}.SemiAxes;
            % Conservative bounds using maximum semi-axis (handles any rotation)
            maxRadius = max(semiAxes);
            xmin = min(xmin, center(1) - maxRadius);
            xmax = max(xmax, center(1) + maxRadius);
            ymin = min(ymin, center(2) - maxRadius);
            ymax = max(ymax, center(2) + maxRadius);
        end
    end

    if isinf(xmin)
        xmin = [];
        return;
    end

    % Add margin (10% of bounds size)
    xmargin = (xmax - xmin) * 0.1;
    ymargin = (ymax - ymin) * 0.1;

    xmin = max(0.5, xmin - xmargin);
    xmax = min(guiData.imageSize(2) + 0.5, xmax + xmargin);
    ymin = max(0.5, ymin - ymargin);
    ymax = min(guiData.imageSize(1) + 0.5, ymax + ymargin);
end

function baseQuads = convertDisplayQuadsToBase(guiData, displayQuads, cfg)
    % Convert quads from rotated display coordinates back to original image coordinates
    baseQuads = displayQuads;

    if isempty(displayQuads)
        return;
    end

    if ~isfield(guiData, 'totalRotation')
        return;
    end

    rotation = guiData.totalRotation;
    if rotation == 0
        return;
    end

    imageSize = guiData.imageSize;
    if isempty(imageSize)
        if isfield(guiData, 'currentImg') && ~isempty(guiData.currentImg)
            imageSize = [size(guiData.currentImg, 1), size(guiData.currentImg, 2)];
        else
            return;
        end
    else
        imageSize = imageSize(1:2);
    end

    if cfg.geomTform.isMultipleOfNinety(rotation, cfg.rotation.angleTolerance)
        [baseQuads, newSize] = cfg.geomTform.geom.rotateQuadsDiscrete(displayQuads, imageSize, -rotation);

        if isfield(guiData, 'baseImageSize') && ~isempty(guiData.baseImageSize)
            targetSize = guiData.baseImageSize(1:2);
            if any(newSize ~= targetSize)
                baseQuads = scaleQuadsForImageSize(baseQuads, newSize, targetSize, cfg.numSquares);
            end
        end
        return;
    end

    % General-angle fallback: geometrically invert the loose imrotate transform
    rotatedSize = imageSize;
    targetSize = rotatedSize;
    if isfield(guiData, 'baseImageSize') && ~isempty(guiData.baseImageSize)
        targetSize = guiData.baseImageSize(1:2);
    end

    numQuads = size(displayQuads, 1);
    baseQuads = zeros(size(displayQuads));

    for i = 1:numQuads
        polyRot = squeeze(displayQuads(i, :, :));
        % Clamp first to avoid NaNs during transform
        polyRot = cfg.geomTform.geom.clampQuadToImage(polyRot, rotatedSize);
        polyBase = cfg.geomTform.geom.inverseRotatePoints(polyRot, rotatedSize, targetSize, rotation, cfg.rotation.angleTolerance);
        baseQuads(i, :, :) = cfg.geomTform.geom.clampQuadToImage(polyBase, targetSize);
    end
end

function displayQuads = convertBaseQuadsToDisplay(baseQuads, baseImageSize, displayImageSize, rotation, cfg)
    displayQuads = baseQuads;
    if isempty(baseQuads)
        return;
    end

    if rotation == 0
        rotatedQuads = baseQuads;
        rotatedSize = baseImageSize;
    elseif cfg.geomTform.isMultipleOfNinety(rotation, cfg.rotation.angleTolerance)
        [rotatedQuads, rotatedSize] = cfg.geomTform.geom.rotateQuadsDiscrete(baseQuads, baseImageSize, rotation);
    else
        % General-angle forward rotation: geometrically apply the imrotate transform
        originalSize = baseImageSize(1:2);
        rotatedSize = displayImageSize(1:2);

        numQuads = size(baseQuads, 1);
        rotatedQuads = zeros(size(baseQuads));

        for i = 1:numQuads
            polyBase = squeeze(baseQuads(i, :, :));
            polyBase = cfg.geomTform.geom.clampQuadToImage(polyBase, originalSize);
            polyRot = cfg.geomTform.geom.forwardRotatePoints(polyBase, originalSize, rotatedSize, rotation, cfg.rotation.angleTolerance);
            rotatedQuads(i, :, :) = cfg.geomTform.geom.clampQuadToImage(polyRot, rotatedSize);
        end
    end

    targetSize = displayImageSize(1:2);
    if any(rotatedSize ~= targetSize)
        displayQuads = scaleQuadsForImageSize(rotatedQuads, rotatedSize, targetSize, size(baseQuads, 1));
    else
        displayQuads = rotatedQuads;
    end
end

function displaySize = computeDisplayImageSize(baseSize, rotation, cfg)
    displaySize = baseSize;
    if rotation == 0
        return;
    end

    if abs(mod(rotation, 90)) < cfg.rotation.angleTolerance
        k = mod(round(rotation / 90), 2);
        if k == 1
            displaySize = fliplr(baseSize);
        end
    end
end

function rotatedImg = applyRotation(img, rotation, cfg)
    % Apply rotation to image (lossless rot90 for 90-deg multiples, bilinear with loose mode otherwise)
    if rotation == 0
        rotatedImg = img;
        return;
    end

    % For exact 90-degree multiples, use lossless rot90
    if abs(mod(rotation, 90)) < cfg.rotation.angleTolerance
        numRotations = mod(round(rotation / 90), 4);
        if numRotations == 0
            rotatedImg = img;
        else
            rotatedImg = rot90(img, -numRotations);
        end
    else
        rotatedImg = imrotate(img, rotation, 'bilinear', 'loose');
    end
end

function displayEllipses = convertBaseEllipsesToDisplay(ellipseData, baseImageSize, displayImageSize, rotation, cfg)
    displayEllipses = ellipseData;
    if isempty(ellipseData)
        return;
    end

    if rotation ~= 0 && cfg.geomTform.isMultipleOfNinety(rotation, cfg.rotation.angleTolerance)
        [rotCenters, rotatedSize] = cfg.geomTform.geom.rotatePointsDiscrete(ellipseData(:, 3:4), baseImageSize, rotation);
    else
        rotCenters = ellipseData(:, 3:4);
        rotatedSize = baseImageSize;
    end

    targetSize = displayImageSize(1:2);
    if any(rotatedSize ~= targetSize)
        scaleX = targetSize(2) / rotatedSize(2);
        scaleY = targetSize(1) / rotatedSize(1);
        rotCenters(:, 1) = rotCenters(:, 1) * scaleX;
        rotCenters(:, 2) = rotCenters(:, 2) * scaleY;

        % Scale ellipse axes to match display image resizing.
        % Use geometric mean to preserve area under uniform scaling.
        axisScale = sqrt(scaleX * scaleY);
        displayEllipses(:, 5) = ellipseData(:, 5) * axisScale;
        displayEllipses(:, 6) = ellipseData(:, 6) * axisScale;
    end

    displayEllipses(:, 3:4) = rotCenters;
    displayEllipses(:, 7) = mod(ellipseData(:, 7) + rotation + 180, 360) - 180;
end

function updateQuadLabelsCallback(quad, varargin)
    % Callback for ROIMoved event to keep labels/colors ordered along dominant axis
    fig = ancestor(quad, 'figure');
    if isempty(fig) || ~ishandle(fig) || ~strcmp(get(fig, 'Type'), 'figure')
        return;
    end

    guiData = get(fig, 'UserData');
    if isempty(guiData) || ~isstruct(guiData)
        return;
    end

    guiData = enforceConcentrationOrdering(guiData);
    set(fig, 'UserData', guiData);
end

function [quads, order, orientation] = assignQuadLabels(quads)
    % Reorder drawpolygon handles so con_0..con_N follow low→high concentration order.
    % Ordering direction matches the dominant layout axis: horizontal strips
    % sort left→right, vertical strips sort bottom→top.
    %
    % Returns:
    %   quads    - Reordered cell array of quadhandles
    %   order       - Permutation indices used for reordering
    %   orientation - 'horizontal' or 'vertical' based on layout
    order = [];
    orientation = 'horizontal';  % Default

    if isempty(quads) || ~iscell(quads)
        return;
    end

    numQuads = numel(quads);
    centroids = nan(numQuads, 2);
    minXs = nan(numQuads, 1);
    maxXs = nan(numQuads, 1);
    minYs = nan(numQuads, 1);
    maxYs = nan(numQuads, 1);
    validMask = false(numQuads, 1);

    for i = 1:numQuads
        if ~isvalid(quads{i})
            continue;
        end

        pos = quads{i}.Position;
        centroids(i, 1) = mean(pos(:, 1));
        centroids(i, 2) = mean(pos(:, 2));
        minXs(i) = min(pos(:, 1));
        maxXs(i) = max(pos(:, 1));
        minYs(i) = min(pos(:, 2));
        maxYs(i) = max(pos(:, 2));
        validMask(i) = true;
    end

    if ~any(validMask)
        return;
    end

    widths = maxXs(validMask) - minXs(validMask);
    heights = maxYs(validMask) - minYs(validMask);

    validWidths = widths(widths > 0);
    if isempty(validWidths)
        widthRef = 1;
    else
        widthRef = median(validWidths);
    end

    validHeights = heights(heights > 0);
    if isempty(validHeights)
        heightRef = 1;
    else
        heightRef = median(validHeights);
    end

    rangeX = max(maxXs(validMask)) - min(minXs(validMask));
    rangeY = max(maxYs(validMask)) - min(minYs(validMask));

    if ~isfinite(rangeX) || rangeX < 0
        rangeX = 0;
    end
    if ~isfinite(rangeY) || rangeY < 0
        rangeY = 0;
    end

    widthDen = max(widthRef, 1e-6);
    heightDen = max(heightRef, 1e-6);

    countX = rangeX / widthDen;
    countY = rangeY / heightDen;

    if ~isfinite(countX)
        countX = 0;
    end
    if ~isfinite(countY)
        countY = 0;
    end

    sortKey = inf(numQuads, 2);
    if countX >= countY
        % Horizontal dominance: prioritize left→right, break ties bottom→top
        orientation = 'horizontal';
        sortKey(validMask, 1) = centroids(validMask, 1);
        sortKey(validMask, 2) = -centroids(validMask, 2);
    else
        % Vertical dominance: prioritize bottom→top, break ties left→right
        orientation = 'vertical';
        sortKey(validMask, 1) = -centroids(validMask, 2);
        sortKey(validMask, 2) = centroids(validMask, 1);
    end

    [~, order] = sortrows(sortKey);
    quads = quads(order);
end

function guiData = enforceConcentrationOrdering(guiData)
    % Ensure drawpolygon handles, colors, and labels follow the dominant axis ordering
    if nargin < 1 || isempty(guiData) || ~isstruct(guiData)
        return;
    end
    if ~isfield(guiData, 'quads') || ~iscell(guiData.quads) || isempty(guiData.quads)
        return;
    end
    if ~isfield(guiData, 'cfg') || ~isfield(guiData.cfg, 'micropadUI')
        return;
    end
    ui = guiData.cfg.micropadUI;

    [sortedQuads, order, newOrientation] = assignQuadLabels(guiData.quads);
    guiData.orientation = newOrientation;  % Always update orientation based on current layout
    numQuads = numel(guiData.quads);
    needsReindex = ~isempty(order) && numel(order) == numQuads && ...
                   ~isequal(order(:).', 1:numQuads);
    if needsReindex
        guiData.quads = sortedQuads;
    end

    hasLabels = isfield(guiData, 'quadLabels') && iscell(guiData.quadLabels) && ...
                ~isempty(guiData.quadLabels);
    if hasLabels
        labelCount = numel(guiData.quadLabels);
        if needsReindex && labelCount >= numQuads
            guiData.quadLabels = guiData.quadLabels(order);
        end

        for idx = 1:min(numQuads, labelCount)
            labelHandle = guiData.quadLabels{idx};
            if isvalid(labelHandle)
                set(labelHandle, 'String', sprintf('con_%d', idx - 1));
            end
        end

        ui.updateQuadLabels(guiData.quads, guiData.quadLabels);
    end

    if needsReindex
        totalForColor = max(numQuads, 1);
        for idx = 1:numQuads
            polyHandle = guiData.quads{idx};
            if isvalid(polyHandle)
                gradColor = ui.getConcentrationColor(idx - 1, totalForColor);
                ui.setQuadColor(polyHandle, gradColor, 0.25);
            end
        end
    end

    if needsReindex || ~isfield(guiData, 'aiBaseColors') || isempty(guiData.aiBaseColors)
        guiData.aiBaseColors = ui.captureQuadColors(guiData.quads);
    end
end

function [sortedQuads, orientation] = sortQuadArrayByX(quads)
    % Determine quad ordering based on dominant axis coverage.
    % Primary ordering follows the axis with the greater spread-to-size ratio:
    %   - Horizontal layouts: left-to-right (primary), bottom-to-top (secondary)
    %   - Vertical layouts: bottom-to-top (primary), left-to-right (secondary)
    %
    % Returns:
    %   sortedQuads - Quads sorted by dominant axis
    %   orientation    - 'horizontal' or 'vertical' based on layout
    sortedQuads = quads;
    orientation = 'horizontal';  % Default
    if isempty(quads)
        return;
    end
    if ndims(quads) ~= 3
        return;
    end

    numQuads = size(quads, 1);
    if numQuads == 0
        return;
    end

    centroids = nan(numQuads, 2);
    minXs = nan(numQuads, 1);
    maxXs = nan(numQuads, 1);
    minYs = nan(numQuads, 1);
    maxYs = nan(numQuads, 1);

    for i = 1:numQuads
        poly = squeeze(quads(i, :, :));
        if isempty(poly)
            continue;
        end

        xs = poly(:, 1);
        ys = poly(:, 2);

        centroids(i, 1) = mean(xs);
        centroids(i, 2) = mean(ys);

        minXs(i) = min(xs);
        maxXs(i) = max(xs);
        minYs(i) = min(ys);
        maxYs(i) = max(ys);
    end

    validMask = all(isfinite(centroids), 2);
    if ~any(validMask)
        return;
    end

    widths = maxXs(validMask) - minXs(validMask);
    heights = maxYs(validMask) - minYs(validMask);

    validWidths = widths(widths > 0);
    if isempty(validWidths)
        widthRef = 1;
    else
        widthRef = median(validWidths);
    end

    validHeights = heights(heights > 0);
    if isempty(validHeights)
        heightRef = 1;
    else
        heightRef = median(validHeights);
    end

    rangeX = max(maxXs(validMask)) - min(minXs(validMask));
    rangeY = max(maxYs(validMask)) - min(minYs(validMask));

    if ~isfinite(rangeX) || rangeX < 0
        rangeX = 0;
    end
    if ~isfinite(rangeY) || rangeY < 0
        rangeY = 0;
    end

    widthDen = max(widthRef, 1e-6);
    heightDen = max(heightRef, 1e-6);

    countX = rangeX / widthDen;
    countY = rangeY / heightDen;

    if ~isfinite(countX)
        countX = 0;
    end
    if ~isfinite(countY)
        countY = 0;
    end

    sortKey = inf(numQuads, 2);

    if countX >= countY
        % Horizontal dominance: prioritize left→right, then bottom→top
        orientation = 'horizontal';
        sortKey(validMask, 1) = centroids(validMask, 1);
        sortKey(validMask, 2) = -centroids(validMask, 2);
    else
        % Vertical dominance: prioritize bottom→top, then left→right
        orientation = 'vertical';
        sortKey(validMask, 1) = -centroids(validMask, 2);
        sortKey(validMask, 2) = centroids(validMask, 1);
    end

    [~, order] = sortrows(sortKey);
    sortedQuads = quads(order, :, :);
end

%% -------------------------------------------------------------------------
%% User Interaction
%% -------------------------------------------------------------------------

function setAction(fig, action)
    guiData = get(fig, 'UserData');
    guiData.action = action;
    set(fig, 'UserData', guiData);
    uiresume(fig);
end




function stopExecution(fig)
    guiData = get(fig, 'UserData');

    % Cancel any in-flight async detection
    if isfield(guiData, 'cfg')
        cancelActiveDetection(fig, guiData, guiData.cfg);
        guiData = get(fig, 'UserData');
    end

    guiData.action = 'stop';
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function terminateExecution(fig)
    % Centralized handler for terminating script execution
    % Called when 'stop' action is detected or figure is closed
    if nargin > 0 && isvalid(fig)
        delete(fig);
    end
    error('cut_micropads:userStopped', 'User stopped execution');
end

function rerunAIDetection(fig, cfg)
    % Re-run AI detection and replace current quads with fresh detections
    guiData = get(fig, 'UserData');

    if ~strcmp(guiData.mode, 'editing')
        return;
    end

    % Validate AI detection prerequisites even if auto-detection is disabled
    if ~isfile(cfg.pythonScriptPath)
        warning('cut_micropads:script_missing', ...
            'Python script not found: %s\nCannot run AI detection.', cfg.pythonScriptPath);
        return;
    end

    if ~isfile(cfg.detectionModel)
        warning('cut_micropads:model_missing', ...
            'Model not found: %s\nCannot run AI detection.', cfg.detectionModel);
        return;
    end

    % Cancel existing detection before rerun
    if isfield(guiData, 'asyncDetection') && guiData.asyncDetection.active
        fprintf('  Manual rerun requested - canceling current detection...\n');
        cancelActiveDetection(fig, guiData, cfg);
    end

    fprintf('  Re-running AI detection asynchronously...\n');
    runDeferredAIDetection(fig, cfg);
end

function keyPressHandler(src, event)
    switch event.Key
        case 'space'
            setAction(src, 'accept');
        case 'escape'
            setAction(src, 'skip');
    end
end

function [action, quadParams, rotation] = waitForUserAction(fig)
    uiwait(fig);

    action = '';
    quadParams = [];
    rotation = 0;

    if isvalid(fig)
        guiData = get(fig, 'UserData');
        action = guiData.action;

        if strcmp(action, 'accept')
            if strcmp(guiData.mode, 'preview')
                quadParams = guiData.savedQuadParams;
                if isfield(guiData, 'savedRotation')
                    rotation = guiData.savedRotation;
                end
            elseif strcmp(guiData.mode, 'editing')
                guiData = enforceConcentrationOrdering(guiData);
                set(fig, 'UserData', guiData);
                quadParams = extractQuadParameters(guiData);
                if isempty(quadParams)
                    action = 'skip';
                else
                    rotation = guiData.totalRotation;
                end
            elseif strcmp(guiData.mode, 'ellipse_editing')
                % Get all ellipse data at once
                numConcentrations = size(guiData.quads, 1);
                numReplicates = guiData.cfg.ellipse.replicatesPerMicropad;
                totalEllipses = numConcentrations * numReplicates;
                boundsPerConcentration = cell(numConcentrations, 1);
                quadSetForBounds = guiData.quads;
                if isfield(guiData, 'displayQuads') && ~isempty(guiData.displayQuads)
                    quadSetForBounds = guiData.displayQuads;
                end
                for concIdx = 1:numConcentrations
                    currentQuad = squeeze(quadSetForBounds(concIdx, :, :));
                    boundsPerConcentration{concIdx} = guiData.cfg.geomTform.geom.computeEllipseAxisBounds(currentQuad, guiData.imageSize, guiData.cfg);
                end

                ellipseData = zeros(totalEllipses, 7);
                ellipseIdx = 1;

                for concIdx = 1:numConcentrations
                    for repIdx = 1:numReplicates
                        if ellipseIdx <= numel(guiData.ellipses) && isvalid(guiData.ellipses{ellipseIdx})
                            ellipse = guiData.ellipses{ellipseIdx};
                            center = ellipse.Center;
                            semiAxes = ellipse.SemiAxes;
                            rotationAngle = ellipse.RotationAngle;

                            bounds = boundsPerConcentration{concIdx};
                            [semiMajor, semiMinor, rotationAngle] = guiData.cfg.geomTform.geom.enforceEllipseAxisLimits(semiAxes(1), semiAxes(2), rotationAngle, bounds);

                            ellipseData(ellipseIdx, :) = [concIdx-1, repIdx-1, round(center), semiMajor, semiMinor, rotationAngle];
                        end
                        ellipseIdx = ellipseIdx + 1;
                    end
                end

                % Transform ellipse coordinates from display space to original image space
                if guiData.rotation ~= 0
                    originalImageSize = guiData.baseImageSize;
                    rotatedImageSize = guiData.imageSize;

                    for i = 1:size(ellipseData, 1)
                        if ellipseData(i, 3) > 0  % Valid ellipse (check x-coordinate)
                            center = ellipseData(i, 3:4);

                            % Transform center from rotated display to original image
                            originalCenter = guiData.cfg.geomTform.geom.inverseRotatePoints(center, rotatedImageSize, ...
                                originalImageSize, guiData.rotation, guiData.cfg.rotation.angleTolerance);
                            ellipseData(i, 3:4) = round(originalCenter);

                            % Transform ellipse rotation angle
                            % Subtract display rotation to get angle in original image frame
                            ellipseData(i, 7) = mod(ellipseData(i, 7) - guiData.rotation + 180, 360) - 180;
                        end
                    end
                end

                % Store ellipse data in guiData for access by Mode 3 caller
                guiData.ellipseData = ellipseData;
                set(fig, 'UserData', guiData);

                % Return Nx7 matrix instead of cell array
                quadParams = ellipseData;
                rotation = guiData.rotation;
            elseif strcmp(guiData.mode, 'ellipse_editing_grid')
                % Mode 3: Ellipse-only with grid layout (no quads)
                numGroups = guiData.cfg.numSquares;
                numReplicates = guiData.cfg.ellipse.replicatesPerMicropad;
                totalEllipses = numGroups * numReplicates;
                gridBounds = guiData.cfg.geomTform.geom.computeEllipseAxisBounds([], guiData.imageSize, guiData.cfg);

                ellipseData = zeros(totalEllipses, 7);
                ellipseIdx = 1;

                for groupIdx = 1:numGroups
                    for repIdx = 1:numReplicates
                        if ellipseIdx <= numel(guiData.ellipses) && isvalid(guiData.ellipses{ellipseIdx})
                            ellipse = guiData.ellipses{ellipseIdx};
                            center = ellipse.Center;
                            semiAxes = ellipse.SemiAxes;
                            rotationAngle = ellipse.RotationAngle;

                            [semiMajor, semiMinor, rotationAngle] = guiData.cfg.geomTform.geom.enforceEllipseAxisLimits(semiAxes(1), semiAxes(2), rotationAngle, gridBounds);

                            ellipseData(ellipseIdx, :) = [groupIdx-1, repIdx-1, round(center), semiMajor, semiMinor, rotationAngle];
                        end
                        ellipseIdx = ellipseIdx + 1;
                    end
                end

                % Transform ellipse coordinates from display space to original image space
                if guiData.rotation ~= 0
                    originalImageSize = guiData.baseImageSize;
                    rotatedImageSize = guiData.imageSize;

                    for i = 1:size(ellipseData, 1)
                        if ellipseData(i, 3) > 0  % Valid ellipse (check x-coordinate)
                            center = ellipseData(i, 3:4);

                            % Transform center from rotated display to original image
                            originalCenter = guiData.cfg.geomTform.geom.inverseRotatePoints(center, rotatedImageSize, ...
                                originalImageSize, guiData.rotation, guiData.cfg.rotation.angleTolerance);
                            ellipseData(i, 3:4) = round(originalCenter);

                            % Transform ellipse rotation angle
                            ellipseData(i, 7) = mod(ellipseData(i, 7) - guiData.rotation + 180, 360) - 180;
                        end
                    end
                end

                % Store ellipse data in guiData for access by caller
                guiData.ellipseData = ellipseData;
                set(fig, 'UserData', guiData);

                quadParams = []; % No quads in grid mode
                rotation = guiData.rotation;
            elseif strcmp(guiData.mode, 'preview_readonly')
                % Mode 4: Read-only preview, no outputs needed
                quadParams = [];
                rotation = 0;
            end
        elseif strcmp(action, 'back')
            % BACK action only valid in ellipse editing mode
            if ~strcmp(guiData.mode, 'ellipse_editing')
                action = '';
            end
        end
    end
end

function quadParams = extractQuadParameters(guiData)
    quadParams = [];

    if ~isfield(guiData, 'quads') || ~iscell(guiData.quads)
        return;
    end

    validMask = cellfun(@isvalid, guiData.quads);
    if ~any(validMask)
        return;
    end

    validQuads = guiData.quads(validMask);
    keptPositions = cell(1, numel(validQuads));

    keepIdx = 0;
    for i = 1:numel(validQuads)
        pos = validQuads{i}.Position;

        if isempty(pos) || size(pos, 1) < 4
            warning('cut_micropads:invalid_quad', ...
                    'Quad %d is missing vertices. Ignoring this quad for extraction.', i);
            continue;
        end

        keepIdx = keepIdx + 1;
        keptPositions{keepIdx} = pos(1:4, :);
    end

    if keepIdx == 0
        quadParams = [];
        return;
    end

    quadParams = zeros(keepIdx, 4, 2);
    for i = 1:keepIdx
        quadParams(i, :, :) = keptPositions{i};
    end

    [quadParams, ~] = sortQuadArrayByX(quadParams);
end

%% -------------------------------------------------------------------------
%% Image Cropping and Coordinate Saving (delegating wrappers to file_io_manager)
%% -------------------------------------------------------------------------

function [quadParams, found, rotation] = loadQuadCoordinates(coordFile, imageName, numExpected)
    fileIO = file_io_manager();
    [quadParams, found, rotation] = fileIO.loadQuadCoordinates(coordFile, imageName, numExpected);
end

function [ellipseData, found] = loadEllipseCoordinates(coordFile, imageName)
    fileIO = file_io_manager();
    [ellipseData, found] = fileIO.loadEllipseCoordinates(coordFile, imageName);
end

function ellipsePositions = createDefaultEllipseGrid(imageSize, numGroups, replicatesPerGroup, cfg)
    % Create default ellipse positions in horizontal grid layout
    %
    % INPUTS:
    %   imageSize         - [height, width] of image
    %   numGroups         - Number of concentration groups (e.g., 7)
    %   replicatesPerGroup - Number of ellipses per group (e.g., 3)
    %   cfg               - Configuration structure with ellipse parameters
    %
    % OUTPUTS:
    %   ellipsePositions - (numGroups * replicatesPerGroup) x 2 matrix of [x, y] centers

    imageHeight = imageSize(1);
    imageWidth = imageSize(2);

    % Vertical placement respects configuration ratio
    yCenter = imageHeight * cfg.ellipse.verticalPositionRatio;
    yCenter = max(1, min(yCenter, imageHeight));

    % Horizontal layout driven by margin-to-spacing ratio
    marginRatio = max(cfg.ellipse.marginToSpacingRatio, eps);
    totalUnits = (2 * marginRatio) + numGroups + max(numGroups - 1, 0) * marginRatio;
    unitWidth = imageWidth / totalUnits;
    outerMargin = marginRatio * unitWidth;
    interGroupSpacing = marginRatio * unitWidth;
    groupWidth = unitWidth;

    % Within-group spacing for replicates (reuse ratio)
    replicateMargin = groupWidth * marginRatio / (marginRatio + 1);
    replicateUsableWidth = max(groupWidth - 2 * replicateMargin, eps);
    if replicatesPerGroup > 1
        replicateSpacing = replicateUsableWidth / (replicatesPerGroup - 1);
    else
        replicateSpacing = 0;
    end

    % Generate positions
    totalEllipses = numGroups * replicatesPerGroup;
    ellipsePositions = zeros(totalEllipses, 2);
    ellipseIdx = 1;

    for groupIdx = 1:numGroups
        % Calculate group start and center
        startX = outerMargin + (groupIdx - 1) * (groupWidth + interGroupSpacing);
        groupCenterX = startX + groupWidth / 2;

        % Position replicates within group
        for repIdx = 1:replicatesPerGroup
            if replicatesPerGroup == 1
                % Single replicate - center in group
                xPos = groupCenterX;
            else
                % Multiple replicates - distribute horizontally
                xPos = startX + replicateMargin + (repIdx - 1) * replicateSpacing;
            end
            
            % Clamp positions to image bounds
            xPos = max(1, min(xPos, imageWidth));
            yPos = max(1, min(yCenter, imageHeight));

            ellipsePositions(ellipseIdx, :) = [xPos, yPos];
            ellipseIdx = ellipseIdx + 1;
        end
    end
end

%% -------------------------------------------------------------------------
%% File I/O Utilities (delegating wrappers to file_io_manager)
%% -------------------------------------------------------------------------

function outputDirs = createOutputDirectory(basePathQuads, basePathEllipses, phoneName, numConcentrations, concFolderPrefix)
    fileIO = file_io_manager();
    outputDirs = fileIO.createOutputDirectory(basePathQuads, basePathEllipses, phoneName, numConcentrations, concFolderPrefix);
end

function files = getImageFiles(dirPath, extensions)
    fileIO = file_io_manager();
    files = fileIO.getImageFiles(dirPath, extensions);
end

%% -------------------------------------------------------------------------
%% Memory System
%% -------------------------------------------------------------------------

function memory = initializeMemory()
    % Initialize empty memory structure
    memory = struct();
    memory.hasSettings = false;
    memory.baseQuads = [];     % Base (unrotated) coordinates
    memory.rotation = 0;          % Image rotation angle
    memory.baseImageSize = [];    % Base image dimensions [height, width]
    memory.ellipses = {};         % Cell array indexed by concentration (0-based)
    memory.hasEllipseSettings = false;  % Boolean flag indicating if ellipse settings are saved
    memory.orientation = 'horizontal';  % Layout orientation: 'horizontal' or 'vertical'
end

function memory = updateMemory(memory, baseQuads, rotation, baseImageSize, ellipseData, cfg, orientation)
    % Update memory with base (unrotated) quad coordinates and rotation
    % When loading, we scale base→base then apply rotation to get display coords
    memory.hasSettings = true;
    memory.baseQuads = baseQuads;
    memory.rotation = rotation;
    memory.baseImageSize = baseImageSize;

    % Store orientation if provided
    if nargin >= 7 && ~isempty(orientation)
        memory.orientation = orientation;
    end

    % Compute display quads for ellipse scaling (needs display coordinates)
    if ~isempty(baseQuads)
        displayImageSize = computeDisplayImageSize(baseImageSize, rotation, cfg);
        displayQuads = convertBaseQuadsToDisplay(baseQuads, baseImageSize, displayImageSize, rotation, cfg);
        memory.quads = displayQuads;
    end

    % Store ellipse data if provided
    if nargin >= 6 && ~isempty(ellipseData)
        displayImageSize = computeDisplayImageSize(baseImageSize, rotation, cfg);
        ellipseDisplay = convertBaseEllipsesToDisplay(ellipseData, baseImageSize, displayImageSize, rotation, cfg);
        % Store ellipse data per concentration for memory persistence
        memory.ellipses = cell(1, max(ellipseDisplay(:, 1)) + 1);
        for i = 1:size(ellipseDisplay, 1)
            concIdx = ellipseDisplay(i, 1) + 1;  % Convert 0-indexed to 1-indexed
            if isempty(memory.ellipses{concIdx})
                memory.ellipses{concIdx} = [];
            end
            memory.ellipses{concIdx} = [memory.ellipses{concIdx}; ellipseDisplay(i, :)];
        end
        memory.hasEllipseSettings = true;
    end
end

function [initialQuads, rotation, source] = getInitialQuadsWithMemory(img, cfg, memory, imageSize)
    % Get initial quads and rotation with progressive AI detection workflow
    % Priority: memory (if available) -> default -> AI updates later
    %
    % Memory stores base (unrotated) quads. We:
    % 1. Scale base quads from old base size to new base size
    % 2. Apply rotation to convert to display coordinates

    % Check memory FIRST (even when AI is enabled)
    if memory.hasSettings && ~isempty(memory.baseQuads) && ~isempty(memory.baseImageSize)
        % Scale base quads from old base size to new base size
        scaledBaseQuads = scaleQuadsForImageSize(memory.baseQuads, memory.baseImageSize, imageSize, cfg.numSquares);

        % If quadcount mismatch, fall back to default geometry
        if isempty(scaledBaseQuads)
            fprintf('  Memory quadcount mismatch - using default geometry\n');
            [imageHeight, imageWidth, ~] = size(img);
            initialQuads = cfg.geomTform.geom.calculateDefaultQuads(imageWidth, imageHeight, cfg);
            rotation = 0;
            source = 'default';
            return;
        end

        % Apply stored rotation to convert base→display coordinates
        displaySize = computeDisplayImageSize(imageSize, memory.rotation, cfg);
        initialQuads = convertBaseQuadsToDisplay(scaledBaseQuads, imageSize, displaySize, memory.rotation, cfg);

        rotation = memory.rotation;
        fprintf('  Using quadshapes and rotation from memory (AI will update if enabled)\n');
        source = 'memory';
        return;
    end

    % No memory available: use default geometry for immediate display
    [imageHeight, imageWidth, ~] = size(img);
    fprintf('  Using default geometry (AI will update if enabled)\n');
    initialQuads = cfg.geomTform.geom.calculateDefaultQuads(imageWidth, imageHeight, cfg);
    rotation = 0;
    source = 'default';

    % NOTE: AI detection will run asynchronously after GUI displays
end

function scaledQuads = scaleQuadsForImageSize(quads, oldSize, newSize, expectedCount)
    % Scale quad coordinates when image dimensions change
    %
    % Inputs:
    %   quads - [N x 4 x 2] array of quadvertices
    %   oldSize - [height, width] of previous image
    %   newSize - [height, width] of current image
    %   expectedCount - expected number of quads (optional)
    %
    % Returns empty if quadcount doesn't match expectedCount

    if isempty(oldSize) || any(oldSize <= 0) || isempty(newSize) || any(newSize <= 0)
        error('cut_micropads:invalid_dimensions', ...
            'Cannot scale quads: invalid dimensions [%d %d] -> [%d %d]', ...
            oldSize(1), oldSize(2), newSize(1), newSize(2));
    end

    % Validate quadcount if expectedCount is provided
    numQuads = size(quads, 1);
    if nargin >= 4 && ~isempty(expectedCount) && numQuads ~= expectedCount
        scaledQuads = [];
        return;
    end

    oldHeight = oldSize(1);
    oldWidth = oldSize(2);
    newHeight = newSize(1);
    newWidth = newSize(2);

    if oldHeight == newHeight && oldWidth == newWidth
        scaledQuads = quads;
        return;
    end

    scaleX = newWidth / oldWidth;
    scaleY = newHeight / oldHeight;

    scaledQuads = zeros(size(quads));

    for i = 1:numQuads
        poly = squeeze(quads(i, :, :));
        poly(:, 1) = poly(:, 1) * scaleX;
        poly(:, 2) = poly(:, 2) * scaleY;
        poly(:, 1) = max(1, min(poly(:, 1), newWidth));
        poly(:, 2) = max(1, min(poly(:, 2), newHeight));
        scaledQuads(i, :, :) = poly;
    end
end



function runDeferredAIDetection(fig, cfg)
    % Run AI detection asynchronously and update quads when complete
    %
    % Called via timer after GUI is fully rendered

    % CRITICAL FIX: Guard against invalid/deleted figure
    if ~ishandle(fig) || ~isvalid(fig)
        return;
    end

    guiData = get(fig, 'UserData');

    % CRITICAL FIX: Guard against empty or non-struct guiData
    if isempty(guiData) || ~isstruct(guiData)
        return;
    end

    if ~strcmp(guiData.mode, 'editing')
        return;
    end

    % Validate AI detection prerequisites (allows manual detection even if auto-detection disabled)
    if ~isfile(cfg.pythonScriptPath) || ~isfile(cfg.detectionModel)
        return;
    end

    % Check if detection is already running
    if guiData.asyncDetection.active
        return;
    end

    % Show progress indicator (starts animation timer)
    showAIProgressIndicator(fig, true);

    % Re-fetch guiData to get breathing timer reference
    guiData = get(fig, 'UserData');

    try
        % Launch async detection on BASE (unrotated) image
        % AI works on original image frame, coordinates are then rotated to match display
        img = guiData.baseImg;
        [~, ~, outputFile, imgPath] = cfg.yoloUtils.detectQuadsYOLO(img, cfg, 'async', true);

        % Store detection state
        guiData.asyncDetection.active = true;
        guiData.asyncDetection.outputFile = outputFile;
        guiData.asyncDetection.imgPath = imgPath;
        guiData.asyncDetection.startTime = tic;
        % Capture launch state for validation (including generation counter)
        guiData.asyncDetection.launchRotation = guiData.totalRotation;
        guiData.asyncDetection.launchImgSize = [size(guiData.currentImg, 2), size(guiData.currentImg, 1)];  % [width, height]
        guiData.asyncDetection.launchGeneration = guiData.asyncDetection.generation;

        % Create polling timer (100ms interval)
        guiData.asyncDetection.pollingTimer = timer(...
            'Period', 0.1, ...
            'ExecutionMode', 'fixedSpacing', ...
            'TimerFcn', @(~,~) pollDetectionStatus(fig, cfg));

        % Save state and start polling
        set(fig, 'UserData', guiData);
        start(guiData.asyncDetection.pollingTimer);

    catch ME
        % Failed to launch - clean up and fall back to default geometry
        warning('cut_micropads:async_launch_failed', ...
                'Failed to launch async detection: %s', ME.message);
        cancelActiveDetection(fig, guiData, cfg);
        guiData = get(fig, 'UserData');
        set(fig, 'UserData', guiData);
    end
end

function pollDetectionStatus(fig, cfg)
    % Poll for async detection completion and update quads when ready
    %
    % Called by polling timer every 100ms

    % Guard against invalid figure
    if ~ishandle(fig) || ~isvalid(fig)
        return;
    end

    guiData = get(fig, 'UserData');

    % Guard against invalid guiData
    if isempty(guiData) || ~isstruct(guiData)
        return;
    end

    % Guard against inactive detection
    if ~guiData.asyncDetection.active
        return;
    end

    % Check for timeout
    elapsed = toc(guiData.asyncDetection.startTime);
    if elapsed > guiData.asyncDetection.timeoutSeconds
        fprintf('  AI detection timeout after %.1f seconds\n', elapsed);
        cleanupAsyncDetection(fig, guiData, false, cfg);
        return;
    end

    % Check if detection completed (validate against base image dimensions)
    [isComplete, quads, confidences, errorMsg] = cfg.yoloUtils.checkDetectionComplete(...
        guiData.asyncDetection.outputFile, ...
        guiData.baseImg);

    if ~isComplete
        return;  % Still running, keep polling
    end

    % Check for error messages
    if ~isempty(errorMsg)
        fprintf('  AI detection failed: %s\n', errorMsg);
    end

    % Detection finished - update quads if successful
    detectionSucceeded = false;

    if ~isempty(quads)
        numDetected = size(quads, 1);

        if numDetected >= cfg.numSquares
            % Validate launch generation matches current generation (invalidates ALL stale detections)
            launchGen = guiData.asyncDetection.launchGeneration;
            currentGen = guiData.asyncDetection.generation;

            if launchGen ~= currentGen
                fprintf('  Discarding stale detection (generation mismatch: %d vs %d). Fresh detection will launch.\n', launchGen, currentGen);
                cleanupAsyncDetection(fig, guiData, false, cfg);
                return;
            end

            % Validate rotation and size as additional safety check
            launchRot = guiData.asyncDetection.launchRotation;
            currentRot = guiData.totalRotation;
            launchSize = guiData.asyncDetection.launchImgSize;
            currentSize = [size(guiData.currentImg, 2), size(guiData.currentImg, 1)];

            if launchRot ~= currentRot || ~isequal(launchSize, currentSize)
                fprintf('  Discarding stale detection (rotation/size changed during detection). Fresh detection will launch.\n');
                cleanupAsyncDetection(fig, guiData, false, cfg);
                return;
            end

            % Use top N detections by confidence
            [~, sortIdx] = sort(confidences, 'descend');
            quads = quads(sortIdx(1:cfg.numSquares), :, :);
            topConfidences = confidences(sortIdx(1:cfg.numSquares));

            % Transform base-frame detections to current display frame
            % AI worked on baseImg, so quads are in base frame - rotate forward to match display
            [newQuads, ~] = cfg.geomTform.geom.rotateQuadsDiscrete(quads, guiData.baseImageSize, guiData.totalRotation);
            
            % Sort by display-frame X coordinate for consistent labeling
            [newQuads, newOrientation] = sortQuadArrayByX(newQuads);
            guiData.orientation = newOrientation;  % Update orientation based on AI detection

            % Apply detected quads with race condition guard
            try
                guiData = applyDetectedQuads(guiData, newQuads, cfg, fig);
                detectionSucceeded = true;

                fprintf('  AI detection complete: %d regions (avg confidence: %.2f)\n', ...
                        cfg.numSquares, mean(topConfidences));
            catch ME
                % Figure was deleted during update - ignore error
                if strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
                    return;
                else
                    rethrow(ME);
                end
            end
        else
            fprintf('  AI detected only %d/%d regions - keeping initial positions\n', ...
                    numDetected, cfg.numSquares);
        end
    else
        fprintf('  AI detection found no regions - keeping initial positions\n');
    end

    % Re-check figure validity after quadupdate (race condition guard)
    if ~ishandle(fig) || ~isvalid(fig)
        return;
    end

    % Clear cached zoom bounds after detection (with race condition guard)
    try
        if isvalid(fig)
            % Force recalculation of zoom bounds from current quadstate
            guiData.autoZoomBounds = [];

            % Save state to figure
            set(fig, 'UserData', guiData);

            % Re-fetch guiData to ensure cleanup gets fresh state (handles concurrent updates)
            guiData = get(fig, 'UserData');

            % Clean up async state
            cleanupAsyncDetection(fig, guiData, detectionSucceeded, cfg);
        end
    catch ME
        % Figure was deleted during final state update - ignore error
        if strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
            return;
        else
            rethrow(ME);
        end
    end
end

function cleanupAsyncDetection(fig, guiData, success, cfg)
    % Clean up async detection resources and stop animation
    %
    % Inputs:
    %   fig - figure handle
    %   guiData - GUI data structure (may be modified)
    %   success - true if detection succeeded and quads updated
    %   cfg - configuration struct (for auto-zoom)

    % Stop and delete polling timer
    if ~isempty(guiData.asyncDetection.pollingTimer)
        cfg.micropadUI.safeStopTimer(guiData.asyncDetection.pollingTimer);
    end

    % Clean up output file
    if ~isempty(guiData.asyncDetection.outputFile) && ...
       exist(guiData.asyncDetection.outputFile, 'file')
        try
            delete(guiData.asyncDetection.outputFile);
        catch
            % Ignore cleanup errors
        end
    end

    % Clean up image file
    if ~isempty(guiData.asyncDetection.imgPath) && ...
       exist(guiData.asyncDetection.imgPath, 'file')
        try
            delete(guiData.asyncDetection.imgPath);
        catch
            % Ignore cleanup errors
        end
    end

    % Reset async state
    guiData.asyncDetection.active = false;
    guiData.asyncDetection.outputFile = '';
    guiData.asyncDetection.imgPath = '';
    guiData.asyncDetection.startTime = [];
    guiData.asyncDetection.pollingTimer = [];
    guiData.asyncDetection.launchRotation = 0;
    guiData.asyncDetection.launchImgSize = [0, 0];

    % Stop animation and apply auto-zoom (with race condition guard)
    try
        if isvalid(fig)
            % Stop animation
            showAIProgressIndicator(fig, false);

            % Update guiData
            set(fig, 'UserData', guiData);

            % Apply auto-zoom if detection succeeded
            if success
                applyAutoZoom(fig, guiData, cfg);
            end
        end
    catch ME
        % Figure was deleted during cleanup - ignore error
        if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
            rethrow(ME);
        end
    end
end

function cancelActiveDetection(fig, guiData, cfg)
    % Cancel any in-flight async detection
    %
    % Inputs:
    %   fig - figure handle
    %   guiData - GUI data structure
    %   cfg - configuration struct

    if ~isfield(guiData, 'asyncDetection') || ~guiData.asyncDetection.active
        return;
    end

    fprintf('  Canceling in-flight AI detection...\n');
    cleanupAsyncDetection(fig, guiData, false, cfg);
end

function updateQuadPositions(quadHandles, newPositions, labelHandles, ui)
    % Update drawpolygon positions smoothly without recreating objects
    %
    % Inputs:
    %   quadHandles - cell array of drawpolygon objects
    %   newPositions - [N x 4 x 2] array of new quad positions
    %   labelHandles - cell array of text objects (optional)

    n = numel(quadHandles);
    if size(newPositions, 1) ~= n
        warning('Quad count mismatch: %d handles vs %d positions', n, size(newPositions, 1));
        return;
    end

    for i = 1:n
        poly = quadHandles{i};
        if ~isvalid(poly)
            continue;
        end

        newPos = squeeze(newPositions(i, :, :));

        % Update position property directly (smooth transition)
        poly.Position = newPos;

        % CRITICAL: Update LastValidPosition appdata to prevent snap-back on next drag
        % (enforceQuadrilateral listener compares against this stored value)
        setappdata(poly, 'LastValidPosition', newPos);
    end

    % NEW: Update labels after quad positions change
    if nargin >= 3 && ~isempty(labelHandles)
        if nargin >= 4 && ~isempty(ui)
            ui.updateQuadLabels(quadHandles, labelHandles);
        end
    end

    drawnow limitrate;
end

%% -------------------------------------------------------------------------
%% Error Handling
%% -------------------------------------------------------------------------

function handleError(ME)
    if strcmp(ME.message, 'User stopped execution')
        fprintf('\n!! Script stopped by user\n');
        return;
    end

    fprintf('\n!! ERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:numel(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end

function cleanupAndClose(fig)
    % Clean up timers and progress indicators before closing figure

    % Guard against invalid figure
    if ~isvalid(fig)
        return;
    end

    guiData = get(fig, 'UserData');

    % Cleanup all timers without cfg dependency (prevents leak on early errors)
    if isstruct(guiData)
        ui = [];
        if isfield(guiData, 'cfg') && isfield(guiData.cfg, 'micropadUI')
            ui = guiData.cfg.micropadUI;
        end
        % Direct timer fields
        timerFields = {'aiTimer', 'aiBreathingTimer'};
        for i = 1:numel(timerFields)
            if isfield(guiData, timerFields{i})
                timerHandle = guiData.(timerFields{i});
                if ~isempty(ui)
                    ui.safeStopTimer(timerHandle);
                elseif isvalid(timerHandle)
                    try
                        stop(timerHandle);
                    catch
                        % Ignore: timer may already be stopped/deleted
                    end
                    delete(timerHandle);
                end
            end
        end

        % Async detection polling timer (nested field)
        if isfield(guiData, 'asyncDetection') && isstruct(guiData.asyncDetection) && ...
           isfield(guiData.asyncDetection, 'pollingTimer')
            if ~isempty(ui)
                ui.safeStopTimer(guiData.asyncDetection.pollingTimer);
            elseif isvalid(guiData.asyncDetection.pollingTimer)
                try
                    stop(guiData.asyncDetection.pollingTimer);
                catch
                    % Ignore: timer may already be stopped/deleted
                end
                delete(guiData.asyncDetection.pollingTimer);
            end
        end
    end

    % Clean up async detection if active (this also stops polling timer via cleanupAsyncDetection)
    if isstruct(guiData) && isfield(guiData, 'cfg')
        cancelActiveDetection(fig, guiData, guiData.cfg);
        guiData = get(fig, 'UserData');
    end

    % CRITICAL FIX: Set action='stop' before deleting so main loop can exit cleanly
    % Without this, waitForUserAction returns empty action, main loop continues,
    % and tries to rebuild UI with deleted figure handle
    if isstruct(guiData)
        guiData.action = 'stop';
        set(fig, 'UserData', guiData);
    end

    % Clean up progress indicator
    showAIProgressIndicator(fig, false);

    % CRITICAL FIX: Resume event loop before deleting
    % This allows waitForUserAction to read the 'stop' action we just set
    if isvalid(fig) && strcmp(get(fig, 'waitstatus'), 'waiting')
        uiresume(fig);
        pause(0.01);  % Allow event loop to process resume
    end

    % Delete figure (with final validity check)
    if isvalid(fig)
        delete(fig);
    end
end
