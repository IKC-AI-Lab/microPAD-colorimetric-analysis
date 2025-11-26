function cut_micropads(varargin)
    %% microPAD Colorimetric Analysis — Unified microPAD Processing Tool
    %% Detect and extract polygonal concentration regions from raw microPAD images
    %% Author: Veysel Y. Yilmaz
    %
    % This script combines rotation adjustment and AI-powered polygon detection
    % to directly process raw microPAD images into concentration region crops.
    %
    % Pipeline stage: 1_dataset → 2_micropads
    %
    % Features:
    %   - Interactive rotation adjustment with memory
    %   - AI-powered polygon detection (YOLOv11s-pose)
    %   - Manual polygon editing and refinement
    %   - Saves polygon coordinates with rotation angle
    %
    % Inputs (Name-Value pairs):
    % - 'numSquares': number of regions to capture per strip (default: 7)
    % - 'aspectRatio': width/height ratio of each region (default: 1.0, perfect squares)
    % - 'coverage': fraction of image width to fill (default: 0.80)
    % - 'gapPercent': gap as percent of region width, 0..1 or 0..100 (default: 0.19)
    % - 'inputFolder' | 'outputFolder': override default I/O folders
    % - 'saveCoordinates': output behavior
    % - 'useAIDetection': use YOLO for initial polygon placement (default: true)
    % - 'detectionModel': path to YOLOv11 pose model (default: 'models/yolo11s-micropad-pose-1280.pt')
    % - 'minConfidence': minimum detection confidence (default: 0.6)
    % - 'inferenceSize': YOLO inference image size in pixels (default: 1280)
    % - 'pythonPath': path to Python executable (default: '' - uses MICROPAD_PYTHON env var)
    %
    % Outputs/Side effects:
    % - Writes PNG polygon crops to 2_micropads/[phone]/con_*/
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
    % - User can manually adjust polygons before saving
    % - Cuts N region crops and saves into con_0..con_(N-1) subfolders for each strip
    % - All polygon coordinates written to single phone-level coordinates.txt
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
        error('micropad:deprecated_parameter', ...
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
    DEFAULT_ENABLE_POLYGON_EDITING = true;
    REPLICATES_PER_CONCENTRATION = 3;

    % Ellipse definitions in normalized micropad coordinates [0,1]×[0,1]
    % Each row: [x, y, semiMajorAxis, semiMinorAxis, rotationAngle]
    % - x, y: center position (0,0 = top-left, 1,1 = bottom-right)
    % - semiMajorAxis, semiMinorAxis: fraction of micropad side length
    % - rotationAngle: degrees, clockwise from horizontal
    % These values define ellipse positions relative to an ideal square micropad.
    % The homography transform will adjust for perspective distortion.
    ELLIPSE_DEFAULT_RECORDS = [
        % x,    y,    semiMajor, semiMinor, rotationAngle
        0.25,  0.40,  0.10,      0.08,      -45;  % Replicate 0 (Urea)
        0.5,  0.35,  0.10,      0.08,        0;  % Replicate 1 (Creatinine)
        0.75,  0.40,  0.10,      0.08,       45   % Replicate 2 (Lactate)
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

    % === UI CONSTANTS ===
    UI_CONST = struct();
    UI_CONST.fontSize = struct(...
        'title', 16, ...
        'path', 12, ...
        'button', 13, ...
        'info', 10, ...
        'instruction', 10, ...
        'preview', 14, ...
        'label', 12, ...
        'status', 13, ...
        'value', 13);
    UI_CONST.colors = struct(...
        'background', 'black', ...
        'foreground', 'white', ...
        'panel', [0.15 0.15 0.15], ...
        'stop', [0.85 0.2 0.2], ...
        'accept', [0.2 0.75 0.3], ...
        'retry', [0.9 0.75 0.2], ...
        'skip', [0.75 0.25 0.25], ...
        'polygon', [0.0 1.0 1.0], ...
        'info', [1.0 1.0 0.3], ...
        'path', [0.75 0.75 0.75], ...
        'apply', [0.2 0.5 0.9]);
    % UI positions use normalized coordinates [x, y, width, height]
    % Origin: (0, 0) = bottom-left, (1, 1) = top-right
    % Standard gap: 0.01 (1% of figure)
    UI_CONST.positions = struct(...
        'figure', [0 0 1 1], ...                      % Full window
        'stopButton', [0.01 0.945 0.06 0.045], ...    % Top-left corner, 6% width
        'title', [0.08 0.945 0.84 0.045], ...         % Top center bar (84% width)
        'pathDisplay', [0.08 0.90 0.84 0.035], ...    % Below title, same width
        'aiStatus', [0.25 0.905 0.50 0.035], ...      % Centered AI status label
        'instructions', [0.01 0.855 0.98 0.035], ...  % Full-width instruction text
        'image', [0.01 0.215 0.98 0.64], ...          % Primary image axes (64% height)
        'runAIButton', [0.01 0.16 0.08 0.045], ...    % "RUN AI" button above rotation panel
        'rotationPanel', [0.01 0.01 0.24 0.14], ...   % Left: rotation preset buttons
        'zoomPanel', [0.26 0.01 0.26 0.14], ...       % Center: zoom slider + controls
        'cutButtonPanel', [0.53 0.01 0.46 0.14], ...  % Right: APPLY/SKIP buttons (46% = 98%-24%-26%-2%)
        'previewPanel', [0.25 0.01 0.50 0.14], ...    % Preview action buttons in main window
        'previewTitle', [0.01 0.92 0.98 0.04], ...    % Preview window title bar
        'previewMeta', [0.01 0.875 0.98 0.035], ...   % Preview metadata text below title
        'previewLeft', [0.01 0.22 0.48 0.64], ...     % Left comparison image (48% width)
        'previewRight', [0.50 0.22 0.49 0.64]);       % Right comparison image (49% width)
    UI_CONST.polygon = struct(...
        'lineWidth', 3, ...
        'borderWidth', 2);
    UI_CONST.dimFactor = 0.2;
    UI_CONST.layout = struct();
    UI_CONST.layout.rotationLabel = [0.05 0.78 0.90 0.18];
    UI_CONST.layout.quickRotationRow1 = {[0.05 0.42 0.42 0.30], [0.53 0.42 0.42 0.30]};
    UI_CONST.layout.quickRotationRow2 = {[0.05 0.08 0.42 0.30], [0.53 0.08 0.42 0.30]};
    UI_CONST.layout.zoomLabel = [0.05 0.78 0.90 0.18];
    UI_CONST.layout.zoomSlider = [0.05 0.42 0.72 0.28];
    UI_CONST.layout.zoomValue = [0.79 0.42 0.16 0.28];
    UI_CONST.layout.zoomResetButton = [0.05 0.08 0.44 0.28];
    UI_CONST.layout.zoomAutoButton = [0.51 0.08 0.44 0.28];
    UI_CONST.rotation = struct(...
        'range', [-180, 180], ...
        'quickAngles', [-90, 0, 90, 180]);
    UI_CONST.zoom = struct(...
        'range', [0, 1], ...
        'defaultValue', 0);

    %% Add helper_scripts to path (contains homography_utils and other utilities)
    scriptDir = fileparts(mfilename('fullpath'));
    helperDir = fullfile(scriptDir, 'helper_scripts');
    if exist(helperDir, 'dir')
        addpath(helperDir);
    end

    %% Load homography utilities for ellipse transformation
    homography = homography_utils();

    %% Build configuration
    cfg = createConfiguration(INPUT_FOLDER, OUTPUT_FOLDER, OUTPUT_FOLDER_ELLIPSES, SAVE_COORDINATES, ...
                              DEFAULT_NUM_SQUARES, DEFAULT_ASPECT_RATIO, DEFAULT_COVERAGE, DEFAULT_GAP_PERCENT, ...
                              DEFAULT_ENABLE_ELLIPSE_EDITING, DEFAULT_ENABLE_POLYGON_EDITING, REPLICATES_PER_CONCENTRATION, ...
                              MARGIN_TO_SPACING_RATIO, VERTICAL_POSITION_RATIO, MIN_AXIS_PERCENT, ...
                              SEMI_MAJOR_DEFAULT_RATIO, SEMI_MINOR_DEFAULT_RATIO, ELLIPSE_DEFAULT_RECORDS, ...
                              DEFAULT_USE_AI_DETECTION, DEFAULT_DETECTION_MODEL, DEFAULT_MIN_CONFIDENCE, DEFAULT_PYTHON_PATH, DEFAULT_INFERENCE_SIZE, ...
                              ROTATION_ANGLE_TOLERANCE, ...
                              COORDINATE_FILENAME, SUPPORTED_FORMATS, ALLOWED_IMAGE_EXTENSIONS, CONC_FOLDER_PREFIX, UI_CONST, homography, varargin{:});

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
                                   defaultEnableEllipseEditing, defaultEnablePolygonEditing, replicatesPerConcentration, ...
                                   marginToSpacingRatio, verticalPositionRatio, minAxisPercent, ...
                                   semiMajorDefaultRatio, semiMinorDefaultRatio, ellipseDefaultRecords, ...
                                   defaultUseAI, defaultDetectionModel, defaultMinConfidence, defaultPythonPath, defaultInferenceSize, ...
                                   rotationAngleTolerance, ...
                                   coordinateFileName, supportedFormats, allowedImageExtensions, concFolderPrefix, UI_CONST, homography, varargin)
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
    parser.addParameter('enablePolygonEditing', defaultEnablePolygonEditing, @islogical);

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

    cfg = addPathConfiguration(cfg, parser.Results.inputFolder, parser.Results.outputFolder, outputFolderEllipses);

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
    cfg.enablePolygonEditing = parser.Results.enablePolygonEditing;
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

    % Store homography utilities for ellipse transformation
    cfg.homography = homography;

    % Rotation configuration
    cfg.rotation.angleTolerance = rotationAngleTolerance;

    % UI configuration
    cfg.ui.fontSize = UI_CONST.fontSize;
    cfg.ui.colors = UI_CONST.colors;
    cfg.ui.positions = UI_CONST.positions;
    cfg.ui.polygon = UI_CONST.polygon;
    cfg.ui.layout = UI_CONST.layout;
    cfg.ui.rotation = UI_CONST.rotation;
    cfg.ui.zoom = UI_CONST.zoom;
    cfg.dimFactor = UI_CONST.dimFactor;
end

function cfg = addPathConfiguration(cfg, inputFolder, outputFolder, outputFolderEllipses)
    projectRoot = find_project_root(inputFolder);

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

function projectRoot = find_project_root(inputFolder)
    searchPath = pwd;
    maxLevels = 5;

    for level = 1:maxLevels
        candidatePath = fullfile(searchPath, inputFolder);
        if isfolder(candidatePath)
            projectRoot = searchPath;
            return;
        end
        parentPath = fileparts(searchPath);
        if strcmp(parentPath, searchPath)
            break;
        end
        searchPath = parentPath;
    end

    warning('cut_micropads:no_input_folder', ...
        'Could not find input folder "%s" within %d directory levels. Using current directory as project root.', ...
        inputFolder, maxLevels);
    projectRoot = pwd;
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

    executeInFolder(cfg.inputPath, @() processPhones(cfg));
end

function processPhones(cfg)
    phoneFolders = getSubFolders('.');
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
    executeInFolder(phoneName, @() processImagesInPhone(phoneName, cfg));
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
        ensurePythonSetup(cfg.pythonPath);
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

    [img, isValid] = loadImage(imageName);
    if ~isValid
        fprintf('  !! Failed to load image\n');
        return;
    end

    % Get image dimensions
    [imageHeight, imageWidth, ~] = size(img);

    % Initialize rotation (may be overridden by mode-specific logic)
    initialRotation = 0;

    % Determine operating mode based on enablePolygonEditing and enableEllipseEditing
    if cfg.enablePolygonEditing && cfg.enableEllipseEditing
        processingMode = 1; % Unified: polygon edit -> ellipse edit -> preview
    elseif cfg.enablePolygonEditing && ~cfg.enableEllipseEditing
        processingMode = 2; % Polygon-only: polygon edit -> preview
    elseif ~cfg.enablePolygonEditing && cfg.enableEllipseEditing
        processingMode = 3; % Ellipse-only: [load/default] -> ellipse edit -> preview
    else
        processingMode = 4; % Read-only preview: [load coords] -> preview
    end

    % Mode 4: Read-only preview (both editing disabled)
    if processingMode == 4
        % Try loading polygon coordinates
        polygonCoordFile = fullfile(outputDirs.polygonDir, cfg.coordinateFileName);
        [loadedPolygons, polygonsFound] = loadPolygonCoordinates(polygonCoordFile, imageName, cfg.numSquares);

        % Try loading ellipse coordinates
        ellipseCoordFile = fullfile(outputDirs.ellipseDir, cfg.coordinateFileName);
        [loadedEllipses, ellipsesFound] = loadEllipseCoordinates(ellipseCoordFile, imageName);

        % Error if neither found
        if ~polygonsFound && ~ellipsesFound
            error('cut_micropads:no_coordinates_for_preview', ...
                ['No coordinate files found for preview mode (image: %s).\n' ...
                 'Enable at least one editing mode or ensure coordinate files exist in:\n' ...
                 '  - %s (polygons)\n' ...
                 '  - %s (ellipses)'], ...
                imageName, polygonCoordFile, ellipseCoordFile);
        end

        % Build read-only preview UI
        buildReadOnlyPreviewUI(fig, img, imageName, phoneName, cfg, loadedPolygons, polygonsFound, ...
                              loadedEllipses, ellipsesFound, initialRotation);

        % Wait for user to advance to next image
        [action, ~, ~] = waitForUserAction(fig);

        if strcmp(action, 'stop')
            return;
        else
            % No save operation in preview mode - just advance to next image
            success = true;
            return;
        end
    end

    % Mode 3: Ellipse-only editing
    if processingMode == 3
        % Try loading polygon coordinates for positioning context
        polygonCoordFile = fullfile(outputDirs.polygonDir, cfg.coordinateFileName);
        [loadedPolygons, polygonsFound] = loadPolygonCoordinates(polygonCoordFile, imageName, cfg.numSquares);

        if polygonsFound
            % Use loaded polygons for positioning
            polygonParams = loadedPolygons;
            fprintf('  Mode 3: Loaded %d polygon coordinates for ellipse positioning\n', size(polygonParams, 1));

            % Go directly to ellipse editing with polygon overlays
            buildEllipseEditingUI(fig, img, imageName, phoneName, cfg, polygonParams, initialRotation, memory);

        else
            % No polygons - use default grid layout
            fprintf('  Mode 3: No polygon coordinates - using default grid layout\n');

            % Create default ellipse positions using grid
            imageSize = [size(img, 1), size(img, 2)];
            numReplicates = cfg.ellipse.replicatesPerMicropad;
            defaultPositions = createDefaultEllipseGrid(imageSize, cfg.numSquares, numReplicates, cfg);

            % Build ellipse editing UI without polygon overlays
            buildEllipseEditingUIGridMode(fig, img, imageName, phoneName, cfg, defaultPositions, initialRotation, memory);
            polygonParams = []; % No polygons in grid mode
        end

        % Wait for user action
        [action, ~, ~] = waitForUserAction(fig);

        if strcmp(action, 'stop')
            return;
        elseif strcmp(action, 'retry')
            % Re-process same image - recurse
            [success, fig, memory] = processOneImage(imageName, outputDirs, cfg, fig, phoneName, memory);
            return;
        elseif strcmp(action, 'accept')
            % Save ONLY ellipse outputs (skip polygon outputs)
            guiData = get(fig, 'UserData');

            % Extract ellipse data from UI (initialize to empty if field doesn't exist)
            ellipseData = [];
            if isfield(guiData, 'ellipseData') && ~isempty(guiData.ellipseData)
                ellipseData = guiData.ellipseData;

                % For Mode 3, we only save ellipse outputs
                % Do NOT save polygon crops or polygon coordinates
                saveEllipseData(img, imageName, polygonParams, ellipseData, outputDirs.ellipseDir, cfg);

                fprintf('  Saved %d elliptical regions\n', size(ellipseData, 1));
            end

            % Update memory for next image (Mode 3)
            if ~isempty(ellipseData)
                if ~isempty(polygonParams)
                    displaySize = computeDisplayImageSize([imageHeight, imageWidth], initialRotation, cfg);
                    displayPolygonsMode3 = convertBasePolygonsToDisplay(polygonParams, [imageHeight, imageWidth], displaySize, initialRotation, cfg);
                    memory = updateMemory(memory, displayPolygonsMode3, initialRotation, [imageHeight, imageWidth], displaySize, ellipseData, cfg);
                else
                    displaySize = computeDisplayImageSize([imageHeight, imageWidth], initialRotation, cfg);
                    memory = updateMemory(memory, [], initialRotation, [imageHeight, imageWidth], displaySize, ellipseData, cfg);
                end
            end

            success = true;
            return;
        end
    end

    % Modes 1 & 2: Get initial polygon positions and rotation (memory or default, NOT AI yet)
    [initialPolygons, initialRotation, ~] = getInitialPolygonsWithMemory(img, cfg, memory, [imageHeight, imageWidth]);

    % Memory polygons are exact display coordinates - use them directly
    [initialPolygons, orientation] = sortPolygonArrayByX(initialPolygons);

    % Use memory orientation if available (polygons may come from memory with known orientation)
    if memory.hasSettings && ~isempty(memory.orientation)
        orientation = memory.orientation;
    end

    % Display GUI immediately with memory/default polygons and rotation
    [polygonParams, displayPolygons, fig, rotation, ellipseData, orientation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig, initialRotation, memory, orientation);

    % NOTE: If AI detection is enabled, it will run asynchronously AFTER GUI is displayed

    if ~isempty(polygonParams)
        saveCroppedRegions(img, imageName, polygonParams, outputDirs.polygonDir, cfg, rotation);
        % Update memory with exact display polygon shapes and rotation
        displayImageSize = computeDisplayImageSize([imageHeight, imageWidth], rotation, cfg);
        memory = updateMemory(memory, displayPolygons, rotation, [imageHeight, imageWidth], displayImageSize, ellipseData, cfg, orientation);

        % Save ellipse data if ellipse editing was enabled
        if cfg.enableEllipseEditing && ~isempty(ellipseData)
            saveEllipseData(img, imageName, polygonParams, ellipseData, outputDirs.ellipseDir, cfg);
        end

        success = true;
    end
end

function polygons = calculateDefaultPolygons(imageWidth, imageHeight, cfg)
    % Generate default polygon positions using geometry parameters
    n = cfg.numSquares;

    % Build world coordinates
    aspect = cfg.geometry.aspectRatio;
    aspect = max(aspect, eps);
    totalGridWidth = 1.0;

    % Compute gap size and individual rectangle width
    gp = cfg.geometry.gapPercentWidth;
    denom = n + max(n-1, 0) * gp;
    if denom <= 0
        denom = max(n, 1);
    end
    w = totalGridWidth / denom;
    gapSizeWorld = gp * w;

    % Calculate height based on individual rectangle width (not total grid width)
    rectHeightWorld = w / aspect;

    % Build world corners
    worldCorners = zeros(n, 4, 2);
    xi = -totalGridWidth / 2;
    yi = -rectHeightWorld / 2;

    for i = 1:n
        worldCorners(i, :, :) = [
            xi,       yi;
            xi + w,   yi;
            xi + w,   yi + rectHeightWorld;
            xi,       yi + rectHeightWorld
        ];
        xi = xi + w + gapSizeWorld;
    end

    % Scale and center to image
    polygons = scaleAndCenterPolygons(worldCorners, imageWidth, imageHeight, cfg);
end

function ellipseParams = transformDefaultEllipsesToPolygon(polygonVertices, cfg, orientation, rotation)
    % Transform normalized ellipse records to pixel coordinates via homography
    %
    % Computes a homography from the unit square [0,1]×[0,1] to the actual
    % polygon vertices, then transforms each ellipse from ELLIPSE_DEFAULT_RECORDS
    % through this homography to get pixel-space ellipse parameters.
    %
    % Args:
    %   polygonVertices - 4×2 matrix of polygon corners [TL; TR; BR; BL] in pixels
    %   cfg             - Configuration struct with cfg.ellipse.defaultRecords
    %                     and cfg.homography
    %   orientation     - 'horizontal' or 'vertical' (strip layout on screen)
    %                     Determines how ELLIPSE_DEFAULT_RECORDS are transformed
    %                     before applying homography. Default: 'horizontal'
    %   rotation        - Rotation angle applied to image (degrees). Used to
    %                     re-align vertex order (Vertex 1 = Visual Top-Left).
    %
    % Returns:
    %   ellipseParams - N×5 matrix [x, y, semiMajor, semiMinor, rotation] in pixels

    if nargin < 3 || isempty(orientation)
        orientation = 'horizontal';
    end

    if nargin < 4
        rotation = 0;
    end

    % Adjust polygon vertices based on rotation to ensure Vertex 1 is always Top-Left
    % relative to the visual display.
    % Rotation is positive clockwise (e.g. 90 = 1x CW).
    % Default vertex order [TL, TR, BR, BL] cycles with rotation.
    if rotation ~= 0
        k = mod(round(rotation / 90), 4);
        if k > 0
            polygonVertices = circshift(polygonVertices, k, 1);
        end
    end

    defaultRecords = cfg.ellipse.defaultRecords;
    homography = cfg.homography;

    % Transform default records based on micropad strip orientation
    % ELLIPSE_DEFAULT_RECORDS is defined for horizontal strip layout (reading position)
    % When strip is vertical (rotated 90° CCW), transform positions accordingly
    if strcmp(orientation, 'vertical')
        % Paper strip rotated 90° CCW: physical left → screen bottom
        % Transform in unit square: (x, y) → (y, 1-x)
        % Ellipse rotation: +90° to account for paper rotation
        transformedRecords = zeros(size(defaultRecords));
        transformedRecords(:, 1) = defaultRecords(:, 2);           % new x = old y
        transformedRecords(:, 2) = 1 - defaultRecords(:, 1);       % new y = 1 - old x
        transformedRecords(:, 3:4) = defaultRecords(:, 3:4);       % axes unchanged
        transformedRecords(:, 5) = defaultRecords(:, 5) + 90;      % rotation +90°
        defaultRecords = transformedRecords;
    end

    % Unit square corners (source reference frame)
    % Order: TL, TR, BR, BL to match polygon vertex order
    unitSquare = [0, 0; 1, 0; 1, 1; 0, 1];

    % Compute homography from unit square to polygon
    tform = homography.compute_homography_from_points(unitSquare, polygonVertices);

    % Compute polygon scale (average side length for axis scaling)
    sides = zeros(4, 1);
    for i = 1:4
        j = mod(i, 4) + 1;
        sides(i) = norm(polygonVertices(i,:) - polygonVertices(j,:));
    end
    polygonScale = mean(sides);

    numEllipses = size(defaultRecords, 1);
    ellipseParams = zeros(numEllipses, 5);

    for i = 1:numEllipses
        % Create ellipse struct for transform_ellipse
        % Note: defaultRecords axes are fractions of unit square side (0-1)
        ellipseIn.center = defaultRecords(i, 1:2);
        ellipseIn.semiMajor = defaultRecords(i, 3);
        ellipseIn.semiMinor = defaultRecords(i, 4);
        ellipseIn.rotation = defaultRecords(i, 5);
        ellipseIn.valid = true;

        % Transform through homography
        ellipseOut = homography.transform_ellipse(ellipseIn, tform);

        if ellipseOut.valid
            % Scale axes by polygon size (normalized -> pixels)
            ellipseParams(i, :) = [
                ellipseOut.center(1), ellipseOut.center(2), ...
                ellipseOut.semiMajor * polygonScale, ...
                ellipseOut.semiMinor * polygonScale, ...
                ellipseOut.rotation
            ];
        else
            % Fallback: simple center transform without shape distortion
            centerPt = homography.transform_polygon(defaultRecords(i, 1:2), tform);
            ellipseParams(i, :) = [
                centerPt(1), centerPt(2), ...
                defaultRecords(i, 3) * polygonScale, ...
                defaultRecords(i, 4) * polygonScale, ...
                defaultRecords(i, 5)
            ];
        end

        % Enforce semiMajor >= semiMinor convention
        if ellipseParams(i, 3) < ellipseParams(i, 4)
            tmp = ellipseParams(i, 3);
            ellipseParams(i, 3) = ellipseParams(i, 4);
            ellipseParams(i, 4) = tmp;
            ellipseParams(i, 5) = ellipseParams(i, 5) + 90;
        end

        % Normalize rotation to [-180, 180]
        ellipseParams(i, 5) = homography.normalizeAngle(ellipseParams(i, 5));
    end
end

function bounds = computeEllipseAxisBounds(~, imageSize, cfg)
    % Compute min/max semi-axis lengths allowed for ellipse editing
    % No overlap safety - ellipses can be any size up to image extent
    imgHeight = imageSize(1);
    imgWidth = imageSize(2);
    baseExtent = max(imgWidth, imgHeight);
    minAxis = max(1, baseExtent * cfg.ellipse.minAxisPercent);
    maxAxis = baseExtent;
    bounds = struct('minAxis', minAxis, 'maxAxis', maxAxis);
end

function [semiMajor, semiMinor, rotationAngle] = enforceEllipseAxisLimits(semiMajor, semiMinor, rotationAngle, bounds)
    % Clamp ellipse axes to configured bounds and normalize rotation
    if semiMinor > semiMajor
        tmp = semiMajor;
        semiMajor = semiMinor;
        semiMinor = tmp;
        rotationAngle = rotationAngle + 90;
    end

    if nargin < 4 || isempty(bounds)
        rotationAngle = mod(rotationAngle + 180, 360) - 180;
        return;
    end

    minAxis = bounds.minAxis;
    maxAxis = bounds.maxAxis;

    semiMajor = min(max(semiMajor, minAxis), maxAxis);
    semiMinor = min(max(semiMinor, minAxis), semiMajor);
    rotationAngle = mod(rotationAngle + 180, 360) - 180;
end

function ellipseHandle = createEllipseROI(axHandle, center, semiMajor, semiMinor, rotationAngle, color, ~, bounds)
    % Helper to instantiate drawellipse overlays with consistent constraints
    if nargin < 8
        bounds = [];
    end

    [semiMajor, semiMinor, rotationAngle] = enforceEllipseAxisLimits(semiMajor, semiMinor, rotationAngle, bounds);

    ellipseHandle = drawellipse(axHandle, ...
        'Center', center, ...
        'SemiAxes', [semiMajor, semiMinor], ...
        'RotationAngle', rotationAngle, ...
        'Color', color, ...
        'LineWidth', 2, ...
        'FaceAlpha', 0.2, ...
        'InteractionsAllowed', 'all');
end

function polygons = scaleAndCenterPolygons(worldCorners, imageWidth, imageHeight, cfg)
    % Scale world coordinates to fit image with coverage factor
    n = size(worldCorners, 1);
    polygons = zeros(n, 4, 2);

    % Find bounding box of all world corners (width only needed for scaling)
    allX = worldCorners(:, :, 1);
    minX = min(allX(:));
    maxX = max(allX(:));

    worldW = maxX - minX;

    % Scale to fit image width with coverage factor
    targetWidth = imageWidth * cfg.coverage;
    scale = targetWidth / worldW;

    % Center in image
    centerX = imageWidth / 2;
    centerY = imageHeight / 2;

    for i = 1:n
        corners = squeeze(worldCorners(i, :, :));
        scaled = corners * scale;
        scaled(:, 1) = scaled(:, 1) + centerX;
        scaled(:, 2) = scaled(:, 2) + centerY;
        polygons(i, :, :) = scaled;
    end
end

%% -------------------------------------------------------------------------
%% Interactive UI
%% -------------------------------------------------------------------------

function [polygonParams, displayPolygons, fig, rotation, ellipseData, orientation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig, initialRotation, memory, orientation)
    % Show interactive GUI with editing, ellipse editing (optional), and preview modes
    polygonParams = [];
    displayPolygons = [];
    rotation = 0;
    ellipseData = [];

    % Create figure if needed
    if isempty(fig) || ~isvalid(fig)
        fig = createFigure(imageName, phoneName, cfg);
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
        % Polygon editing mode
        clearAndRebuildUI(fig, 'editing', img, imageName, phoneName, cfg, initialPolygons, initialRotation, memory, [], orientation);

        [action, userPolygons, userRotation] = waitForUserAction(fig);

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
                basePolygons = convertDisplayPolygonsToBase(guiDataEditing, userPolygons, cfg);
                savedRotation = userRotation;
                savedDisplayPolygons = userPolygons;
                savedBasePolygons = basePolygons;

                % Retrieve updated orientation from editing mode (may have changed due to rotation/AI)
                if isfield(guiDataEditing, 'orientation') && ~isempty(guiDataEditing.orientation)
                    orientation = guiDataEditing.orientation;
                end

                % Ellipse editing mode (if enabled)
                if cfg.enableEllipseEditing
                    viewState = captureViewState(guiDataEditing);
                    clearAndRebuildUI(fig, 'ellipse_editing', img, imageName, phoneName, cfg, savedBasePolygons, savedRotation, memory, viewState, orientation);

                    [ellipseAction, ellipseCoords] = waitForUserAction(fig);

                    % Defensive check
                    if ~isvalid(fig) || isempty(ellipseAction)
                        return;
                    end

                    switch ellipseAction
                        case 'back'
                            % Return to polygon editing
                            initialPolygons = savedDisplayPolygons;
                            initialRotation = savedRotation;
                            continue;
                        case 'skip'
                            return;
                        case 'stop'
                            if isvalid(fig)
                                delete(fig);
                            end
                            error('User stopped execution');
                        case 'accept'
                            % Store ellipse data and proceed to preview
                            savedEllipseData = ellipseCoords;
                    end
                else
                    % Skip ellipse editing, no ellipse data
                    savedEllipseData = [];
                end

                % Preview mode
                clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, savedBasePolygons, savedRotation, memory, [], orientation);

                % Store rotation and polygon params in guiData for preview mode
                guiData = get(fig, 'UserData');
                guiData.savedRotation = savedRotation;
                guiData.savedPolygonParams = savedBasePolygons;
                guiData.savedEllipseData = savedEllipseData;
                set(fig, 'UserData', guiData);

                [prevAction, ~, ~] = waitForUserAction(fig);

                % Defensive check: if figure was closed/deleted, exit cleanly
                if ~isvalid(fig) || isempty(prevAction)
                    return;
                end

                switch prevAction
                    case 'accept'
                        polygonParams = savedBasePolygons;
                        displayPolygons = savedDisplayPolygons;
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
                        % Use edited polygons as new initial positions
                        initialPolygons = savedDisplayPolygons;
                        initialRotation = savedRotation;
                        continue;
                end
        end
    end
end

function clearAndRebuildUI(fig, mode, img, imageName, phoneName, cfg, polygonParams, initialRotation, memory, viewState, orientation)
    % Modes: 'editing' (polygon adjustment), 'ellipse_editing' (ellipse placement), 'preview' (final confirmation)
    if nargin < 11 || isempty(orientation)
        orientation = 'horizontal';
    end

    if nargin < 10
        viewState = [];
    end

    if nargin < 8
        initialRotation = 0;
    end

    if nargin < 9
        memory = initializeMemory();
    end

    guiData = get(fig, 'UserData');
    clearAllUIElements(fig, guiData);

    switch mode
        case 'editing'
            buildEditingUI(fig, img, imageName, phoneName, cfg, polygonParams, initialRotation);

        case 'ellipse_editing'
            buildEllipseEditingUI(fig, img, imageName, phoneName, cfg, polygonParams, initialRotation, memory, orientation);

        case 'preview'
            buildPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams);
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

    % Add polygon ROIs from guiData
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'polygons')
        validPolys = collectValidPolygons(guiData);
        if ~isempty(validPolys)
            toDelete = [toDelete; validPolys];
        end
    end

    % Add polygon labels from guiData
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'polygonLabels')
        numLabels = numel(guiData.polygonLabels);
        validLabels = gobjects(numLabels, 1);
        validCount = 0;
        for i = 1:numLabels
            if isvalid(guiData.polygonLabels{i})
                validCount = validCount + 1;
                validLabels(validCount) = guiData.polygonLabels{i};
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
            safeStopTimer(guiData.aiTimer);
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
        end
    end

    if isfield(viewState, 'ylim')
        try
            ylim(guiData.imgAxes, viewState.ylim);
        catch
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

function polys = collectValidPolygons(guiData)
    polys = [];
    if isempty(guiData) || ~isstruct(guiData) || ~isfield(guiData, 'polygons')
        return;
    end
    if ~iscell(guiData.polygons)
        return;
    end

    validMask = cellfun(@isvalid, guiData.polygons);
    if any(validMask)
        % Clear appdata before collecting for deletion
        validPolys = guiData.polygons(validMask);
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
        polys = [guiData.polygons{validMask}]';
    end
end

function buildEditingUI(fig, img, imageName, phoneName, cfg, initialPolygons, initialRotation)
    % Build UI for polygon editing mode
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
    guiData.titleHandle = createTitle(fig, phoneName, imageName, cfg);
    guiData.pathHandle = createPathDisplay(fig, phoneName, imageName, cfg);

    % Image display (show image with initial rotation if any)
    if initialRotation ~= 0
        displayImg = applyRotation(img, initialRotation, cfg);
        guiData.currentImg = displayImg;
    else
        displayImg = img;
        guiData.currentImg = displayImg;
    end
    guiData.imageSize = [size(displayImg, 1), size(displayImg, 2)];
    [guiData.imgAxes, guiData.imgHandle] = createImageAxes(fig, displayImg, cfg);

    % Create editable polygons
    guiData.polygons = createPolygons(initialPolygons, cfg, fig);
    [guiData.polygons, ~, guiData.orientation] = assignPolygonLabels(guiData.polygons);

    numInitialPolygons = numel(guiData.polygons);
    totalForColor = max(numInitialPolygons, 1);
    guiData.aiBaseColors = zeros(numInitialPolygons, 3);
    for idx = 1:numInitialPolygons
        polyHandle = guiData.polygons{idx};
        if isvalid(polyHandle)
            baseColor = getConcentrationColor(idx - 1, totalForColor);
            setPolygonColor(polyHandle, baseColor, 0.25);
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
    guiData.polygonLabels = addPolygonLabels(guiData.polygons, guiData.imgAxes);

    % Rotation panel (preset buttons only)
    guiData.rotationPanel = createRotationButtonPanel(fig, cfg);

    % Run AI button sits above rotation controls for manual detection refresh
    guiData.runAIButton = createRunAIButton(fig, cfg);

    % Zoom panel
    [guiData.zoomSlider, guiData.zoomValue] = createZoomPanel(fig, cfg);

    % Buttons
    guiData.cutButtonPanel = createEditButtonPanel(fig, cfg);
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.instructionText = createInstructions(fig, cfg);
    guiData.aiStatusLabel = createAIStatusLabel(fig, cfg);

    guiData.action = '';

    % Store guiData before auto-zoom
    set(fig, 'UserData', guiData);

    % Auto-zoom to polygons after all UI is created
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
        start(t);

        % Store timer handle for cleanup
        guiData.aiTimer = t;
        set(fig, 'UserData', guiData);
    end
end

function buildPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams)
    % Build UI for preview mode
    set(fig, 'Name', sprintf('Preview - %s - %s', phoneName, imageName));

    % Check if ellipse data was saved (from previous UI state)
    tempGuiData = get(fig, 'UserData');
    ellipseData = [];
    if ~isempty(tempGuiData) && isstruct(tempGuiData) && isfield(tempGuiData, 'savedEllipseData')
        ellipseData = tempGuiData.savedEllipseData;
    end

    guiData = struct();
    guiData.mode = 'preview';
    guiData.savedPolygonParams = polygonParams;
    guiData.savedEllipseData = ellipseData;

    % Preview titles occupying the top band
    numRegions = size(polygonParams, 1);
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
    [guiData.leftAxes, guiData.rightAxes, guiData.leftImgHandle, guiData.rightImgHandle] = createPreviewAxes(fig, img, polygonParams, ellipseData, cfg);

    % Bottom controls
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.buttonPanel = createPreviewButtons(fig, cfg);

    guiData.action = '';
    set(fig, 'UserData', guiData);
end

function buildEllipseEditingUI(fig, img, imageName, phoneName, cfg, polygonParams, rotation, memory, orientation)
    % Build UI for ellipse editing mode
    set(fig, 'Name', sprintf('Ellipse Editing - %s - %s', phoneName, imageName));

    % Initialize orientation if not provided
    if nargin < 9 || isempty(orientation)
        orientation = 'horizontal';
    end

    guiData = struct();
    guiData.mode = 'ellipse_editing';
    guiData.cfg = cfg;
    guiData.polygons = polygonParams;
    guiData.rotation = rotation;
    guiData.memory = memory;
    guiData.orientation = orientation;  % Store for polygon ordering and ellipse positioning

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
    guiData.displayPolygons = convertBasePolygonsToDisplay(polygonParams, guiData.baseImageSize, guiData.imageSize, rotation, cfg);

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

    % Image display with polygons
    [guiData.imgAxes, guiData.imgHandle] = createImageAxes(fig, displayImg, cfg);

    % Draw all polygons (no highlighting, show all equally)
    numConcentrations = size(polygonParams, 1);
    guiData.polygonHandles = cell(numConcentrations, 1);
    for i = 1:numConcentrations
        vertices = squeeze(guiData.displayPolygons(i, :, :));
        polygonColor = getConcentrationColor(i - 1, numConcentrations);
        guiData.polygonHandles{i} = drawpolygon(guiData.imgAxes, 'Position', vertices, ...
                                                'Color', polygonColor, 'LineWidth', 2, ...
                                                'FaceAlpha', 0.15, 'InteractionsAllowed', 'none');
    end

    % Create ALL ellipses at once (21 total for 7 polygons × 3 replicates)
    numReplicates = cfg.ellipse.replicatesPerMicropad;
    totalEllipses = numConcentrations * numReplicates;
    guiData.ellipses = cell(1, totalEllipses);
    ellipseIdx = 1;

    % Check if memory has ellipse settings from previous image
    hasMemory = false;
    if isfield(guiData, 'memory') && ~isempty(guiData.memory) && ...
       isfield(guiData.memory, 'hasEllipseSettings') && isequal(guiData.memory.hasEllipseSettings, true)
        hasMemory = true;

        % Check if polygon geometry changed (need to scale)
        if isfield(guiData.memory, 'polygons') && ~isempty(guiData.memory.polygons)
            oldPolygons = guiData.memory.polygons;
            newPolygons = guiData.displayPolygons;

            % Scale ellipses for each concentration
            for concIdx = 1:numConcentrations
                if concIdx <= numel(guiData.memory.ellipses) && ~isempty(guiData.memory.ellipses{concIdx})
                    oldPoly = squeeze(oldPolygons(concIdx, :, :));
                    newPoly = squeeze(newPolygons(concIdx, :, :));
                    oldEllipses = guiData.memory.ellipses{concIdx};

                    % Scale ellipses to new polygon geometry
                    scaledEllipses = scaleEllipsesForPolygonChange(oldPoly, newPoly, oldEllipses, cfg, guiData.imageSize);
                    guiData.memory.ellipses{concIdx} = scaledEllipses;
                end
            end
        end
    end

    displayPolygons = guiData.displayPolygons;
    for concIdx = 1:numConcentrations
        polygonColor = getConcentrationColor(concIdx - 1, numConcentrations);
        currentPolygon = squeeze(displayPolygons(concIdx, :, :));

        % Compute axis bounds for this polygon
        ellipseBounds = computeEllipseAxisBounds(currentPolygon, guiData.imageSize, cfg);

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
                    ellipseParams = transformDefaultEllipsesToPolygon(currentPolygon, cfg, orientation, rotation);
                    ellipseCenter = ellipseParams(repIdx, 1:2);
                    ellipseSemiMajor = ellipseParams(repIdx, 3);
                    ellipseSemiMinor = ellipseParams(repIdx, 4);
                    ellipseRotation = ellipseParams(repIdx, 5);
                end

                guiData.ellipses{ellipseIdx} = createEllipseROI(guiData.imgAxes, ellipseCenter, ...
                    ellipseSemiMajor, ellipseSemiMinor, ellipseRotation, polygonColor, cfg, ellipseBounds);
                ellipseIdx = ellipseIdx + 1;
            end
        else
            % No memory - use homography-transformed defaults from ELLIPSE_DEFAULT_RECORDS
            ellipseParams = transformDefaultEllipsesToPolygon(currentPolygon, cfg, orientation, rotation);

            for repIdx = 1:numReplicates
                guiData.ellipses{ellipseIdx} = createEllipseROI(guiData.imgAxes, ellipseParams(repIdx, 1:2), ...
                    ellipseParams(repIdx, 3), ellipseParams(repIdx, 4), ellipseParams(repIdx, 5), polygonColor, cfg, ellipseBounds);
                ellipseIdx = ellipseIdx + 1;
            end
        end
    end

    % Zoom panel (reuse existing)
    guiData.zoomLevel = 0;
    guiData.autoZoomBounds = [];
    [guiData.zoomSlider, guiData.zoomValue] = createZoomPanel(fig, cfg);

    % Action buttons panel
    guiData.ellipseButtonPanel = createEllipseEditingButtonPanel(fig, cfg);
    guiData.stopButton = createStopButton(fig, cfg);

    % Instructions
    instructionText = 'Draw 21 ellipses (3 per micropad). Colors match polygons. Click DONE when finished.';
    guiData.instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', instructionText, ...
                                       'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
                                       'FontSize', cfg.ui.fontSize.instruction, ...
                                       'ForegroundColor', cfg.ui.colors.foreground, ...
                                       'BackgroundColor', cfg.ui.colors.background, ...
                                       'HorizontalAlignment', 'center');

    guiData.action = '';
    set(fig, 'UserData', guiData);
end

function buildEllipseEditingUIGridMode(fig, img, imageName, phoneName, cfg, ellipsePositions, rotation, memory)
    % Build ellipse editing UI for grid mode (no polygon overlays)
    % This is used in Mode 3 when no polygon coordinates exist

    set(fig, 'Name', sprintf('Ellipse Editing (Grid Mode) - %s - %s', phoneName, imageName));

    guiData = struct();
    guiData.mode = 'ellipse_editing_grid';
    guiData.cfg = cfg;
    guiData.polygons = []; % No polygons in grid mode
    guiData.displayPolygons = [];
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

    % Image display (NO polygon overlays in grid mode)
    [guiData.imgAxes, guiData.imgHandle] = createImageAxes(fig, displayImg, cfg);

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
    gridBounds = computeEllipseAxisBounds([], guiData.imageSize, cfg);
    for groupIdx = 1:numGroups
        % Use cold-to-hot color gradient
        ellipseColor = getConcentrationColor(groupIdx - 1, numGroups);

        for repIdx = 1:numReplicates
            centerPos = ellipsePositions(ellipseIdx, :);

            guiData.ellipses{ellipseIdx} = createEllipseROI(guiData.imgAxes, centerPos, ...
                defaultSemiMajor, defaultSemiMinor, defaultRotations(repIdx), ellipseColor, cfg, gridBounds);
            ellipseIdx = ellipseIdx + 1;
        end
    end

    % Zoom panel
    guiData.zoomLevel = 0;
    guiData.autoZoomBounds = [];
    [guiData.zoomSlider, guiData.zoomValue] = createZoomPanel(fig, cfg);

    % Action buttons panel
    guiData.ellipseButtonPanel = createEllipseEditingButtonPanel(fig, cfg);
    guiData.stopButton = createStopButton(fig, cfg);

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
end

function buildReadOnlyPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams, hasPolygons, ...
                               ellipseData, hasEllipses, rotation)
    % Build read-only preview UI for Mode 4 (both editing disabled)
    % Displays existing coordinate overlays without editing capability

    set(fig, 'Name', sprintf('Preview Mode - %s - %s', phoneName, imageName));

    guiData = struct();
    guiData.mode = 'preview_readonly';
    guiData.cfg = cfg;
    guiData.polygons = polygonParams;
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
    [guiData.imgAxes, guiData.imgHandle] = createImageAxes(fig, displayImg, cfg);

    displayPolygonsPreview = convertBasePolygonsToDisplay(polygonParams, guiData.baseImageSize, guiData.imageSize, rotation, cfg);
    displayEllipsesPreview = convertBaseEllipsesToDisplay(ellipseData, guiData.baseImageSize, guiData.imageSize, rotation, cfg);

    % Draw polygon overlays if available
    if hasPolygons && ~isempty(displayPolygonsPreview)
        numPolygons = size(displayPolygonsPreview, 1);
        guiData.polygonHandles = cell(numPolygons, 1);
        for i = 1:numPolygons
            vertices = squeeze(displayPolygonsPreview(i, :, :));
            polygonColor = getConcentrationColor(i - 1, numPolygons);
            guiData.polygonHandles{i} = drawpolygon(guiData.imgAxes, 'Position', vertices, ...
                                                    'Color', polygonColor, 'LineWidth', 2, ...
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

            % Use same color as corresponding polygon
            if hasPolygons
                numConcentrations = size(displayPolygonsPreview, 1);
            else
                numConcentrations = max(displayEllipsesPreview(:, 1)) + 1;
            end
            ellipseColor = getConcentrationColor(concIdx - 1, numConcentrations);

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
    [guiData.zoomSlider, guiData.zoomValue] = createZoomPanel(fig, cfg);

    % NEXT button (replaces ACCEPT in read-only mode)
    uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'NEXT', ...
              'Units', 'normalized', 'Position', [0.80 0.02 0.15 0.06], ...
              'FontSize', cfg.ui.fontSize.button, ...
              'Callback', @(~,~) set(fig, 'UserData', setfield(get(fig, 'UserData'), 'action', 'accept')));

    % STOP button
    guiData.stopButton = createStopButton(fig, cfg);

    % Instructions
    overlayInfo = '';
    if hasPolygons && hasEllipses
        overlayInfo = sprintf('%d polygons and %d ellipses', size(polygonParams, 1), size(ellipseData, 1));
    elseif hasPolygons
        overlayInfo = sprintf('%d polygons', size(polygonParams, 1));
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
end

%% -------------------------------------------------------------------------
%% UI Components
%% -------------------------------------------------------------------------

function fig = createFigure(imageName, phoneName, cfg)
    titleText = sprintf('microPAD Processor - %s - %s', phoneName, imageName);
    fig = figure('Name', titleText, ...
                'Units', 'normalized', 'Position', cfg.ui.positions.figure, ...
                'MenuBar', 'none', 'ToolBar', 'none', ...
                'Color', cfg.ui.colors.background, 'KeyPressFcn', @keyPressHandler, ...
                'CloseRequestFcn', @(src, ~) cleanupAndClose(src));

    drawnow limitrate;
    pause(0.05);
    set(fig, 'WindowState', 'maximized');
    figure(fig);
    drawnow limitrate;
end

function titleHandle = createTitle(fig, phoneName, imageName, cfg)
    titleText = sprintf('%s - %s', phoneName, imageName);
    titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                           'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                           'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                           'ForegroundColor', cfg.ui.colors.foreground, ...
                           'BackgroundColor', cfg.ui.colors.background, ...
                           'HorizontalAlignment', 'center');
end

function pathHandle = createPathDisplay(fig, phoneName, imageName, cfg)
    pathText = sprintf('Path: %s | Image: %s', phoneName, imageName);
    pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                          'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                          'ForegroundColor', cfg.ui.colors.path, ...
                          'BackgroundColor', cfg.ui.colors.background, ...
                          'HorizontalAlignment', 'center');
end

function [imgAxes, imgHandle] = createImageAxes(fig, img, cfg)
    imgAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.image);
    imgHandle = imshow(img, 'Parent', imgAxes, 'InitialMagnification', 'fit');
    axis(imgAxes, 'image');
    axis(imgAxes, 'tight');
    hold(imgAxes, 'on');
end

function stopButton = createStopButton(fig, cfg)
    stopButton = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
                          'String', 'STOP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.stopButton, ...
                          'BackgroundColor', cfg.ui.colors.stop, 'ForegroundColor', cfg.ui.colors.foreground, ...
                          'Callback', @(~,~) stopExecution(fig));
end

function runAIButton = createRunAIButton(fig, cfg)
    runAIButton = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
                           'String', 'RUN AI', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
                           'Units', 'normalized', 'Position', cfg.ui.positions.runAIButton, ...
                           'BackgroundColor', [0.30 0.50 0.70], ...
                           'ForegroundColor', cfg.ui.colors.foreground, ...
                           'TooltipString', 'Run YOLO detection on the current view', ...
                           'Callback', @(~,~) rerunAIDetection(fig, cfg));
end

function polygons = createPolygons(initialPolygons, cfg, ~)
    % Create drawpolygon objects from initial positions with color gradient
    n = size(initialPolygons, 1);
    polygons = cell(1, n);

    for i = 1:n
        pos = squeeze(initialPolygons(i, :, :));

        % Apply color gradient based on concentration index (zero-based)
        concentrationIndex = i - 1;
        polyColor = getConcentrationColor(concentrationIndex, cfg.numSquares);

        polygons{i} = drawpolygon('Position', pos, ...
                                 'Color', polyColor, ...
                                 'LineWidth', cfg.ui.polygon.lineWidth, ...
                                 'MarkerSize', 8, ...
                                 'Selected', false);

        % Ensure consistent face styling even on releases that lack name-value support
        setPolygonColor(polygons{i}, polyColor, 0.25);

        % Store initial valid position
        setappdata(polygons{i}, 'LastValidPosition', pos);

        % Add listener for quadrilateral enforcement
        listenerHandle = addlistener(polygons{i}, 'ROIMoved', @(~,~) enforceQuadrilateral(polygons{i}));
        setappdata(polygons{i}, 'ListenerHandle', listenerHandle);

        % Add listener for label updates when user drags vertices
        labelUpdateListener = addlistener(polygons{i}, 'ROIMoved', @(src, evt) updatePolygonLabelsCallback(src, evt));
        setappdata(polygons{i}, 'LabelUpdateListener', labelUpdateListener);
    end
end

function labelHandles = addPolygonLabels(polygons, axesHandle)
    % Add text labels showing concentration number on each polygon
    % Labels positioned at TOP edge of polygon (per user requirement)
    %
    % Inputs:
    %   polygons - cell array of drawpolygon objects
    %   axesHandle - axes where labels should be drawn
    %
    % Output:
    %   labelHandles - cell array of text object handles

    n = numel(polygons);
    labelHandles = cell(1, n);

    for i = 1:n
        poly = polygons{i};
        if ~isvalid(poly)
            continue;
        end

        % Get polygon position
        pos = poly.Position;
        if isempty(pos) || size(pos, 1) < 3
            continue;
        end

        % CHANGED: Position at TOP of polygon, not center
        % Use image-relative units (polygon height fraction) instead of fixed pixels
        % for zoom/rotation consistency
        centerX = mean(pos(:, 1));
        minY = min(pos(:, 2));  % Top edge (smallest Y value)
        polyHeight = max(pos(:, 2)) - minY;
        labelY = minY - max(15, polyHeight * 0.1);  % 10% of polygon height or 15px minimum

        % Create label text
        concentrationIndex = i - 1;  % Zero-based
        labelText = sprintf('con_%d', concentrationIndex);

        % Create text object with dark background for visibility
        % NOTE: BackgroundColor only supports 3-element RGB (no alpha channel)
        labelHandles{i} = text(axesHandle, centerX, labelY, labelText, ...
                              'HorizontalAlignment', 'center', ...
                              'VerticalAlignment', 'bottom', ...  % CHANGED: anchor bottom to position
                              'FontSize', 12, ...
                              'FontWeight', 'bold', ...
                              'Color', [1 1 1], ...  % White text
                              'BackgroundColor', [0.2 0.2 0.2], ...  % Dark gray (opaque)
                              'EdgeColor', 'none', ...
                              'Margin', 2);
    end
end

function enforceQuadrilateral(polygon)
    % Ensure polygon remains a quadrilateral by reverting invalid changes
    if ~isvalid(polygon)
        return;
    end

    pos = polygon.Position;
    if size(pos, 1) ~= 4
        % Revert to last valid state
        lastValid = getappdata(polygon, 'LastValidPosition');
        if ~isempty(lastValid)
            polygon.Position = lastValid;
        end
        warning('cut_micropads:invalid_polygon', 'Polygon must have exactly 4 vertices. Reverting change.');
    else
        % Store valid state
        setappdata(polygon, 'LastValidPosition', pos);
    end
end

function color = getConcentrationColor(concentrationIndex, totalConcentrations)
    % Generate spectrum gradient: blue (cold) → red (hot)
    % Uses HSV color space for maximum visual distinction
    %
    % Inputs:
    %   concentrationIndex - zero-based index (0 to totalConcentrations-1)
    %   totalConcentrations - total number of concentration regions
    %
    % Output:
    %   color - [R G B] triplet in range [0, 1]

    if totalConcentrations <= 1
        color = [0.0 0.5 1.0];  % Default blue for single region
        return;
    end

    % Normalize index to [0, 1]
    t = concentrationIndex / (totalConcentrations - 1);

    % Interpolate hue from 240° (blue) to 0° (red) through spectrum
    hue = (1 - t) * 240 / 360;  % 240° = blue, 0° = red
    sat = 1.0;  % Full saturation
    val = 1.0;  % Full value/brightness

    % Convert HSV to RGB
    color = hsv2rgb([hue, sat, val]);
end

function setPolygonColor(polygonHandle, colorValue, faceAlpha)
    % Apply edge/face color updates with compatibility guards
    if nargin < 3
        faceAlpha = [];
    end

    if isempty(polygonHandle) || ~isvalid(polygonHandle)
        return;
    end

    if ~isempty(colorValue) && all(isfinite(colorValue))
        set(polygonHandle, 'Color', colorValue);
        if isprop(polygonHandle, 'FaceColor')
            set(polygonHandle, 'FaceColor', colorValue);
        end
    end

    if ~isempty(faceAlpha) && isprop(polygonHandle, 'FaceAlpha')
        set(polygonHandle, 'FaceAlpha', faceAlpha);
    end
end

function cutButtonPanel = createEditButtonPanel(fig, cfg)
    cutButtonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                             'Position', cfg.ui.positions.cutButtonPanel, ...
                             'BackgroundColor', cfg.ui.colors.panel, ...
                             'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground);

    % APPLY button
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'APPLY', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.15 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.apply, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setAction(fig, 'accept'));

    % SKIP button
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'SKIP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.55 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.skip, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setAction(fig, 'skip'));
end

function instructionText = createInstructions(fig, cfg)
    instructionString = 'Mouse = Drag Vertices | Buttons = Rotate | RUN AI = Detect Polygons | Slider = Zoom | APPLY = Save & Continue | SKIP = Skip | STOP = Exit | Space = APPLY | Esc = SKIP';

    instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', instructionString, ...
             'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
             'FontSize', cfg.ui.fontSize.instruction, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.background, 'HorizontalAlignment', 'center');
end

function statusLabel = createAIStatusLabel(fig, cfg)
    if nargin < 2 || isempty(cfg) || ~isfield(cfg, 'ui')
        position = [0.25 0.905 0.50 0.035];
        fontSize = 13;
        infoColor = [1 1 0.3];
        backgroundColor = 'black';
    else
        position = cfg.ui.positions.aiStatus;
        fontSize = cfg.ui.fontSize.status;
        infoColor = cfg.ui.colors.info;
        backgroundColor = cfg.ui.colors.background;
    end

    statusLabel = uicontrol('Parent', fig, 'Style', 'text', ...
                           'String', 'AI DETECTION RUNNING', ...
                           'Units', 'normalized', ...
                           'Position', position, ...
                           'FontSize', fontSize, ...
                           'FontWeight', 'bold', ...
                           'ForegroundColor', infoColor, ...
                           'BackgroundColor', backgroundColor, ...
                           'HorizontalAlignment', 'center', ...
                           'Visible', 'off');
end

function buttonPanel = createPreviewButtons(fig, cfg)
    buttonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                         'Position', cfg.ui.positions.previewPanel, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground, ...
                         'BorderWidth', cfg.ui.polygon.borderWidth);

    buttons = {'ACCEPT', 'RETRY', 'SKIP'};
    positions = {[0.05 0.25 0.25 0.50], [0.375 0.25 0.25 0.50], [0.70 0.25 0.25 0.50]};
    colors = {cfg.ui.colors.accept, cfg.ui.colors.retry, cfg.ui.colors.skip};
    actions = {'accept', 'retry', 'skip'};

    for i = 1:numel(buttons)
        uicontrol('Parent', buttonPanel, 'Style', 'pushbutton', ...
                 'String', buttons{i}, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold', ...
                 'Units', 'normalized', 'Position', positions{i}, ...
                 'BackgroundColor', colors{i}, 'ForegroundColor', cfg.ui.colors.foreground, ...
                 'Callback', @(~,~) setAction(fig, actions{i}));
    end
end

function buttonPanel = createEllipseEditingButtonPanel(fig, cfg)
    % Create button panel for ellipse editing mode (DONE, BACK)
    buttonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                         'Position', cfg.ui.positions.cutButtonPanel, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground);

    % DONE button (green, accepts ellipse data)
    uicontrol('Parent', buttonPanel, 'Style', 'pushbutton', ...
             'String', 'DONE', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.15 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.accept, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setAction(fig, 'accept'));

    % BACK button (red, returns to polygon editing)
    uicontrol('Parent', buttonPanel, 'Style', 'pushbutton', ...
             'String', 'BACK', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.55 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.retry, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setAction(fig, 'back'));
end




function showAIProgressIndicator(fig, show)
    % Toggle AI detection status indicator and polygon breathing animation
    if ~ishandle(fig) || ~strcmp(get(fig, 'Type'), 'figure')
        return;
    end

    guiData = get(fig, 'UserData');
    if isempty(guiData) || ~isstruct(guiData) || ~strcmp(guiData.mode, 'editing')
        return;
    end

    if show
        % Ensure status label exists and is visible
        if ~isfield(guiData, 'aiStatusLabel') || ~isvalid(guiData.aiStatusLabel)
            cfgForLabel = [];
            if isfield(guiData, 'cfg')
                cfgForLabel = guiData.cfg;
            end
            guiData.aiStatusLabel = createAIStatusLabel(fig, cfgForLabel);
        end
        set(guiData.aiStatusLabel, 'String', 'AI DETECTION RUNNING', 'Visible', 'on');
        uistack(guiData.aiStatusLabel, 'top');

        % Capture current polygon colors as animation baseline
        guiData.aiBaseColors = capturePolygonColors(guiData.polygons);

        % Start breathing animation timer if polygons exist
        guiData = stopAIBreathingTimer(guiData);
        if ~isempty(guiData.aiBaseColors)
            guiData.aiBreathingStart = tic;
            guiData.aiBreathingFrequency = 0.8;            % Hz (slow breathing cadence)
            guiData.aiBreathingMixRange = [0.12, 0.36];    % Blend-to-white range (min..max)
            guiData.aiBreathingDimFactor = 0.22;           % Max dim amount during exhale (22%)
            guiData.aiBreathingTimer = timer(...
                'Name', 'microPAD-AI-breathing', ...
                'Period', 1/45, ...                        % ~22 ms (≈45 FPS)
                'ExecutionMode', 'fixedRate', ...
                'BusyMode', 'queue', ...
                'TasksToExecute', Inf, ...
                'TimerFcn', @(~,~) animatePolygonBreathing(fig));
            start(guiData.aiBreathingTimer);
        end

        drawnow limitrate;
    else
        % Hide status label if present
        if isfield(guiData, 'aiStatusLabel') && isvalid(guiData.aiStatusLabel)
            set(guiData.aiStatusLabel, 'Visible', 'off');
        end

        % Stop animation timer and restore base colors
        guiData = stopAIBreathingTimer(guiData);
        if isfield(guiData, 'polygons') && iscell(guiData.polygons) && ~isempty(guiData.aiBaseColors)
            numRestore = min(size(guiData.aiBaseColors, 1), numel(guiData.polygons));
            for idx = 1:numRestore
                poly = guiData.polygons{idx};
                baseColor = guiData.aiBaseColors(idx, :);
                if isvalid(poly) && all(isfinite(baseColor))
                    setPolygonColor(poly, baseColor, 0.25);
                end
            end
            drawnow limitrate;
        end

        % Refresh baseline colors to reflect final state
        guiData.aiBaseColors = capturePolygonColors(guiData.polygons);
    end

    set(fig, 'UserData', guiData);
end

function guiData = stopAIBreathingTimer(guiData)
    if isfield(guiData, 'aiBreathingTimer')
        safeStopTimer(guiData.aiBreathingTimer);
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

function baseColors = capturePolygonColors(polygons)
    baseColors = [];
    if isempty(polygons) || ~iscell(polygons)
        return;
    end

    numPolygons = numel(polygons);
    baseColors = nan(numPolygons, 3);

    for idx = 1:numPolygons
        if isvalid(polygons{idx})
            color = get(polygons{idx}, 'Color');
            if numel(color) == 3
                baseColors(idx, :) = color;
            end
        end
    end
end

function animatePolygonBreathing(fig)
    if ~ishandle(fig) || ~strcmp(get(fig, 'Type'), 'figure')
        return;
    end

    guiData = get(fig, 'UserData');
    if isempty(guiData) || ~isstruct(guiData)
        return;
    end
    if ~isfield(guiData, 'polygons') || ~iscell(guiData.polygons)
        return;
    end
    if ~isfield(guiData, 'aiBaseColors') || isempty(guiData.aiBaseColors)
        return;
    end
    if ~isfield(guiData, 'aiBreathingStart') || isempty(guiData.aiBreathingStart)
        return;
    end
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

    numPolygons = min(size(guiData.aiBaseColors, 1), numel(guiData.polygons));
    for idx = 1:numPolygons
        poly = guiData.polygons{idx};
        baseColor = guiData.aiBaseColors(idx, :);
        if isvalid(poly) && all(isfinite(baseColor))
            whitened = baseColor * (1 - brightenMix) + brightenMix;
            newColor = min(max(whitened * dimScale, 0), 1);
            setPolygonColor(poly, newColor, []);
        end
    end

    drawnow limitrate;
end

function guiData = applyDetectedPolygons(guiData, newPolygons, cfg, fig)
    % Synchronize drawpolygon handles with detection output preserving UI ordering

    if isempty(newPolygons)
        return;
    end

    % Ensure polygons are ordered left-to-right, bottom-to-top in UI space
    [newPolygons, newOrientation] = sortPolygonArrayByX(newPolygons);
    guiData.orientation = newOrientation;  % Update orientation based on new polygon layout
    targetCount = size(newPolygons, 1);

    if targetCount == 0
        return;
    end

    % Determine whether we can reuse existing polygon handles
    hasPolygons = isfield(guiData, 'polygons') && iscell(guiData.polygons) && ~isempty(guiData.polygons);
    validMask = hasPolygons;
    if hasPolygons
        validMask = cellfun(@isvalid, guiData.polygons);
    end
    reusePolygons = hasPolygons && all(validMask) && numel(guiData.polygons) == targetCount;

    if reusePolygons
        updatePolygonPositions(guiData.polygons, newPolygons);
    else
        % Clean up existing polygons if present
        if hasPolygons
            for idx = 1:numel(guiData.polygons)
                if isvalid(guiData.polygons{idx})
                    delete(guiData.polygons{idx});
                end
            end
        end

        guiData.polygons = createPolygons(newPolygons, cfg, fig);
    end

    % Reorder polygons to enforce gradient ordering
    [guiData.polygons, order, newOrientation] = assignPolygonLabels(guiData.polygons);
    guiData.orientation = newOrientation;  % Update orientation based on current layout

    % Synchronize labels
    hasLabels = isfield(guiData, 'polygonLabels') && iscell(guiData.polygonLabels);
    reuseLabels = false;
    if hasLabels
        labelValidMask = cellfun(@isvalid, guiData.polygonLabels);
        reuseLabels = all(labelValidMask) && numel(guiData.polygonLabels) == targetCount;
    end

    if ~reuseLabels
        if hasLabels
            for idx = 1:numel(guiData.polygonLabels)
                if isvalid(guiData.polygonLabels{idx})
                    delete(guiData.polygonLabels{idx});
                end
            end
        end
        guiData.polygonLabels = addPolygonLabels(guiData.polygons, guiData.imgAxes);
    elseif ~isempty(order)
        guiData.polygonLabels = guiData.polygonLabels(order);
    end

    % Apply consistent cold-to-hot gradient and refresh label strings
    numPolygons = numel(guiData.polygons);
    totalForColor = max(numPolygons, 1);
    for idx = 1:numPolygons
        polyHandle = guiData.polygons{idx};
        if isvalid(polyHandle)
            gradColor = getConcentrationColor(idx - 1, totalForColor);
            setPolygonColor(polyHandle, gradColor, 0.25);

        end
    end

    if ~isempty(guiData.polygonLabels)
        for idx = 1:min(numPolygons, numel(guiData.polygonLabels))
            labelHandle = guiData.polygonLabels{idx};
            if isvalid(labelHandle)
                set(labelHandle, 'String', sprintf('con_%d', idx - 1));
            end
        end
        updatePolygonLabels(guiData.polygons, guiData.polygonLabels);
    end

    guiData.aiBaseColors = capturePolygonColors(guiData.polygons);
end

function [leftAxes, rightAxes, leftImgHandle, rightImgHandle] = createPreviewAxes(fig, img, polygonParams, ellipseData, cfg)
    % Left: original with overlays
    leftAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewLeft);
    leftImgHandle = imshow(img, 'Parent', leftAxes, 'InitialMagnification', 'fit');
    axis(leftAxes, 'image');
    axis(leftAxes, 'tight');
    title(leftAxes, 'Original Image', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold');
    hold(leftAxes, 'on');

    % Draw polygon overlays
    for i = 1:size(polygonParams, 1)
        poly = squeeze(polygonParams(i,:,:));
        if size(poly, 1) >= 3
            concentrationIndex = i - 1;
            polyColor = getConcentrationColor(concentrationIndex, cfg.numSquares);

            plot(leftAxes, [poly(:,1); poly(1,1)], [poly(:,2); poly(1,2)], ...
                 'Color', polyColor, 'LineWidth', cfg.ui.polygon.lineWidth);

            centerX = mean(poly(:,1));
            centerY = mean(poly(:,2));
            text(leftAxes, centerX, centerY, sprintf('con_%d', i-1), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'FontSize', cfg.ui.fontSize.info, 'FontWeight', 'bold', ...
                 'Color', cfg.ui.colors.info, 'BackgroundColor', [0 0 0], ...
                 'EdgeColor', 'none');
        end
    end

    % Draw ellipse overlays if ellipse editing was enabled
    if ~isempty(ellipseData) && cfg.enableEllipseEditing
        for i = 1:size(ellipseData, 1)
            if ellipseData(i, 3) > 0
                concIdx = ellipseData(i, 1);
                x = ellipseData(i, 3);
                y = ellipseData(i, 4);
                a = ellipseData(i, 5);
                b = ellipseData(i, 6);
                theta = ellipseData(i, 7);

                ellipseColor = getConcentrationColor(concIdx, cfg.numSquares);

                % Draw ellipse using parametric form
                t = linspace(0, 2*pi, 100);
                theta_rad = deg2rad(theta);
                x_ellipse = a * cos(t);
                y_ellipse = b * sin(t);
                x_rot = x + x_ellipse * cos(theta_rad) - y_ellipse * sin(theta_rad);
                y_rot = y + x_ellipse * sin(theta_rad) + y_ellipse * cos(theta_rad);

                plot(leftAxes, x_rot, y_rot, 'Color', ellipseColor, 'LineWidth', 1.5, 'LineStyle', '--');
            end
        end
    end

    hold(leftAxes, 'off');

    % Right: highlighted regions
    rightAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewRight);
    maskedImg = createMaskedPreview(img, polygonParams, ellipseData, cfg);
    rightImgHandle = imshow(maskedImg, 'Parent', rightAxes, 'InitialMagnification', 'fit');
    axis(rightAxes, 'image');
    axis(rightAxes, 'tight');
    title(rightAxes, 'Masked Preview', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold');
end

function maskedImg = createMaskedPreview(img, polygonParams, ellipseData, cfg)
    [height, width, ~] = size(img);
    totalMask = false(height, width);

    % Add polygon masks
    numRegions = size(polygonParams, 1);
    for i = 1:numRegions
        poly = squeeze(polygonParams(i,:,:));
        if size(poly, 1) >= 3
            regionMask = poly2mask(poly(:,1), poly(:,2), height, width);
            totalMask = totalMask | regionMask;
        end
    end

    % Add ellipse masks if ellipse editing was enabled
    if ~isempty(ellipseData) && cfg.enableEllipseEditing
        for i = 1:size(ellipseData, 1)
            if ellipseData(i, 3) > 0
                x = ellipseData(i, 3);
                y = ellipseData(i, 4);
                a = ellipseData(i, 5);
                b = ellipseData(i, 6);
                theta = ellipseData(i, 7);

                % Create ellipse mask
                [X, Y] = meshgrid(1:width, 1:height);
                theta_rad = deg2rad(theta);
                dx = X - x;
                dy = Y - y;
                x_rot =  dx * cos(theta_rad) + dy * sin(theta_rad);
                y_rot = -dx * sin(theta_rad) + dy * cos(theta_rad);

                ellipseMask = (x_rot ./ a).^2 + (y_rot ./ b).^2 <= 1;
                totalMask = totalMask | ellipseMask;
            end
        end
    end

    dimFactor = cfg.dimFactor;
    maskedImg = double(img);
    dimMultiplier = double(totalMask) + (1 - double(totalMask)) * dimFactor;
    maskedImg = maskedImg .* dimMultiplier;
    maskedImg = uint8(maskedImg);
end

%% -------------------------------------------------------------------------
%% Rotation and Zoom Panel Controls
%% -------------------------------------------------------------------------

function rotationPanel = createRotationButtonPanel(fig, cfg)
    % Create rotation panel with preset angle buttons only
    rotationPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                           'Position', cfg.ui.positions.rotationPanel, ...
                           'BackgroundColor', cfg.ui.colors.panel, ...
                           'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground, ...
                           'BorderWidth', 2);

    % Panel label
    uicontrol('Parent', rotationPanel, 'Style', 'text', 'String', 'Rotation', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.rotationLabel, ...
             'FontSize', cfg.ui.fontSize.label, 'FontWeight', 'bold', ...
             'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.panel, 'HorizontalAlignment', 'center');

    % Rotation preset buttons
    angles = cfg.ui.rotation.quickAngles;
    positions = {cfg.ui.layout.quickRotationRow1{1}, cfg.ui.layout.quickRotationRow1{2}, ...
                 cfg.ui.layout.quickRotationRow2{1}, cfg.ui.layout.quickRotationRow2{2}};

    for i = 1:numel(angles)
        uicontrol('Parent', rotationPanel, 'Style', 'pushbutton', ...
                 'String', sprintf('%d%s', angles(i), char(176)), ...
                 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
                 'Units', 'normalized', 'Position', positions{i}, ...
                 'BackgroundColor', [0.25 0.25 0.25], ...
                 'ForegroundColor', cfg.ui.colors.foreground, ...
                 'Callback', @(~,~) applyRotation_UI(angles(i), fig, cfg));
    end

end

function [zoomSlider, zoomValue] = createZoomPanel(fig, cfg)
    % Create zoom panel with slider and control buttons
    zoomPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                       'Position', cfg.ui.positions.zoomPanel, ...
                       'BackgroundColor', cfg.ui.colors.panel, ...
                       'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground, ...
                       'BorderWidth', 2);

    % Panel label
    uicontrol('Parent', zoomPanel, 'Style', 'text', 'String', 'Zoom', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.zoomLabel, ...
             'FontSize', cfg.ui.fontSize.label, 'FontWeight', 'bold', ...
             'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.panel, 'HorizontalAlignment', 'center');

    % Zoom slider
    zoomSlider = uicontrol('Parent', zoomPanel, 'Style', 'slider', ...
                          'Min', cfg.ui.zoom.range(1), 'Max', cfg.ui.zoom.range(2), ...
                          'Value', cfg.ui.zoom.defaultValue, ...
                          'Units', 'normalized', 'Position', cfg.ui.layout.zoomSlider, ...
                          'BackgroundColor', cfg.ui.colors.panel, ...
                          'Callback', @(src, ~) zoomSliderCallback(src, fig, cfg));

    % Zoom value display
    zoomValue = uicontrol('Parent', zoomPanel, 'Style', 'text', ...
                         'String', '0%', ...
                         'Units', 'normalized', 'Position', cfg.ui.layout.zoomValue, ...
                         'FontSize', cfg.ui.fontSize.value, 'FontWeight', 'bold', ...
                         'ForegroundColor', cfg.ui.colors.foreground, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'HorizontalAlignment', 'center');

    % Reset button (full image view)
    uicontrol('Parent', zoomPanel, 'Style', 'pushbutton', ...
             'String', 'Reset', ...
             'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.zoomResetButton, ...
             'BackgroundColor', [0.25 0.25 0.25], ...
             'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) resetZoom(fig, cfg));

    % Auto button (zoom to polygons)
    uicontrol('Parent', zoomPanel, 'Style', 'pushbutton', ...
             'String', 'Auto', ...
             'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.zoomAutoButton, ...
             'BackgroundColor', [0.25 0.25 0.25], ...
             'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) applyAutoZoom(fig, get(fig, 'UserData'), cfg));
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

    % Convert polygon positions to normalized coordinates [0, 1] before image update
    numPolygons = 0;
    polygonNormalized = {};
    if isfield(guiData, 'polygons') && iscell(guiData.polygons)
        numPolygons = length(guiData.polygons);
        polygonNormalized = cell(numPolygons, 1);
        for i = 1:numPolygons
            if isvalid(guiData.polygons{i})
                posData = guiData.polygons{i}.Position;  % [N x 2] array of vertices
                % Convert to normalized axes coordinates [0, 1]
                polygonNormalized{i} = [(posData(:, 1) - 1) / currentWidth, (posData(:, 2) - 1) / currentHeight];
            end
        end
    end

    % Update image data and spatial extent (preserves all axes children)
    set(guiData.imgHandle, 'CData', guiData.currentImg, ...
                            'XData', [1, newWidth], ...
                            'YData', [1, newHeight]);

    % Snap axes to new image bounds
    axis(guiData.imgAxes, 'image');

    % Update polygon positions to maintain screen-space locations
    for i = 1:numPolygons
        if isvalid(guiData.polygons{i})
            % Convert normalized coordinates back to new data coordinates
            newPos = [1 + polygonNormalized{i}(:, 1) * newWidth, 1 + polygonNormalized{i}(:, 2) * newHeight];
            guiData.polygons{i}.Position = newPos;
        end
    end

    % Reorder polygons to maintain concentration ordering after rotation
    [guiData.polygons, order, newOrientation] = assignPolygonLabels(guiData.polygons);
    guiData.orientation = newOrientation;  % Update orientation based on new layout

    hasLabels = isfield(guiData, 'polygonLabels') && iscell(guiData.polygonLabels);
    if hasLabels && ~isempty(order) && numel(guiData.polygonLabels) >= numel(order)
        guiData.polygonLabels = guiData.polygonLabels(order);
    end

    numPolygons = numel(guiData.polygons);
    totalForColor = max(numPolygons, 1);
    for idx = 1:numPolygons
        polyHandle = guiData.polygons{idx};
        if isvalid(polyHandle)
            gradColor = getConcentrationColor(idx - 1, totalForColor);
            setPolygonColor(polyHandle, gradColor, 0.25);

        end

        if hasLabels && idx <= numel(guiData.polygonLabels)
            labelHandle = guiData.polygonLabels{idx};
            if isvalid(labelHandle)
                set(labelHandle, 'String', sprintf('con_%d', idx - 1));
            end
        end
    end

    if hasLabels
        updatePolygonLabels(guiData.polygons, guiData.polygonLabels);
    end

    guiData.aiBaseColors = capturePolygonColors(guiData.polygons);
    guiData.autoZoomBounds = [];

    % Save guiData
    set(fig, 'UserData', guiData);

    % Re-trigger detection after rotation completes
    if cfg.useAIDetection
        fprintf('  Relaunching detection with new rotation...\n');
        runDeferredAIDetection(fig, cfg);
    end
end

function zoomSliderCallback(slider, fig, cfg)
    % Handle zoom slider changes
    guiData = get(fig, 'UserData');
    if ~strcmp(guiData.mode, 'editing')
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
    if ~strcmp(guiData.mode, 'editing')
        return;
    end

    guiData.zoomLevel = 0;
    set(guiData.zoomSlider, 'Value', 0);
    set(guiData.zoomValue, 'String', '0%');

    % Apply zoom to axes
    applyZoomToAxes(guiData, cfg);

    set(fig, 'UserData', guiData);
end

function applyAutoZoom(fig, guiData, cfg)
    % Auto-zoom to fit all polygons

    if isempty(guiData) || ~isstruct(guiData)
        return;
    end

    if ~isfield(guiData, 'mode') || ~strcmp(guiData.mode, 'editing')
        return;
    end

    % Calculate bounding box of all polygons
    [xmin, xmax, ymin, ymax] = calculatePolygonBounds(guiData);

    if isempty(xmin)
        return;  % No valid polygons
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
    % zoomLevel: 0 = full image, 1 = auto-zoom to polygons

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
            % Calculate bounds from polygons if they exist
            [xmin, xmax, ymin, ymax] = calculatePolygonBounds(guiData);
            if ~isempty(xmin)
                % Use actual polygon bounds
                autoZoomBounds = [xmin, xmax, ymin, ymax];
            else
                % No polygons yet - use center estimate
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

function [xmin, xmax, ymin, ymax] = calculatePolygonBounds(guiData)
    % Calculate bounding box containing all polygons
    xmin = inf;
    xmax = -inf;
    ymin = inf;
    ymax = -inf;

    if ~isfield(guiData, 'polygons') || isempty(guiData.polygons)
        xmin = [];
        return;
    end

    for i = 1:numel(guiData.polygons)
        if isvalid(guiData.polygons{i})
            pos = guiData.polygons{i}.Position;
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
    % Estimate bounds for a single micropad size when no polygons available
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

function basePolygons = convertDisplayPolygonsToBase(guiData, displayPolygons, cfg)
    % Convert polygons from rotated display coordinates back to original image coordinates
    basePolygons = displayPolygons;

    if isempty(displayPolygons)
        return;
    end

    if ~isfield(guiData, 'totalRotation')
        return;
    end

    rotation = guiData.totalRotation;
    if ~isMultipleOfNinety(rotation, cfg.rotation.angleTolerance)
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

    [basePolygons, newSize] = rotatePolygonsDiscrete(displayPolygons, imageSize, -rotation);

    if isfield(guiData, 'baseImageSize') && ~isempty(guiData.baseImageSize)
        targetSize = guiData.baseImageSize(1:2);
        if any(newSize ~= targetSize)
            basePolygons = scalePolygonsForImageSize(basePolygons, newSize, targetSize, cfg.numSquares);
        end
    end
end

function displayPolygons = convertBasePolygonsToDisplay(basePolygons, baseImageSize, displayImageSize, rotation, cfg)
    displayPolygons = basePolygons;
    if isempty(basePolygons)
        return;
    end

    if rotation ~= 0 && isMultipleOfNinety(rotation, cfg.rotation.angleTolerance)
        [rotatedPolygons, rotatedSize] = rotatePolygonsDiscrete(basePolygons, baseImageSize, rotation);
    else
        rotatedPolygons = basePolygons;
        rotatedSize = baseImageSize;
    end

    targetSize = displayImageSize(1:2);
    if any(rotatedSize ~= targetSize)
        displayPolygons = scalePolygonsForImageSize(rotatedPolygons, rotatedSize, targetSize, size(basePolygons, 1));
    else
        displayPolygons = rotatedPolygons;
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

function [rotatedPolygons, newSize] = rotatePolygonsDiscrete(polygons, imageSize, rotation)
    % Rotate polygons by multiples of 90 degrees using the same conventions as rot90
    imageSize = imageSize(1:2);
    [numPolygons, numVertices, ~] = size(polygons);
    rotatedPolygons = polygons;
    newSize = imageSize;

    if isempty(polygons)
        return;
    end

    k = mod(round(rotation / 90), 4);
    if k == 0
        return;
    end

    H = imageSize(1);
    W = imageSize(2);
    rotatedPolygons = zeros(size(polygons));

    switch k
        case 1  % 90 degrees clockwise
            newSize = [W, H];
            for i = 1:numPolygons
                poly = squeeze(polygons(i, :, :));
                transformed = zeros(numVertices, 2);
                transformed(:, 1) = H - poly(:, 2) + 1;
                transformed(:, 2) = poly(:, 1);
                rotatedPolygons(i, :, :) = clampPolygonToImage(transformed, newSize);
            end
        case 2  % 180 degrees
            newSize = [H, W];
            for i = 1:numPolygons
                poly = squeeze(polygons(i, :, :));
                transformed = zeros(numVertices, 2);
                transformed(:, 1) = W - poly(:, 1) + 1;
                transformed(:, 2) = H - poly(:, 2) + 1;
                rotatedPolygons(i, :, :) = clampPolygonToImage(transformed, newSize);
            end
        case 3  % 270 degrees clockwise (or 90 counter-clockwise)
            newSize = [W, H];
            for i = 1:numPolygons
                poly = squeeze(polygons(i, :, :));
                transformed = zeros(numVertices, 2);
                transformed(:, 1) = poly(:, 2);
                transformed(:, 2) = W - poly(:, 1) + 1;
                rotatedPolygons(i, :, :) = clampPolygonToImage(transformed, newSize);
            end
    end
end

function [rotatedPoints, newSize] = rotatePointsDiscrete(points, imageSize, rotation)
    rotatedPoints = points;
    newSize = imageSize(1:2);

    if isempty(points)
        return;
    end

    k = mod(round(rotation / 90), 4);
    if k == 0
        return;
    end

    H = imageSize(1);
    W = imageSize(2);
    rotatedPoints = zeros(size(points));

    switch k
        case 1 % 90 degrees clockwise
            newSize = [W, H];
            rotatedPoints(:, 1) = H - points(:, 2) + 1;
            rotatedPoints(:, 2) = points(:, 1);
        case 2 % 180 degrees
            newSize = [H, W];
            rotatedPoints(:, 1) = W - points(:, 1) + 1;
            rotatedPoints(:, 2) = H - points(:, 2) + 1;
        case 3 % 270 degrees clockwise (90 ccw)
            newSize = [W, H];
            rotatedPoints(:, 1) = points(:, 2);
            rotatedPoints(:, 2) = W - points(:, 1) + 1;
    end
end

function tf = isMultipleOfNinety(angle, tolerance)
    % Determine if an angle is effectively a multiple of 90 degrees
    if isnan(angle) || isinf(angle)
        tf = false;
        return;
    end
    tf = abs(angle / 90 - round(angle / 90)) <= tolerance;
end

function displayEllipses = convertBaseEllipsesToDisplay(ellipseData, baseImageSize, displayImageSize, rotation, cfg)
    displayEllipses = ellipseData;
    if isempty(ellipseData)
        return;
    end

    if rotation ~= 0 && isMultipleOfNinety(rotation, cfg.rotation.angleTolerance)
        [rotCenters, rotatedSize] = rotatePointsDiscrete(ellipseData(:, 3:4), baseImageSize, rotation);
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
    end

    displayEllipses(:, 3:4) = rotCenters;
    displayEllipses(:, 7) = mod(ellipseData(:, 7) + rotation + 180, 360) - 180;
end

function clamped = clampPolygonToImage(poly, imageSize)
    % Clamp polygon coordinates to lie within image extents
    if isempty(poly)
        clamped = poly;
        return;
    end
    width = imageSize(2);
    height = imageSize(1);
    clamped = poly;
    clamped(:, 1) = max(1, min(width, clamped(:, 1)));
    clamped(:, 2) = max(1, min(height, clamped(:, 2)));
end

function transformedPoints = inverseRotatePoints(points, rotatedSize, originalSize, rotation, angleTolerance)
    % Transform points from rotated image frame back to original frame
    % ROTATION CONVENTION: Positive rotation = clockwise in image coordinates
    % TRANSFORMATION: rotated -> original (inverse/reverse rotation)
    if rotation == 0 || isempty(points)
        transformedPoints = points;
        return;
    end

    % Handle exact 90-degree rotations
    if abs(mod(rotation, 90)) < angleTolerance
        numRotations = mod(round(rotation / 90), 4);

        % Map centers back through discrete rotations (with +1 for MATLAB 1-based indexing)
        switch numRotations
            case 1  % -90 degrees (rot90(..., -1))
                x_orig = points(:, 2);
                y_orig = rotatedSize(2) - points(:, 1) + 1;
            case 2  % 180 degrees
                x_orig = rotatedSize(2) - points(:, 1) + 1;
                y_orig = rotatedSize(1) - points(:, 2) + 1;
            case 3  % 90 degrees (rot90(..., 1))
                x_orig = rotatedSize(1) - points(:, 2) + 1;
                y_orig = points(:, 1);
            otherwise  % 0 degrees
                x_orig = points(:, 1);
                y_orig = points(:, 2);
        end

        transformedPoints = [x_orig, y_orig];
    else
        % For non-90-degree rotations, use geometric transform (inverse rotation)
        % Inverse rotation matrix for counter-clockwise in image coordinates (Y-axis down):
        % [x']   [cos(θ)  -sin(θ)]   [x]
        % [y'] = [sin(θ)   cos(θ)]   [y]
        theta = -deg2rad(rotation);  % Inverse rotation
        cosTheta = cos(theta);
        sinTheta = sin(theta);

        % Center of rotated image
        centerRotated = [rotatedSize(2)/2, rotatedSize(1)/2];
        centerOriginal = [originalSize(2)/2, originalSize(1)/2];

        % Translate to origin, rotate, translate back
        pointsCentered = points - centerRotated;
        x_orig = pointsCentered(:, 1) * cosTheta + pointsCentered(:, 2) * sinTheta;
        y_orig = -pointsCentered(:, 1) * sinTheta + pointsCentered(:, 2) * cosTheta;

        transformedPoints = [x_orig + centerOriginal(1), y_orig + centerOriginal(2)];
    end
end

function updatePolygonLabels(polygons, labelHandles)
    % Update label positions to follow polygon top edges
    %
    % Inputs:
    %   polygons - cell array of drawpolygon objects
    %   labelHandles - cell array of text objects

    n = numel(polygons);
    if numel(labelHandles) ~= n
        return;
    end

    for i = 1:n
        poly = polygons{i};
        label = labelHandles{i};

        if ~isvalid(poly) || ~isvalid(label)
            continue;
        end

        pos = poly.Position;
        if isempty(pos) || size(pos, 1) < 3
            continue;
        end

        % CHANGED: Position at TOP edge, not center
        % Use image-relative units (polygon height fraction) for consistency
        centerX = mean(pos(:, 1));
        minY = min(pos(:, 2));
        polyHeight = max(pos(:, 2)) - minY;
        labelY = minY - max(15, polyHeight * 0.1);  % 10% of polygon height or 15px minimum

        set(label, 'Position', [centerX, labelY, 0]);
    end
end

function updatePolygonLabelsCallback(polygon, varargin)
    % Callback for ROIMoved event to keep labels/colors ordered along dominant axis
    fig = ancestor(polygon, 'figure');
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

function [polygons, order, orientation] = assignPolygonLabels(polygons)
    % Reorder drawpolygon handles so con_0..con_N follow low→high concentration order.
    % Ordering direction matches the dominant layout axis: horizontal strips
    % sort left→right, vertical strips sort bottom→top.
    %
    % Returns:
    %   polygons    - Reordered cell array of polygon handles
    %   order       - Permutation indices used for reordering
    %   orientation - 'horizontal' or 'vertical' based on layout
    order = [];
    orientation = 'horizontal';  % Default

    if isempty(polygons) || ~iscell(polygons)
        return;
    end

    numPolygons = numel(polygons);
    centroids = nan(numPolygons, 2);
    minXs = nan(numPolygons, 1);
    maxXs = nan(numPolygons, 1);
    minYs = nan(numPolygons, 1);
    maxYs = nan(numPolygons, 1);
    validMask = false(numPolygons, 1);

    for i = 1:numPolygons
        if ~isvalid(polygons{i})
            continue;
        end

        pos = polygons{i}.Position;
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

    sortKey = inf(numPolygons, 2);
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
    polygons = polygons(order);
end

function guiData = enforceConcentrationOrdering(guiData)
    % Ensure drawpolygon handles, colors, and labels follow the dominant axis ordering
    if nargin < 1 || isempty(guiData) || ~isstruct(guiData)
        return;
    end
    if ~isfield(guiData, 'polygons') || ~iscell(guiData.polygons) || isempty(guiData.polygons)
        return;
    end

    [sortedPolygons, order, newOrientation] = assignPolygonLabels(guiData.polygons);
    guiData.orientation = newOrientation;  % Always update orientation based on current layout
    numPolygons = numel(guiData.polygons);
    needsReindex = ~isempty(order) && numel(order) == numPolygons && ...
                   ~isequal(order(:).', 1:numPolygons);
    if needsReindex
        guiData.polygons = sortedPolygons;
    end

    hasLabels = isfield(guiData, 'polygonLabels') && iscell(guiData.polygonLabels) && ...
                ~isempty(guiData.polygonLabels);
    if hasLabels
        labelCount = numel(guiData.polygonLabels);
        if needsReindex && labelCount >= numPolygons
            guiData.polygonLabels = guiData.polygonLabels(order);
        end

        for idx = 1:min(numPolygons, labelCount)
            labelHandle = guiData.polygonLabels{idx};
            if isvalid(labelHandle)
                set(labelHandle, 'String', sprintf('con_%d', idx - 1));
            end
        end

        updatePolygonLabels(guiData.polygons, guiData.polygonLabels);
    end

    if needsReindex
        totalForColor = max(numPolygons, 1);
        for idx = 1:numPolygons
            polyHandle = guiData.polygons{idx};
            if isvalid(polyHandle)
                gradColor = getConcentrationColor(idx - 1, totalForColor);
                setPolygonColor(polyHandle, gradColor, 0.25);
            end
        end
    end

    if needsReindex || ~isfield(guiData, 'aiBaseColors') || isempty(guiData.aiBaseColors)
        guiData.aiBaseColors = capturePolygonColors(guiData.polygons);
    end
end

function [sortedPolygons, orientation] = sortPolygonArrayByX(polygons)
    % Determine polygon ordering based on dominant axis coverage.
    % Primary ordering follows the axis with the greater spread-to-size ratio:
    %   - Horizontal layouts: left-to-right (primary), bottom-to-top (secondary)
    %   - Vertical layouts: bottom-to-top (primary), left-to-right (secondary)
    %
    % Returns:
    %   sortedPolygons - Polygons sorted by dominant axis
    %   orientation    - 'horizontal' or 'vertical' based on layout
    sortedPolygons = polygons;
    orientation = 'horizontal';  % Default
    if isempty(polygons)
        return;
    end
    if ndims(polygons) ~= 3
        return;
    end

    numPolygons = size(polygons, 1);
    if numPolygons == 0
        return;
    end

    centroids = nan(numPolygons, 2);
    minXs = nan(numPolygons, 1);
    maxXs = nan(numPolygons, 1);
    minYs = nan(numPolygons, 1);
    maxYs = nan(numPolygons, 1);

    for i = 1:numPolygons
        poly = squeeze(polygons(i, :, :));
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

    sortKey = inf(numPolygons, 2);

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
    sortedPolygons = polygons(order, :, :);
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

function rerunAIDetection(fig, cfg)
    % Re-run AI detection and replace current polygons with fresh detections
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

function [action, polygonParams, rotation] = waitForUserAction(fig)
    uiwait(fig);

    action = '';
    polygonParams = [];
    rotation = 0;

    if isvalid(fig)
        guiData = get(fig, 'UserData');
        action = guiData.action;

        if strcmp(action, 'accept')
            if strcmp(guiData.mode, 'preview')
                polygonParams = guiData.savedPolygonParams;
                if isfield(guiData, 'savedRotation')
                    rotation = guiData.savedRotation;
                end
            elseif strcmp(guiData.mode, 'editing')
                guiData = enforceConcentrationOrdering(guiData);
                set(fig, 'UserData', guiData);
                polygonParams = extractPolygonParameters(guiData);
                if isempty(polygonParams)
                    action = 'skip';
                else
                    rotation = guiData.totalRotation;
                end
            elseif strcmp(guiData.mode, 'ellipse_editing')
                % Get all ellipse data at once
                numConcentrations = size(guiData.polygons, 1);
                numReplicates = guiData.cfg.ellipse.replicatesPerMicropad;
                totalEllipses = numConcentrations * numReplicates;
                boundsPerConcentration = cell(numConcentrations, 1);
                polygonSetForBounds = guiData.polygons;
                if isfield(guiData, 'displayPolygons') && ~isempty(guiData.displayPolygons)
                    polygonSetForBounds = guiData.displayPolygons;
                end
                for concIdx = 1:numConcentrations
                    currentPolygon = squeeze(polygonSetForBounds(concIdx, :, :));
                    boundsPerConcentration{concIdx} = computeEllipseAxisBounds(currentPolygon, guiData.imageSize, guiData.cfg);
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
                            [semiMajor, semiMinor, rotationAngle] = enforceEllipseAxisLimits(semiAxes(1), semiAxes(2), rotationAngle, bounds);

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
                            originalCenter = inverseRotatePoints(center, rotatedImageSize, ...
                                originalImageSize, guiData.rotation, guiData.cfg.rotation.angleTolerance);
                            ellipseData(i, 3:4) = round(originalCenter);

                            % Transform ellipse rotation angle
                            % Subtract display rotation to get angle in original image frame
                            ellipseData(i, 7) = mod(ellipseData(i, 7) - guiData.rotation + 180, 360) - 180;
                        end
                    end
                end

                % Return Nx7 matrix instead of cell array
                polygonParams = ellipseData;
                rotation = guiData.rotation;
            elseif strcmp(guiData.mode, 'ellipse_editing_grid')
                % Mode 3: Ellipse-only with grid layout (no polygons)
                numGroups = guiData.cfg.numSquares;
                numReplicates = guiData.cfg.ellipse.replicatesPerMicropad;
                totalEllipses = numGroups * numReplicates;
                gridBounds = computeEllipseAxisBounds([], guiData.imageSize, guiData.cfg);

                ellipseData = zeros(totalEllipses, 7);
                ellipseIdx = 1;

                for groupIdx = 1:numGroups
                    for repIdx = 1:numReplicates
                        if ellipseIdx <= numel(guiData.ellipses) && isvalid(guiData.ellipses{ellipseIdx})
                            ellipse = guiData.ellipses{ellipseIdx};
                            center = ellipse.Center;
                            semiAxes = ellipse.SemiAxes;
                            rotationAngle = ellipse.RotationAngle;

                            [semiMajor, semiMinor, rotationAngle] = enforceEllipseAxisLimits(semiAxes(1), semiAxes(2), rotationAngle, gridBounds);

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
                            originalCenter = inverseRotatePoints(center, rotatedImageSize, ...
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

                polygonParams = []; % No polygons in grid mode
                rotation = guiData.rotation;
            elseif strcmp(guiData.mode, 'preview_readonly')
                % Mode 4: Read-only preview, no outputs needed
                polygonParams = [];
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

function polygonParams = extractPolygonParameters(guiData)
    polygonParams = [];

    if ~isfield(guiData, 'polygons') || ~iscell(guiData.polygons)
        return;
    end

    validMask = cellfun(@isvalid, guiData.polygons);
    if ~any(validMask)
        return;
    end

    validPolygons = guiData.polygons(validMask);
    keptPositions = cell(1, numel(validPolygons));

    keepIdx = 0;
    for i = 1:numel(validPolygons)
        pos = validPolygons{i}.Position;

        if isempty(pos) || size(pos, 1) < 4
            warning('cut_micropads:invalid_polygon', ...
                    'Polygon %d is missing vertices. Ignoring this polygon for extraction.', i);
            continue;
        end

        keepIdx = keepIdx + 1;
        keptPositions{keepIdx} = pos(1:4, :);
    end

    if keepIdx == 0
        polygonParams = [];
        return;
    end

    polygonParams = zeros(keepIdx, 4, 2);
    for i = 1:keepIdx
        polygonParams(i, :, :) = keptPositions{i};
    end

    [polygonParams, ~] = sortPolygonArrayByX(polygonParams);
end

%% -------------------------------------------------------------------------
%% Image Cropping and Coordinate Saving
%% -------------------------------------------------------------------------

function saveCroppedRegions(img, imageName, polygons, outputDir, cfg, rotation)
    [~, baseName, ~] = fileparts(imageName);
    outExt = '.png';

    numRegions = size(polygons, 1);

    for concentration = 0:(numRegions - 1)
        polygon = squeeze(polygons(concentration + 1, :, :));

        croppedImg = cropImageWithPolygon(img, polygon);

        concFolder = sprintf('%s%d', cfg.concFolderPrefix, concentration);
        concPath = fullfile(outputDir, concFolder);

        outputName = sprintf('%s_con_%d%s', baseName, concentration, outExt);
        outputPath = fullfile(concPath, outputName);

        saveImageWithFormat(croppedImg, outputPath, outExt, cfg);

        if cfg.output.saveCoordinates
            appendPolygonCoordinates(outputDir, baseName, concentration, polygon, cfg, rotation);
        end
    end
end

function saveEllipseData(img, imageName, ~, ellipseData, outputDir, cfg)
    % Save ellipse patches and coordinates to 3_elliptical_regions/
    % ellipseData: Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]

    [~, baseName, ~] = fileparts(imageName);

    % Save elliptical patches
    saveEllipticalPatches(img, baseName, ellipseData, outputDir, cfg);

    % Save ellipse coordinates to phone-level coordinates.txt
    if cfg.output.saveCoordinates
        appendEllipseCoordinates(outputDir, baseName, ellipseData, cfg);
    end
end

function croppedImg = cropImageWithPolygon(img, polygonVertices)
    x = polygonVertices(:, 1);
    y = polygonVertices(:, 2);

    minX = floor(min(x));
    maxX = ceil(max(x));
    minY = floor(min(y));
    maxY = ceil(max(y));

    [imgH, imgW, numChannels] = size(img);
    minX = max(1, minX);
    maxX = min(imgW, maxX);
    minY = max(1, minY);
    maxY = min(imgH, maxY);

    % Create binary mask for the polygon
    mask = poly2mask(x, y, imgH, imgW);

    % Apply mask to each color channel (black out pixels outside polygon)
    maskedImg = img;
    for ch = 1:numChannels
        channel = maskedImg(:, :, ch);
        channel(~mask) = 0;
        maskedImg(:, :, ch) = channel;
    end

    % Crop to bounding box
    croppedImg = maskedImg(minY:maxY, minX:maxX, :);
end

function appendPolygonCoordinates(phoneOutputDir, baseName, concentration, polygon, cfg, rotation)
    % Append polygon vertex coordinates to phone-level coordinates file with atomic write
    % Overwrites existing entry for same image/concentration combination

    coordFolder = phoneOutputDir;
    coordPath = fullfile(coordFolder, cfg.coordinateFileName);

    if ~isnumeric(polygon) || size(polygon, 2) ~= 2
        warning('cut_micropads:coord_polygon', 'Polygon must be an Nx2 numeric array. Skipping write for %s.', baseName);
        return;
    end

    nVerts = size(polygon, 1);
    if nVerts ~= 4
        warning('cut_micropads:coord_vertices', ...
            'Expected 4-vertex polygon; got %d. Proceeding may break downstream tools.', nVerts);
    end

    numericCount = 1 + 2 * nVerts + 1; % concentration, vertices, rotation

    headerParts = cell(1, 2 + 2 * nVerts + 1);
    headerParts{1} = 'image';
    headerParts{2} = 'concentration';
    for v = 1:nVerts
        headerParts{2*v+1} = sprintf('x%d', v);
        headerParts{2*v+2} = sprintf('y%d', v);
    end
    headerParts{end} = 'rotation';
    header = strjoin(headerParts, ' ');

    scanFmt = ['%s' repmat(' %f', 1, numericCount)];

    writeSpecs = repmat({'%.6f'}, 1, numericCount);
    writeSpecs{1} = '%.0f';   % concentration index
    writeFmt = ['%s ' strjoin(writeSpecs, ' ') '\n'];

    coords = reshape(polygon.', 1, []);
    newNums = [concentration, coords, rotation];

    [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount);
    [existingNames, existingNums] = filterConflictingEntries(existingNames, existingNums, baseName, concentration);

    allNames = [existingNames; {baseName}];
    allNums = [existingNums; newNums];

    atomicWriteCoordinates(coordPath, header, allNames, allNums, writeFmt, coordFolder);
end

function saveEllipticalPatches(img, baseName, ellipseData, outputDir, cfg)
    % Extract and save elliptical patches from original image
    % ellipseData: Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]

    [imgH, imgW, numChannels] = size(img);

    for i = 1:size(ellipseData, 1)
        if ellipseData(i, 3) > 0
            concIdx = ellipseData(i, 1);
            repIdx = ellipseData(i, 2);
            x = ellipseData(i, 3);
            y = ellipseData(i, 4);
            a = ellipseData(i, 5);
            b = ellipseData(i, 6);
            theta = ellipseData(i, 7);

            % Calculate axis-aligned bounding box
            theta_rad = deg2rad(theta);
            ux = sqrt((a * cos(theta_rad))^2 + (b * sin(theta_rad))^2);
            uy = sqrt((a * sin(theta_rad))^2 + (b * cos(theta_rad))^2);

            x1 = max(1, floor(x - ux));
            y1 = max(1, floor(y - uy));
            x2 = min(imgW, ceil(x + ux));
            y2 = min(imgH, ceil(y + uy));

            % Extract region
            patchRegion = img(y1:y2, x1:x2, :);
            [patchH, patchW, ~] = size(patchRegion);

            % Create elliptical mask
            [Xpatch, Ypatch] = meshgrid(1:patchW, 1:patchH);

            centerX_patch = x - x1 + 1;
            centerY_patch = y - y1 + 1;

            dx = Xpatch - centerX_patch;
            dy = Ypatch - centerY_patch;
            x_rot =  dx * cos(theta_rad) + dy * sin(theta_rad);
            y_rot = -dx * sin(theta_rad) + dy * cos(theta_rad);

            ellipseMask = (x_rot ./ a).^2 + (y_rot ./ b).^2 <= 1;

            % Apply mask (zero out pixels outside ellipse)
            ellipticalPatch = patchRegion;
            inverseMask3D = repmat(~ellipseMask, [1, 1, numChannels]);
            ellipticalPatch(inverseMask3D) = 0;

            % Save patch
            concFolder = fullfile(outputDir, sprintf('%s%d', cfg.concFolderPrefix, concIdx));
            if ~exist(concFolder, 'dir')
                mkdir(concFolder);
            end

            patchFileName = sprintf('%s_con%d_rep%d.png', baseName, concIdx, repIdx);
            patchPath = fullfile(concFolder, patchFileName);
            imwrite(ellipticalPatch, patchPath);
        end
    end
end

function appendEllipseCoordinates(phoneOutputDir, baseName, ellipseData, cfg)
    % Append ellipse coordinates to phone-level coordinates file with atomic write
    % ellipseData: Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]

    coordFolder = phoneOutputDir;
    coordPath = fullfile(coordFolder, cfg.coordinateFileName);

    header = 'image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle';
    numericCount = 7;

    scanFmt = ['%s' repmat(' %f', 1, numericCount)];
    writeFmt = '%s %.0f %.0f %.6f %.6f %.6f %.6f %.6f\n';

    % Read existing entries
    [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount);

    % Filter out entries for this image (remove all existing rows for same image)
    if ~isempty(existingNames)
        existingNames = existingNames(:);
        keepMask = ~strcmp(existingNames, baseName);
        existingNames = existingNames(keepMask);
        if ~isempty(existingNums)
            existingNums = existingNums(keepMask, :);
        end
    end

    % Build new rows for this image (only valid ellipses)
    validMask = ellipseData(:, 3) > 0;
    validData = ellipseData(validMask, :);

    if isempty(validData)
        return;
    end

    numValid = size(validData, 1);
    newNames = cell(numValid, 1);
    newNames(:) = {baseName};
    newNums = validData;

    % Combine and write atomically
    allNames = [existingNames; newNames];
    allNums = [existingNums; newNums];

    atomicWriteCoordinates(coordPath, header, allNames, allNums, writeFmt, coordFolder);
end

function [polygonParams, found] = loadPolygonCoordinates(coordFile, imageName, numExpected)
    % Load polygon coordinates from 2_micropads coordinates.txt file
    %
    % INPUTS:
    %   coordFile   - Full path to coordinates.txt file
    %   imageName   - Base image name to filter rows
    %   numExpected - Expected number of polygons (concentrations)
    %
    % OUTPUTS:
    %   polygonParams - Nx4x2 matrix of polygon vertices (N concentrations, 4 vertices, 2 coords)
    %   found         - Boolean indicating if file exists and contains data

    polygonParams = [];
    found = false;

    % Check if file exists
    if ~isfile(coordFile)
        return;
    end

    % Read file using atomic pattern
    try
        fid = fopen(coordFile, 'rt');
        if fid == -1
            return;
        end

        % Read header
        headerLine = fgetl(fid);
        if ~ischar(headerLine)
            fclose(fid);
            return;
        end

        % Read all data rows
        allRows = {};
        while ~feof(fid)
            line = fgetl(fid);
            if ischar(line) && ~isempty(line)
                allRows{end+1} = line; %#ok<AGROW>
            end
        end
        fclose(fid);

        % Parse rows matching this image
        [~, baseNameNoExt, ~] = fileparts(imageName);
        matchingRows = {};

        for i = 1:length(allRows)
            parts = strsplit(strtrim(allRows{i}));
            if length(parts) >= 11
                rowImageName = parts{1};
                [~, rowBaseNoExt, ~] = fileparts(rowImageName);

                if strcmpi(rowBaseNoExt, baseNameNoExt)
                    matchingRows{end+1} = allRows{i}; %#ok<AGROW>
                end
            end
        end

        if isempty(matchingRows)
            return;
        end

        % Parse matching rows into polygon matrix
        numRows = length(matchingRows);
        polygonParams = zeros(numRows, 4, 2);

        for i = 1:numRows
            parts = strsplit(strtrim(matchingRows{i}));
            if length(parts) >= 11
                % Parse: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation
                concIdx = str2double(parts{2});
                
                % Validate concentration index
                if ~isempty(numExpected) && (concIdx < 0 || concIdx >= numExpected)
                    warning('cut_micropads:invalid_concentration', ...
                        'Invalid concentration index %.0f (expected 0-%d) for image %s - skipping row', ...
                        concIdx, numExpected - 1, imageName);
                    continue;
                end
                
                % Extract vertices (columns 3-10)
                x1 = str2double(parts{3});
                y1 = str2double(parts{4});
                x2 = str2double(parts{5});
                y2 = str2double(parts{6});
                x3 = str2double(parts{7});
                y3 = str2double(parts{8});
                x4 = str2double(parts{9});
                y4 = str2double(parts{10});
                
                % Validate coordinates are finite
                coords = [x1 y1 x2 y2 x3 y3 x4 y4];
                if any(~isfinite(coords))
                    warning('cut_micropads:invalid_coordinates', ...
                        'Invalid polygon coordinates for image %s, concentration %d - skipping row', ...
                        imageName, concIdx);
                    continue;
                end

                % Store as 4x2 matrix: [x1 y1; x2 y2; x3 y3; x4 y4]
                polygonParams(i, :, :) = [x1 y1; x2 y2; x3 y3; x4 y4];
            end
        end

        found = true;

        % Validate polygon count matches expected
        if ~isempty(numExpected) && size(polygonParams, 1) ~= numExpected
            warning('cut_micropads:polygon_count_mismatch', ...
                'Expected %d polygons, found %d for image %s', ...
                numExpected, size(polygonParams, 1), imageName);
        end

    catch ME
        warning('cut_micropads:polygon_load_error', ...
            'Failed to load polygon coordinates from %s: %s', coordFile, ME.message);
        polygonParams = [];
        found = false;
    end
end

function [ellipseData, found] = loadEllipseCoordinates(coordFile, imageName)
    % Load ellipse coordinates from 3_elliptical_regions coordinates.txt file
    %
    % INPUTS:
    %   coordFile  - Full path to coordinates.txt file
    %   imageName  - Base image name to filter rows
    %
    % OUTPUTS:
    %   ellipseData - Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
    %   found       - Boolean indicating if file exists and contains data

    ellipseData = [];
    found = false;

    % Check if file exists
    if ~isfile(coordFile)
        return;
    end

    % Read file using atomic pattern
    try
        fid = fopen(coordFile, 'rt');
        if fid == -1
            return;
        end

        % Read header
        headerLine = fgetl(fid);
        if ~ischar(headerLine)
            fclose(fid);
            return;
        end

        % Read all data rows
        allRows = {};
        while ~feof(fid)
            line = fgetl(fid);
            if ischar(line) && ~isempty(line)
                allRows{end+1} = line; %#ok<AGROW>
            end
        end
        fclose(fid);

        % Parse rows matching this image
        [~, baseNameNoExt, ~] = fileparts(imageName);
        matchingRows = {};

        for i = 1:length(allRows)
            parts = strsplit(strtrim(allRows{i}));
            if length(parts) >= 8
                rowImageName = parts{1};
                [~, rowBaseNoExt, ~] = fileparts(rowImageName);

                if strcmpi(rowBaseNoExt, baseNameNoExt)
                    matchingRows{end+1} = allRows{i}; %#ok<AGROW>
                end
            end
        end

        if isempty(matchingRows)
            return;
        end

        % Parse matching rows into ellipse matrix
        numRows = length(matchingRows);
        ellipseData = zeros(numRows, 7);

        for i = 1:numRows
            parts = strsplit(strtrim(matchingRows{i}));
            if length(parts) >= 8
                % Parse: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
                concIdx = str2double(parts{2});
                repIdx = str2double(parts{3});
                x = str2double(parts{4});
                y = str2double(parts{5});
                semiMajor = str2double(parts{6});
                semiMinor = str2double(parts{7});
                rotation = str2double(parts{8});
                
                % Validate all numeric values are finite
                values = [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation];
                if any(~isfinite(values))
                    warning('cut_micropads:invalid_ellipse', ...
                        'Invalid ellipse data for image %s, concentration %.0f, replicate %.0f - skipping row', ...
                        imageName, concIdx, repIdx);
                    continue;
                end
                
                % Validate ellipse geometry
                if semiMajor <= 0 || semiMinor <= 0
                    warning('cut_micropads:invalid_ellipse_axes', ...
                        'Invalid ellipse axes (semiMajor=%.2f, semiMinor=%.2f) for image %s - skipping row', ...
                        semiMajor, semiMinor, imageName);
                    continue;
                end

                ellipseData(i, :) = [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation];
            end
        end

        found = true;

    catch ME
        warning('cut_micropads:ellipse_load_error', ...
            'Failed to load ellipse coordinates from %s: %s', coordFile, ME.message);
        ellipseData = [];
        found = false;
    end
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

function [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount)
    existingNames = {};
    existingNums = zeros(0, numericCount);

    if ~isfile(coordPath)
        return;
    end

    fid = fopen(coordPath, 'rt');
    if fid == -1
        warning('cut_micropads:coord_read', 'Cannot open coordinates file for reading: %s', coordPath);
        return;
    end

    firstLine = fgetl(fid);
    if ischar(firstLine)
        trimmed = strtrim(firstLine);
        expectedPrefix = 'image concentration';
        if ~strncmpi(trimmed, expectedPrefix, numel(expectedPrefix))
            fseek(fid, 0, 'bof');
        end
    else
        fseek(fid, 0, 'bof');
    end

    data = textscan(fid, scanFmt, 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'CollectOutput', true);
    fclose(fid);

    if ~isempty(data)
        if numel(data) >= 1 && ~isempty(data{1})
            existingNames = data{1};
        end
        if numel(data) >= 2 && ~isempty(data{2})
            nums = data{2};

            % Validate coordinate format (no migration - project is in active development)
            if size(nums, 2) ~= numericCount
                error('cut_micropads:invalid_coord_format', ...
                    ['Coordinate file has invalid format: %d columns found, expected %d.\n' ...
                     'File: %s\n' ...
                     'This project requires the current 10-column format (image, concentration, x1, y1, x2, y2, x3, y3, x4, y4, rotation).\n' ...
                     'NOTE: This project is in active development mode with no backward compatibility.\n' ...
                     'Delete the corrupted file and rerun the stage to regenerate.'], ...
                    size(nums, 2), numericCount, coordPath);
            end

            existingNums = nums;

            % Validate numeric content (skip rotation column for NaN check)
            coordCols = 2:9;  % x1, y1, x2, y2, x3, y3, x4, y4
            invalidRows = any(~isfinite(existingNums(:, coordCols)), 2);
            if any(invalidRows)
                warning('cut_micropads:corrupt_coords', ...
                    'Found %d rows with invalid coordinates in %s. Skipping corrupted entries.', ...
                    sum(invalidRows), coordPath);
                validMask = ~invalidRows;
                existingNames = existingNames(validMask);
                existingNums = existingNums(validMask, :);
            end
        end
    end

    if ~isempty(existingNames) && ~isempty(existingNums)
        rows = min(numel(existingNames), size(existingNums, 1));
        if size(existingNums, 1) ~= numel(existingNames)
            existingNames = existingNames(1:rows);
            existingNums = existingNums(1:rows, :);
        end

        if iscell(existingNames)
            emptyMask = cellfun(@(s) isempty(strtrim(s)), existingNames);
        else
            emptyMask = arrayfun(@(s) isempty(strtrim(s)), existingNames);
        end
        if any(emptyMask)
            existingNames = existingNames(~emptyMask);
            existingNums = existingNums(~emptyMask, :);
        end
    end
end

function [filteredNames, filteredNums] = filterConflictingEntries(existingNames, existingNums, newName, concentration)
    if isempty(existingNames)
        filteredNames = existingNames;
        filteredNums = existingNums;
        return;
    end

    existingNames = existingNames(:);
    sameImageMask = strcmp(existingNames, newName);
    sameConcentrationMask = false(size(sameImageMask));
    if ~isempty(existingNums)
        sameConcentrationMask = sameImageMask & (existingNums(:, 1) == concentration);
    end
    keepMask = ~sameConcentrationMask;

    filteredNames = existingNames(keepMask);
    if isempty(existingNums)
        filteredNums = existingNums;
    else
        filteredNums = existingNums(keepMask, :);
    end
end

function atomicWriteCoordinates(coordPath, header, names, nums, writeFmt, coordFolder)
    tmpPath = tempname(coordFolder);

    fid = fopen(tmpPath, 'wt');
    if fid == -1
        error('cut_micropads:coord_write_failed', ...
              'Cannot open temp coordinates file for writing: %s\nCheck folder permissions.', tmpPath);
    end

    fprintf(fid, '%s\n', header);

    for j = 1:numel(names)
        rowVals = nums(j, :);
        rowVals(isnan(rowVals)) = 0;
        fprintf(fid, writeFmt, names{j}, rowVals);
    end

    fclose(fid);

    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        warning('cut_micropads:coord_move', ...
            'Failed to move temp file to coordinates.txt: %s (%s). Attempting fallback copy.', msg, msgid);
        [copied, cmsg, ~] = copyfile(tmpPath, coordPath, 'f');
        if ~copied
            if isfile(tmpPath)
                delete(tmpPath);
            end
            error('cut_micropads:coord_write_fail', ...
                'Cannot write coordinates to %s: movefile failed (%s), copyfile failed (%s).', ...
                coordPath, msg, cmsg);
        end
        if isfile(tmpPath)
            delete(tmpPath);
        end
    end
end

%% -------------------------------------------------------------------------
%% File I/O Utilities
%% -------------------------------------------------------------------------

function [img, isValid] = loadImage(imageName)
    isValid = false;
    img = [];

    if ~isfile(imageName)
        warning('cut_micropads:missing_file', 'Image file not found: %s', imageName);
        return;
    end

    try
        img = imread_raw(imageName);
        isValid = true;
    catch ME
        warning('cut_micropads:read_error', 'Failed to read image %s: %s', imageName, ME.message);
    end
end

function saveImageWithFormat(img, outPath, ~, ~)
    imwrite(img, outPath);
end

function outputDirs = createOutputDirectory(basePathPolygons, basePathEllipses, phoneName, numConcentrations, concFolderPrefix)
    % Create polygon output directories
    phoneOutputDirPolygons = fullfile(basePathPolygons, phoneName);
    if ~isfolder(phoneOutputDirPolygons)
        mkdir(phoneOutputDirPolygons);
    end

    for i = 0:(numConcentrations - 1)
        concFolder = sprintf('%s%d', concFolderPrefix, i);
        concPath = fullfile(phoneOutputDirPolygons, concFolder);
        if ~isfolder(concPath)
            mkdir(concPath);
        end
    end

    % Create ellipse output directories
    phoneOutputDirEllipses = fullfile(basePathEllipses, phoneName);
    if ~isfolder(phoneOutputDirEllipses)
        mkdir(phoneOutputDirEllipses);
    end

    for i = 0:(numConcentrations - 1)
        concFolder = sprintf('%s%d', concFolderPrefix, i);
        concPath = fullfile(phoneOutputDirEllipses, concFolder);
        if ~isfolder(concPath)
            mkdir(concPath);
        end
    end

    % Return both directories as struct
    outputDirs = struct();
    outputDirs.polygonDir = phoneOutputDirPolygons;
    outputDirs.ellipseDir = phoneOutputDirEllipses;
end

function folders = getSubFolders(dirPath)
    items = dir(dirPath);
    folders = {items([items.isdir]).name};
    folders = folders(~ismember(folders, {'.', '..'}));
end

function files = getImageFiles(dirPath, extensions)
    % Collect files for each extension efficiently
    fileList = cell(numel(extensions), 1);
    for i = 1:numel(extensions)
        foundFiles = dir(fullfile(dirPath, extensions{i}));
        if ~isempty(foundFiles)
            fileList{i} = {foundFiles.name}';
        else
            fileList{i} = {};
        end
    end

    % Concatenate and get unique files
    files = unique(vertcat(fileList{:}));
end

function executeInFolder(folder, func)
    origDir = pwd;
    cleanupObj = onCleanup(@() cd(origDir));
    cd(folder);
    func();
end

%% -------------------------------------------------------------------------
%% YOLO Auto-Detection Integration
%% -------------------------------------------------------------------------

function ensurePythonSetup(pythonPath)
    persistent setupComplete
    if ~isempty(setupComplete) && setupComplete
        return;
    end

    try
        pythonPath = char(pythonPath);

        % Check environment variable first
        envPath = getenv('MICROPAD_PYTHON');
        if ~isempty(envPath)
            pythonPath = envPath;
        end

        % If still empty, try platform-specific search paths
        if isempty(pythonPath)
            pythonPath = searchForPython();
        end

        % Validate Python path is provided
        if isempty(pythonPath)
            error('cut_micropads:python_not_configured', ...
                ['Python path not configured! Options:\n', ...
                 '  1. Set MICROPAD_PYTHON environment variable\n', ...
                 '  2. Pass pythonPath parameter: cut_micropads(''pythonPath'', ''path/to/python'')\n', ...
                 '  3. Ensure Python is in system PATH']);
        end

        if ~isfile(pythonPath)
            error('cut_micropads:python_missing', ...
                'Python executable not found at: %s', pythonPath);
        end

        fprintf('Python configured: %s\n', pythonPath);
        setupComplete = true;
    catch ME
        setupComplete = [];
        rethrow(ME);
    end
end

function pythonPath = searchForPython()
    % Search for Python executable in common platform-specific locations
    pythonPath = '';

    if ispc()
        % Windows: check common conda/miniconda locations
        commonPaths = {
            fullfile(getenv('USERPROFILE'), 'miniconda3\envs\microPAD-python-env\python.exe')
            fullfile(getenv('USERPROFILE'), 'anaconda3\envs\microPAD-python-env\python.exe')
            fullfile(getenv('LOCALAPPDATA'), 'Programs\Python\Python*\python.exe')
        };
    elseif ismac()
        % macOS: check common conda/homebrew locations
        commonPaths = {
            fullfile(getenv('HOME'), 'miniconda3/envs/microPAD-python-env/bin/python')
            fullfile(getenv('HOME'), 'anaconda3/envs/microPAD-python-env/bin/python')
            '/usr/local/bin/python3'
            '/opt/homebrew/bin/python3'
        };
    else
        % Linux: check common conda/system locations
        commonPaths = {
            fullfile(getenv('HOME'), 'miniconda3/envs/microPAD-python-env/bin/python')
            fullfile(getenv('HOME'), 'anaconda3/envs/microPAD-python-env/bin/python')
            '/usr/bin/python3'
        };
    end

    % Check each path
    for i = 1:numel(commonPaths)
        if isfile(commonPaths{i})
            pythonPath = commonPaths{i};
            return;
        end
    end

    % Try system PATH as fallback
    if ispc()
        [status, result] = system('where python');
    else
        [status, result] = system('which python3');
    end

    if status == 0
        lines = strsplit(strtrim(result), newline);
        if ~isempty(lines)
            pythonPath = char(lines{1});
        end
    end
end

function I = imread_raw(fname)
% Read image pixels in their recorded layout without applying EXIF orientation
% metadata. Any user-requested rotation is stored in coordinates.txt and applied
% during downstream processing rather than via image metadata.

    I = imread(fname);
end

function [quads, confidences, outputFile, imgPath] = detectQuadsYOLO(img, cfg, varargin)
    % Run YOLO detection via Python helper script (subprocess interface)
    %
    % Inputs:
    %   img - input image array
    %   cfg - configuration struct
    %   varargin - optional name-value pairs:
    %     'async' - if true, launch non-blocking and return immediately
    %               returns empty quads/confidences, non-empty outputFile
    %
    % Outputs:
    %   quads - detected quadrilaterals (Nx4x2 array) or [] if async
    %   confidences - detection confidences (Nx1 array) or [] if async
    %   outputFile - path to output file for async mode, empty otherwise
    %   imgPath - path to temp image file (for caller cleanup in async mode)

    % Extract image dimensions for validation
    [imageHeight, imageWidth, ~] = size(img);

    p = inputParser;
    addParameter(p, 'async', false, @islogical);
    parse(p, varargin{:});
    asyncMode = p.Results.async;

    % Save image to temporary file
    tmpDir = tempdir;
    [~, tmpName] = fileparts(tempname);
    tmpImgPath = fullfile(tmpDir, sprintf('%s_micropad_detect.png', tmpName));
    imwrite(img, tmpImgPath);

    % Ensure cleanup even if error occurs (only in blocking mode)
    if ~asyncMode
        cleanupObj = onCleanup(@() cleanupTempFile(tmpImgPath));
    end

    % Initialize output variables
    outputFile = '';
    imgPath = '';

    % Create output file for async mode
    if asyncMode
        outputFile = fullfile(tmpDir, sprintf('%s_detection_output.txt', tmpName));
    end

    % Build command with platform-specific syntax
    if asyncMode
        % Platform-specific background execution
        if ispc
            % Windows: Build command with proper quote handling for cmd /c
            innerCmd = sprintf('"%s" "%s" "%s" "%s" --conf %.2f --imgsz %d', ...
                cfg.pythonPath, cfg.pythonScriptPath, tmpImgPath, cfg.detectionModel, ...
                cfg.minConfidence, cfg.inferenceSize);

            % Escape quotes with double-quote for cmd /c context
            escapedOutput = strrep(outputFile, '"', '""');

            % Construct command: double-quotes work inside cmd /c "..."
            cmd = sprintf('start /B "" cmd /c "%s > ""%s"" 2>&1"', innerCmd, escapedOutput);
        else
            % Unix/macOS: Use '&' suffix for background execution
            cmd = sprintf('"%s" "%s" "%s" "%s" --conf %.2f --imgsz %d > "%s" 2>&1 &', ...
                cfg.pythonPath, cfg.pythonScriptPath, tmpImgPath, cfg.detectionModel, ...
                cfg.minConfidence, cfg.inferenceSize, outputFile);
        end
    else
        % Blocking mode: redirect stderr to stdout to capture all output
        cmd = sprintf('"%s" "%s" "%s" "%s" --conf %.2f --imgsz %d 2>&1', ...
            cfg.pythonPath, cfg.pythonScriptPath, tmpImgPath, cfg.detectionModel, ...
            cfg.minConfidence, cfg.inferenceSize);
    end

    % Run detection
    if asyncMode
        % Launch background process and return immediately
        system(cmd);
        quads = [];
        confidences = [];
        imgPath = tmpImgPath;  % Return temp image path for caller cleanup
        return;
    end

    % Blocking mode: continue with original code
    [status, output] = system(cmd);

    if status ~= 0
        error('cut_micropads:detection_failed', 'Python detection failed (exit code %d): %s', status, output);
    end

    % Parse output (split by newlines - R2019b compatible)
    lines = strsplit(output, {'\n', '\r\n', '\r'}, 'CollapseDelimiters', false);
    lines = lines(~cellfun(@isempty, lines));
    [quads, confidences] = parseDetectionOutput(lines, imageHeight, imageWidth);
end

function [isComplete, quads, confidences, errorMsg] = checkDetectionComplete(outputFile, img)
    % Check if async detection has completed and parse results
    %
    % Inputs:
    %   outputFile - path to detection output file
    %   img - input image for dimension validation
    %
    % Outputs:
    %   isComplete - true if detection finished (success or error)
    %   quads - detected quadrilaterals (Nx4x2) or [] if not complete/failed
    %   confidences - detection confidences (Nx1) or [] if not complete/failed
    %   errorMsg - error message string if parsing failed, empty otherwise

    isComplete = false;
    quads = [];
    confidences = [];
    errorMsg = '';

    % Extract image dimensions for validation
    [imageHeight, imageWidth, ~] = size(img);

    % Check if output file exists and has content
    if ~exist(outputFile, 'file')
        return;
    end

    % Try to read the file
    try
        fid = fopen(outputFile, 'rt');
        if fid < 0
            return;
        end

        % Read all content
        content = fread(fid, '*char')';
        fclose(fid);

        % Empty file means process hasn't written yet
        if isempty(content)
            return;
        end

        % Check for error patterns using single regexp
        isError = ~isempty(regexp(content, '(ERROR:|Traceback|Exception|FileNotFoundError)', 'once', 'ignorecase'));

        if isError
            % Extract first 200 characters for error context (simple and reliable)
            errorMsg = content;
            if length(errorMsg) > 200
                errorMsg = [errorMsg(1:200) '...'];
            end
            isComplete = true;
            return;
        end

        % Parse output
        lines = strsplit(content, {'\n', '\r\n', '\r'}, 'CollapseDelimiters', false);
        lines = lines(~cellfun(@isempty, lines));

        % Check for incomplete writes (detection count > 0 but no valid detections parsed)
        if ~isempty(lines)
            numDetections = str2double(lines{1});
            if numDetections > 0 && length(lines) < numDetections + 1
                % Incomplete write - Python still writing output
                return;
            end
        end

        [quads, confidences] = parseDetectionOutput(lines, imageHeight, imageWidth);

        % Additional check: if count > 0 but parsing returned nothing, might be incomplete
        if ~isempty(lines)
            numDetections = str2double(lines{1});
            if numDetections > 0 && isempty(quads)
                return;
            end
        end

        % Successfully parsed - mark complete
        isComplete = true;

    catch ME
        % Error reading or parsing - consider complete but failed
        isComplete = true;
        errorMsg = sprintf('Failed to parse detection output: %s', ME.message);
        warning('cut_micropads:detection_parse_error', '%s', errorMsg);
    end
end

function [quads, confidences] = parseDetectionOutput(lines, imageHeight, imageWidth)
    % Parse YOLO detection output format into polygon arrays
    %
    % Inputs:
    %   lines - cell array of text lines (first line = count, rest = detections)
    %   imageHeight - image height for bounds validation
    %   imageWidth - image width for bounds validation
    %
    % Outputs:
    %   quads - detected quadrilaterals [N x 4 x 2]
    %   confidences - detection confidence scores [N x 1]
    %
    % Format: Each detection line contains 9 space-separated values:
    %   x1 y1 x2 y2 x3 y3 x4 y4 confidence
    %   (0-based coordinates from Python, converted to 1-based for MATLAB)

    quads = [];
    confidences = [];

    if isempty(lines)
        return;
    end

    numDetections = str2double(lines{1});

    if numDetections == 0 || isnan(numDetections)
        return;
    end

    quads = zeros(numDetections, 4, 2);
    confidences = zeros(numDetections, 1);

    for i = 1:numDetections
        if i+1 > length(lines)
            break;
        end

        parts = str2double(split(lines{i+1}));
        if length(parts) < 9 || any(isnan(parts)) || any(isinf(parts))
            warning('cut_micropads:invalid_detection', ...
                    'Skipping detection %d: invalid numeric data', i);
            continue;
        end

        % Parse: x1 y1 x2 y2 x3 y3 x4 y4 confidence (0-based from Python)
        % Convert to MATLAB 1-based indexing
        vertices = parts(1:8) + 1;

        % Validate vertices are within reasonable bounds (2× image size for rotations)
        if any(vertices < 0) || any(vertices > max([imageHeight, imageWidth]) * 2)
            warning('cut_micropads:out_of_bounds', ...
                    'Skipping detection %d: vertices out of bounds', i);
            continue;
        end

        quad = reshape(vertices, 2, 4)';  % Reshape to 4x2 matrix
        quads(i, :, :) = quad;
        confidences(i) = parts(9);
    end

    % Filter out empty detections (where confidence = 0)
    validMask = confidences > 0;
    quads = quads(validMask, :, :);
    confidences = confidences(validMask);

    % NOTE: Polygon ordering is handled by sortPolygonArrayByX in caller
    % (adaptive orientation-aware sorting for horizontal vs vertical strips)
end

function safeStopTimer(timerObj)
    % Safely stop and delete timer without generating warnings
    %
    % Input:
    %   timerObj - timer object to stop and delete
    %
    % Behavior:
    %   - Checks if timer exists and is valid
    %   - Only calls stop() if timer is currently running
    %   - Always calls delete() to free resources

    if ~isempty(timerObj) && isvalid(timerObj)
        if strcmp(timerObj.Running, 'on')
            stop(timerObj);
        end
        delete(timerObj);
    end
end

%% -------------------------------------------------------------------------
%% Memory System
%% -------------------------------------------------------------------------

function memory = initializeMemory()
    % Initialize empty memory structure
    memory = struct();
    memory.hasSettings = false;
    memory.displayPolygons = [];  % Exact display coordinates (preserves quadrilateral shapes)
    memory.rotation = 0;          % Image rotation angle
    memory.displayImageSize = [];
    memory.baseImageSize = [];
    memory.ellipses = {};         % Cell array indexed by concentration (0-based)
    memory.hasEllipseSettings = false;  % Boolean flag indicating if ellipse settings are saved
    memory.orientation = 'horizontal';  % Layout orientation: 'horizontal' or 'vertical'
end

function memory = updateMemory(memory, displayPolygons, rotation, baseImageSize, displayImageSize, ellipseData, cfg, orientation)
    % Update memory with exact display polygon coordinates and rotation
    % These preserve the exact quadrilateral shapes and rotation as seen by the user
    memory.hasSettings = true;
    memory.displayPolygons = displayPolygons;
    memory.rotation = rotation;
    memory.displayImageSize = displayImageSize;
    memory.baseImageSize = baseImageSize;

    % Store orientation if provided
    if nargin >= 8 && ~isempty(orientation)
        memory.orientation = orientation;
    end

    % Store polygon params for ellipse scaling
    if ~isempty(displayPolygons)
        memory.polygons = displayPolygons;
    end

    % Store ellipse data if provided
    if nargin >= 7 && ~isempty(ellipseData)
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

function [initialPolygons, rotation, source] = getInitialPolygonsWithMemory(img, cfg, memory, imageSize)
    % Get initial polygons and rotation with progressive AI detection workflow
    % Priority: memory (if available) -> default -> AI updates later

    % Check memory FIRST (even when AI is enabled)
    if memory.hasSettings && ~isempty(memory.displayPolygons) && ~isempty(memory.displayImageSize)
        % Use exact display polygons and rotation from memory
        scaledPolygons = scalePolygonsForImageSize(memory.displayPolygons, memory.displayImageSize, imageSize, cfg.numSquares);

        % If polygon count mismatch, fall back to default geometry
        if isempty(scaledPolygons)
            fprintf('  Memory polygon count mismatch - using default geometry\n');
            [imageHeight, imageWidth, ~] = size(img);
            initialPolygons = calculateDefaultPolygons(imageWidth, imageHeight, cfg);
            rotation = 0;
            source = 'default';
            return;
        end

        initialPolygons = scaledPolygons;
        rotation = memory.rotation;
        fprintf('  Using exact polygon shapes and rotation from memory (AI will update if enabled)\n');
        source = 'memory';
        return;
    end

    % No memory available: use default geometry for immediate display
    [imageHeight, imageWidth, ~] = size(img);
    fprintf('  Using default geometry (AI will update if enabled)\n');
    initialPolygons = calculateDefaultPolygons(imageWidth, imageHeight, cfg);
    rotation = 0;
    source = 'default';

    % NOTE: AI detection will run asynchronously after GUI displays
end

function scaledPolygons = scalePolygonsForImageSize(polygons, oldSize, newSize, expectedCount)
    % Scale polygon coordinates when image dimensions change
    %
    % Inputs:
    %   polygons - [N x 4 x 2] array of polygon vertices
    %   oldSize - [height, width] of previous image
    %   newSize - [height, width] of current image
    %   expectedCount - expected number of polygons (optional)
    %
    % Returns empty if polygon count doesn't match expectedCount

    if isempty(oldSize) || any(oldSize <= 0) || isempty(newSize) || any(newSize <= 0)
        error('cut_micropads:invalid_dimensions', ...
            'Cannot scale polygons: invalid dimensions [%d %d] -> [%d %d]', ...
            oldSize(1), oldSize(2), newSize(1), newSize(2));
    end

    % Validate polygon count if expectedCount is provided
    numPolygons = size(polygons, 1);
    if nargin >= 4 && ~isempty(expectedCount) && numPolygons ~= expectedCount
        scaledPolygons = [];
        return;
    end

    oldHeight = oldSize(1);
    oldWidth = oldSize(2);
    newHeight = newSize(1);
    newWidth = newSize(2);

    if oldHeight == newHeight && oldWidth == newWidth
        scaledPolygons = polygons;
        return;
    end

    scaleX = newWidth / oldWidth;
    scaleY = newHeight / oldHeight;

    scaledPolygons = zeros(size(polygons));

    for i = 1:numPolygons
        poly = squeeze(polygons(i, :, :));
        poly(:, 1) = poly(:, 1) * scaleX;
        poly(:, 2) = poly(:, 2) * scaleY;
        poly(:, 1) = max(1, min(poly(:, 1), newWidth));
        poly(:, 2) = max(1, min(poly(:, 2), newHeight));
        scaledPolygons(i, :, :) = poly;
    end
end

function scaledEllipses = scaleEllipsesForPolygonChange(oldCorners, newCorners, oldEllipses, cfg, imageSize)
    % Scale ellipse positions when polygon geometry changes between images
    % Preserves relative positions within polygon bounds
    %
    % Inputs:
    %   oldCorners - 4x2 matrix of old polygon vertices
    %   newCorners - 4x2 matrix of new polygon vertices
    %   oldEllipses - Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
    %
    % Outputs:
    %   scaledEllipses - Nx7 matrix of scaled ellipse parameters (same format as input)

    if isempty(oldEllipses)
        scaledEllipses = oldEllipses;
        return;
    end

    % Calculate old polygon centroid and dimensions
    oldCentroid = mean(oldCorners, 1);
    oldMinX = min(oldCorners(:, 1));
    oldMaxX = max(oldCorners(:, 1));
    oldMinY = min(oldCorners(:, 2));
    oldMaxY = max(oldCorners(:, 2));
    oldWidth = oldMaxX - oldMinX;
    oldHeight = oldMaxY - oldMinY;

    % Calculate new polygon centroid and dimensions
    newCentroid = mean(newCorners, 1);
    newMinX = min(newCorners(:, 1));
    newMaxX = max(newCorners(:, 1));
    newMinY = min(newCorners(:, 2));
    newMaxY = max(newCorners(:, 2));
    newWidth = newMaxX - newMinX;
    newHeight = newMaxY - newMinY;

    % Validate polygon dimensions (prevent division by zero)
    if oldWidth <= 0 || oldHeight <= 0 || newWidth <= 0 || newHeight <= 0
        warning('scaleEllipsesForPolygonChange:DegeneratePolygon', ...
                'Polygon has zero or negative dimensions - returning empty ellipses');
        scaledEllipses = [];
        return;
    end

    % Compute uniform scale factor using geometric mean
    scaleX = newWidth / oldWidth;
    scaleY = newHeight / oldHeight;
    axisScale = sqrt(scaleX * scaleY);

    % Scale ellipse parameters
    numEllipses = size(oldEllipses, 1);
    scaledEllipses = zeros(size(oldEllipses));

    bounds = computeEllipseAxisBounds(newCorners, imageSize, cfg);

    for i = 1:numEllipses
        % Extract from Nx7 format: [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
        concIdx = oldEllipses(i, 1);
        repIdx = oldEllipses(i, 2);
        oldX = oldEllipses(i, 3);
        oldY = oldEllipses(i, 4);
        oldSemiMajor = oldEllipses(i, 5);
        oldSemiMinor = oldEllipses(i, 6);
        oldRotation = oldEllipses(i, 7);

        % Transform centers relative to polygon centroid using uniform scaling
        newX = (oldX - oldCentroid(1)) * axisScale + newCentroid(1);
        newY = (oldY - oldCentroid(2)) * axisScale + newCentroid(2);

        % Scale axes uniformly
        newSemiMajor = oldSemiMajor * axisScale;
        newSemiMinor = oldSemiMinor * axisScale;

        % Enforce constraint: semiMajor >= semiMinor
        if newSemiMinor > newSemiMajor
            temp = newSemiMajor;
            newSemiMajor = newSemiMinor;
            newSemiMinor = temp;
            newRotation = mod(oldRotation + 90, 360);
        else
            newRotation = oldRotation;
        end

        % Enforce configured limits
        [newSemiMajor, newSemiMinor, newRotation] = enforceEllipseAxisLimits(newSemiMajor, newSemiMinor, newRotation, bounds);

        % Preserve concIdx and repIdx in output (Nx7 format)
        scaledEllipses(i, :) = [concIdx, repIdx, newX, newY, newSemiMajor, newSemiMinor, newRotation];
    end
end

function runDeferredAIDetection(fig, cfg)
    % Run AI detection asynchronously and update polygons when complete
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
        [~, ~, outputFile, imgPath] = detectQuadsYOLO(img, cfg, 'async', true);

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
    % Poll for async detection completion and update polygons when ready
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
    [isComplete, quads, confidences, errorMsg] = checkDetectionComplete(...
        guiData.asyncDetection.outputFile, ...
        guiData.baseImg);

    if ~isComplete
        return;  % Still running, keep polling
    end

    % Check for error messages
    if ~isempty(errorMsg)
        fprintf('  AI detection failed: %s\n', errorMsg);
    end

    % Detection finished - update polygons if successful
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
            [newPolygons, ~] = rotatePolygonsDiscrete(quads, guiData.baseImageSize, guiData.totalRotation);
            
            % Sort by display-frame X coordinate for consistent labeling
            [newPolygons, newOrientation] = sortPolygonArrayByX(newPolygons);
            guiData.orientation = newOrientation;  % Update orientation based on AI detection

            % Apply detected polygons with race condition guard
            try
                guiData = applyDetectedPolygons(guiData, newPolygons, cfg, fig);
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

    % Re-check figure validity after polygon update (race condition guard)
    if ~ishandle(fig) || ~isvalid(fig)
        return;
    end

    % Clear cached zoom bounds after detection (with race condition guard)
    try
        if isvalid(fig)
            % Force recalculation of zoom bounds from current polygon state
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
    %   success - true if detection succeeded and polygons updated
    %   cfg - configuration struct (for auto-zoom)

    % Stop and delete polling timer
    if ~isempty(guiData.asyncDetection.pollingTimer)
        safeStopTimer(guiData.asyncDetection.pollingTimer);
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

function updatePolygonPositions(polygonHandles, newPositions, labelHandles)
    % Update drawpolygon positions smoothly without recreating objects
    %
    % Inputs:
    %   polygonHandles - cell array of drawpolygon objects
    %   newPositions - [N x 4 x 2] array of new polygon positions
    %   labelHandles - cell array of text objects (optional)

    n = numel(polygonHandles);
    if size(newPositions, 1) ~= n
        warning('Polygon count mismatch: %d handles vs %d positions', n, size(newPositions, 1));
        return;
    end

    for i = 1:n
        poly = polygonHandles{i};
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

    % NEW: Update labels after polygon positions change
    if nargin >= 3 && ~isempty(labelHandles)
        updatePolygonLabels(polygonHandles, labelHandles);
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

function cleanupTempFile(tmpPath)
    % Helper to clean up temporary detection image file
    if isfile(tmpPath)
        try
            delete(tmpPath);
        catch
            % Silently ignore cleanup errors
        end
    end
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
        % Direct timer fields
        timerFields = {'aiTimer', 'aiBreathingTimer'};
        for i = 1:numel(timerFields)
            if isfield(guiData, timerFields{i})
                safeStopTimer(guiData.(timerFields{i}));
            end
        end

        % Async detection polling timer (nested field)
        if isfield(guiData, 'asyncDetection') && isstruct(guiData.asyncDetection) && ...
           isfield(guiData.asyncDetection, 'pollingTimer')
            safeStopTimer(guiData.asyncDetection.pollingTimer);
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
