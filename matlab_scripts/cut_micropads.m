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
    %   - AI-powered polygon detection (YOLOv11n-seg)
    %   - Manual polygon editing and refinement
    %   - Saves polygon coordinates with rotation angle
    %
    % Inputs (Name-Value pairs):
    % - 'numSquares': number of regions to capture per strip (default: 7)
    % - 'aspectRatio': width/height ratio of each region (default: 1.0, perfect squares)
    % - 'coverage': fraction of image width to fill (default: 0.80)
    % - 'gapPercent': gap as percent of region width, 0..1 or 0..100 (default: 0.19)
    % - 'inputFolder' | 'outputFolder': override default I/O folders
    % - 'preserveFormat' | 'jpegQuality' | 'saveCoordinates': output behavior
    % - 'useAIDetection': use YOLO for initial polygon placement (default: true)
    % - 'detectionModel': path to YOLOv11 model (default: 'models/yolo11m_micropad_seg.pt')
    % - 'minConfidence': minimum detection confidence (default: 0.6)
    % - 'inferenceSize': YOLO inference image size in pixels (default: 1280)
    % - 'pythonPath': path to Python executable (default: '' - uses MICROPAD_PYTHON env var)
    %
    % Outputs/Side effects:
    % - Writes polygon crops to 2_micropads/[phone]/con_*/
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

    % === DATASET AND FOLDER STRUCTURE ===
    INPUT_FOLDER = '1_dataset';
    OUTPUT_FOLDER = '2_micropads';

    % === OUTPUT FORMATTING ===
    PRESERVE_FORMAT = true;
    JPEG_QUALITY = 100;
    SAVE_COORDINATES = true;

    % === DEFAULT GEOMETRY / SELECTION ===
    DEFAULT_NUM_SQUARES = 7;
    DEFAULT_ASPECT_RATIO = 1.0;  % width/height ratio: 1.0 = perfect squares
    DEFAULT_COVERAGE = 0.80;     % regions span 80% of image width
    DEFAULT_GAP_PERCENT = 0.19;  % 19% gap between regions

    % === AI DETECTION DEFAULTS ===
    DEFAULT_USE_AI_DETECTION = false;
    DEFAULT_DETECTION_MODEL = 'models/yolo11m_micropad_seg.pt';
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
    UI_CONST.dimFactor = 0.3;
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

    %% Build configuration
    cfg = createConfiguration(INPUT_FOLDER, OUTPUT_FOLDER, PRESERVE_FORMAT, JPEG_QUALITY, SAVE_COORDINATES, ...
                              DEFAULT_NUM_SQUARES, DEFAULT_ASPECT_RATIO, DEFAULT_COVERAGE, DEFAULT_GAP_PERCENT, ...
                              DEFAULT_USE_AI_DETECTION, DEFAULT_DETECTION_MODEL, DEFAULT_MIN_CONFIDENCE, DEFAULT_PYTHON_PATH, DEFAULT_INFERENCE_SIZE, ...
                              ROTATION_ANGLE_TOLERANCE, ...
                              COORDINATE_FILENAME, SUPPORTED_FORMATS, ALLOWED_IMAGE_EXTENSIONS, CONC_FOLDER_PREFIX, UI_CONST, varargin{:});

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

function cfg = createConfiguration(inputFolder, outputFolder, preserveFormat, jpegQuality, saveCoordinates, ...
                                   defaultNumSquares, defaultAspectRatio, defaultCoverage, defaultGapPercent, ...
                                   defaultUseAI, defaultDetectionModel, defaultMinConfidence, defaultPythonPath, defaultInferenceSize, ...
                                   rotationAngleTolerance, ...
                                   coordinateFileName, supportedFormats, allowedImageExtensions, concFolderPrefix, UI_CONST, varargin)
    parser = inputParser;
    parser.addParameter('numSquares', defaultNumSquares, @(x) validateattributes(x, {'numeric'}, {'scalar','integer','>=',1,'<=',20}));

    validateFolder = @(s) validateattributes(s, {'char', 'string'}, {'nonempty', 'scalartext'});
    parser.addParameter('inputFolder', inputFolder, validateFolder);
    parser.addParameter('outputFolder', outputFolder, validateFolder);
    parser.addParameter('preserveFormat', preserveFormat, @(x) islogical(x));
    parser.addParameter('jpegQuality', jpegQuality, @(x) validateattributes(x, {'numeric'}, {'scalar','>=',1,'<=',100}));
    parser.addParameter('saveCoordinates', saveCoordinates, @(x) islogical(x));

    parser.addParameter('aspectRatio', defaultAspectRatio, @(x) validateattributes(x, {'numeric'}, {'scalar','>',0}));
    parser.addParameter('coverage', defaultCoverage, @(x) validateattributes(x, {'numeric'}, {'scalar','>',0,'<=',1}));
    parser.addParameter('gapPercent', defaultGapPercent, @(x) isnumeric(x) && isscalar(x) && x>=0);

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

    cfg = addPathConfiguration(cfg, parser.Results.inputFolder, parser.Results.outputFolder);

    cfg.output.preserveFormat = parser.Results.preserveFormat;
    cfg.output.jpegQuality = parser.Results.jpegQuality;
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

function cfg = addPathConfiguration(cfg, inputFolder, outputFolder)
    projectRoot = find_project_root(inputFolder);

    cfg.projectRoot = projectRoot;
    cfg.inputPath = fullfile(projectRoot, inputFolder);
    cfg.outputPath = fullfile(projectRoot, outputFolder);

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

    outputDir = createOutputDirectory(cfg.outputPath, phoneName, cfg.numSquares, cfg.concFolderPrefix);

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
            [success, persistentFig, memory] = processOneImage(imageList{idx}, outputDir, cfg, persistentFig, phoneName, memory);
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

function [success, fig, memory] = processOneImage(imageName, outputDir, cfg, fig, phoneName, memory)
    success = false;

    fprintf('  -> Processing: %s\n', imageName);

    [img, isValid] = loadImage(imageName);
    if ~isValid
        fprintf('  !! Failed to load image\n');
        return;
    end

    % Get initial polygon positions and rotation (memory or default, NOT AI yet)
    [imageHeight, imageWidth, ~] = size(img);
    [initialPolygons, initialRotation, ~] = getInitialPolygonsWithMemory(img, cfg, memory, [imageHeight, imageWidth]);

    % Memory polygons are exact display coordinates - use them directly
    initialPolygons = sortPolygonArrayByX(initialPolygons);

    % Display GUI immediately with memory/default polygons and rotation
    [polygonParams, displayPolygons, fig, rotation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig, initialRotation);

    % NOTE: If AI detection is enabled, it will run asynchronously AFTER GUI is displayed
    % (Implementation in Phase 2.3)

    if ~isempty(polygonParams)
        saveCroppedRegions(img, imageName, polygonParams, outputDir, cfg, rotation);
        % Update memory with exact display polygon shapes and rotation
        memory = updateMemory(memory, displayPolygons, rotation, [imageHeight, imageWidth]);
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

function [polygonParams, displayPolygons, fig, rotation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig, initialRotation)
    % Show interactive GUI with editing and preview modes
    polygonParams = [];
    displayPolygons = [];
    rotation = 0;

    % Create figure if needed
    if isempty(fig) || ~isvalid(fig)
        fig = createFigure(imageName, phoneName, cfg);
    end

    % Use rotation from memory (or 0 if no memory)
    if nargin < 7
        initialRotation = 0;
    end

    while true
        % Editing mode
        clearAndRebuildUI(fig, 'editing', img, imageName, phoneName, cfg, initialPolygons, initialRotation);

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
                % Store rotation before preview mode
                savedRotation = userRotation;
                savedDisplayPolygons = userPolygons;
                savedBasePolygons = basePolygons;

                % Preview mode
                clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, savedBasePolygons);

                % Store rotation in guiData for preview mode
                guiData = get(fig, 'UserData');
                guiData.savedRotation = savedRotation;
                guiData.savedPolygonParams = savedBasePolygons;
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

function clearAndRebuildUI(fig, mode, img, imageName, phoneName, cfg, polygonParams, initialRotation)
    % Modes: 'editing' (interactive polygon adjustment), 'preview' (final confirmation)

    if nargin < 8
        initialRotation = 0;
    end

    guiData = get(fig, 'UserData');
    clearAllUIElements(fig, guiData);

    switch mode
        case 'editing'
            buildEditingUI(fig, img, imageName, phoneName, cfg, polygonParams, initialRotation);

        case 'preview'
            buildPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams);
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
        end
        if isfield(guiData, 'aiBreathingTimer')
            stopAIBreathingTimer(guiData);
        end
    end

    set(fig, 'UserData', []);
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
    guiData.polygons = assignPolygonLabels(guiData.polygons);

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

    guiData = struct();
    guiData.mode = 'preview';
    guiData.savedPolygonParams = polygonParams;

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
    [guiData.leftAxes, guiData.rightAxes, guiData.leftImgHandle, guiData.rightImgHandle] = createPreviewAxes(fig, img, polygonParams, cfg);

    % Bottom controls
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.buttonPanel = createPreviewButtons(fig, cfg);

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
        labelUpdateListener = addlistener(polygons{i}, 'ROIMoved', @(~,~) updatePolygonLabelsCallback(polygons{i}));
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
    newPolygons = sortPolygonArrayByX(newPolygons);
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
    [guiData.polygons, order] = assignPolygonLabels(guiData.polygons);

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

function [leftAxes, rightAxes, leftImgHandle, rightImgHandle] = createPreviewAxes(fig, img, polygonParams, cfg)
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
    hold(leftAxes, 'off');

    % Right: highlighted regions
    rightAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewRight);
    maskedImg = createMaskedPreview(img, polygonParams, cfg);
    rightImgHandle = imshow(maskedImg, 'Parent', rightAxes, 'InitialMagnification', 'fit');
    axis(rightAxes, 'image');
    axis(rightAxes, 'tight');
    title(rightAxes, 'Masked Preview', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold');
end

function maskedImg = createMaskedPreview(img, polygonParams, cfg)
    [height, width, ~] = size(img);
    totalMask = false(height, width);

    numRegions = size(polygonParams, 1);
    for i = 1:numRegions
        poly = squeeze(polygonParams(i,:,:));
        if size(poly, 1) >= 3
            regionMask = poly2mask(poly(:,1), poly(:,2), height, width);
            totalMask = totalMask | regionMask;
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
    [guiData.polygons, order] = assignPolygonLabels(guiData.polygons);

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
            basePolygons = scalePolygonsForImageSize(basePolygons, newSize, targetSize);
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

function tf = isMultipleOfNinety(angle, tolerance)
    % Determine if an angle is effectively a multiple of 90 degrees
    if isnan(angle) || isinf(angle)
        tf = false;
        return;
    end
    tf = abs(angle / 90 - round(angle / 90)) <= tolerance;
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

function alignedQuads = normalizePolygonsForDisplay(detectedQuads, totalRotation, imgSize, baseImgSize, angleTolerance)
    % Convert YOLO detections into axis-aligned horizontal rectangles for display
    %
    % Inputs:
    %   detectedQuads - YOLO output [N x 4 x 2] in currentImg (rotated frame)
    %   totalRotation - cumulative rotation in degrees
    %   imgSize - [height, width] of currentImg
    %   baseImgSize - [height, width] of base image
    %   angleTolerance - tolerance for rotation angle comparisons
    %
    % Outputs:
    %   alignedQuads - [N x 4 x 2] axis-aligned horizontal rectangles

    if isempty(detectedQuads)
        alignedQuads = detectedQuads;
        return;
    end

    numQuads = size(detectedQuads, 1);
    alignedQuads = zeros(size(detectedQuads));

    % Step 1: Compute axis-aligned bounding boxes directly in currentImg frame
    for i = 1:numQuads
        quad = squeeze(detectedQuads(i, :, :));  % [4 x 2] vertices
        quadX = quad(:, 1);
        quadY = quad(:, 2);

        xMin = min(quadX);
        xMax = max(quadX);
        yMin = min(quadY);
        yMax = max(quadY);

        % Build horizontal rectangle (clockwise from TL)
        alignedQuads(i, :, :) = [xMin, yMin; xMax, yMin; xMax, yMax; xMin, yMax];
    end

    % Step 2: Sort by base-frame X coordinate for stable left-to-right labeling
    centroids = squeeze(mean(alignedQuads, 2));  % [N x 2]
    baseCentroids = inverseRotatePoints(centroids, imgSize, baseImgSize, totalRotation, angleTolerance);
    [~, sortIdx] = sort(baseCentroids(:, 1));  % Sort by base-frame X
    alignedQuads = alignedQuads(sortIdx, :, :);
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

function updatePolygonLabelsCallback(polygon, ~)
    % Callback for ROIMoved event to update ALL labels
    % Note: Updates all labels for simplicity (performance impact negligible for ~7 polygons)
    fig = ancestor(polygon, 'figure');
    if isempty(fig)
        return;
    end
    guiData = get(fig, 'UserData');
    if isfield(guiData, 'polygons') && isfield(guiData, 'polygonLabels')
        updatePolygonLabels(guiData.polygons, guiData.polygonLabels);
    end
end

function basePoints = inverseRotatePoints(rotatedPoints, ~, baseSize, rotation, angleTolerance)
    % Transform points from rotated image frame back to base image frame
    %
    % Inputs:
    %   rotatedPoints - [N x 2] points in rotated frame
    %   baseSize - [height, width] of base image
    %   rotation - rotation angle in degrees (cumulative)
    %   angleTolerance - tolerance for detecting exact 90-degree rotations
    %
    % Outputs:
    %   basePoints - [N x 2] points in base image frame

    if isempty(rotatedPoints)
        basePoints = rotatedPoints;
        return;
    end

    % Only handle multiples of 90 degrees
    if ~isMultipleOfNinety(rotation, angleTolerance)
        basePoints = rotatedPoints;
        return;
    end

    k = mod(round(rotation / 90), 4);
    if k == 0
        basePoints = rotatedPoints;
        return;
    end

    H_base = baseSize(1);
    W_base = baseSize(2);

    basePoints = zeros(size(rotatedPoints));

    % Inverse transformations (opposite of rotatePolygonsDiscrete)
    switch k
        case 1  % Rotated 90 CW, so inverse is 90 CCW (270 CW)
            % Original: x' = H - y + 1, y' = x
            % Inverse: x = y', y = H - x' + 1
            basePoints(:, 1) = rotatedPoints(:, 2);
            basePoints(:, 2) = H_base - rotatedPoints(:, 1) + 1;

        case 2  % Rotated 180, so inverse is also 180
            % Original: x' = W - x + 1, y' = H - y + 1
            % Inverse: x = W - x' + 1, y = H - y' + 1
            basePoints(:, 1) = W_base - rotatedPoints(:, 1) + 1;
            basePoints(:, 2) = H_base - rotatedPoints(:, 2) + 1;

        case 3  % Rotated 270 CW (90 CCW), so inverse is 90 CW
            % Original: x' = y, y' = W - x + 1
            % Inverse: x = W - y' + 1, y = x'
            basePoints(:, 1) = W_base - rotatedPoints(:, 2) + 1;
            basePoints(:, 2) = rotatedPoints(:, 1);
    end
end

function [polygons, order] = assignPolygonLabels(polygons)
    % Reorder drawpolygon handles so con0..conN map left-to-right, bottom-to-top
    % con_0 (blue) at BOTTOM (large Y), con_max (red) at TOP (small Y)
    order = [];

    if isempty(polygons) || ~iscell(polygons)
        return;
    end

    numPolygons = numel(polygons);
    centroids = zeros(numPolygons, 2);
    validMask = false(numPolygons, 1);

    for i = 1:numPolygons
        if isvalid(polygons{i})
            pos = polygons{i}.Position;
            centroids(i, 1) = mean(pos(:, 1));
            centroids(i, 2) = mean(pos(:, 2));
            validMask(i) = true;
        else
            centroids(i, :) = inf;
        end
    end

    if ~any(validMask)
        return;
    end

    % Sort by Y DESCENDING (bottom→top, primary), then X ascending (left→right, secondary)
    % This puts bottom row first (large Y), sorted left-to-right
    sortKey = [-centroids(:, 2), centroids(:, 1)];
    [~, order] = sortrows(sortKey);
    polygons = polygons(order);
end

function sortedPolygons = sortPolygonArrayByX(polygons)
    % Sort numeric polygon array by centroid: left→right, bottom→top in UI window
    % con_0 (blue) at BOTTOM (large Y), con_max (red) at TOP (small Y)
    sortedPolygons = polygons;
    if isempty(polygons)
        return;
    end
    if ndims(polygons) ~= 3
        return;
    end

    numPolygons = size(polygons, 1);
    centroids = zeros(numPolygons, 2);

    if numPolygons == 0
        sortedPolygons = polygons;
        return;
    end

    for i = 1:numPolygons
        poly = squeeze(polygons(i, :, :));
        if isempty(poly)
            centroids(i, :) = inf;
        else
            centroids(i, 1) = mean(poly(:, 1));
            centroids(i, 2) = mean(poly(:, 2));
        end
    end

    % Sort by Y DESCENDING (bottom→top, primary), then X ascending (left→right, secondary)
    % This puts bottom row first (large Y), sorted left-to-right
    sortKey = [-centroids(:, 2), centroids(:, 1)];
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

    % Avoid launching another job while one is already running
    if isfield(guiData, 'asyncDetection') && guiData.asyncDetection.active
        fprintf('  AI detection already running - ignoring manual rerun request\n');
        return;
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
                polygonParams = extractPolygonParameters(guiData);
                if isempty(polygonParams)
                    action = 'skip';
                else
                    rotation = guiData.totalRotation;
                end
            end
        end
    end
end

function polygonParams = extractPolygonParameters(guiData)
    polygonParams = [];
    if isfield(guiData, 'polygons') && iscell(guiData.polygons)
        numPolygons = numel(guiData.polygons);
        polygonParams = zeros(numPolygons, 4, 2);
        for i = 1:numPolygons
            if isvalid(guiData.polygons{i})
                polygonParams(i,:,:) = guiData.polygons{i}.Position;
            end
        end
    end
end

%% -------------------------------------------------------------------------
%% Image Cropping and Coordinate Saving
%% -------------------------------------------------------------------------

function saveCroppedRegions(img, imageName, polygons, outputDir, cfg, rotation)
    [~, baseName, ~] = fileparts(imageName);
    [~, extOrig] = fileparts(imageName);
    outExt = determineOutputExtension(extOrig, cfg.output.supportedFormats, cfg.output.preserveFormat);

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

function croppedImg = cropImageWithPolygon(img, polygonVertices)
    x = polygonVertices(:, 1);
    y = polygonVertices(:, 2);

    minX = floor(min(x));
    maxX = ceil(max(x));
    minY = floor(min(y));
    maxY = ceil(max(y));

    [imgH, imgW, ~] = size(img);
    minX = max(1, minX);
    maxX = min(imgW, maxX);
    minY = max(1, minY);
    maxY = min(imgH, maxY);

    croppedImg = img(minY:maxY, minX:maxX, :);
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
                     'Delete the corrupted file and rerun the stage to regenerate.'], ...
                    size(nums, 2), numericCount, coordPath);
            end

            existingNums = nums;
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

function saveImageWithFormat(img, outPath, outExt, cfg)
    if strcmpi(outExt, '.jpg') || strcmpi(outExt, '.jpeg')
        imwrite(img, outPath, 'jpg', 'Quality', cfg.output.jpegQuality);
    else
        imwrite(img, outPath);
    end
end

function outExt = determineOutputExtension(extOrig, supported, preserveFormat)
    if preserveFormat && any(strcmpi(extOrig, supported))
        outExt = lower(extOrig);
    else
        outExt = '.jpg';
    end
end

function outputDir = createOutputDirectory(basePath, phoneName, numConcentrations, concFolderPrefix)
    phoneOutputDir = fullfile(basePath, phoneName);
    if ~isfolder(phoneOutputDir)
        mkdir(phoneOutputDir);
    end

    for i = 0:(numConcentrations - 1)
        concFolder = sprintf('%s%d', concFolderPrefix, i);
        concPath = fullfile(phoneOutputDir, concFolder);
        if ~isfolder(concPath)
            mkdir(concPath);
        end
    end

    outputDir = phoneOutputDir;
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
        % Check environment variable first
        envPath = getenv('MICROPAD_PYTHON');
        if ~isempty(envPath)
            pythonPath = envPath;
        end

        % Validate Python path is provided
        pythonPath = char(pythonPath);
        if isempty(pythonPath)
            error('cut_micropads:python_not_configured', ...
                ['Python path not configured! Options:\n', ...
                 '  1. Set MICROPAD_PYTHON environment variable\n', ...
                 '  2. Pass pythonPath parameter: cut_micropads(''pythonPath'', ''path/to/python'')\n', ...
                 '  3. Edit DEFAULT_PYTHON_PATH in script (line 79)']);
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
    tmpImgPath = fullfile(tmpDir, sprintf('%s_micropad_detect.jpg', tmpName));
    imwrite(img, tmpImgPath, 'JPEG', 'Quality', 95);

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

    % Sort by centroid X coordinate for consistent left-to-right ordering
    if ~isempty(quads)
        centroids = squeeze(mean(quads, 2));
        if isvector(centroids)
            centroids = centroids(:).';
        end
        [~, order] = sort(centroids(:, 1), 'ascend');
        quads = quads(order, :, :);
        confidences = confidences(order);
    end
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
    memory.imageSize = [];
end

function memory = updateMemory(memory, displayPolygons, rotation, imageSize)
    % Update memory with exact display polygon coordinates and rotation
    % These preserve the exact quadrilateral shapes and rotation as seen by the user
    memory.hasSettings = true;
    memory.displayPolygons = displayPolygons;
    memory.rotation = rotation;
    memory.imageSize = imageSize;
end

function [initialPolygons, rotation, source] = getInitialPolygonsWithMemory(img, cfg, memory, imageSize)
    % Get initial polygons and rotation with progressive AI detection workflow
    % Priority: memory (if available) -> default -> AI updates later

    % Check memory FIRST (even when AI is enabled)
    if memory.hasSettings && ~isempty(memory.displayPolygons) && ~isempty(memory.imageSize)
        % Use exact display polygons and rotation from memory
        scaledPolygons = scalePolygonsForImageSize(memory.displayPolygons, memory.imageSize, imageSize);
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

function scaledPolygons = scalePolygonsForImageSize(polygons, oldSize, newSize)
    % Scale polygon coordinates when image dimensions change
    if isempty(oldSize) || any(oldSize <= 0) || isempty(newSize) || any(newSize <= 0)
        scaledPolygons = polygons;
        warning('cut_micropads:invalid_image_size', ...
                'Invalid image dimensions for scaling: old=[%d %d], new=[%d %d]', ...
                oldSize(1), oldSize(2), newSize(1), newSize(2));
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

    numPolygons = size(polygons, 1);
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
        % Launch async detection
        img = guiData.currentImg;
        [~, ~, outputFile, imgPath] = detectQuadsYOLO(img, cfg, 'async', true);

        % Store detection state
        guiData.asyncDetection.active = true;
        guiData.asyncDetection.outputFile = outputFile;
        guiData.asyncDetection.imgPath = imgPath;
        guiData.asyncDetection.startTime = tic;

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
        guiData.asyncDetection.active = false;
        showAIProgressIndicator(fig, false);
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

    % Check if detection completed
    [isComplete, quads, confidences, errorMsg] = checkDetectionComplete(...
        guiData.asyncDetection.outputFile, ...
        guiData.currentImg);

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
            % Use top N detections by confidence
            [~, sortIdx] = sort(confidences, 'descend');
            quads = quads(sortIdx(1:cfg.numSquares), :, :);
            topConfidences = confidences(sortIdx(1:cfg.numSquares));

            % Normalize and apply polygons respecting UI ordering
            newPolygons = normalizePolygonsForDisplay(quads, guiData.totalRotation, ...
                                                      guiData.imageSize, guiData.baseImageSize, ...
                                                      cfg.rotation.angleTolerance);

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
        % Force recalculation of zoom bounds from current polygon state
        guiData.autoZoomBounds = [];

        % Save state to figure
        set(fig, 'UserData', guiData);

        % Re-fetch guiData to ensure cleanup gets fresh state (handles concurrent updates)
        guiData = get(fig, 'UserData');

        % Clean up async state
        cleanupAsyncDetection(fig, guiData, detectionSucceeded, cfg);
    catch ME
        % Figure was deleted during final state update - ignore error
        if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
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

    % Stop animation
    showAIProgressIndicator(fig, false);

    % Update guiData
    set(fig, 'UserData', guiData);

    % Apply auto-zoom if detection succeeded
    if success
        applyAutoZoom(fig, guiData, cfg);
    end
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

    % Clean up async detection if active
    if isstruct(guiData) && isfield(guiData, 'asyncDetection')
        if guiData.asyncDetection.active
            % Stop polling timer
            if ~isempty(guiData.asyncDetection.pollingTimer)
                safeStopTimer(guiData.asyncDetection.pollingTimer);
            end

            % Clean up temp files
            if ~isempty(guiData.asyncDetection.outputFile) && ...
               exist(guiData.asyncDetection.outputFile, 'file')
                try
                    delete(guiData.asyncDetection.outputFile);
                catch
                    % Ignore cleanup errors
                end
            end

            % Clean up temp image file
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

            % Update guiData after async cleanup
            set(fig, 'UserData', guiData);
        end
    end

    % Stop and delete AI timer if exists
    if isstruct(guiData) && isfield(guiData, 'aiTimer')
        safeStopTimer(guiData.aiTimer);
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
    if strcmp(get(fig, 'waitstatus'), 'waiting')
        uiresume(fig);
    end

    % Delete figure
    delete(fig);
end
