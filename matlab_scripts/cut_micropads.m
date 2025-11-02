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
    DEFAULT_USE_AI_DETECTION = true;
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
    UI_CONST.positions = struct(...
        'figure', [0 0 1 1], ...
        'stopButton', [0.01 0.945 0.06 0.045], ...
        'title', [0.08 0.945 0.84 0.045], ...
        'pathDisplay', [0.08 0.90 0.84 0.035], ...
        'instructions', [0.01 0.855 0.98 0.035], ...
        'image', [0.01 0.16 0.98 0.68], ...
        'rotationPanel', [0.01 0.01 0.24 0.14], ...
        'zoomPanel', [0.26 0.01 0.26 0.14], ...
        'cutButtonPanel', [0.53 0.01 0.46 0.14], ...
        'previewPanel', [0.25 0.01 0.50 0.14], ...
        'previewLeft', [0.01 0.16 0.48 0.73], ...
        'previewRight', [0.50 0.16 0.49 0.73]);
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

    % Get initial polygon positions (AI detection, memory, or default geometry)
    [imageHeight, imageWidth, ~] = size(img);
    [initialPolygons, initialSource] = getInitialPolygonsWithMemory(img, cfg, memory, [imageHeight, imageWidth]);
    useMemoryRotation = strcmp(initialSource, 'memory');

    if useMemoryRotation && memory.hasSettings
        rotationAngle = memory.rotation;
        if isMultipleOfNinety(rotationAngle, cfg.rotation.angleTolerance)
            [initialPolygons, ~] = rotatePolygonsDiscrete(initialPolygons, [imageHeight, imageWidth], rotationAngle);
        end
    end
    initialPolygons = sortPolygonArrayByX(initialPolygons);

    % Interactive region selection with persistent window
    [polygonParams, fig, rotation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig, memory, useMemoryRotation);

    if ~isempty(polygonParams)
        saveCroppedRegions(img, imageName, polygonParams, outputDir, cfg, rotation);
        % Update memory with current polygons and rotation
        memory = updateMemory(memory, polygonParams, rotation, [imageHeight, imageWidth]);
        success = true;
    end
end

function [polygons, detectionSucceeded] = getInitialPolygons(img, cfg)
    % Attempt AI detection for initial polygons
    polygons = [];
    detectionSucceeded = false;

    if ~cfg.useAIDetection
        return;
    end

    try
        [detectedQuads, confidences] = detectQuadsYOLO(img, cfg);

        if isempty(detectedQuads)
            fprintf('  No AI detections\n');
            return;
        end

        numDetected = size(detectedQuads, 1);

        if numDetected == cfg.numSquares
            fprintf('  AI detected %d regions (avg confidence: %.2f)\n', ...
                numDetected, mean(confidences));
            polygons = detectedQuads;
            detectionSucceeded = true;
            return;
        elseif numDetected > cfg.numSquares
            fprintf('  AI detected %d regions, filtering to top %d by confidence...\n', ...
                numDetected, cfg.numSquares);

            [~, sortIdx] = sort(confidences, 'descend');
            topIdx = sortIdx(1:cfg.numSquares);
            detectedQuads = detectedQuads(topIdx, :, :);
            confidences = confidences(topIdx);

            centroids = squeeze(mean(detectedQuads, 2));
            [~, spatialIdx] = sort(centroids(:, 1));
            detectedQuads = detectedQuads(spatialIdx, :, :);
            confidences = confidences(spatialIdx);

            fprintf('  Using top %d detections (avg confidence: %.2f)\n', ...
                cfg.numSquares, mean(confidences));
            polygons = detectedQuads;
            detectionSucceeded = true;
            return;
        else
            fprintf('  AI detected only %d/%d regions\n', numDetected, cfg.numSquares);
        end
    catch ME
        fprintf('  AI detection failed (%s)\n', ME.message);
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

function [polygonParams, fig, rotation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig, memory, useMemoryRotation)
    % Show interactive GUI with editing and preview modes
    polygonParams = [];
    rotation = 0;

    if nargin < 8
        useMemoryRotation = false;
    end

    % Create figure if needed
    if isempty(fig) || ~isvalid(fig)
        fig = createFigure(imageName, phoneName, cfg);
    end

    % Initialize rotation from memory if available
    initialRotation = 0;
    if useMemoryRotation && memory.hasSettings
        initialRotation = memory.rotation;
    end

    while true
        % Editing mode
        clearAndRebuildUI(fig, 'editing', img, imageName, phoneName, cfg, initialPolygons, initialRotation);

        [action, userPolygons, userRotation] = waitForUserAction(fig);

        switch action
            case 'skip'
                return;
            case 'stop'
                close(fig);
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

                switch prevAction
                    case 'accept'
                        polygonParams = savedBasePolygons;
                        rotation = savedRotation;
                        return;
                    case {'skip', 'stop'}
                        if strcmp(prevAction, 'stop')
                            close(fig);
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
            if isvalid(validPolys{i}) && isappdata(validPolys{i}, 'LastValidPosition')
                rmappdata(validPolys{i}, 'LastValidPosition');
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

    % Initialize rotation data (from memory or default to 0)
    guiData.baseImg = img;
    guiData.baseImageSize = [size(img, 1), size(img, 2)];
    guiData.currentImg = img;
    guiData.memoryRotation = initialRotation;
    guiData.adjustmentRotation = 0;
    guiData.totalRotation = initialRotation;

    % Initialize zoom state
    guiData.zoomLevel = 0;  % 0 = full image, 1 = single micropad size

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
    guiData.imgAxes = createImageAxes(fig, displayImg, cfg);

    % Create editable polygons
    guiData.polygons = createPolygons(initialPolygons, cfg);
    guiData.polygons = assignPolygonLabels(guiData.polygons);

    % Rotation panel (preset buttons only)
    guiData.rotationPanel = createRotationButtonPanel(fig, cfg);

    % Zoom panel
    [guiData.zoomSlider, guiData.zoomValue] = createZoomPanel(fig, cfg);

    % Buttons
    guiData.cutButtonPanel = createEditButtonPanel(fig, cfg);
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.instructionText = createInstructions(fig, cfg);

    guiData.action = '';

    % Store guiData before auto-zoom
    set(fig, 'UserData', guiData);

    % Auto-zoom to polygons after all UI is created
    guiData = get(fig, 'UserData');
    applyAutoZoom(fig, guiData, cfg);
end

function buildPreviewUI(fig, img, imageName, phoneName, cfg, polygonParams)
    % Build UI for preview mode
    set(fig, 'Name', sprintf('PREVIEW - %s - %s', phoneName, imageName));

    guiData = struct();
    guiData.mode = 'preview';
    guiData.savedPolygonParams = polygonParams;

    % Title and path
    titleText = sprintf('PREVIEW: %s - %s', phoneName, imageName);
    guiData.titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                                   'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                                   'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                                   'ForegroundColor', cfg.ui.colors.foreground, ...
                                   'BackgroundColor', cfg.ui.colors.background, ...
                                   'HorizontalAlignment', 'center');

    pathText = sprintf('PREVIEW - Path: %s | Image: %s', phoneName, imageName);
    guiData.pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                                  'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                                  'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                                  'ForegroundColor', cfg.ui.colors.path, ...
                                  'BackgroundColor', cfg.ui.colors.background, ...
                                  'HorizontalAlignment', 'center');

    % Preview axes
    [guiData.leftAxes, guiData.rightAxes] = createPreviewAxes(fig, img, polygonParams, cfg);

    % Buttons
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
                'Color', cfg.ui.colors.background, 'KeyPressFcn', @keyPressHandler);

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

function imgAxes = createImageAxes(fig, img, cfg)
    imgAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.image);
    imshow(img, 'Parent', imgAxes, 'InitialMagnification', 'fit');
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

function polygons = createPolygons(initialPolygons, cfg)
    % Create drawpolygon objects from initial positions
    n = size(initialPolygons, 1);
    polygons = cell(1, n);

    for i = 1:n
        pos = squeeze(initialPolygons(i, :, :));
        polygons{i} = drawpolygon('Position', pos, ...
                                 'Color', cfg.ui.colors.polygon, ...
                                 'LineWidth', cfg.ui.polygon.lineWidth, ...
                                 'MarkerSize', 8, ...
                                 'Selected', false);

        % Store initial valid position
        setappdata(polygons{i}, 'LastValidPosition', pos);

        % Add listener for quadrilateral enforcement
        addlistener(polygons{i}, 'ROIMoved', @(~,~) enforceQuadrilateral(polygons{i}));
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
    instructionString = 'Mouse = Drag Vertices | Buttons = Rotate | Slider = Zoom | APPLY = Save & Continue | SKIP = Skip | STOP = Exit | Space = APPLY | Esc = SKIP';

    instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', instructionString, ...
             'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
             'FontSize', cfg.ui.fontSize.instruction, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.background, 'HorizontalAlignment', 'center');
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

function [leftAxes, rightAxes] = createPreviewAxes(fig, img, polygonParams, cfg)
    % Left: original with overlays
    leftAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewLeft);
    imshow(img, 'Parent', leftAxes, 'InitialMagnification', 'fit');
    axis(leftAxes, 'image');
    axis(leftAxes, 'tight');
    title(leftAxes, sprintf('Original with %d Concentration Regions', size(polygonParams, 1)), ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold');
    hold(leftAxes, 'on');

    % Draw polygon overlays
    for i = 1:size(polygonParams, 1)
        poly = squeeze(polygonParams(i,:,:));
        if size(poly, 1) >= 3
            plot(leftAxes, [poly(:,1); poly(1,1)], [poly(:,2); poly(1,2)], ...
                 'Color', cfg.ui.colors.polygon, 'LineWidth', cfg.ui.polygon.lineWidth);

            centerX = mean(poly(:,1));
            centerY = mean(poly(:,2));
            text(leftAxes, centerX, centerY, sprintf('C%d', i-1), ...
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
    imshow(maskedImg, 'Parent', rightAxes, 'InitialMagnification', 'fit');
    axis(rightAxes, 'image');
    axis(rightAxes, 'tight');
    title(rightAxes, 'Highlighted Concentration Regions', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold');
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

    % Capture state before rotation changes
    oldRotation = guiData.totalRotation;
    oldImageSize = guiData.imageSize;
    if numel(oldImageSize) > 2
        oldImageSize = oldImageSize(1:2);
    end
    if isfield(guiData, 'baseImageSize') && ~isempty(guiData.baseImageSize)
        baseImageSize = guiData.baseImageSize;
    else
        baseImageSize = [size(guiData.baseImg, 1), size(guiData.baseImg, 2)];
    end

    % Save polygon positions BEFORE clearing axes
    savedPositions = extractPolygonPositions(guiData);

    % Update rotation state (quick buttons are absolute presets)
    guiData.adjustmentRotation = angle;
    guiData.memoryRotation = angle;
    guiData.totalRotation = angle;

    % Apply rotation to image
    guiData.currentImg = applyRotation(guiData.baseImg, guiData.totalRotation, cfg);
    guiData.imageSize = [size(guiData.currentImg, 1), size(guiData.currentImg, 2)];

    % Update image display
    axes(guiData.imgAxes);
    cla(guiData.imgAxes);
    imshow(guiData.currentImg, 'Parent', guiData.imgAxes, 'InitialMagnification', 'fit');
    axis(guiData.imgAxes, 'image');
    axis(guiData.imgAxes, 'tight');
    hold(guiData.imgAxes, 'on');

    % Prepare fallback polygons in case AI is disabled or fails
    fallbackPolygons = remapPolygonsForRotation(savedPositions, oldImageSize, baseImageSize, ...
        oldRotation, guiData.totalRotation, cfg.rotation.angleTolerance);
    fallbackPolygons = sortPolygonArrayByX(fallbackPolygons);

    % Re-run AI detection if enabled and recreate polygons
    if cfg.useAIDetection
        try
            [detectedQuads, confidences] = detectQuadsYOLO(guiData.currentImg, cfg);

            if ~isempty(detectedQuads)
                numDetected = size(detectedQuads, 1);

                if numDetected == cfg.numSquares
                    % Perfect match - use all detections
                    guiData.polygons = createPolygons(detectedQuads, cfg);
                    guiData.polygons = assignPolygonLabels(guiData.polygons);
                    fprintf('  AI re-detected %d regions after rotation (avg confidence: %.2f)\n', ...
                        numDetected, mean(confidences));

                elseif numDetected > cfg.numSquares
                    % Too many detections - keep top N by confidence
                    fprintf('  AI detected %d regions after rotation, filtering to top %d...\n', ...
                        numDetected, cfg.numSquares);

                    % Sort by confidence (descending) and keep top N
                    [~, sortIdx] = sort(confidences, 'descend');
                    topIdx = sortIdx(1:cfg.numSquares);
                    detectedQuads = detectedQuads(topIdx, :, :);
                    confidences = confidences(topIdx);

                    % Sort spatially (left to right)
                    centroids = squeeze(mean(detectedQuads, 2));
                    [~, spatialIdx] = sort(centroids(:, 1));
                    detectedQuads = detectedQuads(spatialIdx, :, :);

                    guiData.polygons = createPolygons(detectedQuads, cfg);
                    guiData.polygons = assignPolygonLabels(guiData.polygons);
                    fprintf('  Using top %d detections (avg confidence: %.2f)\n', ...
                        cfg.numSquares, mean(confidences));

                else
                    % Too few detections - use saved positions
                    fprintf('  AI detected only %d regions after rotation, using previous positions\n', numDetected);
                    guiData.polygons = createPolygons(fallbackPolygons, cfg);
                    guiData.polygons = assignPolygonLabels(guiData.polygons);
                end
            else
                % No detections - use saved positions
                guiData.polygons = createPolygons(fallbackPolygons, cfg);
                guiData.polygons = assignPolygonLabels(guiData.polygons);
            end
        catch ME
            fprintf('  AI detection after rotation failed: %s\n', ME.message);
            % Recreate polygons at their previous positions
            guiData.polygons = createPolygons(fallbackPolygons, cfg);
            guiData.polygons = assignPolygonLabels(guiData.polygons);
        end
    else
        % No AI detection - recreate polygons at their previous positions
        guiData.polygons = createPolygons(fallbackPolygons, cfg);
        guiData.polygons = assignPolygonLabels(guiData.polygons);
    end

    % Save guiData before auto-zoom
    set(fig, 'UserData', guiData);

    % Auto-zoom to polygons after rotation (will update guiData internally)
    guiData = get(fig, 'UserData');
    applyAutoZoom(fig, guiData, cfg);
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
    if ~strcmp(guiData.mode, 'editing')
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
    set(guiData.zoomSlider, 'Value', 1);
    set(guiData.zoomValue, 'String', '100%');

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
            % If no auto-zoom bounds calculated yet, use center single micropad estimate
            [autoZoomBounds] = estimateSingleMicropadBounds(guiData, cfg);
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

function positions = extractPolygonPositions(guiData)
    % Extract current polygon positions from valid polygon objects
    % Must be called BEFORE clearing axes to preserve positions

    if ~isfield(guiData, 'polygons') || isempty(guiData.polygons)
        positions = [];
        return;
    end

    numPolygons = numel(guiData.polygons);
    positions = zeros(numPolygons, 4, 2);

    for i = 1:numPolygons
        if isvalid(guiData.polygons{i})
            positions(i, :, :) = guiData.polygons{i}.Position;
        else
            warning('cut_micropads:invalid_polygon', 'Polygon %d is invalid before extraction', i);
        end
    end
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

function remappedPolygons = remapPolygonsForRotation(polygons, oldImageSize, baseImageSize, oldRotation, newRotation, tolerance)
    % Remap polygon coordinates when the displayed image rotation changes
    if isempty(polygons)
        remappedPolygons = polygons;
        return;
    end

    if nargin < 6 || isempty(tolerance)
        tolerance = 1e-6;
    end

    if isempty(oldImageSize)
        remappedPolygons = polygons;
        return;
    end

    oldImageSize = oldImageSize(1:2);
    baseImageSize = baseImageSize(1:2);

    if ~isMultipleOfNinety(oldRotation, tolerance) || ~isMultipleOfNinety(newRotation, tolerance)
        remappedPolygons = polygons;
        return;
    end

    [basePolygons, ~] = rotatePolygonsDiscrete(polygons, oldImageSize, -oldRotation);
    [remappedPolygons, ~] = rotatePolygonsDiscrete(basePolygons, baseImageSize, newRotation);
    remappedPolygons = sortPolygonArrayByX(remappedPolygons);
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

function polygons = assignPolygonLabels(polygons)
    % Reorder drawpolygon handles so con0..conN map left-to-right (ties by top-to-bottom)
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

    [~, order] = sortrows(centroids, [1 2]);
    polygons = polygons(order);
end

function sortedPolygons = sortPolygonArrayByX(polygons)
    % Sort numeric polygon array by centroid (X primary, Y secondary)
    sortedPolygons = polygons;
    if isempty(polygons)
        return;
    end
    if ndims(polygons) ~= 3
        return;
    end

    numPolygons = size(polygons, 1);
    centroids = zeros(numPolygons, 2);

    for i = 1:numPolygons
        poly = squeeze(polygons(i, :, :));
        if isempty(poly)
            centroids(i, :) = inf;
        else
            centroids(i, 1) = mean(poly(:, 1));
            centroids(i, 2) = mean(poly(:, 2));
        end
    end

    [~, order] = sortrows(centroids, [1 2]);
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
            if size(nums, 2) >= numericCount
                existingNums = nums(:, 1:numericCount);
            else
                pad = nan(size(nums, 1), numericCount - size(nums, 2));
                existingNums = [nums, pad];

                if size(nums, 2) == numericCount - 1
                    warning('cut_micropads:coord_migration', ...
                        'Migrating 9-column coordinates to 10-column format (rotation=0): %s', coordPath);
                    existingNums(:, end) = 0;
                end
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
        warning('cut_micropads:coord_open', 'Cannot open temp coordinates file for writing: %s', tmpPath);
        return;
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
    % Read image with EXIF orientation handling for microPAD pipeline
    %
    % This function reads images while preserving raw sensor layout by
    % inverting EXIF 90-degree rotation tags. This ensures polygon coordinates
    % remain valid across pipeline stages.
    %
    % Inputs:
    %   fname - Path to image file (char or string)
    %
    % Outputs:
    %   I - Image array with EXIF rotations inverted
    %
    % EXIF Orientation Handling:
    %   - Tags 5/6/7/8 (90-degree rotations): INVERTED to preserve raw layout
    %   - Tags 2/3/4 (flips/180): IGNORED (not inverted)
    %   - Tag 1 or missing: No modification
    %
    % Example:
    %   img = imread_raw('micropad_photo.jpg');

    % Read image without automatic orientation
    try
        I = imread(fname, 'AutoOrient', false);
    catch
        I = imread(fname);
    end

    % Get EXIF orientation tag
    try
        info = imfinfo(fname);
        if ~isfield(info, 'Orientation')
            return;
        end
        ori = double(info.Orientation);
    catch
        return;
    end

    % Invert 90-degree EXIF rotations to preserve raw sensor layout
    switch ori
        case 5
            I = rot90(I, +1);
            I = fliplr(I);
        case 6
            I = rot90(I, -1);
        case 7
            I = rot90(I, -1);
            I = fliplr(I);
        case 8
            I = rot90(I, +1);
    end
end

function [quads, confidences] = detectQuadsYOLO(img, cfg)
    % Run YOLO detection via Python helper script (subprocess interface)

    % Save image to temporary file
    tmpDir = tempdir;
    [~, tmpName] = fileparts(tempname);
    tmpImgPath = fullfile(tmpDir, sprintf('%s_micropad_detect.jpg', tmpName));
    imwrite(img, tmpImgPath, 'JPEG', 'Quality', 95);

    % Ensure cleanup even if error occurs
    cleanupObj = onCleanup(@() cleanupTempFile(tmpImgPath));

    % Build command (redirect stderr to stdout to capture all output)
    cmdRedirect = '2>&1';  % Works on both Windows and Unix

    cmd = sprintf('"%s" "%s" "%s" "%s" --conf %.2f --imgsz %d %s', ...
        cfg.pythonPath, cfg.pythonScriptPath, tmpImgPath, cfg.detectionModel, ...
        cfg.minConfidence, cfg.inferenceSize, cmdRedirect);

    % Run detection
    [status, output] = system(cmd);

    if status ~= 0
        error('cut_micropads:detection_failed', 'Python detection failed (exit code %d): %s', status, output);
    end

    % Parse output (split by newlines - R2019b compatible)
    lines = strsplit(output, {'\n', '\r\n', '\r'}, 'CollapseDelimiters', false);
    lines = lines(~cellfun(@isempty, lines));  % Remove empty lines

    if isempty(lines)
        quads = [];
        confidences = [];
        return;
    end

    numDetections = str2double(lines{1});

    if numDetections == 0 || isnan(numDetections)
        quads = [];
        confidences = [];
        return;
    end

    quads = zeros(numDetections, 4, 2);
    confidences = zeros(numDetections, 1);

    for i = 1:numDetections
        if i+1 > length(lines)
            break;
        end

        parts = str2double(split(lines{i+1}));
        if length(parts) < 9
            continue;
        end

        % Parse: x1 y1 x2 y2 x3 y3 x4 y4 confidence (0-based from Python)
        % Convert to MATLAB 1-based indexing
        vertices = parts(1:8) + 1;
        quad = reshape(vertices, 2, 4)';  % 4x2 matrix
        quads(i, :, :) = quad;
        confidences(i) = parts(9);
    end

    % Filter out empty detections
    validMask = confidences > 0;
    quads = quads(validMask, :, :);
    confidences = confidences(validMask);

    % Ensure consistent left-to-right ordering by centroid X coordinate
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

%% -------------------------------------------------------------------------
%% Memory System
%% -------------------------------------------------------------------------

function memory = initializeMemory()
    % Initialize empty memory structure
    memory = struct();
    memory.hasSettings = false;
    memory.polygonPositions = [];
    memory.rotation = 0;
    memory.imageSize = [];
end

function memory = updateMemory(memory, polygonParams, rotation, imageSize)
    % Update memory with current settings
    memory.hasSettings = true;
    memory.polygonPositions = polygonParams;
    memory.rotation = rotation;
    memory.imageSize = imageSize;
end

function [initialPolygons, source] = getInitialPolygonsWithMemory(img, cfg, memory, imageSize)
    % Get initial polygons prioritizing AI detection with memory fallback
    source = 'default';

    aiPolygons = [];
    detectionSucceeded = false;
    if cfg.useAIDetection
        [aiPolygons, detectionSucceeded] = getInitialPolygons(img, cfg);
    end

    if detectionSucceeded
        initialPolygons = aiPolygons;
        source = 'ai';
        return;
    end

    if memory.hasSettings && ~isempty(memory.polygonPositions) && ~isempty(memory.imageSize)
        scaledPolygons = scalePolygonsForImageSize(memory.polygonPositions, memory.imageSize, imageSize);
        initialPolygons = scaledPolygons;
        fprintf('  Using polygon positions from memory (scaled if needed)\n');
        source = 'memory';
        return;
    end

    [imageHeight, imageWidth, ~] = size(img);
    fprintf('  Using default geometry for initial polygons\n');
    initialPolygons = calculateDefaultPolygons(imageWidth, imageHeight, cfg);
end

function scaledPolygons = scalePolygonsForImageSize(polygons, oldSize, newSize)
    % Scale polygon coordinates when image dimensions change
    if isempty(oldSize) || any(oldSize <= 0)
        scaledPolygons = polygons;
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
