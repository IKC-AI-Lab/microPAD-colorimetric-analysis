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
    % - 'aspectRatio': width/height ratio of each region (default: 0.90, slightly taller than wide)
    % - 'coverage': fraction of image width to fill (default: 0.80)
    % - 'gapPercent': gap as percent of region width, 0..1 or 0..100 (default: 0.19)
    % - 'inputFolder' | 'outputFolder': override default I/O folders
    % - 'preserveFormat' | 'jpegQuality' | 'saveCoordinates': output behavior
    % - 'useAIDetection': use YOLO for initial polygon placement (default: true)
    % - 'detectionModel': path to YOLOv11 model (default: 'models/yolo11n_micropad_seg.pt')
    % - 'minConfidence': minimum detection confidence (default: 0.6)
    % - 'inferenceSize': YOLO inference image size in pixels (default: 640)
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
    DEFAULT_ASPECT_RATIO = 0.90;  % width/height ratio: rectangles are slightly taller than wide
    DEFAULT_COVERAGE = 0.80;       % rectangles span 80% of image width
    DEFAULT_GAP_PERCENT = 0.19;    % 19% gap between rectangles

    % === AI DETECTION DEFAULTS ===
    DEFAULT_USE_AI_DETECTION = true;
    DEFAULT_DETECTION_MODEL = 'models/yolo11n_micropad_seg.pt';
    DEFAULT_MIN_CONFIDENCE = 0.6;
    DEFAULT_PYTHON_PATH = 'C:\Users\veyse\miniconda3\envs\microPAD-python-env\python.exe';
    DEFAULT_INFERENCE_SIZE = 640;

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

    % Validate model file if AI detection enabled
    if cfg.useAIDetection && ~isfile(cfg.detectionModel)
        warning('cut_micropads:model_missing', ...
            'AI detection enabled but model not found: %s\nDisabling AI detection.', cfg.detectionModel);
        cfg.useAIDetection = false;
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
        validateModelFile(cfg.detectionModel);
        loadYOLOModel(cfg.detectionModel);
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
    initialPolygons = getInitialPolygonsWithMemory(img, cfg, memory, [imageHeight, imageWidth]);

    % Interactive region selection with persistent window
    [polygonParams, fig, rotation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig, memory);

    if ~isempty(polygonParams)
        saveCroppedRegions(img, imageName, polygonParams, outputDir, cfg, rotation);
        % Update memory with current polygons and rotation
        memory = updateMemory(memory, polygonParams, rotation, [imageHeight, imageWidth]);
        success = true;
    end
end

function initialPolygons = getInitialPolygons(img, cfg)
    % Get initial polygon positions using AI detection (if enabled) or default geometry
    if cfg.useAIDetection
        try
            [detectedQuads, confidences] = detectQuadsYOLO(img, cfg.minConfidence, cfg.inferenceSize);

            if ~isempty(detectedQuads) && size(detectedQuads, 1) == cfg.numSquares
                fprintf('  AI detected %d regions (avg confidence: %.2f)\n', ...
                    size(detectedQuads, 1), mean(confidences));
                initialPolygons = detectedQuads;
                return;
            elseif ~isempty(detectedQuads)
                fprintf('  AI detected %d regions but expected %d, using default geometry\n', ...
                    size(detectedQuads, 1), cfg.numSquares);
            else
                fprintf('  No AI detections, using default geometry\n');
            end
        catch ME
            fprintf('  AI detection failed (%s), using default geometry\n', ME.message);
        end
    end

    % Fallback to default geometry
    [imageHeight, imageWidth, ~] = size(img);
    initialPolygons = calculateDefaultPolygons(imageWidth, imageHeight, cfg);
end

function polygons = calculateDefaultPolygons(imageWidth, imageHeight, cfg)
    % Generate default polygon positions using geometry parameters
    n = cfg.numSquares;

    % Build world coordinates
    aspect = cfg.geometry.aspectRatio;
    aspect = max(aspect, eps);
    totalGridWidth = 1.0;
    rectHeightWorld = 1.0 / aspect;

    % Compute gap size
    gp = cfg.geometry.gapPercentWidth;
    denom = n + max(n-1, 0) * gp;
    if denom <= 0
        denom = max(n, 1);
    end
    w = totalGridWidth / denom;
    gapSizeWorld = gp * w;

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

function [polygonParams, fig, rotation] = showInteractiveGUI(img, imageName, phoneName, cfg, initialPolygons, fig, memory)
    % Show interactive GUI with editing and preview modes
    polygonParams = [];
    rotation = 0;

    % Create figure if needed
    if isempty(fig) || ~isvalid(fig)
        fig = createFigure(imageName, phoneName, cfg);
    end

    % Initialize rotation from memory if available
    initialRotation = 0;
    if nargin >= 7 && memory.hasSettings
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
                % Preview mode - store rotation in savedRotation field
                clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, userPolygons);

                % Store rotation in guiData for preview mode
                guiData = get(fig, 'UserData');
                guiData.savedRotation = userRotation;
                set(fig, 'UserData', guiData);

                [prevAction, ~, ~] = waitForUserAction(fig);

                switch prevAction
                    case 'accept'
                        polygonParams = userPolygons;
                        rotation = userRotation;
                        return;
                    case {'skip', 'stop'}
                        if strcmp(prevAction, 'stop')
                            close(fig);
                            error('User stopped execution');
                        end
                        return;
                    case 'retry'
                        % Use edited polygons as new initial positions
                        initialPolygons = userPolygons;
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
    guiData.currentImg = img;
    guiData.memoryRotation = initialRotation;
    guiData.adjustmentRotation = 0;
    guiData.totalRotation = initialRotation;

    % Initialize zoom state
    guiData.zoomLevel = 0;  % 0 = full image, 1 = single micropad size
    guiData.imageSize = size(img);

    % Title and path
    guiData.titleHandle = createTitle(fig, phoneName, imageName, cfg);
    guiData.pathHandle = createPathDisplay(fig, phoneName, imageName, cfg);

    % Image display (show image with initial rotation if any)
    if initialRotation ~= 0
        displayImg = applyRotation(img, initialRotation);
        guiData.currentImg = displayImg;
    else
        displayImg = img;
    end
    guiData.imgAxes = createImageAxes(fig, displayImg, cfg);

    % Create editable polygons
    guiData.polygons = createPolygons(initialPolygons, cfg);

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

    % Update rotation state
    guiData.adjustmentRotation = angle;
    guiData.totalRotation = guiData.memoryRotation + angle;

    % Apply rotation to image
    guiData.currentImg = applyRotation(guiData.baseImg, angle);

    % Save polygon positions BEFORE clearing axes
    savedPositions = extractPolygonPositions(guiData);

    % Update image display
    axes(guiData.imgAxes);
    cla(guiData.imgAxes);
    imshow(guiData.currentImg, 'Parent', guiData.imgAxes, 'InitialMagnification', 'fit');
    axis(guiData.imgAxes, 'image');
    axis(guiData.imgAxes, 'tight');
    hold(guiData.imgAxes, 'on');

    % Re-run AI detection if enabled and recreate polygons
    if cfg.useAIDetection
        try
            [detectedQuads, ~] = detectQuadsYOLO(guiData.currentImg, ...
                cfg.minConfidence, cfg.inferenceSize);

            if ~isempty(detectedQuads) && size(detectedQuads, 1) == cfg.numSquares
                % Recreate polygons with AI-detected positions
                guiData.polygons = createPolygons(detectedQuads, cfg);
                fprintf('  AI re-detected %d regions after rotation\n', size(detectedQuads, 1));
            else
                % Recreate polygons at their previous positions if AI detection failed
                guiData.polygons = createPolygons(savedPositions, cfg);
            end
        catch ME
            fprintf('  AI detection after rotation failed: %s\n', ME.message);
            % Recreate polygons at their previous positions
            guiData.polygons = createPolygons(savedPositions, cfg);
        end
    else
        % No AI detection - recreate polygons at their previous positions
        guiData.polygons = createPolygons(savedPositions, cfg);
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

function rotatedImg = applyRotation(img, rotation)
    % Apply rotation to image (use loose mode to preserve entire image)
    if rotation == 0
        rotatedImg = img;
        return;
    end

    % For exact 90-degree multiples, use lossless rot90
    if abs(mod(rotation, 90)) < 1e-6
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
    coordPath = fullfile(phoneOutputDir, cfg.coordinateFileName);

    % Updated to 10-column format with rotation
    scanFmt = '%s %d %f %f %f %f %f %f %f %f %f';
    writeFmt = '%s %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n';
    numericCount = 10;

    [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount);

    [filteredNames, filteredNums] = filterConflictingEntries(existingNames, existingNums, baseName, concentration);

    newRow = [concentration, polygon(1,:), polygon(2,:), polygon(3,:), polygon(4,:), rotation];
    filteredNames{end+1} = baseName;
    filteredNums = [filteredNums; newRow];

    header = 'image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation';
    atomicWriteCoordinates(coordPath, header, filteredNames, filteredNums, writeFmt, phoneOutputDir);
end

function [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, ~)
    existingNames = {};
    existingNums = [];

    if ~isfile(coordPath)
        return;
    end

    fid = fopen(coordPath, 'rt');
    if fid == -1
        warning('cut_micropads:coord_read', 'Could not open coordinates file: %s', coordPath);
        return;
    end

    headerLine = fgetl(fid);
    if headerLine == -1
        fclose(fid);
        return;
    end

    rowData = textscan(fid, scanFmt, 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
    fclose(fid);

    if isempty(rowData) || isempty(rowData{1})
        return;
    end

    existingNames = rowData{1};
    % Vectorize numeric data extraction
    existingNums = cell2mat(rowData(2:end)');

    % Validate and migrate coordinate format
    if ~isempty(existingNums) && size(existingNums, 2) ~= 10
        if size(existingNums, 2) == 9
            % Migrate old 9-column format to 10-column by adding zero rotation
            warning('cut_micropads:old_coord_format', ...
                'Migrating old 9-column coordinate format to 10-column (adding rotation=0): %s', coordPath);
            existingNums = [existingNums, zeros(size(existingNums, 1), 1)];
        else
            error('cut_micropads:invalid_coord_format', ...
                'Coordinate file has %d columns, expected 10 (with rotation): %s\nDelete this file to regenerate.', ...
                size(existingNums, 2), coordPath);
        end
    end
end

function [filteredNames, filteredNums] = filterConflictingEntries(existingNames, existingNums, newName, concentration)
    if isempty(existingNames)
        filteredNames = {};
        filteredNums = [];
        return;
    end

    matchMask = strcmp(existingNames, newName) & (existingNums(:, 1) == concentration);

    filteredNames = existingNames(~matchMask);
    filteredNums = existingNums(~matchMask, :);
end

function atomicWriteCoordinates(coordPath, header, names, nums, writeFmt, coordFolder)
    tmpPath = tempname(coordFolder);

    fid = fopen(tmpPath, 'wt');
    if fid == -1
        error('cut_micropads:coord_write', 'Cannot open temp file for coordinates: %s', tmpPath);
    end

    fprintf(fid, '%s\n', header);

    for i = 1:numel(names)
        fprintf(fid, writeFmt, names{i}, nums(i, :));
    end

    fclose(fid);

    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        error('cut_micropads:coord_move', 'Failed to move temp coordinate file: %s (%s)', msg, msgid);
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
                ['Python executable not configured.\n', ...
                 'Resolution options:\n', ...
                 '1. Set MICROPAD_PYTHON environment variable\n', ...
                 '2. Pass pythonPath parameter: cut_micropads(''pythonPath'', ''C:\\path\\to\\python.exe'')']);
        end

        if ~isfile(pythonPath)
            error('cut_micropads:python_missing', ...
                'Python executable not found at: %s', pythonPath);
        end

    currentPython = pyenv;
    if strcmp(currentPython.Version, '')
        fprintf('Configuring Python environment for auto-detection...\n');
        pyenv('Version', pythonPath);
        fprintf('[OK] Python environment configured\n');
    elseif ~strcmp(currentPython.Executable, pythonPath)
        warning('cut_micropads:python_mismatch', ...
            'Python already initialized with different executable:\n  Current: %s\n  Expected: %s\nRestart MATLAB to switch environments.', ...
            currentPython.Executable, pythonPath);
    end

    try
        py.importlib.import_module('ultralytics');
    catch
        error('cut_micropads:ultralytics_missing', ...
            'Ultralytics package not found in Python environment.\nRun: pip install ultralytics');
    end

        try
            py.importlib.import_module('cv2');
        catch
            error('cut_micropads:opencv_missing', ...
                'OpenCV package not found in Python environment.\nRun: pip install opencv-python');
        end

        fprintf('[OK] Setup complete - ready for auto-detection\n\n');
        setupComplete = true;
    catch ME
        setupComplete = [];  % Clear on error
        rethrow(ME);
    end
end

function validateModelFile(modelPath)
    if ~isfile(modelPath)
        error('cut_micropads:model_missing', ...
            'YOLO model not found: %s\nEnsure model file exists at specified path.', ...
            modelPath);
    end
end

function loadYOLOModel(modelPath)
    [currentModel, currentPath] = modelCache();
    if ~isempty(currentModel) && ~isempty(currentPath) && strcmp(currentPath, modelPath)
        return;
    end

    fprintf('Loading YOLO model: %s\n', modelPath);

    YOLO = py.getattr(py.importlib.import_module('ultralytics'), 'YOLO');
    newModel = YOLO(modelPath);
    modelCache(newModel, modelPath);

    fprintf('[OK] YOLO model loaded and cached\n');
end

function model = getYOLOModel()
    model = modelCache();

    if isempty(model)
        error('cut_micropads:model_not_loaded', ...
            'YOLO model not loaded. Call loadYOLOModel() first.');
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

function [model, modelPath] = modelCache(newModel, newPath)
    persistent cachedModel
    persistent cachedModelPath

    if nargin == 0
        model = cachedModel;
        modelPath = cachedModelPath;
    elseif nargin == 2
        cachedModel = newModel;
        cachedModelPath = newPath;
        model = cachedModel;
        modelPath = cachedModelPath;
    else
        error('cut_micropads:invalid_cache_args', 'modelCache requires 0 or 2 arguments');
    end
end

function [quads, confidences] = detectQuadsYOLO(img, confThreshold, inferenceSize)
    pyImg = py.numpy.array(img);

    model = getYOLOModel();

    results = model.predict(pyImg, ...
        pyargs('imgsz', int32(inferenceSize), ...
               'conf', confThreshold, ...
               'verbose', false));

    result = results{1};

    if isempty(result.masks) || result.masks.data.size(int32(0)) == 0
        quads = [];
        confidences = [];
        return;
    end

    numDetections = int32(result.masks.data.size(int32(0)));

    quads = zeros(double(numDetections), 4, 2);
    confidences = zeros(double(numDetections), 1);

    % Use YOLO's pre-scaled contours (already in original image coordinates)
    for i = 1:double(numDetections)
        pyIdx = int32(i - 1);

        % Get confidence
        confTensor = py.operator.getitem(result.boxes.conf, pyIdx);
        confCPU = confTensor.cpu();
        if isa(confCPU, 'py.numpy.ndarray') || isa(confCPU, 'py.float') || isa(confCPU, 'py.int')
            conf = double(confCPU);
        else
            conf = double(confCPU.item());
        end

        % Get pre-scaled contour from YOLO (masks.xy provides coordinates in original image space)
        contourPy = py.operator.getitem(result.masks.xy, pyIdx);
        if isa(contourPy, 'py.numpy.ndarray')
            contourNumpy = contourPy;
        else
            contourNumpy = contourPy.numpy();
        end
        contour = double(contourNumpy);

        % Fit quadrilateral from contour points
        quad = fitQuadFromContourPoints(contour);

        % Validate quad has exactly 4 vertices
        if isempty(quad) || size(quad, 1) ~= 4
            continue;
        end

        % Convert from 0-indexed to 1-indexed MATLAB coordinates
        quad = quad + 1;

        quads(i, :, :) = quad;
        confidences(i) = conf;
    end

    validMask = confidences > 0;
    quads = quads(validMask, :, :);
    confidences = confidences(validMask);
end

function quad = fitQuadFromContourPoints(contour)
    % Fit quadrilateral from YOLO contour points (already in original image coordinates)
    % Input: contour is N×2 array of (x,y) points
    % Output: quad is 4×2 array of corner points in clockwise order from top-left

    if size(contour, 1) < 4
        quad = [];
        return;
    end

    % Use Douglas-Peucker simplification to get 4 corners
    % Start with aggressive epsilon
    perimeter = 0;
    for i = 1:size(contour, 1)-1
        perimeter = perimeter + norm(contour(i+1,:) - contour(i,:));
    end

    % Try different epsilon values to get exactly 4 points
    epsilonFactors = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1];

    for k = 1:length(epsilonFactors)
        epsilon = epsilonFactors(k) * perimeter;
        approx = douglasPeucker(contour, epsilon);

        if size(approx, 1) == 4
            quad = orderQuadVertices(approx);
            return;
        elseif size(approx, 1) >= 4 && size(approx, 1) <= 6
            % Close enough, select best 4 corners
            quad = selectBest4Corners(approx);
            if ~isempty(quad)
                quad = orderQuadVertices(quad);
                return;
            end
        end
    end

    % Fallback: select 4 extreme points
    quad = selectBest4Corners(contour);
    if ~isempty(quad)
        quad = orderQuadVertices(quad);
    end
end

function simplified = douglasPeucker(points, epsilon)
    % Simple Douglas-Peucker algorithm for polygon simplification
    if size(points, 1) < 3
        simplified = points;
        return;
    end

    % Find point with maximum distance from line between first and last
    dmax = 0;
    index = 0;
    endIdx = size(points, 1);

    for i = 2:(endIdx-1)
        d = pointLineDistance(points(i,:), points(1,:), points(endIdx,:));
        if d > dmax
            index = i;
            dmax = d;
        end
    end

    % If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon
        % Recursive call
        rec1 = douglasPeucker(points(1:index,:), epsilon);
        rec2 = douglasPeucker(points(index:endIdx,:), epsilon);

        % Build result
        simplified = [rec1(1:end-1,:); rec2];
    else
        simplified = [points(1,:); points(endIdx,:)];
    end
end

function dist = pointLineDistance(point, lineStart, lineEnd)
    % Calculate perpendicular distance from point to line segment
    if isequal(lineStart, lineEnd)
        dist = norm(point - lineStart);
        return;
    end

    num = abs((lineEnd(1)-lineStart(1))*(lineStart(2)-point(2)) - ...
              (lineStart(1)-point(1))*(lineEnd(2)-lineStart(2)));
    den = norm(lineEnd - lineStart);
    dist = num / den;
end

function quad = selectBest4Corners(vertices)
    % Select 4 best corner points from N vertices (N > 4)
    % Strategy: find extreme points in 4 directions from centroid

    if size(vertices, 1) < 4
        quad = [];
        return;
    end

    if size(vertices, 1) == 4
        quad = vertices;
        return;
    end

    % Compute centroid
    centroid = mean(vertices, 1);

    % Compute angles from centroid
    angles = atan2(vertices(:, 2) - centroid(2), vertices(:, 1) - centroid(1));

    % Divide into 4 quadrants and pick furthest point in each
    quadrants = {[], [], [], []};

    for i = 1:size(vertices, 1)
        angle = angles(i);

        % Assign to quadrant (4 ranges covering -pi to pi)
        if angle >= -pi && angle < -pi/2
            quadIdx = 1;  % Bottom-left
        elseif angle >= -pi/2 && angle < 0
            quadIdx = 2;  % Bottom-right
        elseif angle >= 0 && angle < pi/2
            quadIdx = 3;  % Top-right
        else
            quadIdx = 4;  % Top-left
        end

        % Compute distance from centroid
        dist = norm(vertices(i, :) - centroid);

        % Store [vertex_index, distance]
        quadrants{quadIdx} = [quadrants{quadIdx}; i, dist];
    end

    % Select furthest point from each quadrant
    quad = zeros(4, 2);
    quadCount = 0;

    for q = 1:4
        if ~isempty(quadrants{q})
            [~, maxIdx] = max(quadrants{q}(:, 2));
            vertexIdx = quadrants{q}(maxIdx, 1);
            quadCount = quadCount + 1;
            quad(quadCount, :) = vertices(vertexIdx, :);
        end
    end

    % If we didn't get 4 points, return empty
    if quadCount < 4
        quad = [];
    else
        quad = quad(1:4, :);
    end
end

function quadOrdered = orderQuadVertices(quad)
    % Order vertices clockwise from top-left: TL, TR, BR, BL
    % Matches training label format from augment_dataset.m

    centroid = mean(quad, 1);

    % Sort by angle from centroid (counterclockwise from right horizontal)
    angles = atan2(quad(:, 2) - centroid(2), quad(:, 1) - centroid(1));
    [~, order] = sort(angles);
    quadOrdered = quad(order, :);

    % Rotate array so top-left corner is first
    % Top-left corner has minimum (x + y) value
    [~, topLeftIdx] = min(sum(quadOrdered, 2));
    quadOrdered = circshift(quadOrdered, -topLeftIdx + 1, 1);
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

function initialPolygons = getInitialPolygonsWithMemory(img, cfg, memory, imageSize)
    % Get initial polygons considering memory (scaled if image dimensions changed)

    % If memory has settings and image size matches (or can be scaled)
    if memory.hasSettings && ~isempty(memory.polygonPositions) && ~isempty(memory.imageSize)
        % Scale polygons if image dimensions changed
        scaledPolygons = scalePolygonsForImageSize(memory.polygonPositions, memory.imageSize, imageSize);
        initialPolygons = scaledPolygons;
        fprintf('  Using polygon positions from memory (scaled if needed)\n');
        return;
    end

    % Otherwise use AI detection or default geometry
    initialPolygons = getInitialPolygons(img, cfg);
end

function scaledPolygons = scalePolygonsForImageSize(polygons, oldSize, newSize)
    % Scale polygon coordinates when image dimensions change
    oldHeight = oldSize(1);
    oldWidth = oldSize(2);
    newHeight = newSize(1);
    newWidth = newSize(2);

    if oldHeight == newHeight && oldWidth == newWidth
        % No scaling needed
        scaledPolygons = polygons;
        return;
    end

    % Compute scale factors
    scaleX = newWidth / oldWidth;
    scaleY = newHeight / oldHeight;

    % Apply scaling to all polygons
    numPolygons = size(polygons, 1);
    scaledPolygons = zeros(size(polygons));

    for i = 1:numPolygons
        poly = squeeze(polygons(i, :, :));
        poly(:, 1) = poly(:, 1) * scaleX;  % Scale X coordinates
        poly(:, 2) = poly(:, 2) * scaleY;  % Scale Y coordinates
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
