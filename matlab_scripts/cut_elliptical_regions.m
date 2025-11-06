function cut_elliptical_regions(varargin)
    %% microPAD Colorimetric Analysis — Elliptical Patch Cutting Tool
    %% Select and save elliptical ROI patches for microPAD images.
    %% Author: Veysel Y. Yilmaz
    %
    % Inputs (Name-Value pairs):
    % - 'saveCoordinates' (logical): write coordinates.txt entries (default true)
    %
    % Outputs/Side effects:
    % - Writes PNG elliptical patches to 3_elliptical_regions/[phone]/con_*/
    % - Writes consolidated coordinates.txt at phone level with format:
    %   image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
    %   where rotationAngle is in degrees (-180 to 180, clockwise from horizontal major axis)
    % - No duplicate rows per image (existing entries are replaced)
    %
    % ROTATION SEMANTICS:
    %   The rotation column in 2_micropads/coordinates.txt is a UI-only alignment
    %   hint from cut_micropads.m. This script reads polygon crops (which are in
    %   the original unrotated reference frame) and produces ellipse coordinates
    %   where rotationAngle represents the ellipse's geometric orientation, NOT
    %   the UI rotation hint. Downstream processing uses only ellipse rotationAngle.
    %
    % Behavior:
    % - Reads single-concentration polygon crops from 2_micropads/[phone]/con_*/
    % - Captures elliptical ROI masks for each replicate and logs coordinates to phone-level file

%% ========================================================================

    % Error handling for deprecated format parameters
    if ~isempty(varargin) && (any(strcmpi(varargin(1:2:end), 'preserveFormat')) || any(strcmpi(varargin(1:2:end), 'jpegQuality')))
        error('micropad:deprecated_parameter', ...
              ['JPEG format no longer supported. Pipeline outputs PNG exclusively.\n' ...
               'Remove ''preserveFormat'' and ''jpegQuality'' parameters from function call.']);
    end

%% ========================================================================
    %% EXPERIMENT CONFIGURATION CONSTANTS - Modify here for different setups
    %% ========================================================================
    %
    % All experimental constants are centralized here for easy modification.
    % Change these values to adapt the script for different experimental setups
    % without hunting through the code.
    %
    
    % === DATASET AND FOLDER STRUCTURE ===
    INPUT_FOLDER = '2_micropads';   % Source folder from previous step
    OUTPUT_FOLDER = '3_elliptical_regions';     % Output elliptical patches folder
    
    % === NAMING / FILE CONSTANTS ===
    CONC_FOLDER_PREFIX = 'con_';                         % Concentration folder prefix
    COORDINATE_FILENAME = 'coordinates.txt';             % Coordinates log filename
    PATCH_FILENAME_FORMAT = '%s_con%d_rep%d%s';          % Patch filename format
    
    % === IMAGE FILTERING / SEARCH ===
    ALLOWED_IMAGE_EXTENSIONS = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'}; % File patterns
    MAX_PROJECT_ROOT_SEARCH_LEVELS = 5;                 % Up-search depth for project root

    % === OUTPUT FORMATTING ===
    SAVE_COORDINATES = true;                            % Save ellipse coordinates to coordinates.txt
    SUPPORTED_FORMATS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}; % Formats to preserve when enabled

    % === microPAD LAYOUT PARAMETERS ===
    REPLICATES_PER_CONCENTRATION = 3;                   % Replicates per concentration

    % === LAYOUT PROPORTIONS ===
    MARGIN_TO_SPACING_RATIO = 1/3;                      % Margin = rectangleWidth / 3
    VERTICAL_POSITION_RATIO = 1/3;                      % Ellipse center at 1/3 of image height

    % === SAFETY AND QUALITY FACTORS ===
    OVERLAP_SAFETY_FACTOR = 2.1;                       % Prevent ellipse overlap
    DIM_FACTOR = 0.2;                                  % Dimming factor for preview (0.2 = 80% dimmed)

    % === ELLIPSE RANGE PARAMETERS ===
    MIN_AXIS_PERCENT = 0.005;                          % 0.5% of width as minimum axis length
    SEMI_MAJOR_DEFAULT_RATIO = 0.70;                   % Default semi-major axis (70% of max)
    SEMI_MINOR_DEFAULT_RATIO = 0.85;                   % Default semi-minor/semi-major ratio
    ROTATION_DEFAULT_ANGLE = 0;                        % Default rotation (degrees)

    % === ROTATION CONSTANTS ===
    ROTATION_ANGLE_TOLERANCE = 1e-6;                   % Tolerance for detecting exact 90-degree rotations

    % === UI CONSTANTS ===
    UI_CONST = struct();
    UI_CONST.fontSize = struct('title',16,'path',12,'button',12,'label',10,'value',8,'instruction',10,'preview',14);
    UI_CONST.colors = struct('background','black','foreground','white','panel',[0.1 0.1 0.1], ...
                             'stop',[0.8 0.2 0.2],'accept',[0.2 0.7 0.2],'retry',[0.8 0.8 0.2],'skip',[0.7 0.2 0.2], ...
                             'ellipse','cyan','value','yellow','preview','yellow','path',[0.7 0.7 0.7],'apply',[0.2 0.4 0.8]);
    UI_CONST.positions = struct('figure',[0 0 1 1],'title',[0.1 0.93 0.8 0.04], ...
                                'image',[0.02 0.18 0.96 0.68], ...
                                'rotationPanel',[0.02 0.02 0.20 0.12], ...
                                'cutButtonPanel',[0.55 0.02 0.43 0.12],'previewPanel',[0.25 0.02 0.50 0.12], ...
                                'stopButton',[0.02 0.93 0.06 0.05],'instructions',[0.02 0.145 0.96 0.025]);
    UI_CONST.layout = struct();
    UI_CONST.layout.rotationLabel = [0.05 0.78 0.90 0.18];
    UI_CONST.layout.quickRotationRow1 = {[0.05 0.42 0.42 0.30], [0.53 0.42 0.42 0.30]};
    UI_CONST.layout.quickRotationRow2 = {[0.05 0.08 0.42 0.30], [0.53 0.08 0.42 0.30]};
    UI_CONST.rotation = struct('range',[-180 180],'quickAngles',[-90 0 90 180],'angleTolerance',ROTATION_ANGLE_TOLERANCE);

    %% ========================================================================
    
    % Project folder structure constants
    
    % Configuration - All defaults calculated from these base constants
    cfg = createConfiguration(INPUT_FOLDER, OUTPUT_FOLDER, REPLICATES_PER_CONCENTRATION, ...
                              MARGIN_TO_SPACING_RATIO, VERTICAL_POSITION_RATIO, ...
                              OVERLAP_SAFETY_FACTOR, DIM_FACTOR, ...
                              MIN_AXIS_PERCENT, SEMI_MAJOR_DEFAULT_RATIO, SEMI_MINOR_DEFAULT_RATIO, ROTATION_DEFAULT_ANGLE, ...
                              CONC_FOLDER_PREFIX, ...
                              COORDINATE_FILENAME, PATCH_FILENAME_FORMAT, ...
                              ALLOWED_IMAGE_EXTENSIONS, MAX_PROJECT_ROOT_SEARCH_LEVELS, SAVE_COORDINATES, SUPPORTED_FORMATS, varargin{:});

    % Apply UI constants (separate from functionality constants)
    cfg.ui = createUIConfiguration(UI_CONST);

    % Rotation configuration
    cfg.rotation = struct('angleTolerance', ROTATION_ANGLE_TOLERANCE);

    % Output behavior is configured by createConfiguration (supports overrides)
    
    try
        processAllFolders(cfg);
        fprintf('>> Elliptical patch cutting completed successfully!\n');
    catch ME
        handleError(ME);
    end
end

function cfg = createConfiguration(inputFolder, outputFolder, replicatesPerConcentration, ...
                             marginToSpacingRatio, verticalPositionRatio, ...
                             overlapSafetyFactor, dimFactor, ...
                             minAxisPercent, semiMajorDefaultRatio, semiMinorDefaultRatio, rotationDefaultAngle, ...
                             concFolderPrefix, ...
                             coordinateFileName, patchFilenameFormat, imageExtensions, ...
                             maxProjectRootSearchLevels, saveCoordinates, supportedFormats, varargin)
    %% Create configuration with validation and calculated defaults from centralized constants
    % Validate core numeric parameters
    validateattributes(replicatesPerConcentration, {'numeric'},{'scalar','integer','>=',1},mfilename,'replicatesPerConcentration');
    validateattributes(marginToSpacingRatio, {'numeric'},{'scalar','>',0},mfilename,'marginToSpacingRatio');
    validateattributes(verticalPositionRatio, {'numeric'},{'scalar','>',0,'<',1},mfilename,'verticalPositionRatio');
    validateattributes(overlapSafetyFactor, {'numeric'},{'scalar','>',1},mfilename,'overlapSafetyFactor');
    validateattributes(dimFactor, {'numeric'},{'scalar','>=',0,'<=',1},mfilename,'dimFactor');
    validateattributes(minAxisPercent, {'numeric'},{'scalar','>',0,'<',1},mfilename,'minAxisPercent');
    validateattributes(semiMajorDefaultRatio, {'numeric'},{'scalar','>=',0,'<=',1},mfilename,'semiMajorDefaultRatio');
    validateattributes(semiMinorDefaultRatio, {'numeric'},{'scalar','>=',0,'<=',1},mfilename,'semiMinorDefaultRatio');
    validateattributes(rotationDefaultAngle, {'numeric'},{'scalar','>=',-180,'<=',180},mfilename,'rotationDefaultAngle');

    parser = inputParser;
    parser.addParameter('saveCoordinates', saveCoordinates, @(x) islogical(x));
    parser.parse(varargin{:});

    % Validate logical consistency of configuration parameters
    if ~parser.Results.saveCoordinates
        warning('cutEllipticalPatches:NoCoordinates', ...
                'saveCoordinates=false will prevent feature extraction in subsequent pipeline stages.');
    end

    cfg.replicatesPerConcentration = replicatesPerConcentration;
    cfg.numEllipses = cfg.replicatesPerConcentration;  % One concentration rectangle per image

    % Layout proportions (using centralized constants)
    cfg.marginToSpacingRatio = marginToSpacingRatio;  % margin = rectangleWidth * ratio
    cfg.verticalPositionRatio = verticalPositionRatio;

    % Safety and quality factors (using centralized constants)
    cfg.overlapSafetyFactor = overlapSafetyFactor;
    cfg.dimFactor = dimFactor;

    % Ellipse geometry range parameters (using centralized constants)
    cfg.minAxisPercent = minAxisPercent;
    cfg.semiMajorDefaultRatio = semiMajorDefaultRatio;
    cfg.semiMinorDefaultRatio = semiMinorDefaultRatio;
    cfg.rotationDefaultAngle = rotationDefaultAngle;
    cfg.rotationDefaultAngles = determineDefaultRotationAngles(cfg.replicatesPerConcentration, cfg.rotationDefaultAngle);
    
    % File/folder conventions and parsing
    cfg.concFolderPrefix = concFolderPrefix;
    % Build folder-matching regex from configured prefix (prefix-aware parsing)
    try
        cfg.concFolderPattern = ['^' regexptranslate('escape', concFolderPrefix) '(\d+)$'];
    catch %#ok<CTCH>
        % Fallback if regexptranslate is unavailable; escape underscore at minimum
        safePrefix = strrep(concFolderPrefix, '_', '\\_');
        cfg.concFolderPattern = ['^' safePrefix '(\d+)$'];
    end
    cfg.coordinateFileName = coordinateFileName;
    cfg.patchFilenameFormat = patchFilenameFormat;
    % Output behavior
    cfg.output = struct();
    cfg.output.saveCoordinates = parser.Results.saveCoordinates;
    cfg.output.supportedFormats = supportedFormats;
    
    % Image search and project root search depth
    cfg.allowedImageExtensions = imageExtensions;
    cfg.maxProjectRootSearchLevels = maxProjectRootSearchLevels;
    
    % Path configuration - Dynamic path resolution
    cfg = addPathConfiguration(cfg, inputFolder, outputFolder);
    
    % UI configuration is injected at top-level via UI_CONST
end

function rotationAngles = determineDefaultRotationAngles(replicatesPerConcentration, baseAngle)
    %% Compute default rotation angles per replicate
    rotationAngles = repmat(baseAngle, replicatesPerConcentration, 1);
    if replicatesPerConcentration == 3
        rotationAngles = [-45; 0; 45];
    end
end

function uiCfg = createUIConfiguration(uiDefaults)
    %% Return provided UI defaults (centralized at top)
    uiCfg = uiDefaults;
end

%% Layout Calculation Functions
function layoutParams = calculateLayoutParameters(imageWidth, cfg)
    %% Calculate layout parameters for single concentration rectangle
    % Layout: [margin] [rectangle containing all ellipses] [margin]
    % Image width = 2*margin + rectangleWidth
    % Margin proportional to rectangle width via marginToSpacingRatio

    marginRatio = cfg.marginToSpacingRatio;
    % Solve: imageWidth = 2*margin + rectangleWidth, where margin = rectangleWidth * marginRatio
    % imageWidth = 2*(rectangleWidth * marginRatio) + rectangleWidth
    % imageWidth = rectangleWidth * (2*marginRatio + 1)
    rectangleWidth = imageWidth / (2 * marginRatio + 1);
    margin = rectangleWidth * marginRatio;

    layoutParams.rectangleWidth = rectangleWidth;
    layoutParams.margin = margin;
    layoutParams.maxSafeEllipseDimension = rectangleWidth / (cfg.replicatesPerConcentration * cfg.overlapSafetyFactor);
    layoutParams.verticalCenter = NaN; % Will be set when image height is known
end

function geometryParams = calculateGeometryParameters(imageWidth, cfg)
    %% Calculate ellipse geometry limits based on image width and configuration
    layoutParams = calculateLayoutParameters(imageWidth, cfg);

    geometryParams.minPixels = imageWidth * cfg.minAxisPercent;
    geometryParams.maxPixels = layoutParams.maxSafeEllipseDimension;
    geometryParams.range = geometryParams.maxPixels - geometryParams.minPixels;
    geometryParams.rotationMin = -180;
    geometryParams.rotationMax = 180;
end

function ellipseParams = calculateDefaultEllipseGeometry(geometryParams, cfg)
    %% Calculate default ellipse parameters based on configuration
    baseSemiMajor = geometryParams.minPixels + (geometryParams.range * cfg.semiMajorDefaultRatio);
    semiMajorAxis = clampToRange(baseSemiMajor, geometryParams.minPixels, geometryParams.maxPixels);
    semiMinorAxis = semiMajorAxis * cfg.semiMinorDefaultRatio;
    semiMinorAxis = clampToRange(semiMinorAxis, geometryParams.minPixels, semiMajorAxis);
    rotationAngle = cfg.rotationDefaultAngle;

    ellipseParams = struct('semiMajorAxis', semiMajorAxis, ...
                          'semiMinorAxis', semiMinorAxis, ...
                          'rotationAngle', rotationAngle);
end

function positions = calculateDefaultEllipsePositions(imageWidth, imageHeight, cfg)
    %% Calculate default ellipse positions for single concentration rectangle
    layoutParams = calculateLayoutParameters(imageWidth, cfg);
    layoutParams.verticalCenter = imageHeight * cfg.verticalPositionRatio;

    % Distribute ellipses evenly across the rectangle width
    % Ellipse i is centered at: margin + (i + 0.5) * ellipseSpacing
    ellipseIndices = (0:cfg.numEllipses-1).';  % 0-based indices [0, 1, 2, ...]
    ellipseSpacing = layoutParams.rectangleWidth / cfg.replicatesPerConcentration;
    centerX = layoutParams.margin + (ellipseIndices + 0.5) * ellipseSpacing;

    % Create position matrix
    positions = [centerX, repmat(layoutParams.verticalCenter, cfg.numEllipses, 1)];
end

function value = clampToRange(value, minVal, maxVal)
    %% Clamp value to specified range
    value = max(minVal, min(value, maxVal));
end

%% Main Processing Functions
function processAllFolders(cfg)
    % Process all phone folders
    validatePaths(cfg);
    phoneList = getSubFolders(cfg.inputPath);
    
    if isempty(phoneList)
        error('cutEllipticalPatches:NoPhones', 'No phone folders found in: %s', cfg.inputPath);
    end
    
    fprintf('Found %d phone(s): %s\n', length(phoneList), strjoin(phoneList, ', '));
    
    % Process each phone with error handling
    executeInFolder(cfg.inputPath, @() arrayfun(@(i) processPhone(phoneList{i}, cfg), 1:length(phoneList)));
end

function processPhone(phoneName, cfg)
    fprintf('\n=== Processing Phone: %s ===\n', phoneName);
    if ~(exist(phoneName, 'dir') == 7)
        warning('cutEllipticalPatches:MissingPhone', 'Phone folder missing: %s', phoneName);
        return;
    end
    executeInFolder(phoneName, @() processImagesInPhone(phoneName, cfg));
end

function processImagesInPhone(phoneName, cfg)
    concFolders = findConcentrationFolders('.', cfg.concFolderPrefix);
    if isempty(concFolders)
        warning('cutEllipticalPatches:NoConcentrationFolders', 'No concentration folders in: %s (expected %s*)', composePathSegments(phoneName), cfg.concFolderPrefix);
        return;
    end

    outputDir = createOutputStructure(cfg.outputPath, phoneName, cfg.projectRoot);
    fprintf('Output directory: %s\n', outputDir);

    try
        for c = 1:numel(concFolders)
            concName = concFolders{c};
            tokens = regexp(concName, cfg.concFolderPattern, 'tokens', 'once');
            if isempty(tokens)
                warning('cutEllipticalPatches:UnexpectedFolder', 'Skipping unexpected folder: %s', concName);
                continue;
            end
            concIdx = str2double(tokens{1});
            executeInFolder(concName, @() processImagesInConFolder(phoneName, concIdx, cfg, outputDir));
        end
    catch ME
        rethrow(ME);
    end

    fprintf('Completed: %s\n', composePathSegments(phoneName));
end


function processImagesInConFolder(phoneName, concIdx, cfg, outputDir)
    % Process images inside a specific concentration folder
    imageList = getImageFiles('.', cfg.allowedImageExtensions);
    if isempty(imageList)
        warning('cutEllipticalPatches:NoImagesInConcentration', 'No images in: %s/%s%d', composePathSegments(phoneName), cfg.concFolderPrefix, concIdx);
        return;
    end
    fprintf('%s%d: Found %d images\n', cfg.concFolderPrefix, concIdx, length(imageList));

    % Ensure output concentration folder exists (coords file created lazily on first accept)
    concOutputFolder = fullfile(outputDir, sprintf('%s%d', cfg.concFolderPrefix, concIdx));
    if ~exist(concOutputFolder, 'dir')
        mkdir(concOutputFolder);
    end

    memory = initializeMemory();
    persistentFig = [];
    try
        for idx = 1:length(imageList)
            [memory, success, persistentFig] = processOneImageWithPersistentWindow(...
                imageList{idx}, phoneName, concIdx, cfg, memory, idx == 1, outputDir, persistentFig);
            logProcessingResult(success, idx == 1);
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
end

%% Image Processing with Persistent Window
function [updatedMemory, success, fig] = processOneImageWithPersistentWindow(imageName, phoneName, concIdx, cfg, memory, isFirst, outputDir, fig)
    % Process single image with persistent window
    updatedMemory = memory;
    success = false;

    % Load image
    [img, isValid] = loadImage(imageName);
    if ~isValid, return; end

    % Display processing info
    displayProcessingInfo(imageName, memory, isFirst);

    % Interactive ellipse selection with persistent window
    [coords, fig] = showInteractiveGUIWithPersistentWindow(img, imageName, phoneName, cfg, memory, isFirst, fig);

    if ~isempty(coords)
        if isfield(cfg, 'output') && isfield(cfg.output, 'saveCoordinates') && cfg.output.saveCoordinates
            saveCoordinates(outputDir, imageName, concIdx, coords, cfg);
        end
        saveCutPatchesToConcentrationFolders(img, imageName, concIdx, coords, outputDir, cfg);

        % Get rotation from fig if available, otherwise default to 0
        rotationTotal = 0;
        if isvalid(fig)
            guiData = get(fig, 'UserData');
            if isfield(guiData, 'rotation')
                rotationTotal = guiData.rotation.total;
            end
        end

        updatedMemory = updateMemory(coords, rotationTotal, img);
        success = true;
    else
        fprintf('  Image skipped by user\n');
    end
end

function [coords, fig] = showInteractiveGUIWithPersistentWindow(img, imageName, phoneName, cfg, memory, isFirst, fig)
    % Show interactive GUI with persistent window and retry loop for preview
    coords = [];
    
    % Create or reuse figure
    if isempty(fig) || ~isvalid(fig)
        fig = createFigure(imageName, phoneName, cfg);
    end
    
    while true
        % Clear and rebuild UI for elliptical patch cutting mode
        clearAndRebuildUI(fig, 'cutting', img, imageName, phoneName, cfg, memory, isFirst);
        
        [action, userCoords] = waitForUserAction(fig);
        
        switch action
            case 'skip'
                return;
            case 'stop'
                close(fig);
                error('User stopped execution');
            case 'accept'
                coords = userCoords;
                
                % Clear and rebuild UI for preview mode
                clearAndRebuildUI(fig, 'preview', img, imageName, phoneName, cfg, memory, isFirst, coords);
                
                [prevAction, previewCoords] = waitForUserAction(fig);
                
                switch prevAction
                    case 'accept'
                        if ~isempty(previewCoords)
                            coords = previewCoords;
                        end
                        return;
                    case {'skip', 'stop'}
                        if strcmp(prevAction, 'stop')
                            close(fig);
                            error('User stopped execution');
                        end
                        coords = [];
                        return;
                    case 'retry'
                        % Continue loop to redraw ellipse interface
                        continue;
                end
        end
    end
end

%% Unified UI Clear and Rebuild Function
function clearAndRebuildUI(fig, mode, img, imageName, phoneName, cfg, memory, isFirst, coords)
    % Unified function to clear all UI elements and rebuild for specified mode
    if ~exist('coords', 'var') || isempty(coords)
        coords = [];
    end
    
    % Get existing GUI data
    guiData = get(fig, 'UserData');
    
    % Clear ALL UI elements (both cutting and preview components)
    clearAllUIElements(fig, guiData);
    
    % Rebuild UI based on mode
    switch mode
        case 'cutting'
            buildEllipticalPatchCuttingUI(fig, img, imageName, phoneName, cfg, memory, isFirst);
        case 'preview'
            buildPreviewUI(fig, img, imageName, phoneName, cfg, coords);
    end
end

function clearAllUIElements(fig, guiData)
    % Comprehensive clearing of all UI elements
    % Single-pass type-based deletion for optimal performance

    % Get all objects except the figure itself
    allObjects = findall(fig, '-not', 'Type', 'figure');

    % Delete objects based on type (handles invalid objects gracefully)
    for i = 1:length(allObjects)
        obj = allObjects(i);
        if ~isvalid(obj), continue; end

        objType = get(obj, 'Type');
        switch objType
            case 'axes'
                % Clear before deleting to release resources
                cla(obj);
                delete(obj);
            case {'uicontrol', 'uipanel', 'images.roi.Ellipse'}
                % Direct deletion for UI controls, panels, and ROI objects
                delete(obj);
            % Other types (lines, text, etc.) automatically deleted with parent
        end
    end

    % Clear ellipses from guiData if they exist (redundant safety check)
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'ellipses')
        if iscell(guiData.ellipses)
            for i = 1:length(guiData.ellipses)
                try
                    if isvalid(guiData.ellipses{i})
                        delete(guiData.ellipses{i});
                    end
                catch %#ok<CTCH>
                    % Object already deleted or invalid - safe to ignore
                end
            end
        end
    end

    % Reset UserData
    set(fig, 'UserData', []);
end

function buildEllipticalPatchCuttingUI(fig, img, imageName, phoneName, cfg, memory, isFirst)
    % Build UI for elliptical patch cutting mode

    % Update figure name
    set(fig, 'Name', sprintf('Elliptical Patch Cutting Tool - %s - %s', composePathSegments(phoneName), imageName));

    % Create new GUI elements
    guiData = struct();
    guiData.mode = 'cutting';

    % Create UI components
    guiData.titleHandle = createTitle(fig, phoneName, imageName, isFirst, cfg);

    % Initialize rotation state (UI alignment only, no rotation on disk)
    initialRotation = 0;
    if memory.hasMemory && ~isFirst && isfield(memory, 'rotation')
        initialRotation = memory.rotation;
    end
    guiData.rotation = struct('memory', initialRotation, 'offset', 0, 'total', initialRotation, 'angleTolerance', cfg.rotation.angleTolerance);
    guiData.baseImg = img;

    % Apply rotation to image for display
    guiData.currentImg = applyRotationToImage(img, guiData.rotation.total, cfg);
    [imageHeight, imageWidth, ~] = size(guiData.currentImg);

    % Display image
    [guiData.imgAxes, guiData.imgHandle] = createImageAxes(fig, guiData.currentImg, cfg);

    % Calculate parameters and create ellipses
    geometryParams = calculateGeometryParameters(imageWidth, cfg);
    initialGeometry = getInitialEllipseGeometry(imageWidth, memory, isFirst, geometryParams, cfg);

    % Create interactive ellipses
    guiData.ellipses = createEllipses(guiData.currentImg, guiData.baseImg, guiData.rotation, initialGeometry, geometryParams, cfg, memory, isFirst);

    % Rotation panel
    guiData.rotationPanel = createRotationPanel(fig, cfg);

    % Create control buttons
    guiData.cutButtonPanel = createEllipticalPatchCuttingButtonPanel(fig, cfg);
    guiData.stopButton = createStopButton(fig, cfg);
    guiData.instructionText = createInstructions(fig, memory, isFirst, cfg);

    % Initialize GUI data with cutting-specific fields
    guiData = initializeGUIData(guiData, imageWidth, imageHeight);
    guiData.action = '';

    set(fig, 'UserData', guiData);
end

function buildPreviewUI(fig, img, imageName, phoneName, cfg, coords)
    % Build UI for preview mode
    
    % Update figure name
    set(fig, 'Name', sprintf('PREVIEW - %s - %s', composePathSegments(phoneName), imageName));
    
    % Create new GUI data
    guiData = struct();
  	guiData.mode = 'preview';
    guiData.savedCoords = coords;
    
    % Create preview title (no path display)
    titleText = sprintf('PREVIEW: %s - %s', composePathSegments(phoneName), imageName);
    guiData.titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                                   'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                                   'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                                   'ForegroundColor', cfg.ui.colors.foreground, ...
                                   'BackgroundColor', cfg.ui.colors.background, ...
                                   'HorizontalAlignment', 'center');
    
    % Create preview display
    guiData.imgAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.image);
    guiData.imgHandle = createPreviewInExistingAxes(guiData.imgAxes, img, coords, cfg);
    
    % Create stop button (always visible)
    guiData.stopButton = createStopButton(fig, cfg);
    
    % Create preview buttons (simplified without instruction text)
    guiData.buttonPanel = createPreviewButtons(fig, cfg);
    
    guiData.action = '';
    
    set(fig, 'UserData', guiData);
end

%% GUI Helper Functions
function fig = createFigure(imageName, phoneName, cfg)
    % Create main figure window
    titleText = sprintf('Elliptical Patch Cutting Tool - %s - %s', composePathSegments(phoneName), imageName);
    fig = figure('Name', titleText, ...
                'Units', 'normalized', 'Position', cfg.ui.positions.figure, ...
                'MenuBar', 'none', 'ToolBar', 'none', ...
                'WindowState', 'maximized', 'Color', cfg.ui.colors.background, ...
                'KeyPressFcn', @keyPressHandler, 'Tag', 'cut_elliptical_regions');
end

function titleHandle = createTitle(fig, phoneName, imageName, isFirst, cfg)
    % Create title text
    if isFirst
        titleText = sprintf('%s - %s (FIRST IMAGE - positions will be saved)', composePathSegments(phoneName), imageName);
    else
        titleText = sprintf('%s - %s (using saved positions)', composePathSegments(phoneName), imageName);
    end
    
    titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                           'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                           'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                           'ForegroundColor', cfg.ui.colors.foreground, ...
                           'BackgroundColor', cfg.ui.colors.background, ...
                           'HorizontalAlignment', 'center');
end

function [imgAxes, imgHandle] = createImageAxes(fig, img, cfg)
    % Create axes and display image
    imgAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.image);
    imgHandle = imshow(img, 'Parent', imgAxes, 'InitialMagnification', 'fit');
    hold(imgAxes, 'on');
end

function stopButton = createStopButton(fig, cfg)
    % Create STOP button
    stopButton = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
                          'String', 'STOP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.stopButton, ...
                          'BackgroundColor', cfg.ui.colors.stop, 'ForegroundColor', cfg.ui.colors.foreground, ...
                          'Callback', @(~,~) stopExecution(fig));
end

function cutButtonPanel = createEllipticalPatchCuttingButtonPanel(fig, cfg)
    % Create panel for elliptical patch cutting mode buttons (APPLY and SKIP)
    cutButtonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                                'Position', cfg.ui.positions.cutButtonPanel, ...
                                'BackgroundColor', cfg.ui.colors.panel, ...
                                'BorderType', 'line', 'HighlightColor', cfg.ui.colors.foreground);

    % APPLY button (blue)
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'APPLY', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.15 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.apply, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setEllipticalPatchCuttingAction(fig, 'accept'));

    % SKIP button (red)
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'SKIP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.55 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.skip, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', @(~,~) setEllipticalPatchCuttingAction(fig, 'skip'));
end

function setEllipticalPatchCuttingAction(fig, action)
    % Set action for elliptical patch cutting mode buttons
    guiData = get(fig, 'UserData');

    guiData.action = action;
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function initialGeometry = getInitialEllipseGeometry(imageWidth, memory, isFirst, geometryParams, cfg)
    % Determine starting ellipse geometry using memory when available
    defaults = calculateDefaultEllipseGeometry(geometryParams, cfg);
    semiMajorPixels = defaults.semiMajorAxis;
    semiMinorPixels = defaults.semiMinorAxis;
    rotationAngle = defaults.rotationAngle;

    if memory.hasMemory && ~isFirst
        widthScale = 1;
        if ~isempty(memory.imageWidth) && memory.imageWidth > 0
            widthScale = imageWidth / memory.imageWidth;
        end

        if ~isempty(memory.semiMajorAxis) && memory.semiMajorAxis > 0
            semiMajorPixels = memory.semiMajorAxis * widthScale;
        end
        if ~isempty(memory.semiMinorAxis) && memory.semiMinorAxis > 0
            semiMinorPixels = memory.semiMinorAxis * widthScale;
        end
        if ~isempty(memory.rotationAngle)
            rotationAngle = memory.rotationAngle;
        end
    end

    semiMajorPixels = clampToRange(semiMajorPixels, geometryParams.minPixels, geometryParams.maxPixels);
    semiMinorPixels = clampToRange(semiMinorPixels, geometryParams.minPixels, semiMajorPixels);
    rotationAngle = clampToRange(rotationAngle, geometryParams.rotationMin, geometryParams.rotationMax);

    initialGeometry = struct('semiMajor', semiMajorPixels, ...
                             'semiMinor', semiMinorPixels, ...
                             'rotation', rotationAngle);
end

function ellipses = createEllipses(img, baseImg, rotationState, initialGeometry, geometryParams, cfg, memory, isFirst)
    % createEllipses - Create interactive ellipse overlays with rotation support
    %
    % Transforms ellipse positions from memory (base frame) to current display frame
    % (potentially rotated) while scaling for dimension changes between images.
    %
    % Inputs:
    %   img             - Current rotated image for display (may differ from baseImg)
    %   baseImg         - Original unrotated base image (reference frame for coordinates)
    %   rotationState   - Struct with fields: total (degrees), angleTolerance
    %   initialGeometry - Struct: semiMajor, semiMinor, rotation (base frame values)
    %   geometryParams  - Struct: minPixels, maxPixels, rotationMin, rotationMax
    %   cfg             - Configuration struct
    %   memory          - Memory struct with positions from previous image (base frame)
    %   isFirst         - Boolean, true if first image in batch
    %
    % Outputs:
    %   ellipses - Cell array of drawellipse objects positioned in rotated display frame
    %
    % Coordinate transformation pipeline:
    %   1. Memory positions (previous base frame) → scaled base frame (current image)
    %   2. Scaled base frame → rotated display frame (if rotation applied)
    %   3. Ellipse angles adjusted by rotation offset

    [imageHeight, imageWidth, ~] = size(img);
    baseSize = [size(baseImg, 1), size(baseImg, 2)];
    rotatedSize = [imageHeight, imageWidth];
    appliedRotation = rotationState.total;

    baseSemiMajor = clampToRange(initialGeometry.semiMajor, geometryParams.minPixels, geometryParams.maxPixels);
    baseSemiMinor = clampToRange(initialGeometry.semiMinor, geometryParams.minPixels, baseSemiMajor);
    baseRotation = clampToRange(initialGeometry.rotation, geometryParams.rotationMin, geometryParams.rotationMax);

    defaultPositions = calculateDefaultEllipsePositions(imageWidth, imageHeight, cfg);
    ellipses = cell(1, cfg.numEllipses);

    useMemory = memory.hasMemory && ~isFirst && ~isempty(memory.positions);

    for i = 1:cfg.numEllipses
        if useMemory && i <= size(memory.positions, 1) && memory.positions(i, 1) > 0
            % Memory positions are in base (unrotated) frame
            % Step 1: Scale positions using base-to-base dimensions
            widthScale = 1;
            heightScale = 1;
            if ~isempty(memory.imageWidth) && memory.imageWidth > 0
                widthScale = baseSize(2) / memory.imageWidth;
            end
            if ~isempty(memory.imageHeight) && memory.imageHeight > 0
                heightScale = baseSize(1) / memory.imageHeight;
            end
            % Use geometric mean to preserve ellipse area under anisotropic scaling
            scaleFactor = sqrt(widthScale * heightScale);
            if ~isfinite(scaleFactor) || scaleFactor <= 0
                scaleFactor = 1;
                warning('cutEllipticalPatches:InvalidScaleFactor', ...
                        'Invalid scale factor (widthScale=%.2f, heightScale=%.2f). Using default scale=1.', ...
                        widthScale, heightScale);
            end

            % Warn if aspect ratio changed significantly (indicates mixed-orientation batch)
            aspectRatioDiff = abs(widthScale - heightScale) / max(widthScale, heightScale);
            if aspectRatioDiff > 0.05
                warning('cutEllipticalPatches:AspectRatioChange', ...
                        'Aspect ratio changed by %.1f%% between images (widthScale=%.2f, heightScale=%.2f). Ellipses may be distorted.', ...
                        aspectRatioDiff * 100, widthScale, heightScale);
            end

            scaledBaseCenter = [memory.positions(i, 1) * widthScale, memory.positions(i, 2) * heightScale];

            % Scale semi-axes uniformly using geometric mean (preserves area)
            % For mixed-aspect batches, geometric mean minimizes distortion
            major = memory.positions(i, 3) * scaleFactor;
            minor = memory.positions(i, 4) * scaleFactor;
            baseEllipseRotation = memory.positions(i, 5);

            % Step 2: Transform center from base frame to rotated frame
            if abs(appliedRotation) > rotationState.angleTolerance
                rotatedCenter = applyRotationToPoints(scaledBaseCenter, baseSize, rotatedSize, appliedRotation, rotationState.angleTolerance);
                displayRotation = baseEllipseRotation + appliedRotation;
                displayRotation = mod(displayRotation + 180, 360) - 180;
            else
                rotatedCenter = scaledBaseCenter;
                displayRotation = baseEllipseRotation;
            end

            centerX = rotatedCenter(1);
            centerY = rotatedCenter(2);
            rotation = displayRotation;

            % Clamp to valid ranges
            major = clampToRange(major, geometryParams.minPixels, geometryParams.maxPixels);
            minor = clampToRange(minor, geometryParams.minPixels, min(major, geometryParams.maxPixels));
        else
            centerX = defaultPositions(i, 1);
            centerY = defaultPositions(i, 2);
            major = baseSemiMajor;
            minor = baseSemiMinor;
            rotation = baseRotation;
            if isfield(cfg, 'rotationDefaultAngles') && numel(cfg.rotationDefaultAngles) >= i
                rotation = cfg.rotationDefaultAngles(i);
            end
        end
        rotation = clampToRange(rotation, geometryParams.rotationMin, geometryParams.rotationMax);

        ellipses{i} = drawellipse('Center', [centerX, centerY], ...
                                 'SemiAxes', [major, minor], ...
                                 'RotationAngle', rotation, ...
                                 'StripeColor', cfg.ui.colors.ellipse, 'LineWidth', 2);
    end
end


function instructionText = createInstructions(fig, memory, isFirst, cfg)
    % Add instruction text with updated position
    memoryText = '';
    if memory.hasMemory && ~isFirst
        memoryText = sprintf(' | Loaded: a=%.0f b=%.0f rot=%.0f deg', memory.semiMajorAxis, memory.semiMinorAxis, memory.rotationAngle);
    elseif isFirst
        memoryText = ' | First image - will save positions';
    end

    instructionText = ['Mouse = Drag/Resize Ellipses | APPLY = Accept & Save | SKIP = Skip Image | STOP = Exit' memoryText];

    instructionText = uicontrol('Parent', fig, 'Style', 'text', ...
             'String', instructionText, ...
             'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
             'FontSize', cfg.ui.fontSize.instruction, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.background, 'HorizontalAlignment', 'center');
end


function buttonPanel = createPreviewButtons(fig, cfg)
    % Create simplified preview action buttons without instruction text
    buttonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                         'Position', cfg.ui.positions.previewPanel, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'BorderType', 'line', 'HighlightColor', cfg.ui.colors.foreground, ...
                         'BorderWidth', 2);
    
    % Create buttons with adjusted positions (more centered vertically)
    createActionButton(buttonPanel, 'ACCEPT', [0.05 0.25 0.25 0.50], cfg.ui.colors.accept, cfg.ui.colors.foreground, 'accept', fig, cfg);
    createActionButton(buttonPanel, 'RETRY', [0.375 0.25 0.25 0.50], cfg.ui.colors.retry, cfg.ui.colors.background, 'retry', fig, cfg);
    createActionButton(buttonPanel, 'SKIP', [0.70 0.25 0.25 0.50], cfg.ui.colors.skip, cfg.ui.colors.foreground, 'skip', fig, cfg);
end

function imgHandle = createPreviewInExistingAxes(axesHandle, img, coords, cfg)
    % Create preview in existing axes maintaining size and position

    % Create dimmed background with highlighted patches
    maskedImg = createMaskedImage(img, coords, cfg.dimFactor);
    imgHandle = imshow(maskedImg, 'Parent', axesHandle, 'InitialMagnification', 'fit');
    
    % ASCII-only preview title to avoid encoding surprises on some platforms
    previewTitle = sprintf('Preview: %d elliptical patches (%d replicates per concentration)', size(coords, 1), cfg.replicatesPerConcentration);
    title(axesHandle, previewTitle, ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold');
    hold(axesHandle, 'on');
    
    % Add simple patch numbers with small, unobtrusive labels
    for i = 1:size(coords, 1)
        if coords(i, 1) > 0 && coords(i, 2) > 0
            % Add simple patch number
            text(coords(i, 1), coords(i, 2), num2str(i), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', cfg.ui.fontSize.value, 'FontWeight', 'bold', ...
                'Color', cfg.ui.colors.preview, ...
                'BackgroundColor', [0 0 0 0.5], ...
                'EdgeColor', 'none', ...
                'Parent', axesHandle);
        end
    end
    hold(axesHandle, 'off');
end

function maskedImg = createMaskedImage(img, coords, dimFactor)
    % Create masked image showing only ellipse regions (cached meshgrid for performance)
    % Vectorized mask application across all channels
    [height, width, numChannels] = size(img);

    % Persistent cache for meshgrid (reused across preview calls in same session)
    % NOTE: Assumes single-threaded execution (standard for MATLAB interactive scripts)
    persistent lastSize Xcache Ycache; 
    if isempty(lastSize) || numel(lastSize) ~= 2 || any(lastSize ~= [height, width])
        [Xcache, Ycache] = meshgrid(1:width, 1:height);
        lastSize = [height, width];
    end

    mask = false(height, width);

    % Build combined mask for all ellipses (use cached meshgrid)
    for i = 1:size(coords, 1)
        if coords(i, 1) > 0 && coords(i, 2) > 0
            x = coords(i, 1);
            y = coords(i, 2);
            a = coords(i, 3);  % semiMajorAxis
            b = coords(i, 4);  % semiMinorAxis
            theta = deg2rad(coords(i, 5));  % rotationAngle

            % Rotate coordinates to ellipse's principal axes frame
            dx = Xcache - x;
            dy = Ycache - y;
            x_rot =  dx * cos(theta) + dy * sin(theta);
            y_rot = -dx * sin(theta) + dy * cos(theta);

            % Ellipse equation: (x_rot/a)^2 + (y_rot/b)^2 <= 1
            ellipseMask = (x_rot ./ a).^2 + (y_rot ./ b).^2 <= 1;
            mask = mask | ellipseMask;
        end
    end

    % Vectorized dimming: apply mask to all channels at once
    maskedImg = double(img);
    inverseMask3D = repmat(~mask, [1, 1, numChannels]);
    maskedImg(inverseMask3D) = maskedImg(inverseMask3D) * dimFactor;
    maskedImg = uint8(maskedImg);
end

%% User Interaction
function keyPressHandler(src, event)
    % Handle keyboard input
    guiData = get(src, 'UserData');
    
    switch event.Key
        case 'space'
            guiData.action = 'accept';
        case 'escape'
            guiData.action = 'skip';
        otherwise
            return;
    end
    
    set(src, 'UserData', guiData);
    uiresume(src);
end

function [action, coords] = waitForUserAction(fig)
    % Wait for user input and return results
    uiwait(fig);

    action = '';
    coords = [];

    if isvalid(fig)
        guiData = get(fig, 'UserData');
        action = guiData.action;

        if strcmp(action, 'accept')
            if strcmp(guiData.mode, 'preview') && isfield(guiData, 'savedCoords')
                coords = guiData.savedCoords;
            elseif strcmp(guiData.mode, 'cutting')
                coords = getEllipseCoordinates(guiData.ellipses);

                % Transform coordinates back to original (unrotated) frame
                if isfield(guiData, 'rotation') && ~isempty(coords)
                    rotationTotal = guiData.rotation.total;
                    if abs(rotationTotal) > guiData.rotation.angleTolerance
                        imageSize = [size(guiData.currentImg, 1), size(guiData.currentImg, 2)];
                        baseSize = [size(guiData.baseImg, 1), size(guiData.baseImg, 2)];

                        % Transform ellipse centers back to original frame
                        coords(:, 1:2) = inverseRotatePoints(coords(:, 1:2), imageSize, baseSize, rotationTotal, guiData.rotation.angleTolerance);

                        % Adjust ellipse rotation angles (applied AFTER axis constraint)
                        coords(:, 5) = coords(:, 5) - rotationTotal;

                        % Normalize angles to [-180, 180]
                        coords(:, 5) = mod(coords(:, 5) + 180, 360) - 180;
                    end
                end
            end

            if isempty(coords)
                action = 'skip';
            end
        end
    end
end

function coords = getEllipseCoordinates(ellipses)
    % Get coordinates from ellipse objects (returns x, y, semiMajor, semiMinor, rotation)
    coords = zeros(length(ellipses), 5);

    for i = 1:length(ellipses)
        if isvalid(ellipses{i})
            center = ellipses{i}.Center;
            semiAxes = ellipses{i}.SemiAxes;

            % Enforce constraint: semiMajorAxis >= semiMinorAxis
            semiMajorAxis = max(semiAxes);
            semiMinorAxis = min(semiAxes);
            rotation = ellipses{i}.RotationAngle;

            % Adjust rotation if axes were swapped
            if semiAxes(1) < semiAxes(2)
                rotation = rotation + 90;
            end

            % Normalize rotation to [-180, 180]
            rotation = mod(rotation + 180, 360) - 180;

            % Only round pixel positions; preserve decimal precision for dimensions and angles
            coords(i, :) = [round(center), semiMajorAxis, semiMinorAxis, rotation];
        end
    end

    % Remove invalid entries (x > 0)
    coords = coords(coords(:, 1) > 0, :);
end

function stopExecution(fig)
    % Handle stop button
    guiData = get(fig, 'UserData');
    guiData.action = 'stop';
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function setPreviewAction(fig, action)
    % Set action from preview buttons
    guiData = get(fig, 'UserData');
    guiData.action = action;
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

%% File Operations
function saveCoordinates(phoneOutputDir, imageName, concIdx, coords, cfg)
    % Save coordinates to phone-level coordinates.txt for an image.
    % Replaces existing entries for that image if present using atomic write.
    %
    % Inputs:
    %   phoneOutputDir - Phone-level output directory path
    %   imageName      - Name of source image (used as unique key for deduplication)
    %   concIdx        - Concentration index (integer)
    %   coords         - [N x 5] matrix: [x, y, semiMajorAxis, semiMinorAxis, rotationAngle]
    %   cfg            - Configuration structure with coordinateFileName field
    %
    % File format: Space-delimited with header
    %   image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
    %
    % Behavior:
    %   - Lazily creates coordinates.txt with header on first write
    %   - Ensures no duplicate rows per image (existing entries replaced)
    %   - Uses atomic write (temp file + move) to prevent corruption

    COORD_NUM_COLUMNS = 7;  % concentration, replicate, x, y, a, b, theta

    coordFolder = phoneOutputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, cfg.coordinateFileName);

    header = 'image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle';

    % Read existing entries (if any) and drop rows for this image
    existingNames = {};
    existingNums = zeros(0, COORD_NUM_COLUMNS);
    if exist(coordPath, 'file') == 2
        try
            % Read entire file and parse line by line (more robust than mixed fgetl/textscan)
            fileContent = fileread(coordPath);
            lines = strsplit(fileContent, {'\n', '\r\n'}, 'CollapseDelimiters', false);

            % Estimate capacity from line count for pre-allocation
            estimatedLines = length(lines);
            existingNames = cell(estimatedLines, 1);
            existingNums = zeros(estimatedLines, COORD_NUM_COLUMNS);
            rowCount = 0;

            % First line should be header; skip if it matches expected format
            startIdx = 1;
            if ~isempty(lines) && ~isempty(lines{1}) && contains(lower(lines{1}), 'image concentration')
                startIdx = 2;  % Skip header
            end

            % Parse data lines
            for i = startIdx:length(lines)
                trimmedLine = strtrim(lines{i});
                if isempty(trimmedLine), continue; end

                parts = strsplit(trimmedLine);
                if numel(parts) >= (COORD_NUM_COLUMNS + 1)
                    nums = str2double(parts(2:(COORD_NUM_COLUMNS + 1)));
                    if all(~isnan(nums))
                        rowCount = rowCount + 1;
                        existingNames{rowCount} = parts{1};
                        existingNums(rowCount, :) = nums;
                    end
                end
            end

            % Trim to actual size
            existingNames = existingNames(1:rowCount);
            existingNums = existingNums(1:rowCount, :);
        catch ME
            error('cutEllipticalPatches:CorruptCoordinateFile', ...
                  'Cannot parse coordinates file %s: %s\nAborting to prevent data loss. Please fix or delete the corrupted file manually.', ...
                  coordPath, ME.message);
        end
    end

    % Filter out any existing rows for this image (log if overwriting)
    if ~isempty(existingNames)
        keepMask = ~strcmp(existingNames, imageName);
        numOverwritten = sum(~keepMask);
        if numOverwritten > 0
            fprintf('  [INFO] Replacing %d existing coordinate(s) for %s\n', ...
                    numOverwritten, imageName);
        end
        existingNames = existingNames(keepMask);
        if ~isempty(existingNums)
            existingNums = existingNums(keepMask, :);
        end
    end

    % Build new rows for this image (pre-allocate arrays to avoid dynamic growth)
    validMask = coords(:, 1) > 0;  % Valid coordinates (x > 0)
    numValid = sum(validMask);
    newNames = cell(numValid, 1);
    newNums = zeros(numValid, COORD_NUM_COLUMNS);

    validIdx = 0;
    for i = 1:size(coords, 1)
        if coords(i, 1) > 0  % Valid coordinate (x > 0)
            validIdx = validIdx + 1;
            replicate = i;  % 1-based replicate index
            newNames{validIdx} = imageName;
            % coords format: [x, y, semiMajor, semiMinor, rotation]
            newNums(validIdx, :) = [concIdx, replicate, coords(i, 1), coords(i, 2), coords(i, 3), coords(i, 4), coords(i, 5)];
        end
    end

    % Rewrite atomically: write to a temp file in the same folder, then move over
    % Generate temp file in target directory with guaranteed uniqueness
    [~, baseName] = fileparts(cfg.coordinateFileName);
    [~, tmpSuffix] = fileparts(tempname);
    tmpPath = fullfile(coordFolder, sprintf('.%s_%s_tmp', baseName, tmpSuffix));

    fid_w = fopen(tmpPath, 'wt');
    if fid_w == -1
        warning('cutEllipticalPatches:CoordOpen', 'Cannot open temp coordinates file for writing: %s', tmpPath);
        return;
    end
    fprintf(fid_w, '%s\n', header);
    % Write kept existing rows (conc and rep as integers, x and y as integers, a b theta as floats)
    for j = 1:numel(existingNames)
        fprintf(fid_w, '%s %d %d %d %d %.2f %.2f %.2f\n', existingNames{j}, existingNums(j, 1), existingNums(j, 2), ...
                existingNums(j, 3), existingNums(j, 4), existingNums(j, 5), existingNums(j, 6), existingNums(j, 7));
    end
    % Write new rows for current image (conc and rep as integers, x and y as integers, a b theta as floats)
    for j = 1:numel(newNames)
        fprintf(fid_w, '%s %d %d %d %d %.2f %.2f %.2f\n', newNames{j}, newNums(j, 1), newNums(j, 2), ...
                newNums(j, 3), newNums(j, 4), newNums(j, 5), newNums(j, 6), newNums(j, 7));
    end
    fclose(fid_w);

    % Move temp file over destination (force overwrite)
    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        % Log the actual error before fallback
        warning('cutEllipticalPatches:CoordMove', ...
                'Failed to move temp file: %s (%s). Attempting fallback copy.', msg, msgid);

        % Fallback: direct copy (not ideal but better than nothing)
        try
            copyfile(tmpPath, coordPath, 'f');
            delete(tmpPath);
        catch ME2
            error('cutEllipticalPatches:CoordWriteFailed', ...
                  'Cannot save coordinates to %s. Original error: %s. Fallback error: %s', ...
                  coordPath, msg, ME2.message);
        end
    end
end

function saveCutPatchesToConcentrationFolders(img, imageName, concIdx, coords, outputDir, cfg)
    % Save individual elliptical patch images in concentration-based folders
    [~, nameNoExt, ~] = fileparts(imageName);

    fprintf('  Saving elliptical patches to concentration folders...\n');

    % Pre-determine output extension once (constant across all patches)
    outExt = '.png';

    % Create concentration folder
    concFolder = fullfile(outputDir, sprintf('%s%d', cfg.concFolderPrefix, concIdx));
    if ~exist(concFolder, 'dir')
        mkdir(concFolder);
    end

    for i = 1:size(coords, 1)
        if coords(i, 1) > 0 && coords(i, 2) > 0
            replicate = i;  % 1-based replicate index

            % Extract ellipse parameters from coords
            x = coords(i, 1);
            y = coords(i, 2);
            a = coords(i, 3);  % semiMajorAxis
            b = coords(i, 4);  % semiMinorAxis
            theta = coords(i, 5);  % rotationAngle

            % Calculate axis-aligned bounding box for rotated ellipse
            % For ellipse rotated by theta with semi-axes a, b:
            %   Half-width:  ux = sqrt((a*cos(theta))^2 + (b*sin(theta))^2)
            %   Half-height: uy = sqrt((a*sin(theta))^2 + (b*cos(theta))^2)
            % Derivation: envelope of rotated ellipse in axis-aligned frame
            theta_rad = deg2rad(theta);
            ux = sqrt((a * cos(theta_rad))^2 + (b * sin(theta_rad))^2);
            uy = sqrt((a * sin(theta_rad))^2 + (b * cos(theta_rad))^2);

            x1 = max(1, floor(x - ux));
            y1 = max(1, floor(y - uy));
            x2 = min(size(img, 2), ceil(x + ux));
            y2 = min(size(img, 1), ceil(y + uy));

            % Validate bounding box dimensions
            if (x2 - x1 < 1) || (y2 - y1 < 1)
                warning('cutEllipticalPatches:DegenerateBoundingBox', ...
                        'Ellipse %d (%s_con%d_rep%d) has degenerate bounding box [%d:%d, %d:%d]. Skipping.', ...
                        i, nameNoExt, concIdx, replicate, y1, y2, x1, x2);
                continue;
            end

            % Extract region containing the ellipse
            patchRegion = img(y1:y2, x1:x2, :);

            % Create elliptical mask using vectorized operations
            [patchH, patchW, numChannels] = size(patchRegion);
            [Xpatch, Ypatch] = meshgrid(1:patchW, 1:patchH);

            % Adjust ellipse center relative to the patch
            centerX_patch = x - x1 + 1;
            centerY_patch = y - y1 + 1;

            % Rotate coordinates to ellipse's principal axes frame
            dx = Xpatch - centerX_patch;
            dy = Ypatch - centerY_patch;
            x_rot =  dx * cos(theta_rad) + dy * sin(theta_rad);
            y_rot = -dx * sin(theta_rad) + dy * cos(theta_rad);

            % Create elliptical mask: (x_rot/a)^2 + (y_rot/b)^2 <= 1
            ellipseMask = (x_rot ./ a).^2 + (y_rot ./ b).^2 <= 1;

            % Validate that patch contains pixels inside the ellipse
            if ~any(ellipseMask(:))
                warning('cutEllipticalPatches:EllipseOutOfBounds', ...
                        'Ellipse %d (%s_con%d_rep%d) lies outside image bounds. Skipped (expected for boundary cases).', ...
                        i, nameNoExt, concIdx, replicate);
                continue;
            end

            % Vectorized mask application across all channels
            ellipticalPatch = patchRegion;
            inverseMask3D = repmat(~ellipseMask, [1, 1, numChannels]);
            ellipticalPatch(inverseMask3D) = 0;

            % Save with format-aware settings
            patchFileName = sprintf(cfg.patchFilenameFormat, nameNoExt, concIdx, replicate, outExt);
            patchPath = fullfile(concFolder, patchFileName);
            saveImageWithFormat(ellipticalPatch, patchPath, outExt, cfg);
        end
    end

    fprintf('  >> Saved %d elliptical patches\n', size(coords, 1));
end

%% Utility Functions
function memory = initializeMemory()
    % Initialize memory structure
    memory = struct('hasMemory', false, ...
                   'positions', [], ...
                   'semiMajorAxis', [], ...
                   'semiMinorAxis', [], ...
                   'rotationAngle', [], ...
                   'imageWidth', [], ...
                   'imageHeight', []);
end

function memory = updateMemory(coords, rotationTotal, img)
    % Update memory with current settings derived from accepted ellipses
    [h, w, ~] = size(img);

    memory.hasMemory = ~isempty(coords);
    memory.positions = coords;
    memory.rotation = rotationTotal;  % Store UI rotation for next image
    if memory.hasMemory
        memory.semiMajorAxis = coords(1, 3);
        memory.semiMinorAxis = coords(1, 4);
        memory.rotationAngle = coords(1, 5);
    else
        memory.semiMajorAxis = [];
        memory.semiMinorAxis = [];
        memory.rotationAngle = [];
    end
    memory.imageWidth = w;
    memory.imageHeight = h;

    if memory.hasMemory
        fprintf('  Memory updated: %d ellipses, a=%.1f, b=%.1f, rot=%.1f deg, UI rotation=%.1f deg\n', ...
                size(coords, 1), memory.semiMajorAxis, memory.semiMinorAxis, memory.rotationAngle, rotationTotal);
    else
        fprintf('  Memory cleared for this folder (no accepted ellipses)\n');
    end
end

function guiData = initializeGUIData(guiData, width, height)
    % Initialize GUI data structure for cutting mode
    guiData.imageSize = struct('width', width, 'height', height);
    guiData.action = '';
    % mode is already set in buildEllipticalPatchCuttingUI
end

function [img, isValid] = loadImage(imageName)
    % Load and validate image
    isValid = false;
    try
        img = imread_raw(imageName);
        isValid = true;
    catch ME
        warning('cutEllipticalPatches:ReadFailure', 'Cannot read %s: %s', imageName, ME.message);
        img = [];
    end
end

function displayProcessingInfo(imageName, memory, isFirst)
    % Display processing information
    fprintf('\nImage: %s', imageName);
    if isFirst
        fprintf(' (First image - will establish defaults)\n');
    elseif memory.hasMemory
        fprintf(' (Using saved: a=%.1f px, b=%.1f px, rot=%.1f deg)\n', memory.semiMajorAxis, memory.semiMinorAxis, memory.rotationAngle);
    else
        fprintf(' (No memory available, using defaults)\n');
    end
end

function logProcessingResult(success, isFirst)
    % Log processing result
    if success
        if isFirst
            fprintf('>> Memory established for this folder\n');
        else
            fprintf('>> Using and updating memory\n');
        end
    else
        fprintf('!! Image skipped\n');
    end
end

function btn = createActionButton(parent, text, pos, bgColor, fgColor, action, fig, cfg)
    % Create action button for preview
    btn = uicontrol('Parent', parent, 'Style', 'pushbutton', ...
                   'String', text, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold', ...
                   'Units', 'normalized', 'Position', pos, ...
                   'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                   'Callback', @(~,~) setPreviewAction(fig, action));
end

%% Path and File Utilities
function validatePaths(cfg)
    % Validate and create directories
    if ~exist(cfg.inputPath, 'dir')
        error('cutEllipticalPatches:MissingInputDirectory', ...
              'Input directory not found: %s\n\nExpected stage 2 output from cut_micropads.m', ...
              cfg.inputPath);
    end

    % Validate stage numbering consistency (expected: 2_ -> 3_)
    if ~contains(cfg.inputPath, '2_') || ~contains(cfg.outputPath, '3_')
        warning('cutEllipticalPatches:StageNumbering', ...
                'Input/output folders do not follow expected pipeline stage numbering (2_ -> 3_).\nInput: %s\nOutput: %s', ...
                cfg.inputPath, cfg.outputPath);
    end

    fullOutputPath = fullfile(cfg.projectRoot, cfg.outputPath);
    if ~exist(fullOutputPath, 'dir')
        mkdir(fullOutputPath);
        fprintf('Created output directory: %s\n', fullOutputPath);
    end
end

function outputDir = createOutputStructure(basePath, phoneName, projectRoot)
    outputDir = fullfile(projectRoot, basePath, phoneName);
    if ~(exist(outputDir, 'dir') == 7)
        mkdir(outputDir);
        fprintf('Created output directory: %s\n', outputDir);
    end
end

function pathStr = composePathSegments(varargin)
    parts = varargin(~cellfun(@isempty, varargin));
    if isempty(parts)
        pathStr = '';
    else
        pathStr = strjoin(parts, '/');
    end
end

function executeInFolder(folder, func)
    % Execute function in folder with error handling
    if isempty(folder)
        folder = '.';
    end
    currentDir = pwd;
    cd(folder);
    try
        func();
    catch ME
        cd(currentDir);
        rethrow(ME);
    end
    cd(currentDir);
end

function folders = getSubFolders(path)
    % Get list of subdirectories
    items = dir(path);
    isFolder = [items.isdir];
    names = {items(isFolder).name};
    folders = names(~ismember(names, {'.', '..'}));
end

function concFolders = findConcentrationFolders(dirPath, prefix)
    if ~exist('dirPath', 'var') || isempty(dirPath)
        dirPath = '.';
    end
    if ~exist('prefix', 'var') || isempty(prefix)
        prefix = 'con_';
    end
    entries = dir(fullfile(dirPath, [prefix '*']));
    entries = entries([entries.isdir]);
    concFolders = sort({entries.name});
end


function files = getImageFiles(path, extensions)
    % Get list of image files, unique and sorted
    % Single dir() call with post-filtering by extension
    allFiles = dir(path);
    allFiles = allFiles(~[allFiles.isdir]);  % Remove directories

    if isempty(allFiles)
        files = {};
        return;
    end

    % Extract extensions and convert patterns to actual extensions
    fileNames = {allFiles.name};
    validExts = cellfun(@(x) lower(strrep(x, '*', '')), extensions, 'UniformOutput', false);

    % Pre-allocate for worst case (all files match)
    files = cell(1, length(fileNames));
    fileCount = 0;

    % Filter by extension
    for i = 1:length(fileNames)
        [~, ~, ext] = fileparts(fileNames{i});
        if any(strcmpi(ext, validExts))
            fileCount = fileCount + 1;
            files{fileCount} = fileNames{i};
        end
    end

    % Trim to actual size
    files = files(1:fileCount);

    if ~isempty(files)
        files = unique(files);  % unique() returns sorted results by default
    end
end

function handleError(ME)
    % Handle errors appropriately
    if strcmp(ME.message, 'User stopped execution')
        fprintf('!! Script stopped by user\n');
    else
        fprintf('!! Error: %s\n', ME.message);
        rethrow(ME);
    end
end

%% Dynamic Path Resolution Functions
function cfg = addPathConfiguration(cfg, inputFolder, outputFolder)
    % Add path configuration with dynamic resolution
    projectRoot = findProjectRoot(inputFolder, cfg.maxProjectRootSearchLevels);
    cfg.projectRoot = projectRoot;
    cfg.inputPath = fullfile(projectRoot, inputFolder);
    cfg.outputPath = outputFolder;  % Store relative name for createOutputStructure
end

function projectRoot = findProjectRoot(inputFolder, maxLevels)
    % Find project root by searching for the input folder
    currentDir = pwd;
    searchDir = currentDir;
    levelsChecked = 0;
    % maxLevels provided by caller (safety limit to prevent infinite loops)
    
    while levelsChecked < maxLevels
        if exist(fullfile(searchDir, inputFolder), 'dir')
            projectRoot = searchDir;
            return;
        end
        
        [parentDir, ~] = fileparts(searchDir);
        
        % Move up one level; stop if already at filesystem root
        if strcmp(searchDir, parentDir)
            break; % Reached root directory
        end
        
        searchDir = parentDir;
        levelsChecked = levelsChecked + 1;
    end
    
    % Fallback: use current directory and issue warning
    warning('cutEllipticalPatches:ProjectRootFallback', 'Could not find input folder "%s". Using current directory as project root.', inputFolder);
    projectRoot = currentDir;
end

%% Output formatting helpers
function saveImageWithFormat(img, outPath, ~, ~)
    imwrite(img, outPath);
end

%% -------------------------------------------------------------------------
%% Rotation Helper Functions
%% -------------------------------------------------------------------------

function rotationPanel = createRotationPanel(fig, cfg)
    % Create rotation panel with preset angle buttons
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

function applyRotation_UI(angle, fig, cfg)
    % Apply preset rotation angle (UI alignment only)
    guiData = get(fig, 'UserData');
    if ~strcmp(guiData.mode, 'cutting')
        return;
    end

    % Skip update if rotation angle hasn't changed
    if abs(guiData.rotation.total - angle) < cfg.rotation.angleTolerance
        return;
    end

    % Update rotation state
    guiData.rotation.total = angle;

    % Validate image dimensions (defensive check)
    currentWidth = guiData.imageSize.width;
    currentHeight = guiData.imageSize.height;
    if currentWidth <= 0 || currentHeight <= 0
        warning('cutEllipticalPatches:InvalidDimensions', ...
                'Image dimensions are invalid (width=%d, height=%d). Rotation aborted.', ...
                currentWidth, currentHeight);
        return;
    end

    % Convert ellipse positions to normalized axes coordinates [0, 1] before rotation
    numEllipses = length(guiData.ellipses);
    ellipseNormalized = zeros(numEllipses, 2);
    for i = 1:numEllipses
        if isvalid(guiData.ellipses{i})
            centerData = guiData.ellipses{i}.Center;  % Data coordinates [x, y]
            % Convert to normalized axes coordinates [0, 1]
            ellipseNormalized(i, 1) = (centerData(1) - 1) / currentWidth;
            ellipseNormalized(i, 2) = (centerData(2) - 1) / currentHeight;
        end
    end

    % Apply rotation to image
    guiData.currentImg = applyRotationToImage(guiData.baseImg, guiData.rotation.total, cfg);
    [newHeight, newWidth, ~] = size(guiData.currentImg);

    % Update image display - direct CData update preserves existing axes children
    if ~isfield(guiData, 'imgHandle') || ~isvalid(guiData.imgHandle)
        error('cutEllipticalPatches:InvalidImageHandle', ...
              'Image handle is missing or invalid. Cannot update rotation.');
    end

    set(guiData.imgHandle, 'CData', guiData.currentImg, ...
                            'XData', [1, newWidth], ...
                            'YData', [1, newHeight]);

    % Snap axes to new image bounds
    axis(guiData.imgAxes, 'image');

    % Update ellipse positions to maintain screen-space locations
    for i = 1:numEllipses
        if isvalid(guiData.ellipses{i})
            % Convert normalized coordinates back to new data coordinates
            newX = 1 + ellipseNormalized(i, 1) * newWidth;
            newY = 1 + ellipseNormalized(i, 2) * newHeight;
            guiData.ellipses{i}.Center = [newX, newY];
        end
    end

    % Update image size
    guiData.imageSize.width = newWidth;
    guiData.imageSize.height = newHeight;

    set(fig, 'UserData', guiData);
end

function rotatedImg = applyRotationToImage(img, rotation, cfg)
    % Apply rotation to image (lossless rot90 for 90-deg multiples, bilinear otherwise)
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

function transformedPoints = inverseRotatePoints(points, rotatedSize, originalSize, rotation, angleTolerance)
    % Transform points from rotated image frame back to original frame
    % ROTATION CONVENTION: Positive rotation = clockwise in image coordinates
    %                      (standard MATLAB imrotate convention)
    % TRANSFORMATION: rotated → original (inverse/reverse rotation)
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
                % rotated: (x_r, y_r), original: (y_r, W_r - x_r + 1)
                x_orig = points(:, 2);
                y_orig = rotatedSize(2) - points(:, 1) + 1;
            case 2  % 180 degrees
                x_orig = rotatedSize(2) - points(:, 1) + 1;
                y_orig = rotatedSize(1) - points(:, 2) + 1;
            case 3  % 90 degrees (rot90(..., 1))
                % rotated: (x_r, y_r), original: (H_r - y_r + 1, x_r)
                x_orig = rotatedSize(1) - points(:, 2) + 1;
                y_orig = points(:, 1);
            otherwise  % 0 degrees
                x_orig = points(:, 1);
                y_orig = points(:, 2);
        end

        transformedPoints = [x_orig, y_orig];
    else
        % For non-90-degree rotations, use geometric transform
        theta = -deg2rad(rotation);  % Inverse rotation
        cosTheta = cos(theta);
        sinTheta = sin(theta);

        % Center of rotated image
        centerRotated = [rotatedSize(2)/2, rotatedSize(1)/2];
        centerOriginal = [originalSize(2)/2, originalSize(1)/2];

        % Translate to origin, rotate, translate back
        pointsCentered = points - centerRotated;
        x_orig = pointsCentered(:, 1) * cosTheta - pointsCentered(:, 2) * sinTheta;
        y_orig = pointsCentered(:, 1) * sinTheta + pointsCentered(:, 2) * cosTheta;

        transformedPoints = [x_orig + centerOriginal(1), y_orig + centerOriginal(2)];
    end
end

function transformedPoints = applyRotationToPoints(points, originalSize, rotatedSize, rotation, angleTolerance)
    % Transform points from original image frame to rotated frame (forward rotation)
    % ROTATION CONVENTION: Positive rotation = clockwise in image coordinates
    %                      (standard MATLAB imrotate convention)
    % TRANSFORMATION: original → rotated (forward rotation)
    if rotation == 0 || isempty(points)
        transformedPoints = points;
        return;
    end

    % Handle exact 90-degree rotations
    if abs(mod(rotation, 90)) < angleTolerance
        numRotations = mod(round(rotation / 90), 4);

        % Map centers forward through discrete rotations (with +1 for MATLAB 1-based indexing)
        switch numRotations
            case 1  % 90 degrees (rot90(..., -1))
                % original: (x, y), rotated: (H - y + 1, x)
                x_rot = originalSize(1) - points(:, 2) + 1;
                y_rot = points(:, 1);
            case 2  % 180 degrees
                x_rot = originalSize(2) - points(:, 1) + 1;
                y_rot = originalSize(1) - points(:, 2) + 1;
            case 3  % -90 degrees (rot90(..., 1))
                % original: (x, y), rotated: (y, W - x + 1)
                x_rot = points(:, 2);
                y_rot = originalSize(2) - points(:, 1) + 1;
            otherwise  % 0 degrees
                x_rot = points(:, 1);
                y_rot = points(:, 2);
        end

        transformedPoints = [x_rot, y_rot];
    else
        % For non-90-degree rotations, use geometric transform
        theta = deg2rad(rotation);  % Forward rotation
        cosTheta = cos(theta);
        sinTheta = sin(theta);

        % Center of original image
        centerOriginal = [originalSize(2)/2, originalSize(1)/2];
        centerRotated = [rotatedSize(2)/2, rotatedSize(1)/2];

        % Translate to origin, rotate, translate back
        pointsCentered = points - centerOriginal;
        x_rot = pointsCentered(:, 1) * cosTheta - pointsCentered(:, 2) * sinTheta;
        y_rot = pointsCentered(:, 1) * sinTheta + pointsCentered(:, 2) * cosTheta;

        transformedPoints = [x_rot + centerRotated(1), y_rot + centerRotated(2)];
    end
end

%% -------------------------------------------------------------------------
%% Image Reading with EXIF Handling
%% -------------------------------------------------------------------------

function I = imread_raw(fname)
% Read image pixels in their recorded layout without applying EXIF orientation
% metadata. Any user-requested rotation is stored in coordinates.txt and applied
% during downstream processing rather than via image metadata.

    I = imread(fname);
end

