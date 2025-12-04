function ui = micropad_ui()
    %% MICROPAD_UI Returns a struct of function handles for UI operations
    %
    % This utility module provides functions for creating and managing the
    % microPAD processing GUI. Includes figure management, control panels,
    % quadrilateral visualization, and zoom functionality.
    %
    % Usage:
    %   ui = micropad_ui();
    %   cfg = ui.getDefaultUIConfig();
    %   fig = ui.createFigure('image.jpg', 'Phone1', cfg);
    %   [axes, handle] = ui.createImageAxes(fig, img, cfg);
    %
    % Configuration:
    %   The UI configuration structure contains:
    %     .fontSize  - Font sizes for different UI elements
    %     .colors    - Color definitions for UI elements
    %     .positions - Normalized positions [x, y, width, height]
    %     .layout    - Panel-internal layout positions
    %     .quad      - Quadrilateral styling options
    %     .rotation  - Rotation control settings
    %     .zoom      - Zoom control settings
    %
    % See also: cut_micropads, coordinate_io, geometry_transform

    %% Public API
    % Configuration
    ui.getDefaultUIConfig = @getDefaultUIConfig;
    ui.mergeUIConfig = @mergeUIConfig;

    % Figure and display
    ui.createFigure = @createFigure;
    ui.createTitle = @createTitle;
    ui.createPathDisplay = @createPathDisplay;
    ui.createImageAxes = @createImageAxes;
    ui.createInstructions = @createInstructions;

    % Action buttons
    ui.createStopButton = @createStopButton;
    ui.createRunAIButton = @createRunAIButton;
    ui.createEditButtonPanel = @createEditButtonPanel;
    ui.createPreviewButtons = @createPreviewButtons;
    ui.createEllipseEditingButtonPanel = @createEllipseEditingButtonPanel;

    % Control panels
    ui.createRotationButtonPanel = @createRotationButtonPanel;
    ui.createZoomPanel = @createZoomPanel;
    ui.createEllipseZoomPanel = @createEllipseZoomPanel;
    ui.createAIStatusLabel = @createAIStatusLabel;

    % Quadrilateral visualization
    ui.createQuads = @createQuads;
    ui.addQuadLabels = @addQuadLabels;
    ui.updateQuadLabels = @updateQuadLabels;
    ui.getConcentrationColor = @getConcentrationColor;
    ui.setQuadColor = @setQuadColor;
    ui.createEllipseROI = @createEllipseROI;

    % Preview
    ui.createPreviewAxes = @createPreviewAxes;
    ui.createMaskedPreview = @createMaskedPreview;

    % Zoom functions
    ui.applyZoomToAxes = @applyZoomToAxes;
    ui.calculateQuadBounds = @calculateQuadBounds;
    ui.calculateEllipseBounds = @calculateEllipseBounds;

    % State management
    ui.setAction = @setAction;
    ui.stopExecution = @stopExecution;
    ui.keyPressHandler = @keyPressHandler;

    % UI cleanup
    ui.clearAllUIElements = @clearAllUIElements;
    ui.captureViewState = @captureViewState;
    ui.applyViewState = @applyViewState;

    % AI progress indicator
    ui.showAIProgressIndicator = @showAIProgressIndicator;
    ui.captureQuadColors = @captureQuadColors;

    % Timer utilities
    ui.safeStopTimer = @safeStopTimer;
end

%% =========================================================================
%% CONFIGURATION
%% =========================================================================

function cfg = getDefaultUIConfig()
    % Get default UI configuration structure
    %
    % OUTPUTS:
    %   cfg - Struct with UI configuration settings
    %
    % The configuration includes:
    %   - fontSize: Font sizes for title, path, button, info, etc.
    %   - colors: Color definitions for background, foreground, panels, buttons
    %   - positions: Normalized positions for all UI elements
    %   - layout: Internal layout positions within panels
    %   - quad: Quadrilateral styling (lineWidth, borderWidth)
    %   - rotation: Rotation control settings (range, quickAngles)
    %   - zoom: Zoom control settings (range, defaultValue)

    cfg = struct();

    % Font sizes
    cfg.fontSize = struct(...
        'title', 16, ...
        'path', 12, ...
        'button', 13, ...
        'info', 10, ...
        'instruction', 10, ...
        'preview', 14, ...
        'label', 12, ...
        'status', 13, ...
        'value', 13);

    % Colors
    cfg.colors = struct(...
        'background', 'black', ...
        'foreground', 'white', ...
        'panel', [0.15 0.15 0.15], ...
        'stop', [0.85 0.2 0.2], ...
        'accept', [0.2 0.75 0.3], ...
        'retry', [0.9 0.75 0.2], ...
        'skip', [0.75 0.25 0.25], ...
        'quad', [0.0 1.0 1.0], ...
        'info', [1.0 1.0 0.3], ...
        'path', [0.75 0.75 0.75], ...
        'apply', [0.2 0.5 0.9]);

    % UI positions (normalized coordinates [x, y, width, height])
    % Origin: (0, 0) = bottom-left, (1, 1) = top-right
    cfg.positions = struct(...
        'figure', [0 0 1 1], ...
        'stopButton', [0.01 0.945 0.06 0.045], ...
        'title', [0.08 0.945 0.84 0.045], ...
        'pathDisplay', [0.08 0.90 0.84 0.035], ...
        'aiStatus', [0.25 0.905 0.50 0.035], ...
        'instructions', [0.01 0.855 0.98 0.035], ...
        'image', [0.01 0.215 0.98 0.64], ...
        'runAIButton', [0.01 0.16 0.08 0.045], ...
        'rotationPanel', [0.01 0.01 0.24 0.14], ...
        'zoomPanel', [0.26 0.01 0.26 0.14], ...
        'cutButtonPanel', [0.53 0.01 0.46 0.14], ...
        'previewPanel', [0.25 0.01 0.50 0.14], ...
        'previewTitle', [0.01 0.92 0.98 0.04], ...
        'previewMeta', [0.01 0.875 0.98 0.035], ...
        'previewLeft', [0.01 0.22 0.48 0.64], ...
        'previewRight', [0.50 0.22 0.49 0.64]);

    % Panel-internal layouts
    cfg.layout = struct();
    cfg.layout.rotationLabel = [0.05 0.78 0.90 0.18];
    cfg.layout.quickRotationRow1 = {[0.05 0.42 0.42 0.30], [0.53 0.42 0.42 0.30]};
    cfg.layout.quickRotationRow2 = {[0.05 0.08 0.42 0.30], [0.53 0.08 0.42 0.30]};
    cfg.layout.zoomLabel = [0.05 0.78 0.90 0.18];
    cfg.layout.zoomSlider = [0.05 0.42 0.72 0.28];
    cfg.layout.zoomValue = [0.79 0.42 0.16 0.28];
    cfg.layout.zoomResetButton = [0.05 0.08 0.44 0.28];
    cfg.layout.zoomAutoButton = [0.51 0.08 0.44 0.28];
    cfg.layout.ellipseZoomPrevButton = [0.05 0.42 0.18 0.28];
    cfg.layout.ellipseZoomIndicator = [0.25 0.42 0.35 0.28];
    cfg.layout.ellipseZoomNextButton = [0.62 0.42 0.18 0.28];
    cfg.layout.ellipseZoomResetButton = [0.25 0.08 0.50 0.28];

    % Quadrilateral styling
    cfg.quad = struct(...
        'lineWidth', 3, ...
        'borderWidth', 2);

    % Rotation settings
    cfg.rotation = struct(...
        'range', [-180, 180], ...
        'quickAngles', [-90, 0, 90, 180]);

    % Zoom settings
    cfg.zoom = struct(...
        'range', [0, 1], ...
        'defaultValue', 0);

    % Dim factor for non-selected regions
    cfg.dimFactor = 0.2;
end

function merged = mergeUIConfig(base, overrides)
    % Merge UI configuration with overrides
    %
    % INPUTS:
    %   base      - Base UI configuration struct
    %   overrides - Struct with override values
    %
    % OUTPUTS:
    %   merged - Merged configuration
    %
    % Recursively merges struct fields, with overrides taking precedence.

    merged = base;

    if isempty(overrides)
        return;
    end

    fields = fieldnames(overrides);
    for i = 1:numel(fields)
        field = fields{i};
        if isfield(base, field) && isstruct(base.(field)) && isstruct(overrides.(field))
            merged.(field) = mergeUIConfig(base.(field), overrides.(field));
        else
            merged.(field) = overrides.(field);
        end
    end
end

%% =========================================================================
%% FIGURE AND DISPLAY
%% =========================================================================

function fig = createFigure(imageName, phoneName, cfg, keyPressCallback, closeCallback)
    % Create main figure window
    %
    % INPUTS:
    %   imageName        - Name of current image
    %   phoneName        - Name of current phone folder
    %   cfg              - UI configuration struct
    %   keyPressCallback - (Optional) Function handle for key press events
    %   closeCallback    - (Optional) Function handle for close request
    %
    % OUTPUTS:
    %   fig - Figure handle

    if nargin < 4 || isempty(keyPressCallback)
        keyPressCallback = @keyPressHandler;
    end

    if nargin < 5 || isempty(closeCallback)
        closeCallback = @(src, ~) defaultCloseCallback(src);
    end

    titleText = sprintf('microPAD Processor - %s - %s', phoneName, imageName);
    fig = figure('Name', titleText, ...
                'Units', 'normalized', 'Position', cfg.ui.positions.figure, ...
                'MenuBar', 'none', 'ToolBar', 'none', ...
                'Color', cfg.ui.colors.background, ...
                'KeyPressFcn', keyPressCallback, ...
                'CloseRequestFcn', closeCallback);

    drawnow limitrate;
    pause(0.05);
    set(fig, 'WindowState', 'maximized');
    figure(fig);
    drawnow limitrate;
end

function defaultCloseCallback(fig)
    % Default close callback - set stop action and close
    guiData = get(fig, 'UserData');
    if ~isempty(guiData) && isstruct(guiData)
        guiData.action = 'stop';
        set(fig, 'UserData', guiData);
    end
    delete(fig);
end

function titleHandle = createTitle(fig, phoneName, imageName, cfg)
    % Create title text control
    %
    % INPUTS:
    %   fig       - Parent figure
    %   phoneName - Phone folder name
    %   imageName - Image file name
    %   cfg       - UI configuration
    %
    % OUTPUTS:
    %   titleHandle - Handle to title text control

    titleText = sprintf('%s - %s', phoneName, imageName);
    titleHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', titleText, ...
                           'Units', 'normalized', 'Position', cfg.ui.positions.title, ...
                           'FontSize', cfg.ui.fontSize.title, 'FontWeight', 'bold', ...
                           'ForegroundColor', cfg.ui.colors.foreground, ...
                           'BackgroundColor', cfg.ui.colors.background, ...
                           'HorizontalAlignment', 'center');
end

function pathHandle = createPathDisplay(fig, phoneName, imageName, cfg)
    % Create path display text control
    %
    % INPUTS:
    %   fig       - Parent figure
    %   phoneName - Phone folder name
    %   imageName - Image file name
    %   cfg       - UI configuration
    %
    % OUTPUTS:
    %   pathHandle - Handle to path text control

    pathText = sprintf('Path: %s | Image: %s', phoneName, imageName);
    pathHandle = uicontrol('Parent', fig, 'Style', 'text', 'String', pathText, ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.pathDisplay, ...
                          'FontSize', cfg.ui.fontSize.path, 'FontWeight', 'normal', ...
                          'ForegroundColor', cfg.ui.colors.path, ...
                          'BackgroundColor', cfg.ui.colors.background, ...
                          'HorizontalAlignment', 'center');
end

function [imgAxes, imgHandle] = createImageAxes(fig, img, cfg)
    % Create axes for image display
    %
    % INPUTS:
    %   fig - Parent figure
    %   img - Image to display
    %   cfg - UI configuration
    %
    % OUTPUTS:
    %   imgAxes   - Axes handle
    %   imgHandle - Image handle

    imgAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.image);
    imgHandle = imshow(img, 'Parent', imgAxes, 'InitialMagnification', 'fit');
    axis(imgAxes, 'image');
    axis(imgAxes, 'tight');
    hold(imgAxes, 'on');
end

function instructionText = createInstructions(fig, cfg, customString)
    % Create instruction text
    %
    % INPUTS:
    %   fig          - Parent figure
    %   cfg          - UI configuration
    %   customString - (Optional) Custom instruction string
    %
    % OUTPUTS:
    %   instructionText - Handle to instruction text control

    if nargin < 3 || isempty(customString)
        customString = 'Mouse = Drag Vertices | Buttons = Rotate | RUN AI = Detect Quads | Slider = Zoom | APPLY = Save & Continue | SKIP = Skip | STOP = Exit | Space = APPLY | Esc = SKIP';
    end

    instructionText = uicontrol('Parent', fig, 'Style', 'text', 'String', customString, ...
             'Units', 'normalized', 'Position', cfg.ui.positions.instructions, ...
             'FontSize', cfg.ui.fontSize.instruction, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.background, 'HorizontalAlignment', 'center');
end

%% =========================================================================
%% ACTION BUTTONS
%% =========================================================================

function stopButton = createStopButton(fig, cfg, callback)
    % Create STOP button
    %
    % INPUTS:
    %   fig      - Parent figure
    %   cfg      - UI configuration
    %   callback - (Optional) Callback function
    %
    % OUTPUTS:
    %   stopButton - Button handle

    if nargin < 3 || isempty(callback)
        callback = @(~,~) stopExecution(fig);
    end

    stopButton = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
                          'String', 'STOP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
                          'Units', 'normalized', 'Position', cfg.ui.positions.stopButton, ...
                          'BackgroundColor', cfg.ui.colors.stop, 'ForegroundColor', cfg.ui.colors.foreground, ...
                          'Callback', callback);
end

function runAIButton = createRunAIButton(fig, cfg, callback)
    % Create RUN AI button
    %
    % INPUTS:
    %   fig      - Parent figure
    %   cfg      - UI configuration
    %   callback - Callback function for AI detection
    %
    % OUTPUTS:
    %   runAIButton - Button handle

    if nargin < 3
        callback = [];
    end

    runAIButton = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
                           'String', 'RUN AI', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
                           'Units', 'normalized', 'Position', cfg.ui.positions.runAIButton, ...
                           'BackgroundColor', [0.30 0.50 0.70], ...
                           'ForegroundColor', cfg.ui.colors.foreground, ...
                           'TooltipString', 'Run YOLO detection on the current view');

    if ~isempty(callback)
        set(runAIButton, 'Callback', callback);
    end
end

function cutButtonPanel = createEditButtonPanel(fig, cfg, applyCallback, skipCallback)
    % Create APPLY/SKIP button panel
    %
    % INPUTS:
    %   fig           - Parent figure
    %   cfg           - UI configuration
    %   applyCallback - (Optional) Callback for APPLY button
    %   skipCallback  - (Optional) Callback for SKIP button
    %
    % OUTPUTS:
    %   cutButtonPanel - Panel handle

    if nargin < 3 || isempty(applyCallback)
        applyCallback = @(~,~) setAction(fig, 'accept');
    end
    if nargin < 4 || isempty(skipCallback)
        skipCallback = @(~,~) setAction(fig, 'skip');
    end

    cutButtonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                             'Position', cfg.ui.positions.cutButtonPanel, ...
                             'BackgroundColor', cfg.ui.colors.panel, ...
                             'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground);

    % APPLY button
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'APPLY', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.15 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.apply, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', applyCallback);

    % SKIP button
    uicontrol('Parent', cutButtonPanel, 'Style', 'pushbutton', ...
             'String', 'SKIP', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.55 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.skip, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', skipCallback);
end

function buttonPanel = createPreviewButtons(fig, cfg, callbacks)
    % Create ACCEPT/RETRY/SKIP button panel for preview mode
    %
    % INPUTS:
    %   fig       - Parent figure
    %   cfg       - UI configuration
    %   callbacks - (Optional) Struct with .accept, .retry, .skip callbacks
    %
    % OUTPUTS:
    %   buttonPanel - Panel handle

    if nargin < 3 || isempty(callbacks)
        callbacks = struct();
    end
    if ~isfield(callbacks, 'accept')
        callbacks.accept = @(~,~) setAction(fig, 'accept');
    end
    if ~isfield(callbacks, 'retry')
        callbacks.retry = @(~,~) setAction(fig, 'retry');
    end
    if ~isfield(callbacks, 'skip')
        callbacks.skip = @(~,~) setAction(fig, 'skip');
    end

    buttonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                         'Position', cfg.ui.positions.previewPanel, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground, ...
                         'BorderWidth', cfg.ui.quad.borderWidth);

    buttons = {'ACCEPT', 'RETRY', 'SKIP'};
    positions = {[0.05 0.25 0.25 0.50], [0.375 0.25 0.25 0.50], [0.70 0.25 0.25 0.50]};
    colors = {cfg.ui.colors.accept, cfg.ui.colors.retry, cfg.ui.colors.skip};
    callbackList = {callbacks.accept, callbacks.retry, callbacks.skip};

    for i = 1:numel(buttons)
        uicontrol('Parent', buttonPanel, 'Style', 'pushbutton', ...
                 'String', buttons{i}, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold', ...
                 'Units', 'normalized', 'Position', positions{i}, ...
                 'BackgroundColor', colors{i}, 'ForegroundColor', cfg.ui.colors.foreground, ...
                 'Callback', callbackList{i});
    end
end

function buttonPanel = createEllipseEditingButtonPanel(fig, cfg, doneCallback, backCallback)
    % Create DONE/BACK button panel for ellipse editing mode
    %
    % INPUTS:
    %   fig          - Parent figure
    %   cfg          - UI configuration
    %   doneCallback - (Optional) Callback for DONE button
    %   backCallback - (Optional) Callback for BACK button
    %
    % OUTPUTS:
    %   buttonPanel - Panel handle

    if nargin < 3 || isempty(doneCallback)
        doneCallback = @(~,~) setAction(fig, 'accept');
    end
    if nargin < 4 || isempty(backCallback)
        backCallback = @(~,~) setAction(fig, 'back');
    end

    buttonPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                         'Position', cfg.ui.positions.cutButtonPanel, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground);

    % DONE button (green)
    uicontrol('Parent', buttonPanel, 'Style', 'pushbutton', ...
             'String', 'DONE', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.15 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.accept, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', doneCallback);

    % BACK button (yellow)
    uicontrol('Parent', buttonPanel, 'Style', 'pushbutton', ...
             'String', 'BACK', 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', [0.55 0.35 0.30 0.35], ...
             'BackgroundColor', cfg.ui.colors.retry, 'ForegroundColor', cfg.ui.colors.foreground, ...
             'Callback', backCallback);
end

%% =========================================================================
%% CONTROL PANELS
%% =========================================================================

function rotationPanel = createRotationButtonPanel(fig, cfg, rotationCallback)
    % Create rotation button panel with preset angle buttons
    %
    % INPUTS:
    %   fig              - Parent figure
    %   cfg              - UI configuration
    %   rotationCallback - Function handle @(angle, fig, cfg) for rotation
    %
    % OUTPUTS:
    %   rotationPanel - Panel handle

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
        if nargin >= 3 && ~isempty(rotationCallback)
            cb = @(~,~) rotationCallback(angles(i), fig, cfg);
        else
            cb = [];
        end

        uicontrol('Parent', rotationPanel, 'Style', 'pushbutton', ...
                 'String', sprintf('%d%s', angles(i), char(176)), ...
                 'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
                 'Units', 'normalized', 'Position', positions{i}, ...
                 'BackgroundColor', [0.25 0.25 0.25], ...
                 'ForegroundColor', cfg.ui.colors.foreground, ...
                 'Callback', cb);
    end
end

function [zoomSlider, zoomValue] = createZoomPanel(fig, cfg, sliderCallback, resetCallback, autoCallback)
    % Create zoom panel with slider and control buttons
    %
    % INPUTS:
    %   fig            - Parent figure
    %   cfg            - UI configuration
    %   sliderCallback - (Optional) Callback for slider changes
    %   resetCallback  - (Optional) Callback for reset button
    %   autoCallback   - (Optional) Callback for auto button
    %
    % OUTPUTS:
    %   zoomSlider - Slider handle
    %   zoomValue  - Value text handle

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
                          'BackgroundColor', cfg.ui.colors.panel);

    if nargin >= 3 && ~isempty(sliderCallback)
        set(zoomSlider, 'Callback', sliderCallback);
    end

    % Zoom value display
    zoomValue = uicontrol('Parent', zoomPanel, 'Style', 'text', ...
                         'String', '0%', ...
                         'Units', 'normalized', 'Position', cfg.ui.layout.zoomValue, ...
                         'FontSize', cfg.ui.fontSize.value, 'FontWeight', 'bold', ...
                         'ForegroundColor', cfg.ui.colors.foreground, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'HorizontalAlignment', 'center');

    % Reset button
    resetBtn = uicontrol('Parent', zoomPanel, 'Style', 'pushbutton', ...
             'String', 'Reset', ...
             'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.zoomResetButton, ...
             'BackgroundColor', [0.25 0.25 0.25], ...
             'ForegroundColor', cfg.ui.colors.foreground);

    if nargin >= 4 && ~isempty(resetCallback)
        set(resetBtn, 'Callback', resetCallback);
    end

    % Auto button
    autoBtn = uicontrol('Parent', zoomPanel, 'Style', 'pushbutton', ...
             'String', 'Auto', ...
             'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.zoomAutoButton, ...
             'BackgroundColor', [0.25 0.25 0.25], ...
             'ForegroundColor', cfg.ui.colors.foreground);

    if nargin >= 5 && ~isempty(autoCallback)
        set(autoBtn, 'Callback', autoCallback);
    end
end

function [prevButton, zoomIndicator, nextButton, resetButton] = createEllipseZoomPanel(fig, cfg, prevCallback, nextCallback, resetCallback)
    % Create zoom panel with quadrilateral navigation for ellipse editing
    %
    % INPUTS:
    %   fig           - Parent figure
    %   cfg           - UI configuration
    %   prevCallback  - (Optional) Callback for previous button
    %   nextCallback  - (Optional) Callback for next button
    %   resetCallback - (Optional) Callback for reset button
    %
    % OUTPUTS:
    %   prevButton    - Previous button handle
    %   zoomIndicator - Indicator text handle
    %   nextButton    - Next button handle
    %   resetButton   - Reset button handle

    zoomPanel = uipanel('Parent', fig, 'Units', 'normalized', ...
                       'Position', cfg.ui.positions.zoomPanel, ...
                       'BackgroundColor', cfg.ui.colors.panel, ...
                       'BorderType', 'etchedin', 'HighlightColor', cfg.ui.colors.foreground, ...
                       'BorderWidth', 2);

    % Panel label
    uicontrol('Parent', zoomPanel, 'Style', 'text', 'String', 'Quad Zoom', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.zoomLabel, ...
             'FontSize', cfg.ui.fontSize.label, 'FontWeight', 'bold', ...
             'ForegroundColor', cfg.ui.colors.foreground, ...
             'BackgroundColor', cfg.ui.colors.panel, 'HorizontalAlignment', 'center');

    % Previous button
    prevButton = uicontrol('Parent', zoomPanel, 'Style', 'pushbutton', ...
             'String', '<', ...
             'FontSize', cfg.ui.fontSize.button + 2, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.ellipseZoomPrevButton, ...
             'BackgroundColor', [0.25 0.25 0.25], ...
             'ForegroundColor', cfg.ui.colors.foreground);

    if nargin >= 3 && ~isempty(prevCallback)
        set(prevButton, 'Callback', prevCallback);
    end

    % Quadrilateral indicator text
    zoomIndicator = uicontrol('Parent', zoomPanel, 'Style', 'text', ...
                         'String', 'All', ...
                         'Units', 'normalized', 'Position', cfg.ui.layout.ellipseZoomIndicator, ...
                         'FontSize', cfg.ui.fontSize.value, 'FontWeight', 'bold', ...
                         'ForegroundColor', cfg.ui.colors.foreground, ...
                         'BackgroundColor', cfg.ui.colors.panel, ...
                         'HorizontalAlignment', 'center');

    % Next button
    nextButton = uicontrol('Parent', zoomPanel, 'Style', 'pushbutton', ...
             'String', '>', ...
             'FontSize', cfg.ui.fontSize.button + 2, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.ellipseZoomNextButton, ...
             'BackgroundColor', [0.25 0.25 0.25], ...
             'ForegroundColor', cfg.ui.colors.foreground);

    if nargin >= 4 && ~isempty(nextCallback)
        set(nextButton, 'Callback', nextCallback);
    end

    % Reset button
    resetButton = uicontrol('Parent', zoomPanel, 'Style', 'pushbutton', ...
             'String', 'Reset View', ...
             'FontSize', cfg.ui.fontSize.button, 'FontWeight', 'bold', ...
             'Units', 'normalized', 'Position', cfg.ui.layout.ellipseZoomResetButton, ...
             'BackgroundColor', [0.25 0.25 0.25], ...
             'ForegroundColor', cfg.ui.colors.foreground);

    if nargin >= 5 && ~isempty(resetCallback)
        set(resetButton, 'Callback', resetCallback);
    end
end

function statusLabel = createAIStatusLabel(fig, cfg)
    % Create AI detection status label (hidden by default)
    %
    % INPUTS:
    %   fig - Parent figure
    %   cfg - UI configuration
    %
    % OUTPUTS:
    %   statusLabel - Label handle

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

%% =========================================================================
%% QUADRILATERAL VISUALIZATION
%% =========================================================================

function quads = createQuads(initialQuads, cfg, labelCallback)
    % Create drawpolygon objects from initial positions
    %
    % INPUTS:
    %   initialQuads  - [N x 4 x 2] array of quadrilateral vertices
    %   cfg           - Configuration with numSquares and ui.quad settings
    %   labelCallback - (Optional) Callback for label updates
    %
    % OUTPUTS:
    %   quads - Cell array of drawpolygon handles

    n = size(initialQuads, 1);
    quads = cell(1, n);

    for i = 1:n
        pos = squeeze(initialQuads(i, :, :));

        % Apply color gradient based on concentration index
        concentrationIndex = i - 1;
        quadColor = getConcentrationColor(concentrationIndex, cfg.numSquares);

        quads{i} = drawpolygon('Position', pos, ...
                                 'Color', quadColor, ...
                                 'LineWidth', cfg.ui.quad.lineWidth, ...
                                 'MarkerSize', 8, ...
                                 'Selected', false);

        % Apply face styling
        setQuadColor(quads{i}, quadColor, 0.25);

        % Store initial valid position
        setappdata(quads{i}, 'LastValidPosition', pos);

        % Add listener for quadrilateral enforcement
        listenerHandle = addlistener(quads{i}, 'ROIMoved', @(~,~) enforceQuadrilateral(quads{i}));
        setappdata(quads{i}, 'ListenerHandle', listenerHandle);

        % Add listener for label updates
        if nargin >= 3 && ~isempty(labelCallback)
            labelUpdateListener = addlistener(quads{i}, 'ROIMoved', labelCallback);
            setappdata(quads{i}, 'LabelUpdateListener', labelUpdateListener);
        end
    end
end

function enforceQuadrilateral(quad)
    % Ensure quadrilateral remains a quadrilateral
    if ~isvalid(quad)
        return;
    end

    pos = quad.Position;
    if size(pos, 1) ~= 4
        lastValid = getappdata(quad, 'LastValidPosition');
        if ~isempty(lastValid)
            quad.Position = lastValid;
        end
        warning('micropad_ui:invalid_quad', 'Quadrilateral must have exactly 4 vertices. Reverting change.');
    else
        setappdata(quad, 'LastValidPosition', pos);
    end
end

function labelHandles = addQuadLabels(quads, axesHandle)
    % Add text labels showing concentration number on each quadrilateral
    %
    % INPUTS:
    %   quads      - Cell array of drawpolygon objects
    %   axesHandle - Axes where labels should be drawn
    %
    % OUTPUTS:
    %   labelHandles - Cell array of text handles

    n = numel(quads);
    labelHandles = cell(1, n);

    for i = 1:n
        quad = quads{i};
        if ~isvalid(quad)
            continue;
        end

        pos = quad.Position;
        if isempty(pos) || size(pos, 1) < 3
            continue;
        end

        % Position at top of quadrilateral
        centerX = mean(pos(:, 1));
        minY = min(pos(:, 2));
        quadHeight = max(pos(:, 2)) - minY;
        labelY = minY - max(15, quadHeight * 0.1);

        concentrationIndex = i - 1;
        labelText = sprintf('con_%d', concentrationIndex);

        labelHandles{i} = text(axesHandle, centerX, labelY, labelText, ...
                              'HorizontalAlignment', 'center', ...
                              'VerticalAlignment', 'bottom', ...
                              'FontSize', 12, ...
                              'FontWeight', 'bold', ...
                              'Color', [1 1 1], ...
                              'BackgroundColor', [0.2 0.2 0.2], ...
                              'EdgeColor', 'none', ...
                              'Margin', 2);
    end
end

function updateQuadLabels(quads, labelHandles)
    % Update label positions to match quadrilateral positions
    %
    % INPUTS:
    %   quads        - Cell array of drawpolygon objects
    %   labelHandles - Cell array of text handles

    if isempty(labelHandles)
        return;
    end

    numQuads = numel(quads);
    numLabels = numel(labelHandles);

    for i = 1:min(numQuads, numLabels)
        if ~isvalid(quads{i}) || ~isvalid(labelHandles{i})
            continue;
        end

        pos = quads{i}.Position;
        if isempty(pos) || size(pos, 1) < 3
            continue;
        end

        % Position at top of quadrilateral
        centerX = mean(pos(:, 1));
        minY = min(pos(:, 2));
        quadHeight = max(pos(:, 2)) - minY;
        labelY = minY - max(15, quadHeight * 0.1);

        set(labelHandles{i}, 'Position', [centerX, labelY, 0]);
    end
end

function color = getConcentrationColor(concentrationIndex, totalConcentrations)
    % Generate spectrum gradient color: blue (cold) to red (hot)
    %
    % INPUTS:
    %   concentrationIndex  - Zero-based index (0 to totalConcentrations-1)
    %   totalConcentrations - Total number of concentration regions
    %
    % OUTPUTS:
    %   color - [R G B] triplet in range [0, 1]

    if totalConcentrations <= 1
        color = [0.0 0.5 1.0];  % Default blue
        return;
    end

    % Normalize index to [0, 1]
    t = concentrationIndex / (totalConcentrations - 1);

    % Interpolate hue from 240° (blue) to 0° (red)
    hue = (1 - t) * 240 / 360;
    sat = 1.0;
    val = 1.0;

    % Convert HSV to RGB
    color = hsv2rgb([hue, sat, val]);
end

function setQuadColor(quadHandle, colorValue, faceAlpha)
    % Apply edge/face color updates to quadrilateral
    %
    % INPUTS:
    %   quadHandle - Quadrilateral ROI handle
    %   colorValue - [R G B] color
    %   faceAlpha  - (Optional) Face transparency

    if nargin < 3
        faceAlpha = [];
    end

    if isempty(quadHandle) || ~isvalid(quadHandle)
        return;
    end

    if ~isempty(colorValue) && all(isfinite(colorValue))
        set(quadHandle, 'Color', colorValue);
        if isprop(quadHandle, 'FaceColor')
            set(quadHandle, 'FaceColor', colorValue);
        end
    end

    if ~isempty(faceAlpha) && isprop(quadHandle, 'FaceAlpha')
        set(quadHandle, 'FaceAlpha', faceAlpha);
    end
end

function ellipseHandle = createEllipseROI(axHandle, center, semiMajor, semiMinor, rotationAngle, color, bounds, cfg)
    % Helper to instantiate drawellipse overlays with consistent constraints
    %
    % INPUTS:
    %   axHandle      - Axes handle
    %   center        - [x, y] center
    %   semiMajor     - Semi-major axis length
    %   semiMinor     - Semi-minor axis length
    %   rotationAngle - Rotation in degrees
    %   color         - ROI color
    %   bounds        - (Optional) Axis limits struct (.minAxis, .maxAxis)
    %   cfg           - (Optional) Configuration struct
    %
    % OUTPUTS:
    %   ellipseHandle - drawellipse object

    if nargin < 7
        bounds = [];
    end

    % Enforce semiMajor >= semiMinor constraint locally
    if semiMinor > semiMajor
        tmp = semiMajor;
        semiMajor = semiMinor;
        semiMinor = tmp;
        rotationAngle = rotationAngle + 90;
    end

    % Apply bounds if provided
    if ~isempty(bounds)
        minAxis = bounds.minAxis;
        maxAxis = bounds.maxAxis;
        semiMajor = min(max(semiMajor, minAxis), maxAxis);
        semiMinor = min(max(semiMinor, minAxis), semiMajor);
    end

    % Normalize rotation
    rotationAngle = mod(rotationAngle + 180, 360) - 180;

    ellipseHandle = drawellipse(axHandle, ...
        'Center', center, ...
        'SemiAxes', [semiMajor, semiMinor], ...
        'RotationAngle', rotationAngle, ...
        'Color', color, ...
        'LineWidth', 2, ...
        'FaceAlpha', 0.2, ...
        'InteractionsAllowed', 'all');
end

%% =========================================================================
%% PREVIEW
%% =========================================================================

function [leftAxes, rightAxes, leftImgHandle, rightImgHandle] = createPreviewAxes(fig, img, quadParams, ellipseData, cfg)
    % Create preview axes with original and masked images
    %
    % INPUTS:
    %   fig         - Parent figure
    %   img         - Original image
    %   quadParams  - [N x 4 x 2] quadrilateral coordinates
    %   ellipseData - (Optional) Ellipse data [M x 7]
    %   cfg         - UI configuration
    %
    % OUTPUTS:
    %   leftAxes       - Left axes handle (original)
    %   rightAxes      - Right axes handle (masked)
    %   leftImgHandle  - Left image handle
    %   rightImgHandle - Right image handle

    if nargin < 4
        ellipseData = [];
    end

    % Left: original with overlays
    leftAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewLeft);
    leftImgHandle = imshow(img, 'Parent', leftAxes, 'InitialMagnification', 'fit');
    axis(leftAxes, 'image');
    axis(leftAxes, 'tight');
    title(leftAxes, 'Original Image', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold');
    hold(leftAxes, 'on');

    % Draw quadrilateral overlays
    for i = 1:size(quadParams, 1)
        quad = squeeze(quadParams(i,:,:));
        if size(quad, 1) >= 3
            concentrationIndex = i - 1;
            quadColor = getConcentrationColor(concentrationIndex, cfg.numSquares);

            plot(leftAxes, [quad(:,1); quad(1,1)], [quad(:,2); quad(1,2)], ...
                 'Color', quadColor, 'LineWidth', cfg.ui.quad.lineWidth);

            % Label at bottom-right
            bottomRightX = max(quad(:,1));
            bottomRightY = max(quad(:,2));
            hText = text(leftAxes, bottomRightX, bottomRightY, sprintf('con_%d', i-1), ...
                 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
                 'FontSize', cfg.ui.fontSize.info, 'FontWeight', 'bold', ...
                 'Color', cfg.ui.colors.info, 'BackgroundColor', [0 0 0 0.6], ...
                 'EdgeColor', 'none', 'Margin', 2);
            uistack(hText, 'top');
        end
    end

    % Draw ellipse overlays if present
    if ~isempty(ellipseData) && isfield(cfg, 'enableEllipseEditing') && cfg.enableEllipseEditing
        for i = 1:size(ellipseData, 1)
            if ellipseData(i, 3) > 0  % Valid ellipse
                concIdx = ellipseData(i, 1);
                x = ellipseData(i, 3);
                y = ellipseData(i, 4);
                a = ellipseData(i, 5);
                b = ellipseData(i, 6);
                theta = ellipseData(i, 7);

                ellipseColor = getConcentrationColor(concIdx, cfg.numSquares);

                % Draw ellipse using parametric form
                % Use CW rotation to match image coordinate convention
                t = linspace(0, 2*pi, 100);
                theta_rad = deg2rad(theta);
                x_ellipse = a * cos(t);
                y_ellipse = b * sin(t);
                x_rot = x + x_ellipse * cos(theta_rad) - y_ellipse * sin(theta_rad);
                y_rot = y + x_ellipse * sin(theta_rad) + y_ellipse * cos(theta_rad);

                plot(leftAxes, x_rot, y_rot, 'Color', ellipseColor, 'LineWidth', 1.5);
            end
        end
    end

    hold(leftAxes, 'off');

    % Right: masked preview
    rightAxes = axes('Parent', fig, 'Units', 'normalized', 'Position', cfg.ui.positions.previewRight);
    maskedImg = createMaskedPreview(img, quadParams, ellipseData, cfg);
    rightImgHandle = imshow(maskedImg, 'Parent', rightAxes, 'InitialMagnification', 'fit');
    axis(rightAxes, 'image');
    axis(rightAxes, 'tight');
    title(rightAxes, 'Masked Preview', ...
          'Color', cfg.ui.colors.foreground, 'FontSize', cfg.ui.fontSize.preview, 'FontWeight', 'bold');
end

function maskedImg = createMaskedPreview(img, quadParams, ellipseData, cfg)
    % Create masked preview image
    %
    % INPUTS:
    %   img         - Original image
    %   quadParams  - Quadrilateral coordinates
    %   ellipseData - Ellipse data
    %   cfg         - Configuration with dimFactor
    %
    % OUTPUTS:
    %   maskedImg - Image with non-selected regions dimmed

    [height, width, ~] = size(img);
    totalMask = false(height, width);

    % Add quadrilateral masks
    numRegions = size(quadParams, 1);
    for i = 1:numRegions
        quad = squeeze(quadParams(i,:,:));
        if size(quad, 1) >= 3
            regionMask = poly2mask(quad(:,1), quad(:,2), height, width);
            totalMask = totalMask | regionMask;
        end
    end

    % Add ellipse masks if present
    if ~isempty(ellipseData) && isfield(cfg, 'enableEllipseEditing') && cfg.enableEllipseEditing
        for i = 1:size(ellipseData, 1)
            if ellipseData(i, 3) > 0
                x = ellipseData(i, 3);
                y = ellipseData(i, 4);
                a = ellipseData(i, 5);
                b = ellipseData(i, 6);
                theta = ellipseData(i, 7);

                [X, Y] = meshgrid(1:width, 1:height);
                % Inverse rotation (CCW) to map world points to unrotated ellipse frame
                theta_rad = deg2rad(-theta);
                dx = X - x;
                dy = Y - y;
                % Match cut_micropads mask rotation (Y-down image coordinates)
                x_rot =  dx * cos(theta_rad) - dy * sin(theta_rad);
                y_rot =  dx * sin(theta_rad) + dy * cos(theta_rad);

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

%% =========================================================================
%% ZOOM FUNCTIONS
%% =========================================================================

function applyZoomToAxes(guiData, ~)
    % Apply current zoom level to image axes
    %
    % INPUTS:
    %   guiData - GUI data struct with imgAxes, imageSize, zoomLevel
    %   ~       - Configuration (unused, kept for API compatibility)

    imgHeight = guiData.imageSize(1);
    imgWidth = guiData.imageSize(2);

    if guiData.zoomLevel == 0
        % Full image view
        xlim(guiData.imgAxes, [0.5, imgWidth + 0.5]);
        ylim(guiData.imgAxes, [0.5, imgHeight + 0.5]);
    else
        % Calculate target bounds
        if isfield(guiData, 'autoZoomBounds') && ~isempty(guiData.autoZoomBounds)
            autoZoomBounds = guiData.autoZoomBounds;
        else
            [xmin, xmax, ymin, ymax] = calculateQuadBounds(guiData);
            if ~isempty(xmin)
                autoZoomBounds = [xmin, xmax, ymin, ymax];
            else
                autoZoomBounds = [0.5, imgWidth + 0.5, 0.5, imgHeight + 0.5];
            end
        end

        % Interpolate between full image and auto-zoom bounds
        fullBounds = [0.5, imgWidth + 0.5, 0.5, imgHeight + 0.5];
        targetBounds = autoZoomBounds;

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
    % Calculate bounding box containing all quadrilaterals
    %
    % INPUTS:
    %   guiData - GUI data struct with quads cell array
    %
    % OUTPUTS:
    %   xmin, xmax, ymin, ymax - Bounding box (with 10% margin)

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

    % Add 10% margin
    xmargin = (xmax - xmin) * 0.1;
    ymargin = (ymax - ymin) * 0.1;

    xmin = max(0.5, xmin - xmargin);
    xmax = min(guiData.imageSize(2) + 0.5, xmax + xmargin);
    ymin = max(0.5, ymin - ymargin);
    ymax = min(guiData.imageSize(1) + 0.5, ymax + ymargin);
end

function [xmin, xmax, ymin, ymax] = calculateEllipseBounds(guiData)
    % Calculate bounding box containing all ellipses
    %
    % INPUTS:
    %   guiData - GUI data struct with ellipses cell array
    %
    % OUTPUTS:
    %   xmin, xmax, ymin, ymax - Bounding box (with 10% margin)

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
            maxAxis = max(semiAxes);

            xmin = min(xmin, center(1) - maxAxis);
            xmax = max(xmax, center(1) + maxAxis);
            ymin = min(ymin, center(2) - maxAxis);
            ymax = max(ymax, center(2) + maxAxis);
        end
    end

    if isinf(xmin)
        xmin = [];
        return;
    end

    % Add 10% margin
    xmargin = (xmax - xmin) * 0.1;
    ymargin = (ymax - ymin) * 0.1;

    xmin = max(0.5, xmin - xmargin);
    xmax = min(guiData.imageSize(2) + 0.5, xmax + xmargin);
    ymin = max(0.5, ymin - ymargin);
    ymax = min(guiData.imageSize(1) + 0.5, ymax + ymargin);
end

%% =========================================================================
%% STATE MANAGEMENT
%% =========================================================================

function setAction(fig, action)
    % Set action and resume UI wait
    %
    % INPUTS:
    %   fig    - Figure handle
    %   action - Action string ('accept', 'skip', 'retry', 'stop', etc.)

    guiData = get(fig, 'UserData');
    guiData.action = action;
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function stopExecution(fig)
    % Stop execution and close figure
    %
    % INPUTS:
    %   fig - Figure handle

    guiData = get(fig, 'UserData');
    guiData.action = 'stop';
    set(fig, 'UserData', guiData);
    uiresume(fig);
end

function keyPressHandler(src, event)
    % Default keyboard event handler
    %
    % INPUTS:
    %   src   - Source figure
    %   event - Key event data
    %
    % Space -> accept, Escape -> skip

    switch event.Key
        case 'space'
            setAction(src, 'accept');
        case 'escape'
            setAction(src, 'skip');
    end
end

%% =========================================================================
%% UI CLEANUP
%% =========================================================================

function clearAllUIElements(fig, guiData)
    % Delete all UI elements from figure
    %
    % INPUTS:
    %   fig     - Figure handle
    %   guiData - GUI data struct

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

    % Add quadrilateral ROIs
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'quads')
        validQuads = collectValidHandles(guiData.quads);
        if ~isempty(validQuads)
            toDelete = [toDelete; validQuads];
        end
    end

    % Add quadrilateral labels
    if ~isempty(guiData) && isstruct(guiData) && isfield(guiData, 'quadLabels')
        validLabels = collectValidHandles(guiData.quadLabels);
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

    % Clean up timers
    if ~isempty(guiData) && isstruct(guiData)
        if isfield(guiData, 'aiTimer')
            safeStopTimer(guiData.aiTimer);
        end
        if isfield(guiData, 'aiBreathingTimer')
            safeStopTimer(guiData.aiBreathingTimer);
        end
    end

    set(fig, 'UserData', []);
end

function validHandles = collectValidHandles(cellArray)
    % Collect valid handles from cell array
    if isempty(cellArray) || ~iscell(cellArray)
        validHandles = [];
        return;
    end

    numItems = numel(cellArray);
    validHandles = gobjects(numItems, 1);
    validCount = 0;

    for i = 1:numItems
        if isvalid(cellArray{i})
            validCount = validCount + 1;
            validHandles(validCount) = cellArray{i};
        end
    end

    validHandles = validHandles(1:validCount);
end

function viewState = captureViewState(guiData)
    % Capture current view state for restoration
    %
    % INPUTS:
    %   guiData - GUI data struct
    %
    % OUTPUTS:
    %   viewState - Struct with xlim, ylim, zoom settings

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
    % Apply saved view state to figure
    %
    % INPUTS:
    %   fig       - Figure handle
    %   viewState - View state struct from captureViewState

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
            % Ignore if limits are invalid
        end
    end

    if isfield(viewState, 'ylim')
        try
            ylim(guiData.imgAxes, viewState.ylim);
        catch
            % Ignore if limits are invalid
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
        set(fig, 'UserData', guiData);
    end
end

%% =========================================================================
%% AI PROGRESS INDICATOR
%% =========================================================================

function showAIProgressIndicator(fig, show, cfg)
    % Toggle AI detection status indicator
    %
    % INPUTS:
    %   fig  - Figure handle
    %   show - Boolean to show/hide indicator
    %   cfg  - (Optional) Configuration for creating label

    if ~ishandle(fig) || ~strcmp(get(fig, 'Type'), 'figure')
        return;
    end

    guiData = get(fig, 'UserData');
    if isempty(guiData) || ~isstruct(guiData)
        return;
    end

    if show
        % Ensure status label exists
        if ~isfield(guiData, 'aiStatusLabel') || ~isvalid(guiData.aiStatusLabel)
            if nargin >= 3 && ~isempty(cfg)
                guiData.aiStatusLabel = createAIStatusLabel(fig, cfg);
            else
                guiData.aiStatusLabel = createAIStatusLabel(fig, []);
            end
        end
        set(guiData.aiStatusLabel, 'String', 'AI DETECTION RUNNING', 'Visible', 'on');
        uistack(guiData.aiStatusLabel, 'top');

        % Capture current quadrilateral colors
        if isfield(guiData, 'quads')
            guiData.aiBaseColors = captureQuadColors(guiData.quads);
        end

        drawnow limitrate;
    else
        % Hide status label
        if isfield(guiData, 'aiStatusLabel') && isvalid(guiData.aiStatusLabel)
            set(guiData.aiStatusLabel, 'Visible', 'off');
        end

        % Restore base colors
        if isfield(guiData, 'quads') && iscell(guiData.quads) && ...
           isfield(guiData, 'aiBaseColors') && ~isempty(guiData.aiBaseColors)
            numRestore = min(size(guiData.aiBaseColors, 1), numel(guiData.quads));
            for idx = 1:numRestore
                quad = guiData.quads{idx};
                baseColor = guiData.aiBaseColors(idx, :);
                if isvalid(quad) && all(isfinite(baseColor))
                    setQuadColor(quad, baseColor, 0.25);
                end
            end
            drawnow limitrate;
        end

        % Refresh baseline colors
        if isfield(guiData, 'quads')
            guiData.aiBaseColors = captureQuadColors(guiData.quads);
        end
    end

    set(fig, 'UserData', guiData);
end

function baseColors = captureQuadColors(quads)
    % Capture current colors of all quadrilaterals
    %
    % INPUTS:
    %   quads - Cell array of quadrilateral handles
    %
    % OUTPUTS:
    %   baseColors - [N x 3] matrix of RGB colors

    baseColors = [];
    if isempty(quads) || ~iscell(quads)
        return;
    end

    numQuads = numel(quads);
    baseColors = nan(numQuads, 3);

    for idx = 1:numQuads
        if isvalid(quads{idx})
            color = get(quads{idx}, 'Color');
            if numel(color) == 3
                baseColors(idx, :) = color;
            end
        end
    end
end

%% =========================================================================
%% TIMER UTILITIES
%% =========================================================================

function safeStopTimer(timerHandle)
    % Safely stop and delete a timer
    %
    % INPUTS:
    %   timerHandle - Timer handle or empty

    if isempty(timerHandle)
        return;
    end

    try
        if isvalid(timerHandle)
            if strcmp(timerHandle.Running, 'on')
                stop(timerHandle);
            end
            delete(timerHandle);
        end
    catch
        % Timer already invalid
    end
end
