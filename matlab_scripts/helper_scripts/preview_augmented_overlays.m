function preview_augmented_overlays(varargin)
%PREVIEW_AUGMENTED_OVERLAYS Visual integrity check for augmented overlays.
%   Displays augmented scenes from augmented_1_dataset with concentration
%   polygons from augmented_2_micropads and ellipse fits
%   from augmented_3_elliptical_regions overlaid on top.
%
%   This viewer aggregates all polygons and ellipses per base scene image,
%   matching the workflow of preview_overlays.m but for augmented data.
%
%   COORDINATE FORMATS:
%   - Stage 2 (augmented_2_micropads): 11 columns with rotation
%     Format: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation
%     Note: Matches standard pipeline coordinates.txt format.
%   - Stage 3 (augmented_3_elliptical_regions): 8 columns
%     Format: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
%
%   Example usage:
%       addpath('matlab_scripts/helper_scripts');
%       preview_augmented_overlays('maxSamples', 12);
%
%   Controls:
%       Click 'Next' or press 'n' to advance to next image
%       Press 'q' to quit

% Configuration constants
MISSING_IMAGE_HEIGHT = 480;
MISSING_IMAGE_WIDTH = 640;
ELLIPSE_RENDER_POINTS = 72;
OVERLAY_COLOR_RGB = [0.48, 0.99, 0.00];  % Fluorescent green
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

% Coordinate column name constants (11 columns for stage 2, matching standard pipeline format)
STAGE2_COORD_COLUMNS = {'image','concentration','x1','y1','x2','y2','x3','y3','x4','y4','rotation'};
STAGE3_COORD_COLUMNS = {'image','concentration','replicate','x','y','semiMajorAxis','semiMinorAxis','rotationAngle'};

parser = inputParser;
parser.FunctionName = mfilename;

addParameter(parser, 'stage1Folder', 'augmented_1_dataset', ...
    @(s) validateattributes(s, {'char','string'}, {'scalartext'}));
addParameter(parser, 'stage2Folder', 'augmented_2_micropads', ...
    @(s) validateattributes(s, {'char','string'}, {'scalartext'}));
addParameter(parser, 'stage3Folder', 'augmented_3_elliptical_regions', ...
    @(s) validateattributes(s, {'char','string'}, {'scalartext'}));
addParameter(parser, 'phones', string.empty, ...
    @(c) isstring(c) || iscellstr(c));
addParameter(parser, 'maxSamples', Inf, ...
    @(n) validateattributes(n, {'numeric'}, {'scalar','positive'}));

parse(parser, varargin{:});
opts = parser.Results;

requestedPhones = string(opts.phones);
requestedPhones = requestedPhones(requestedPhones ~= "");

stage1Root = resolve_repo_path(opts.stage1Folder);
stage2Root = resolve_repo_path(opts.stage2Folder);

% Stage 3 (ellipse) folder is optional
try
    stage3Root = resolve_repo_path(opts.stage3Folder);
catch
    stage3Root = '';
    fprintf('Note: Elliptical regions folder not found (%s) - skipping ellipse overlays\n', opts.stage3Folder);
end

phoneDirs = list_phone_dirs(stage2Root);
if isempty(phoneDirs)
    error('preview_augmented_overlays:noPhones', ...
        'No phone folders found in %s.', stage2Root);
end

if ~isempty(requestedPhones)
    phoneDirs = intersect(phoneDirs, requestedPhones, 'stable');
    if isempty(phoneDirs)
        error('preview_augmented_overlays:noRequestedPhones', ...
            'Requested phone folder(s) not present inside %s.', stage2Root);
    end
end

if isempty(stage3Root)
    fprintf('Previewing augmented overlays from:\n  Stage 1: %s\n  Stage 2: %s\n  Stage 3: (skipped)\n', ...
        stage1Root, stage2Root);
else
    fprintf('Previewing augmented overlays from:\n  Stage 1: %s\n  Stage 2: %s\n  Stage 3: %s\n', ...
        stage1Root, stage2Root, stage3Root);
end

% Build preview plan aggregating all overlays per base scene
plan = build_augmented_plan(stage1Root, stage2Root, stage3Root, phoneDirs, ...
    opts.maxSamples, SUPPORTED_IMAGE_EXTENSIONS, STAGE2_COORD_COLUMNS, STAGE3_COORD_COLUMNS);

if isempty(plan)
    error('preview_augmented_overlays:noPlan', ...
        'No valid preview entries found. Check that coordinates.txt files exist.');
end

fprintf('Found %d augmented scene(s) for preview\n', numel(plan));
fprintf('Controls: Click Next or press ''n'' to advance, ''q'' to quit.\n\n');

% UI setup
state = struct();
state.plan = plan;
state.idx = 1;
state.overlayColor = OVERLAY_COLOR_RGB;
state.ellipseRenderPoints = ELLIPSE_RENDER_POINTS;
state.missingImageHeight = MISSING_IMAGE_HEIGHT;
state.missingImageWidth = MISSING_IMAGE_WIDTH;

state.fig = figure('Name', 'Augmented Overlay Preview', 'Color', 'k', ...
    'NumberTitle', 'off', 'Units', 'normalized', ...
    'Position', [0.1 0.1 0.8 0.8], ...
    'CloseRequestFcn', @(h,~) on_close(h));
state.ax = axes('Parent', state.fig);
set(state.ax, 'Position', [0.05 0.12 0.9 0.83]);
axis(state.ax, 'image');
axis(state.ax, 'off');

% Next button
uicontrol('Parent', state.fig, 'Style', 'pushbutton', 'String', 'Next', ...
    'Units', 'normalized', 'Position', [0.85 0.02 0.10 0.06], ...
    'FontSize', 12, 'Callback', @(h,~) on_next());

% Info text
state.infoText = uicontrol('Parent', state.fig, 'Style', 'text', ...
    'Units', 'normalized', 'Position', [0.05 0.02 0.78 0.06], ...
    'BackgroundColor', [0 0 0], 'ForegroundColor', [1 1 1], ...
    'HorizontalAlignment', 'left', 'String', '');

% Key press handler
set(state.fig, 'KeyPressFcn', @(~,e) on_key(e));

guidata(state.fig, state);

% Initial draw
draw_current();

    function on_key(e)
        if isfield(e, 'Key')
            if strcmpi(e.Key, 'n')
                on_next();
            elseif strcmpi(e.Key, 'q')
                on_close(state.fig);
            end
        end
    end

    function on_next()
        st = guidata(gcf);
        if st.idx < numel(st.plan)
            st.idx = st.idx + 1;
        else
            st.idx = 1; % loop
        end
        guidata(gcf, st);
        draw_current();
    end

    function draw_current()
        st = guidata(gcf);
        entry = st.plan(st.idx);
        cla(st.ax);

        titleStr = sprintf('%d/%d  %s | %s', st.idx, numel(st.plan), ...
            entry.phoneName, entry.imageName);

        try
            if entry.imageMissing
                img = zeros(st.missingImageHeight, st.missingImageWidth, 3, 'uint8');
                imshow(img, 'Parent', st.ax);
                hold(st.ax, 'on');
                draw_augmented_overlays(st.ax, entry, st.ellipseRenderPoints, st.overlayColor);
                hold(st.ax, 'off');
            else
                img = imread(entry.imagePath);
                imshow(img, 'Parent', st.ax);
                hold(st.ax, 'on');
                draw_augmented_overlays(st.ax, entry, st.ellipseRenderPoints, st.overlayColor);
                hold(st.ax, 'off');
            end
        catch ME
            warning('preview_augmented_overlays:display_error', ...
                'Failed to display %s: %s', entry.imagePath, ME.message);
        end

        set(st.infoText, 'String', titleStr);
        drawnow;
    end

    function on_close(fig)
        if ishghandle(fig)
            % Clean up plan data structure from guidata
            st = guidata(fig);
            if isfield(st, 'plan')
                st.plan = [];
            end
            guidata(fig, st);

            delete(fig);
        end
    end
end

%% Helper functions

function plan = build_augmented_plan(stage1Root, stage2Root, stage3Root, phoneDirs, maxSamples, supportedExts, stage2Cols, stage3Cols)
% Build preview plan aggregating all polygons and ellipses per base scene

plan = struct('phoneName', {}, 'imageName', {}, 'imagePath', {}, ...
    'imageMissing', {}, 'polygons', {}, 'ellipses', {});

entryCount = 0;
imageMap = containers.Map('KeyType', 'char', 'ValueType', 'int32');

for pIdx = 1:numel(phoneDirs)
    if entryCount >= maxSamples
        break;
    end

    phone = phoneDirs(pIdx);
    phoneStr = char(phone);

    % Read stage-2 polygon coordinates
    s2CoordPath = fullfile(stage2Root, phoneStr, 'coordinates.txt');
    if ~isfile(s2CoordPath)
        warning('preview_augmented_overlays:missingS2Coords', ...
            'Missing stage-2 coordinates.txt for %s', phoneStr);
        continue;
    end

    s2Table = read_polygon_table(s2CoordPath, stage2Cols);
    if isempty(s2Table)
        continue;
    end

    % Read stage-3 ellipse coordinates (optional)
    if isempty(stage3Root) || ~isfolder(stage3Root)
        s3Table = table();
    else
        s3CoordPath = fullfile(stage3Root, phoneStr, 'coordinates.txt');
        if isfile(s3CoordPath)
            s3Table = read_ellipse_table(s3CoordPath, stage3Cols);
        else
            s3Table = table();
        end
    end

    % Group polygons by base scene name
    for rowIdx = 1:height(s2Table)
        if entryCount >= maxSamples
            break;
        end

        imageName = s2Table.image(rowIdx);
        concentration = s2Table.concentration(rowIdx);
        imageChar = char(imageName);

        % Extract base scene name (strip _con_X suffix)
        % Stage-2 crops have _con_X suffix: IMG_0957_aug_001_con_0.jpeg
        % Stage-1 originals have no suffix: IMG_0957_aug_001.jpg
        % Note: augment_dataset.m may write different extensions for stage 1 vs 2
        % (e.g., .jpg -> .jpeg), so we must search for stage-1 file by basename only
        [~, baseName, ~] = fileparts(imageChar);  % Remove extension first
        conIdx = strfind(baseName, '_con_');
        if ~isempty(conIdx)
            baseNameNoConc = baseName(1:conIdx(1)-1);
        else
            baseNameNoConc = baseName;
        end

        % Create or retrieve plan entry for this base scene
        key = sprintf('%s|%s', phoneStr, baseNameNoConc);
        if isKey(imageMap, key)
            idx = imageMap(key);
        else
            % Create new entry - find actual stage-1 image file
            stage1PhoneDir = fullfile(stage1Root, phoneStr);
            scenePath = find_image_file(stage1PhoneDir, baseNameNoConc, supportedExts);

            if isempty(scenePath)
                warning('preview_augmented_overlays:missingStage1', ...
                    'Cannot find stage-1 image for %s/%s', phoneStr, baseNameNoConc);
                continue;
            end

            entryCount = entryCount + 1;
            idx = entryCount;
            imageMap(key) = idx;

            [~, ~, ext] = fileparts(scenePath);
            plan(idx).phoneName = phoneStr;
            plan(idx).imageName = [baseNameNoConc, ext];
            plan(idx).imagePath = scenePath;
            plan(idx).imageMissing = ~isfile(scenePath);
            plan(idx).polygons = {};
            plan(idx).ellipses = {};
        end

        % Add polygon to this entry
        polygon = [
            s2Table.x1(rowIdx), s2Table.y1(rowIdx);
            s2Table.x2(rowIdx), s2Table.y2(rowIdx);
            s2Table.x3(rowIdx), s2Table.y3(rowIdx);
            s2Table.x4(rowIdx), s2Table.y4(rowIdx);
        ];
        plan(idx).polygons{end+1} = polygon;

        % Find matching ellipses for this concentration polygon
        if ~isempty(s3Table)
            % Match by the full stage-2 filename (with extension)
            nameMask = strcmpi(s3Table.image, string(imageChar));
            concMask = s3Table.concentration == concentration;
            ellipseRows = s3Table(nameMask & concMask, :);

            if ~isempty(ellipseRows)
                % Transform ellipse coordinates from polygon-crop space to scene space
                minXY = [min(polygon(:,1)), min(polygon(:,2))];
                for eIdx = 1:height(ellipseRows)
                    ellipse = struct();
                    ellipse.center = [ellipseRows.x(eIdx), ellipseRows.y(eIdx)] + minXY;
                    ellipse.semiMajorAxis = ellipseRows.semiMajorAxis(eIdx);
                    ellipse.semiMinorAxis = ellipseRows.semiMinorAxis(eIdx);
                    ellipse.rotationAngle = ellipseRows.rotationAngle(eIdx);
                    plan(idx).ellipses{end+1} = ellipse;
                end
            end
        end
    end
end

% Sort by display name
if ~isempty(plan)
    displayNames = arrayfun(@(p) sprintf('%s/%s', p.phoneName, p.imageName), ...
        plan, 'UniformOutput', false);
    [~, order] = sort(displayNames);
    plan = plan(order);
end
end

function imgPath = find_image_file(phoneDir, baseName, ~)
% Search for image file matching baseName with any supported extension
%
% Delegates to image_io.findImageFile for consistent file discovery
% across all scripts. Uses persistent cache for performance.
%
% Note: Third argument (supportedExts) is ignored - image_io uses its
% built-in list of supported extensions for consistency.
%
% See also: image_io.findImageFile

persistent imageIO cache
if isempty(imageIO)
    imageIO = image_io();
    cache = imageIO.createCaches();
end

imgPath = imageIO.findImageFile(phoneDir, baseName, cache);
end

function draw_augmented_overlays(ax, entry, ellipseRenderPoints, overlayColor)
% Draw all concentration polygons and ellipses for this augmented scene

persistent theta cosTheta sinTheta lastRenderPoints

% Pre-compute parametric angles for ellipse rendering
if isempty(lastRenderPoints) || lastRenderPoints ~= ellipseRenderPoints
    theta = linspace(0, 2*pi, ellipseRenderPoints);
    cosTheta = cos(theta);
    sinTheta = sin(theta);
    lastRenderPoints = ellipseRenderPoints;
end

% Draw all concentration polygons
numPolygons = numel(entry.polygons);
for i = 1:numPolygons
    P = entry.polygons{i};
    plot(ax, [P(:,1); P(1,1)], [P(:,2); P(1,2)], '-', ...
        'Color', overlayColor, 'LineWidth', 0.5);
end

% Draw all ellipses (if available)
numEllipses = numel(entry.ellipses);
if numEllipses > 0
    for i = 1:numEllipses
        ellipse = entry.ellipses{i};
        center = ellipse.center;
        a = ellipse.semiMajorAxis;
        b = ellipse.semiMinorAxis;
        theta_rad = deg2rad(ellipse.rotationAngle);

        % Parametric ellipse with rotation
        ellipseX = center(1) + a * cosTheta * cos(theta_rad) - b * sinTheta * sin(theta_rad);
        ellipseY = center(2) + a * cosTheta * sin(theta_rad) + b * sinTheta * cos(theta_rad);

        plot(ax, ellipseX, ellipseY, '-', 'Color', overlayColor, 'LineWidth', 0.5);
    end
end
% Note: Silently skip ellipse overlay if no data available (not an error)
end

function rootPath = resolve_repo_path(targetFolder)
% Locate targetFolder relative to current working directory or its parents
if isfolder(targetFolder)
    rootPath = char(targetFolder);
    return;
end

startDir = pwd;
searchDir = startDir;
for depth = 1:5
    candidate = fullfile(searchDir, targetFolder);
    if isfolder(candidate)
        rootPath = candidate;
        return;
    end
    parent = fileparts(searchDir);
    if strcmp(parent, searchDir)
        break;
    end
    searchDir = parent;
end

error('preview_augmented_overlays:missingFolder', ...
    'Could not find folder "%s" relative to %s.', targetFolder, startDir);
end

function names = list_phone_dirs(rootPath)
% Return phone folder names as string array
entries = dir(rootPath);
isPhone = [entries.isdir] & ~ismember({entries.name}, {'.', '..'});
names = string({entries(isPhone).name});
end

function tbl = read_polygon_table(coordPath, ~)
% Read concentration polygon coordinate table using coordinate_io
%
% Delegates to coordinate_io.parsePolygonCoordinateFileAsTable for consistent
% parsing across all scripts. See coordinate_io.m for authoritative format docs.
%
% Note: Second argument (expectedColumns) is ignored - coordinate_io uses
% standardized column names for consistency.
%
% See also: coordinate_io.parsePolygonCoordinateFileAsTable

persistent coordIO
if isempty(coordIO)
    coordIO = coordinate_io();
end

tbl = coordIO.parsePolygonCoordinateFileAsTable(coordPath);
end

function tbl = read_ellipse_table(coordPath, ~)
% Read ellipse metadata table using coordinate_io
%
% Delegates to coordinate_io.parseEllipseCoordinateFileAsTable for consistent
% parsing across all scripts. See coordinate_io.m for authoritative format docs.
%
% Note: Second argument (expectedColumns) is ignored - coordinate_io uses
% standardized column names for consistency.
%
% See also: coordinate_io.parseEllipseCoordinateFileAsTable

persistent coordIO
if isempty(coordIO)
    coordIO = coordinate_io();
end

tbl = coordIO.parseEllipseCoordinateFileAsTable(coordPath);
end
