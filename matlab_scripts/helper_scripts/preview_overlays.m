function preview_overlays(varargin)
    %% Preview multi-stage overlays from dataset through elliptical patches
    %% Author: Veysel Y. Yilmaz
    %
    % Overlay rectangular crops, concentration polygons, and elliptical patches
    % on top of the original captures in 1_dataset/ for integrity checks.
    % Stage dependencies are verified before visualization.
    %
    % INPUTS (name-value pairs):
    % - datasetFolder : root of original captures (default '1_dataset')
% - coordsFolder  : root of concentration polygons (default '2_micropads')
% - ellipseFolder : root of elliptical patches (default '3_elliptical_regions')
    %
    % OUTPUTS: none (opens a viewer window)
    %
    % USAGE:
    %   addpath('matlab_scripts/helper_scripts'); preview_overlays
%   preview_overlays('datasetFolder','1_dataset', ...
%                    'coordsFolder','2_micropads', ...
%                    'ellipseFolder','3_elliptical_regions')
    %
    % NOTES:
    % - Navigation: Click 'Next' or press 'n' to advance; press 'q' to close.

    % CONFIGURATION CONSTANTS
    MISSING_IMAGE_HEIGHT = 480;
    MISSING_IMAGE_WIDTH = 640;
    PROJECT_ROOT_SEARCH_DEPTH = 5;
    ELLIPSE_RENDER_POINTS = 60;
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    OVERLAY_COLOR_RGB = [0.48, 0.99, 0.00];  % Fluorescent green

    % Validate Image Processing Toolbox availability
    if ~license('test', 'image_toolbox')
        error('preview_concentration_overlays:missing_toolbox', ...
            'Image Processing Toolbox required');
    end

    % Parse and validate inputs
    parser = inputParser;

    addParameter(parser, 'datasetFolder', '1_dataset', @(s) validateattributes(s, {'char','string'}, {'scalartext'}));
    addParameter(parser, 'coordsFolder', '2_micropads', @(s) validateattributes(s, {'char','string'}, {'scalartext'}));
    addParameter(parser, 'ellipseFolder', '3_elliptical_regions', @(s) validateattributes(s, {'char','string'}, {'scalartext'}));

    parse(parser, varargin{:});

    datasetRootIn = char(parser.Results.datasetFolder);
    coordsRootIn = char(parser.Results.coordsFolder);
    ellipseRootIn = char(parser.Results.ellipseFolder);

    % Resolve paths relative to repo root using standard findProjectRoot
    repoRoot = findProjectRoot(datasetRootIn, PROJECT_ROOT_SEARCH_DEPTH);
    datasetRoot = resolve_folder(repoRoot, datasetRootIn);
    coordsRoot = resolve_folder(repoRoot, coordsRootIn);
    ellipseRoot = resolve_folder(repoRoot, ellipseRootIn);

    validate_folder_exists(datasetRoot, 'preview_concentration_overlays:missing_dataset_folder', 'Dataset folder not found: %s\nExpected path relative to project root.', datasetRootIn);
    validate_folder_exists(coordsRoot, 'preview_concentration_overlays:missing_coords_folder', 'Coordinates folder not found: %s\nExpected path relative to project root.', coordsRootIn);
    validate_folder_exists(ellipseRoot, 'preview_concentration_overlays:missing_ellipse_folder', 'Elliptical patch folder not found: %s\nExpected path relative to project root.', ellipseRootIn);

    % Validate that coordinates folder contains phone subdirectories
    coordPhones = dir(coordsRoot);
    coordPhones = coordPhones([coordPhones.isdir] & ~ismember({coordPhones.name}, {'.', '..'}));
    if isempty(coordPhones)
        error('preview_concentration_overlays:empty_coords_folder', ...
            'No phone subdirectories found in coordinates folder: %s', coordsRoot);
    end

    fprintf('Dataset root: %s\n', datasetRoot);
    fprintf('Concentration root: %s\n', coordsRoot);
    fprintf('Ellipse root: %s\n', ellipseRoot);

    % Build mapping from image path -> list of polygons and ellipses
    plan = build_plan(datasetRoot, coordsRoot, ellipseRoot, SUPPORTED_IMAGE_EXTENSIONS);
    if isempty(plan)
        error('preview_concentration_overlays:no_entries', ...
            ['Coordinate integrity failure: no valid entries found under %s.' ...
             '\nEnsure coordinates.txt exists and is populated for every phone in stages 2-4.'], coordsRoot);
    end

    fprintf('Found %d image entries for preview\n', length(plan));

    % UI setup
    state = struct();
    state.plan = plan;
    state.idx = 1;
    state.overlayColor = OVERLAY_COLOR_RGB;
    state.ellipseRenderPoints = ELLIPSE_RENDER_POINTS;
    state.fig = figure('Name','Concentration Polygons Preview', 'Color','k', 'NumberTitle','off', ...
                       'Units','normalized', 'Position',[0.1 0.1 0.8 0.8], ...
                       'CloseRequestFcn', @(h,~) on_close(h));
    state.ax = axes('Parent', state.fig);
    set(state.ax, 'Position', [0.05 0.12 0.9 0.83]);
    axis(state.ax, 'image');
    axis(state.ax, 'off');

    % Next button
    uicontrol('Parent', state.fig, 'Style','pushbutton', 'String','Next', ...
              'Units','normalized', 'Position',[0.85 0.02 0.10 0.06], ...
              'FontSize', 12, 'Callback', @(h,~) on_next());

    % Info text
    state.infoText = uicontrol('Parent', state.fig, 'Style','text', ...
                               'Units','normalized', 'Position',[0.05 0.02 0.78 0.06], ...
                               'BackgroundColor',[0 0 0], 'ForegroundColor',[1 1 1], ...
                               'HorizontalAlignment','left', 'String','');

    % Key press: 'n' for next, 'q' to quit
    set(state.fig, 'KeyPressFcn', @(~,e) on_key(e));

    % Store state
    guidata(state.fig, state);

    % Initial draw
    draw_current();

    % Nested callbacks and helpers use guidata to access/update state
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
        titleStr = sprintf('%d/%d  %s', st.idx, numel(st.plan), entry.displayName);
        try
            if entry.imageMissing
                % Draw black canvas if missing
                img = zeros(MISSING_IMAGE_HEIGHT, MISSING_IMAGE_WIDTH, 3, 'uint8');
                imshow(img, 'Parent', st.ax);
                hold(st.ax, 'on');
                draw_overlays(st.ax, entry, st.ellipseRenderPoints, st.overlayColor);
                hold(st.ax, 'off');
            else
                img = imread_raw(entry.imagePath);
                imshow(img, 'Parent', st.ax);
                hold(st.ax, 'on');
                draw_overlays(st.ax, entry, st.ellipseRenderPoints, st.overlayColor);
                hold(st.ax, 'off');
            end
        catch ME
            warning('preview_concentration_overlays:display_error', 'Failed to display %s: %s', entry.imagePath, ME.message);
        end
        set(st.infoText, 'String', titleStr);
        drawnow;
    end
    function on_close(fig)
        if ishghandle(fig)
            st = guidata(fig);
            if ~isempty(st)
                % Clear large data structures to release memory
                st.plan = [];
            end
            guidata(fig, []);
            delete(fig);
        end
        % Clear persistent caches
        clear_preview_caches();
    end
end

%% ------------------------------------------------------------------------
function plan = build_plan(datasetRoot, coordsRoot, ellipseRoot, supportedExts)
    %% Build preview plan mapping images to multi-stage overlays in dataset space

    coordFiles = find_concentration_coordinate_files(coordsRoot);
    plan = struct('phoneName', {}, 'imagePath', {}, 'displayName', {}, ...
                  'polygons', {}, 'ellipses', {}, 'rectPolygon', {}, ...
                  'imageMissing', {}, 'rectMeta', {});
    if isempty(coordFiles)
        return;
    end

    idxMap = containers.Map('KeyType','char','ValueType','int32');
    polygonCounts = containers.Map('KeyType','char','ValueType','int32');

    totalPolygonRows = 0;
    for k = 1:numel(coordFiles)
        cfile = coordFiles{k};
        T = read_coordinates_table(cfile);
        if isempty(T)
            continue;
        end
        T = standardize_coord_vars(T, cfile);
        numRows = height(T);
        totalPolygonRows = totalPolygonRows + numRows;

        [cdir, ~, ~] = fileparts(cfile);
        [relDir, okRel] = relative_subpath(coordsRoot, cdir);
        if ~okRel
            error('preview_concentration_overlays:invalid_coords_path', ...
                  'Unable to resolve phone folder for coordinates file: %s', cfile);
        end
        if isempty(relDir)
            [~, phoneName, ~] = fileparts(cdir);
        else
            tokens = strsplit(relDir, '/');
            if isempty(tokens) || isempty(tokens{1})
                error('preview_concentration_overlays:invalid_phone_folder', ...
                      'Cannot determine phone name from %s', cdir);
            end
            phoneName = tokens{1};
        end

        baseNames = cellstr(string(T.image));
        for r = 1:numRows
            baseName = standardize_base_name(baseNames{r});
            key = sprintf('%s|%s', phoneName, baseName);
            if isKey(polygonCounts, key)
                polygonCounts(key) = polygonCounts(key) + 1;
            else
                polygonCounts(key) = 1;
            end
        end
    end

    if totalPolygonRows == 0
        error('preview_concentration_overlays:empty_polygon_data', ...
              'No concentration polygons found under %s.', coordsRoot);
    end

    allPolygonData = repmat(struct('phoneName', '', 'imageName', '', 'concentration', 0, 'polygon', []), totalPolygonRows, 1);
    polygonIdx = 0;

    rectPhoneCache = containers.Map('KeyType','char','ValueType','any');
    polygonIndices = containers.Map('KeyType','char','ValueType','int32');

    for k = 1:numel(coordFiles)
        cfile = coordFiles{k};
        [cdir, ~, ~] = fileparts(cfile);
        [relDir, okRel] = relative_subpath(coordsRoot, cdir);
        if ~okRel
            continue;
        end
        if isempty(relDir)
            [~, phoneName, ~] = fileparts(cdir);
        else
            tokens = strsplit(relDir, '/');
            if isempty(tokens) || isempty(tokens{1})
                continue;
            end
            phoneName = tokens{1};
        end

        T = read_coordinates_table(cfile);
        if isempty(T)
            continue;
        end
        T = standardize_coord_vars(T, cfile);

        concValues = extract_concentration_column(T);
        baseNames = cellstr(string(T.image));
        polygonCoords = [T.x1, T.y1, T.x2, T.y2, T.x3, T.y3, T.x4, T.y4];

        numRows = height(T);
        for r = 1:numRows
            baseName = standardize_base_name(baseNames{r});
            rectEntry = fetch_rectangle_entry(rectPhoneCache, datasetRoot, rectRoot, phoneName, baseName, supportedExts);
            key = sprintf('%s|%s', phoneName, baseName);

            if isKey(idxMap, key)
                idx = idxMap(key);
                polyIdx = polygonIndices(key);
            else
                idx = numel(plan) + 1;
                entry = struct();
                entry.phoneName = phoneName;
                entry.imagePath = rectEntry.originalPath;
                entry.displayName = compute_display_name(datasetRoot, rectEntry.originalPath);
                entry.polygons = cell(1, polygonCounts(key));
                entry.ellipses = {};
                entry.rectPolygon = rectEntry.rectPolygonStage1;
                entry.imageMissing = false;
                entry.rectMeta = rectEntry;
                plan(idx) = entry;
                idxMap(key) = idx;
                polygonIndices(key) = 1;
                polyIdx = 1;
            end

            polyRect = reshape(polygonCoords(r,:), 2, 4)';
            polyStage1 = map_rect_points_to_stage1(polyRect, plan(idx).rectMeta);
            plan(idx).polygons{polyIdx} = polyStage1;
            polygonIndices(key) = polygonIndices(key) + 1;

            concValue = concValues(r);
            if isnan(concValue)
                error('preview_concentration_overlays:missing_concentration', ...
                      'Concentration missing for %s/%s (row %d) in %s.', phoneName, baseName, r, cfile);
            end

            polygonIdx = polygonIdx + 1;
            allPolygonData(polygonIdx) = struct('phoneName', phoneName, ...
                                                'imageName', baseName, ...
                                                'concentration', concValue, ...
                                                'polygon', polyRect);
        end
    end

    allPolygonData = allPolygonData(1:polygonIdx);

    ellipseFiles = find_concentration_coordinate_files(ellipseRoot);
    if isempty(ellipseFiles)
        error('preview_concentration_overlays:missing_ellipse_coords', ...
              'No coordinates.txt files found in %s. Run cut_elliptical_regions first.', ellipseRoot);
    end

    ellipseTables = cell(numel(ellipseFiles), 1);
    ellipsePhones = cell(numel(ellipseFiles), 1);
    totalEllipseRows = 0;
    for k = 1:numel(ellipseFiles)
        efile = ellipseFiles{k};
        [edir, ~, ~] = fileparts(efile);
        [relDir, okRel] = relative_subpath(ellipseRoot, edir);
        if ~okRel
            error('preview_concentration_overlays:invalid_ellipse_path', ...
                  'Unable to resolve phone folder for ellipse coordinates file: %s', efile);
        end
        if isempty(relDir)
            [~, phoneName, ~] = fileparts(edir);
        else
            tokens = strsplit(relDir, '/');
            if isempty(tokens) || isempty(tokens{1})
                error('preview_concentration_overlays:invalid_ellipse_phone', ...
                      'Cannot determine phone name from ellipse path: %s', edir);
            end
            phoneName = tokens{1};
        end

        T = read_ellipse_coordinates_table(efile);
        if isempty(T)
            error('preview_concentration_overlays:empty_ellipse_table', ...
                  'Ellipse coordinates file has no entries: %s', efile);
        end
        T = standardize_ellipse_coord_vars(T, efile);
        ellipseTables{k} = T;
        ellipsePhones{k} = phoneName;
        totalEllipseRows = totalEllipseRows + height(T);
    end

    ellipsePhoneSet = containers.Map('KeyType','char','ValueType','logical');
    for k = 1:numel(ellipsePhones)
        ellipsePhoneSet(ellipsePhones{k}) = true;
    end
    phonesInPlan = unique({plan.phoneName});
    for i = 1:numel(phonesInPlan)
        if ~isKey(ellipsePhoneSet, phonesInPlan{i})
            error('preview_concentration_overlays:missing_phone_ellipses', ...
                  'Elliptical coordinates missing for phone %s in %s.', phonesInPlan{i}, ellipseRoot);
        end
    end

    allEllipseData = repmat(struct('phoneName', '', 'imageName', '', 'x', 0, 'y', 0, ...
                                   'semiMajorAxis', 0, 'semiMinorAxis', 0, 'rotationAngle', 0, ...
                                   'concentration', 0, 'replicate', 0), totalEllipseRows, 1);
    ellipseIdx = 0;

    for k = 1:numel(ellipseTables)
        T = ellipseTables{k};
        if isempty(T)
            continue;
        end
        phoneName = ellipsePhones{k};

        numRows = height(T);
        for r = 1:numRows
            ellipseIdx = ellipseIdx + 1;
            imageName = char(T.image(r));
            baseName = standardize_base_name(imageName);
            allEllipseData(ellipseIdx) = struct('phoneName', phoneName, ...
                                                'imageName', baseName, ...
                                                'x', double(T.x(r)), ...
                                                'y', double(T.y(r)), ...
                                                'semiMajorAxis', double(T.semiMajorAxis(r)), ...
                                                'semiMinorAxis', double(T.semiMinorAxis(r)), ...
                                                'rotationAngle', double(T.rotationAngle(r)), ...
                                                'concentration', double(T.concentration(r)), ...
                                                'replicate', double(T.replicate(r)));
        end
    end

    allEllipseData = allEllipseData(1:ellipseIdx);
    transformedEllipses = transform_ellipse_coordinates(allEllipseData, allPolygonData);

    planMap = containers.Map();
    for idx = 1:numel(plan)
        key = sprintf('%s|%s', plan(idx).phoneName, plan(idx).rectMeta.baseName);
        planMap(key) = idx;
    end

    for i = 1:numel(transformedEllipses)
        ellipse = transformedEllipses(i);
        key = sprintf('%s|%s', ellipse.phoneName, ellipse.imageName);
        if ~isKey(planMap, key)
            error('preview_concentration_overlays:ellipse_no_match', ...
                  'Ellipse entry references missing polygon: phone=%s image=%s.', ellipse.phoneName, ellipse.imageName);
        end
        idx = planMap(key);
        rectEntry = plan(idx).rectMeta;
        centerStage1 = map_rect_points_to_stage1([ellipse.x, ellipse.y], rectEntry);
        thetaStage1 = normalize_angle(rectEntry.rotation + ellipse.rotationAngle);
        plan(idx).ellipses{end+1} = [centerStage1(1), centerStage1(2), ellipse.semiMajorAxis, ellipse.semiMinorAxis, thetaStage1];
    end

    for idx = 1:numel(plan)
        if isempty(plan(idx).ellipses)
            baseName = plan(idx).rectMeta.baseName;
            error('preview_concentration_overlays:missing_ellipses_for_image', ...
                  'No elliptical patches found for %s/%s. Complete cut_elliptical_regions before preview.', ...
                  plan(idx).phoneName, baseName);
        end
    end

    if ~isempty(plan)
        [~, order] = sort({plan.displayName});
        plan = plan(order);
        plan = rmfield(plan, 'rectMeta');
    end
end

function rectEntry = fetch_rectangle_entry(rectPhoneCache, datasetRoot, rectRoot, phoneName, baseName, supportedExts)
    if isKey(rectPhoneCache, phoneName)
        phoneData = rectPhoneCache(phoneName);
    else
        phoneData = load_phone_rect_data(datasetRoot, rectRoot, phoneName, supportedExts);
        rectPhoneCache(phoneName) = phoneData;
    end

    if ~isKey(phoneData.entries, baseName)
        error('preview_concentration_overlays:missing_rect_coordinates', ...
              'Rectangular coordinates missing for %s/%s in %s.', ...
              phoneName, baseName, fullfile(rectRoot, phoneName));
    end
    rectEntry = phoneData.entries(baseName);
end

function phoneData = load_phone_rect_data(datasetRoot, rectRoot, phoneName, supportedExts)
    rectPhoneDir = fullfile(rectRoot, phoneName);
    if ~isfolder(rectPhoneDir)
        error('preview_concentration_overlays:missing_rect_phone', ...
              'Rectangular folder missing for phone %s at %s', phoneName, rectPhoneDir);
    end

    coordFile = fullfile(rectPhoneDir, 'coordinates.txt');
    if ~isfile(coordFile)
        error('preview_concentration_overlays:missing_rect_coord_file', ...
              'coordinates.txt not found for phone %s in %s', phoneName, rectPhoneDir);
    end

    datasetPhoneDir = fullfile(datasetRoot, phoneName);
    if ~isfolder(datasetPhoneDir)
        error('preview_concentration_overlays:missing_dataset_phone', ...
              'Dataset folder missing for phone %s at %s', phoneName, datasetPhoneDir);
    end

    T = read_rectangle_coordinates_table(coordFile);
    if isempty(T)
        error('preview_concentration_overlays:empty_rect_table', ...
              'Rectangular coordinates file has no entries: %s', coordFile);
    end
    T = standardize_rect_coord_vars(T, coordFile);

    entries = containers.Map('KeyType','char','ValueType','any');
    for r = 1:height(T)
        imageName = char(T.image(r));
        baseName = standardize_base_name(imageName);
        if isempty(baseName)
            error('preview_concentration_overlays:invalid_rect_image', ...
                  'Invalid image name in %s (row %d).', coordFile, r);
        end
        if isKey(entries, baseName)
            error('preview_concentration_overlays:duplicate_rect_image', ...
                  'Duplicate rectangular coordinates for %s/%s.', phoneName, baseName);
        end

        rectParams = struct('x', double(T.x(r)), ...
                            'y', double(T.y(r)), ...
                            'width', double(T.width(r)), ...
                            'height', double(T.height(r)));
        rotation = double(T.rotation(r));
        if ~isfinite(rotation)
            rotation = 0;
        end

        rectImagePath = fullfile(rectPhoneDir, imageName);
        if ~isfile(rectImagePath)
            error('preview_concentration_overlays:missing_rect_crop', ...
                  'Rectangular crop missing for %s/%s at %s', phoneName, imageName, rectImagePath);
        end

        origExact = fullfile(datasetPhoneDir, imageName);
        if isfile(origExact)
            origPath = origExact;
        else
            origPath = find_image_file(datasetPhoneDir, baseName, supportedExts);
        end
        if isempty(origPath) || ~isfile(origPath)
            error('preview_concentration_overlays:missing_dataset_image', ...
                  'Dataset image missing for %s/%s in %s', phoneName, baseName, datasetPhoneDir);
        end
        origPath = char(origPath);

        info = imfinfo(origPath);
        transform = compute_rotation_transform(double(info.Width), double(info.Height), rotation);
        rectPolygonStage1 = compute_rect_polygon_stage1(rectParams, transform);

        rectEntry = struct('phoneName', phoneName, ...
                           'baseName', baseName, ...
                           'imageName', imageName, ...
                           'rectParams', rectParams, ...
                           'rotation', rotation, ...
                           'originalPath', origPath, ...
                           'rectPolygonStage1', rectPolygonStage1, ...
                           'transform', transform);
        entries(baseName) = rectEntry;
    end

    phoneData = struct('entries', entries);
end

function T = read_rectangle_coordinates_table(coordFile)
    try
        opts = detectImportOptions(coordFile, 'FileType', 'text');
        opts.Delimiter = {' ', '\t'};
        opts.ConsecutiveDelimitersRule = 'join';
        T = readtable(coordFile, opts);
    catch ME
        error('preview_concentration_overlays:rect_read_failed', ...
              'Failed to parse rectangular coordinates file %s: %s', coordFile, ME.message);
    end
end

function T = standardize_rect_coord_vars(T, sourceName)
    v = lower(string(T.Properties.VariableNames));
    expected = ["image","x","y","width","height","rotation"];
    for i = 1:numel(expected)
        matchIdx = find(v == expected(i), 1);
        if ~isempty(matchIdx)
            T.Properties.VariableNames{matchIdx} = char(expected(i));
        end
    end

    missing = setdiff(cellstr(expected), T.Properties.VariableNames);
    if ~isempty(missing)
        error('preview_concentration_overlays:rect_columns', ...
              'Missing columns in %s: %s', sourceName, strjoin(missing, ', '));
    end

    if ~iscellstr(T.image) && ~isstring(T.image)
        T.image = string(T.image);
    end
    numericVars = {'x','y','width','height','rotation'};
    for i = 1:numel(numericVars)
        varName = numericVars{i};
        if ~isnumeric(T.(varName))
            T.(varName) = str2double(string(T.(varName)));
        else
            T.(varName) = double(T.(varName));
        end
    end
end

function rectPoly = compute_rect_polygon_stage1(rectParams, transform)
    rotCorners = [rectParams.x, rectParams.y;
                  rectParams.x + rectParams.width - 1, rectParams.y;
                  rectParams.x + rectParams.width - 1, rectParams.y + rectParams.height - 1;
                  rectParams.x, rectParams.y + rectParams.height - 1];
    rectPoly = map_rotated_to_original(rotCorners, transform);
end

function transform = compute_rotation_transform(imageWidth, imageHeight, rotationDeg)
    if ~isfinite(rotationDeg)
        rotationDeg = 0;
    end
    theta = deg2rad(rotationDeg);
    cosT = cos(theta);
    sinT = sin(theta);
    centerX = (imageWidth + 1) / 2;
    centerY = (imageHeight + 1) / 2;

    corners = [1, 1;
               imageWidth, 1;
               imageWidth, imageHeight;
               1, imageHeight];
    centered = corners - [centerX, centerY];
    R = [cosT, sinT; -sinT, cosT];
    rotated = (R * centered')';
    rotated = rotated + [centerX, centerY];
    offsetX = 1 - min(rotated(:,1));
    offsetY = 1 - min(rotated(:,2));

    transform = struct('cosTheta', cosT, ...
                       'sinTheta', sinT, ...
                       'offsetX', offsetX, ...
                       'offsetY', offsetY, ...
                       'centerX', centerX, ...
                       'centerY', centerY, ...
                       'rotationDeg', rotationDeg);
end

function pointsStage1 = map_rect_points_to_stage1(pointsRect, rectEntry)
    if isempty(pointsRect)
        pointsStage1 = zeros(0, 2);
        return;
    end
    rectParams = rectEntry.rectParams;
    rotPoints = [rectParams.x + pointsRect(:,1) - 1, ...
                 rectParams.y + pointsRect(:,2) - 1];
    pointsStage1 = map_rotated_to_original(rotPoints, rectEntry.transform);
end

function pointsOrig = map_rotated_to_original(rotPoints, transform)
    if isempty(rotPoints)
        pointsOrig = zeros(0, 2);
        return;
    end
    xShift = rotPoints(:,1) - transform.offsetX;
    yShift = rotPoints(:,2) - transform.offsetY;
    dx = xShift - transform.centerX;
    dy = yShift - transform.centerY;
    cosT = transform.cosTheta;
    sinT = transform.sinTheta;
    xOrig = dx * cosT - dy * sinT + transform.centerX;
    yOrig = dx * sinT + dy * cosT + transform.centerY;
    pointsOrig = [xOrig, yOrig];
end

function angle = normalize_angle(angleDeg)
    % Normalize rotation angle to [-180, 180] degree range
    % Ensures consistency with ellipse rotationAngle domain (0-180° clockwise from horizontal)
    % and handles angle arithmetic across coordinate space transformations
    angle = mod(angleDeg + 180, 360) - 180;
end

function baseName = standardize_base_name(inputName)
    if isstring(inputName)
        inputName = char(inputName);
    end
    if iscell(inputName)
        inputName = char(inputName);
    end
    if isempty(inputName)
        baseName = '';
        return;
    end
    [~, baseName, ~] = fileparts(char(inputName));
    if isempty(baseName)
        baseName = char(inputName);
    end
end

function files = find_concentration_coordinate_files(rootDir)
    % Returns cell array of phone-level coordinates.txt paths
    %
    % Searches only at phone directory level (rootDir/<phone>/coordinates.txt)
    % and does NOT recurse into subdirectories. This matches the pipeline structure
    % where concentration coordinates are consolidated at the phone level, not
    % scattered across con_N subfolders.
    files = {};
    if ~isfolder(rootDir), return; end

    % Get phone directories
    phones = dir(rootDir);
    phones = phones([phones.isdir] & ~ismember({phones.name}, {'.', '..'}));

    % Pre-allocate for phone-level coordinates.txt files
    files = cell(length(phones), 1);
    fileCount = 0;

    for p = 1:length(phones)
        phoneDir = fullfile(rootDir, phones(p).name);
        phoneCoord = fullfile(phoneDir, 'coordinates.txt');
        if isfile(phoneCoord)
            fileCount = fileCount + 1;
            files{fileCount} = phoneCoord;
        end
    end

    files = files(1:fileCount);
end

function d = scan_subdirs_for_file(rootDir, targetName)
    % Recursively find files matching targetName
    d = [];
    if ~isfolder(rootDir), return; end

    % Get all entries at once
    allEntries = dir(rootDir);

    % Vectorized filtering for files matching target name
    isTargetFile = ~[allEntries.isdir] & strcmpi({allEntries.name}, targetName);
    d = allEntries(isTargetFile);

    % Get subdirectories
    subdirs = allEntries([allEntries.isdir] & ~ismember({allEntries.name}, {'.', '..'}));

    % Recursively scan subdirectories with cell accumulation
    if ~isempty(subdirs)
        results = cell(length(subdirs), 1);
        for i = 1:length(subdirs)
            subpath = fullfile(rootDir, subdirs(i).name);
            results{i} = scan_subdirs_for_file(subpath, targetName);
        end
        d = [d; vertcat(results{:})];
    end
end


function [rel, ok] = relative_subpath(ancestor, descendant)
    % Compute descendant path relative to ancestor
    % Returns empty string when paths are equal
    a = char(ancestor); d = char(descendant);
    % Normalize separators and case (Windows-insensitive)
    a = normalize_sep(a); d = normalize_sep(d);
    if startsWith(d, [a '/'], 'IgnoreCase', true)
        rel = d(numel(a)+2:end);
        ok = true;
    elseif strcmpi(d, a)
        rel = '';
        ok = true;
    else
        rel = '';
        ok = false;
    end
    % Convert back to platform separator
    rel = strrep(rel, '/', filesep);
end

function s = normalize_sep(p)
    % Normalize path separators to forward slashes
    persistent cache maxCacheSize;
    if isempty(cache)
        cache = containers.Map('KeyType', 'char', 'ValueType', 'char');
        maxCacheSize = 100;
    end

    pChar = char(p);
    if isKey(cache, pChar)
        s = cache(pChar);
        return;
    end

    s = strrep(pChar, '\', '/');
    % Replace regex with iterative strrep (faster for typical paths)
    while contains(s, '//')
        s = strrep(s, '//', '/');
    end

    % Cache result if under size limit
    if length(cache) < maxCacheSize
        cache(pChar) = s;
    end
end

function T = read_coordinates_table(coordFile)
    T = [];
    try
        opts = detectImportOptions(coordFile, 'FileType', 'text');
        opts.Delimiter = {' ', '\t'};
        opts.ConsecutiveDelimitersRule = 'join';
        T = readtable(coordFile, opts);
    catch ME
        warning('preview_concentration_overlays:coord_read_fallback', ...
            'Import failed for %s: %s\nFalling back to manual read.', coordFile, ME.message);
        % Fallback manual read
        fid = fopen(coordFile, 'rt');
        if fid == -1
            warning('preview_concentration_overlays:coord_open', ...
                'Cannot open coordinates file: %s\nCheck file permissions and path.', coordFile);
            return;
        end
        try
            C = textscan(fid, '%s %f %f %f %f %f %f %f %f %f', 'HeaderLines', 1, ...
                           'Delimiter', {' ', '	'}, 'MultipleDelimsAsOne', true);
        catch ME
            fclose(fid);
            warning('preview_concentration_overlays:coord_textscan_error', ...
                'Failed to parse coordinates file: %s\nError: %s', coordFile, ME.message);
            return;
        end
        fclose(fid);
        if isempty(C) || isempty(C{1})
            return;
        end
        if numel(C) < 10
            warning('preview_concentration_overlays:coord_parse', ...
                'Unexpected coordinate format in %s\nExpected 10 columns: image concentration x1 y1 x2 y2 x3 y3 x4 y4', coordFile);
            return;
        end
        % Validate all columns have equal row counts
        rowCounts = cellfun(@length, C);
        if any(rowCounts ~= rowCounts(1))
            warning('preview_concentration_overlays:coord_parse', ...
                'Inconsistent row counts in %s\nColumn counts: %s', coordFile, mat2str(rowCounts));
            return;
        end
        T = table(C{1}, C{2}, C{3}, C{4}, C{5}, C{6}, C{7}, C{8}, C{9}, C{10}, ...
                  'VariableNames', {'image','concentration','x1','y1','x2','y2','x3','y3','x4','y4'});
    end
end

function T = read_ellipse_coordinates_table(coordFile)
    % Read elliptical patch coordinates.txt files
    % Format: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
    T = [];
    try
        opts = detectImportOptions(coordFile, 'FileType', 'text');
        opts.Delimiter = {' ', '\t'};
        opts.ConsecutiveDelimitersRule = 'join';
        T = readtable(coordFile, opts);
    catch ME
        warning('preview_concentration_overlays:ellipse_read_fallback', ...
            'Import failed for ellipse coordinates %s: %s\nFalling back to manual read.', coordFile, ME.message);
        % Fallback manual read
        fid = fopen(coordFile, 'rt');
        if fid == -1
            warning('preview_concentration_overlays:ellipse_coord_open', ...
                'Cannot open ellipse coordinates file: %s\nCheck file permissions and path.', coordFile);
            return;
        end
        try
            C = textscan(fid, '%s %f %f %f %f %f %f %f', 'HeaderLines', 1);
        catch ME
            fclose(fid);
            warning('preview_concentration_overlays:ellipse_textscan_error', ...
                'Failed to parse ellipse coordinates file: %s\nError: %s', coordFile, ME.message);
            return;
        end
        fclose(fid);
        if isempty(C) || isempty(C{1})
            return;
        end
        if numel(C) < 8
            warning('preview_concentration_overlays:ellipse_coord_parse', ...
                'Unexpected ellipse coordinate format in %s\nExpected 8 columns: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle', coordFile);
            return;
        end
        % Validate all columns have equal row counts
        rowCounts = cellfun(@length, C);
        if any(rowCounts ~= rowCounts(1))
            warning('preview_concentration_overlays:ellipse_coord_parse', ...
                'Inconsistent row counts in %s\nColumn counts: %s', coordFile, mat2str(rowCounts));
            return;
        end
        T = table(C{1}, C{2}, C{3}, C{4}, C{5}, C{6}, C{7}, C{8}, ...
                  'VariableNames', {'image','concentration','replicate','x','y','semiMajorAxis','semiMinorAxis','rotationAngle'});
    end
end

function absFolder = resolve_folder(repoRoot, folderIn)
    % Resolve a folder path, trying:
    % 1) as-is; 2) relative to repo root; 3) relative to current folder
    absFolder = char(folderIn);
    if isfolder(absFolder), return; end
    cand = fullfile(repoRoot, folderIn);
    if isfolder(cand), absFolder = cand; return; end
    % Try relative to pwd as a last resort
    cand = fullfile(pwd, folderIn);
    if isfolder(cand), absFolder = cand; return; end
    % Leave as original (caller will error)
end

function projectRoot = findProjectRoot(inputFolder, maxLevels)
    % Find project root by searching upward from current directory
    currentDir = pwd;
    searchDir = currentDir;

    for level = 1:maxLevels
        [parentDir, ~] = fileparts(searchDir);

        if exist(fullfile(searchDir, inputFolder), 'dir')
            projectRoot = searchDir;
            return;
        end

        if strcmp(searchDir, parentDir)
            break;
        end

        searchDir = parentDir;
    end

    projectRoot = currentDir;
end


function T = standardize_coord_vars(T, sourceName)
    v = lower(string(T.Properties.VariableNames));
    % Expected variables
    expected = ["image","concentration","x1","y1","x2","y2","x3","y3","x4","y4"];
    % If variable names differ only by case or spaces, normalize
    for i = 1:numel(expected)
        match = find(v == expected(i), 1);
        if ~isempty(match) && ~strcmp(T.Properties.VariableNames{match}, char(expected(i)))
            T.Properties.VariableNames{match} = char(expected(i));
        end
    end
    % If missing columns, error out gracefully
    missing = setdiff(cellstr(expected), T.Properties.VariableNames);
    if ~isempty(missing)
        error('preview_concentration_overlays:coord_columns', 'Missing columns in %s: %s', char(sourceName), strjoin(missing, ','));
    end
end

function T = standardize_ellipse_coord_vars(T, sourceName)
    v = lower(string(T.Properties.VariableNames));
    % Expected variables for ellipse coordinates
    expected = ["image","concentration","replicate","x","y","semiMajorAxis","semiMinorAxis","rotationAngle"];
    % If variable names differ only by case or spaces, normalize
    for i = 1:numel(expected)
        match = find(v == expected(i), 1);
        if ~isempty(match) && ~strcmp(T.Properties.VariableNames{match}, char(expected(i)))
            T.Properties.VariableNames{match} = char(expected(i));
        end
    end
    % If missing columns, error out gracefully
    missing = setdiff(cellstr(expected), T.Properties.VariableNames);
    if ~isempty(missing)
        error('preview_concentration_overlays:ellipse_coord_columns', 'Missing columns in %s: %s', char(sourceName), strjoin(missing, ','));
    end
end

function transformedEllipses = transform_ellipse_coordinates(ellipseData, polygonData)
    % Transform ellipse coordinates from concentration region space to rectangular region space
    %
    % INPUTS:
    %   ellipseData - Struct array with fields:
    %       phoneName      (char)   - Device identifier
    %       imageName      (char)   - Concentration region image name (with _con_ tag)
    %       x              (double) - Center x in concentration region space
    %       y              (double) - Center y in concentration region space
    %       semiMajorAxis  (double) - Semi-major axis length
    %       semiMinorAxis  (double) - Semi-minor axis length
    %       rotationAngle  (double) - Rotation angle (degrees)
    %       concentration  (double) - Concentration level
    %       replicate      (double) - Replicate number
    %
    %   polygonData - Struct array with fields:
    %       phoneName     (char)      - Device identifier
    %       imageName     (char)      - Base image name (without _con_ tag)
    %       concentration (double)    - Concentration level
    %       polygon       (4×2 double) - Polygon coordinates [x,y]
    %
    % OUTPUTS:
    %   transformedEllipses - Struct array with fields:
    %       phoneName      (char)   - Device identifier
    %       imageName      (char)   - Base image name (without _con_ tag)
    %       x              (double) - Center x in rectangular region space
    %       y              (double) - Center y in rectangular region space
    %       semiMajorAxis  (double) - Semi-major axis length
    %       semiMinorAxis  (double) - Semi-minor axis length
    %       rotationAngle  (double) - Rotation angle (degrees)

    if isempty(ellipseData)
        transformedEllipses = struct('phoneName', {}, 'imageName', {}, 'x', {}, 'y', {}, 'semiMajorAxis', {}, 'semiMinorAxis', {}, 'rotationAngle', {});
        return;
    end

    numEllipses = length(ellipseData);
    transformedEllipses = repmat(struct('phoneName', '', 'imageName', '', 'x', 0, 'y', 0, 'semiMajorAxis', 0, 'semiMinorAxis', 0, 'rotationAngle', 0), numEllipses, 1);
    outputIdx = 0;

    % Build polygon lookup map with pre-computed keys
    polygonMap = containers.Map();
    validPolygons = ~arrayfun(@(p) isnan(p.concentration), polygonData);
    validPolyData = polygonData(validPolygons);

    for j = 1:length(validPolyData)
        key = strjoin({validPolyData(j).phoneName, validPolyData(j).imageName, format_concentration_key(validPolyData(j).concentration)}, '|');
        if ~isKey(polygonMap, key)
            polygonMap(key) = validPolyData(j);
        end
    end

    % Pre-extract arrays to avoid struct indexing in loop
    imageNames = {ellipseData.imageName};
    phoneNames = {ellipseData.phoneName};
    concentrations = [ellipseData.concentration];
    semiMajorAxes = [ellipseData.semiMajorAxis];
    semiMinorAxes = [ellipseData.semiMinorAxis];
    rotationAngles = [ellipseData.rotationAngle];
    xCoords = [ellipseData.x];
    yCoords = [ellipseData.y];

    % Extract base names using vectorized operations
    conIndices = cellfun(@(name) strfind(name, '_con_'), imageNames, 'UniformOutput', false);
    hasConTag = ~cellfun(@isempty, conIndices);
    ellipseBaseNames = imageNames; % Default: use full name
    ellipseBaseNames(hasConTag) = cellfun(@(name, idx) name(1:idx(1)-1), ...
        imageNames(hasConTag), conIndices(hasConTag), 'UniformOutput', false);

    % Pre-build all lookup keys in batch
    concStrs = arrayfun(@(x) format_concentration_key(x), concentrations, 'UniformOutput', false);
    lookupKeys = strcat(phoneNames, '|', ellipseBaseNames, '|', concStrs);

    % Filter valid ellipses (non-NaN concentration)
    validEllipses = ~isnan(concentrations);

    % Main transformation loop with pre-computed keys
    for i = 1:numEllipses
        if ~validEllipses(i)
            error('preview_concentration_overlays:ellipse_missing_conc', ...
                ['Ellipse concentration missing for %s/%s (ellipse %d/%d).' ...
                 '\nUpdate the ellipse coordinates.txt to include concentration values before previewing.'], ...
                phoneNames{i}, imageNames{i}, i, numEllipses);
        end

        key = lookupKeys{i};
        if isKey(polygonMap, key)
            matchingPolygon = polygonMap(key);
            poly = matchingPolygon.polygon;
            min_x = min(poly(:,1));
            min_y = min(poly(:,2));

            transformed_x = xCoords(i) + min_x;
            transformed_y = yCoords(i) + min_y;

            outputIdx = outputIdx + 1;
            transformedEllipses(outputIdx) = struct('phoneName', phoneNames{i}, ...
                                                 'imageName', ellipseBaseNames{i}, ...
                                                 'x', transformed_x, ...
                                                 'y', transformed_y, ...
                                                 'semiMajorAxis', semiMajorAxes(i), ...
                                                 'semiMinorAxis', semiMinorAxes(i), ...
                                                 'rotationAngle', rotationAngles(i));
        else
            error('preview_concentration_overlays:no_matching_polygon', ...
                ['No polygon found for ellipse %s/%s at concentration %d (ellipse %d/%d).' ...
                 '\nVerify coordinates.txt files for stages 2-4 before retrying.'], ...
                phoneNames{i}, imageNames{i}, concentrations(i), i, numEllipses);
        end
    end

    transformedEllipses = transformedEllipses(1:outputIdx);
end

function imgPath = find_image_file(rectPhoneDir, baseName, supportedExts)
    % Search for image file matching baseName with supported extension
    persistent dirCache validExtSet;
    if isempty(dirCache)
        dirCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end
    if isempty(validExtSet)
        validExtSet = containers.Map();
        for i = 1:length(supportedExts)
            validExtSet(lower(supportedExts{i})) = true;
        end
    end

    imgPath = '';
    if ~isfolder(rectPhoneDir), return; end

    % Try direct matches first (fastest path)
    for i = 1:length(supportedExts)
        candidate = fullfile(rectPhoneDir, [baseName supportedExts{i}]);
        if isfile(candidate)
            imgPath = candidate;
            return;
        end
    end

    % Fallback: case-insensitive search using cached directory listings
    if ~isKey(dirCache, rectPhoneDir)
        % Cache directory contents with optimized parsing
        dirInfo = dir(rectPhoneDir);
        files = dirInfo(~[dirInfo.isdir]);

        if ~isempty(files)
            fileNames = {files.name};
            % Vectorized fileparts decomposition
            numFiles = length(fileNames);
            bases = cell(numFiles, 1);
            fileExts = cell(numFiles, 1);
            for i = 1:numFiles
                [~, bases{i}, fileExts{i}] = fileparts(fileNames{i});
            end
            dirCache(rectPhoneDir) = struct('fileNames', {fileNames}, 'bases', {bases}, 'exts', {fileExts});
        else
            dirCache(rectPhoneDir) = struct('fileNames', {{}}, 'bases', {{}}, 'exts', {{}});
        end
    end

    cached = dirCache(rectPhoneDir);
    if isempty(cached.bases), return; end

    % Find matching base names (case-insensitive)
    baseMatches = strcmpi(cached.bases, baseName);

    % Find matching extensions using pre-computed set (faster than cellfun)
    extMatches = false(size(cached.exts));
    for i = 1:length(cached.exts)
        extMatches(i) = isKey(validExtSet, lower(cached.exts{i}));
    end

    % Find files that match both criteria
    validFiles = baseMatches & extMatches;

    if any(validFiles)
        matchIdx = find(validFiles, 1); % Get first match
        imgPath = fullfile(rectPhoneDir, cached.fileNames{matchIdx});
    end
end

function draw_overlays(ax, entry, ellipseRenderPoints, overlayColor)
    % Draw coordinate overlays on preview axes
    % Renders polygons and ellipses with optimized persistent trigonometric caching
    persistent theta cosTheta sinTheta lastRenderPoints

    % Pre-compute parametric angles (persistent across redraws for performance)
    % Avoids recomputing cos/sin for every ellipse in every frame
    % Only recalculates if ellipseRenderPoints changes
    if isempty(lastRenderPoints) || lastRenderPoints ~= ellipseRenderPoints
        theta = linspace(0, 2*pi, ellipseRenderPoints);
        cosTheta = cos(theta);
        sinTheta = sin(theta);
        lastRenderPoints = ellipseRenderPoints;
    end

    % Draw rectangular crop outline if available
    if isfield(entry, 'rectPolygon') && ~isempty(entry.rectPolygon)
        RP = entry.rectPolygon;
        plot(ax, [RP(:,1); RP(1,1)], [RP(:,2); RP(1,2)], '--', 'Color', overlayColor, 'LineWidth', 0.5);
    end

    % Draw polygons with vectorized operations where possible
    numPolygons = numel(entry.polygons);
    if numPolygons > 0
        for i = 1:numPolygons
            P = entry.polygons{i}; % 4x2
            plot(ax, [P(:,1); P(1,1)], [P(:,2); P(1,2)], '-', 'Color', overlayColor, 'LineWidth', 0.5);
        end
    end

    % Draw ellipses if available using parametric equations with rotation
    if isfield(entry, 'ellipses') && ~isempty(entry.ellipses)
        numEllipses = numel(entry.ellipses);
        for i = 1:numEllipses
            ellipse = entry.ellipses{i}; % [x, y, semiMajorAxis, semiMinorAxis, rotationAngle]
            xc = ellipse(1); yc = ellipse(2);
            a = ellipse(3); b = ellipse(4);
            theta_deg = ellipse(5);

            % Convert rotation angle to radians
            theta_rad = deg2rad(theta_deg);

            % Parametric ellipse equations with rotation
            % x(t) = xc + a*cos(t)*cos(θ) - b*sin(t)*sin(θ)
            % y(t) = yc + a*cos(t)*sin(θ) + b*sin(t)*cos(θ)
            ellipseX = xc + a * cosTheta * cos(theta_rad) - b * sinTheta * sin(theta_rad);
            ellipseY = yc + a * cosTheta * sin(theta_rad) + b * sinTheta * cos(theta_rad);
            plot(ax, ellipseX, ellipseY, '-', 'Color', overlayColor, 'LineWidth', 0.5);
        end
    end
end

function validate_folder_exists(pathStr, errId, msg, showPath)
    % Throws error with the given ID if folder does not exist
    if ~isfolder(pathStr)
        error(errId, msg, showPath);
    end
end

function name = compute_display_name(root, pathStr)
    % Compute a relative display name from root to pathStr (file or folder)
    r = normalize_sep(root);
    p = normalize_sep(pathStr);
    if startsWith(lower(p), [lower(r) '/'])
        name = p(numel(r)+2:end);
    else
        name = pathStr; % fallback to original
    end
    name = strrep(name, '/', filesep);
end

function I = imread_raw(fname)
    % Read image with orientation normalized to upright display

    % Read (some builds honor AutoOrient=false; some ignore it silently)
    try
        I = imread(fname, 'AutoOrient', false);
    catch
        I = imread(fname);
    end

    % Get EXIF orientation (if present)
    try
        info = imfinfo(fname);
        if ~isfield(info, 'Orientation') || isempty(info.Orientation), return; end
        ori = double(info.Orientation);
    catch
        return; % no EXIF -> done
    end

    % Always invert only the 90 deg EXIF cases
    switch ori
        case 5  % mirror H + rotate -90 (to display upright)
            I = rot90(I, +1); I = fliplr(I);   % invert: +90 then mirror H
        case 6  % rotate +90
            I = rot90(I, -1);                  % invert: -90 (== rot90(...,3))
        case 7  % mirror H + rotate +90
            I = rot90(I, -1); I = fliplr(I);   % invert: -90 then mirror H
        case 8  % rotate +270 (== -90)
            I = rot90(I, +1);                  % invert: +90
        otherwise
            % 1,2,3,4 -> leave unchanged (no risk of double-undo)
    end
end

function concValues = extract_concentration_column(T)
    % Vectorized extraction and conversion of concentration column
    % Returns numeric array with NaN for missing/invalid values

    if ~ismember('concentration', T.Properties.VariableNames)
        concValues = nan(height(T), 1);
        return;
    end

    rawConc = T.concentration;

    % Handle different data types efficiently
    if isnumeric(rawConc)
        concValues = double(rawConc);
    elseif iscell(rawConc)
        % Vectorized cell array processing
        isEmpty = cellfun(@isempty, rawConc);
        concValues = nan(length(rawConc), 1);
        concValues(~isEmpty) = cellfun(@(x) convert_to_numeric(x), rawConc(~isEmpty));
    elseif isa(rawConc, 'categorical') || isstring(rawConc) || ischar(rawConc)
        % Convert categorical/string to numeric
        concValues = str2double(string(rawConc));
    else
        concValues = nan(height(T), 1);
    end

    % Ensure scalar values and convert to double
    nonScalar = arrayfun(@(x) ~isscalar(x), concValues);
    concValues(nonScalar) = NaN;
end

function val = convert_to_numeric(x)
    % Helper to convert single value to numeric
    if isnumeric(x)
        val = double(x);
    elseif isstring(x) || ischar(x)
        val = str2double(x);
    elseif isa(x, 'categorical')
        val = str2double(string(x));
    else
        val = NaN;
    end
end

function key = format_concentration_key(val)
    % Format concentration values consistently when composing lookup keys
    if isnan(val)
        key = 'NaN';
        return;
    end
    key = char(string(val));
end

function clear_preview_caches()
    % Clear persistent caches used by helper functions

    % Clear find_image_file cache
    clear find_image_file;

    % Clear draw_overlays cache
    clear draw_overlays;

    % Clear normalize_sep cache
    clear normalize_sep;
end
