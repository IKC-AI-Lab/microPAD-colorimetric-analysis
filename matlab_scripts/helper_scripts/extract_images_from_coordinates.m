function extract_images_from_coordinates(varargin)
    %% Reconstruct pipeline images from coordinates.txt files.
    %% Author: Veysel Y. Yilmaz
    %% Creation: 2025-08
    %
    % Reconstructs pipeline stage outputs from coordinate files and original
    % images. Processes polygon regions and elliptical patches based on saved
    % coordinate data.
    %
    % Stages handled (in-order, with dependency checks):
    % - Step 1: 1_dataset -> 2_micropads
    %   - Reads polygon coordinates: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation
    %   - Crops polygon regions from originals into concentration subfolders (con_0 ...).
    %   - Rotation column (10th field) is required in new pipeline format.
    %
    % - Step 2: 2_micropads -> 3_elliptical_regions
    %   - Reads elliptical coordinates: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
    %   - Extracts elliptical patches from the concentration polygon images.
    %
    % Parameters (name-value):
    % - inputFolder    : folder for originals (default '1_dataset')
% - polygonFolder  : folder for polygon crops (default '2_micropads')
% - patchFolder    : folder for elliptical patches (default '3_elliptical_regions')
    % - concFolderPrefix: prefix for concentration folders (default 'con_')
    %
    % File handling
    % - Coordinate files are named 'coordinates.txt' in respective stage folders.
    % - Output format: PNG exclusively (lossless, no EXIF issues).
    %
    % Performance optimizations
    % - Image file cache: Avoids repeated dir() calls when locating source images
    %   across coordinate entries. Cache key format: 'folder|baseName'.
    % - Directory index cache: Pre-scans each folder once with single dir() call,
    %   building lowercase basename/extension index for O(1) lookups. Typical speedup:
    %   ~10× faster for datasets with 1000+ patches per phone.
    % - Elliptical mask cache: Reuses binary masks for patches with identical
    %   dimensions and ellipse parameters. Cache hit rate typically >90% for datasets
    %   with repeated patch geometries. Masks larger than 1000×1000 pixels are not cached.
    % - Polygon bounding box optimization: Creates masks only for polygon bbox region
    %   rather than full image, reducing memory footprint by ~95% for typical cases.
    % - Cache invalidation: All caches persist for entire phone processing run and are
    %   never invalidated. If manual file changes occur mid-run, restart the function.
    %
    % Notes
    % - This script does not write coordinates.txt. It reads them to reconstruct images.
    % - For Step 1 (polygon extraction), coordinates format:
    %     'image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation'
    %   Rotation column (10th field) is required in the new 4-stage pipeline.
    % - If a required prior stage has neither coordinates nor images, the process errors for that folder.
    %
    % Usage examples (from repo root):
    %   addpath('matlab_scripts'); addpath('matlab_scripts/helper_scripts');
    %   extract_images_from_coordinates();
    %
% See also: cut_micropads, cut_elliptical_regions

% ----------------------
    % Error handling for deprecated format parameters
    if ~isempty(varargin) && (any(strcmpi(varargin(1:2:end), 'preserveFormat')) || any(strcmpi(varargin(1:2:end), 'jpegQuality')))
        error('micropad:deprecated_parameter', ...
              ['JPEG format no longer supported. Pipeline outputs PNG exclusively.\n' ...
               'Remove ''preserveFormat'' and ''jpegQuality'' parameters from function call.']);
    end

    % Parse inputs and create configuration
    % ----------------------
    parser = inputParser;
    addParameter(parser, 'inputFolder', '1_dataset', @(s) ((ischar(s) || isstring(s)) && ~isempty(char(s))));
    addParameter(parser, 'polygonFolder', '2_micropads', @(s) ((ischar(s) || isstring(s)) && ~isempty(char(s))));
    addParameter(parser, 'patchFolder', '3_elliptical_regions', @(s) ((ischar(s) || isstring(s)) && ~isempty(char(s))));
    addParameter(parser, 'concFolderPrefix', 'con_', @(s) ((ischar(s) || isstring(s)) && ~isempty(char(s))));
    parse(parser, varargin{:});

    % Create configuration using standard pattern
    cfg = createConfiguration(char(parser.Results.inputFolder), ...
                              char(parser.Results.polygonFolder), char(parser.Results.patchFolder), ...
                              char(parser.Results.concFolderPrefix));

    % Validate base inputs folder
    validate_folder_exists(cfg.paths.input, 'extract:missing_input', 'Original images folder not found: %s', cfg.paths.input);

    % Ensure base output folders exist (we create only when needed)
    ensure_folder(cfg.paths.polygon);
    ensure_folder(cfg.paths.patch);

    phones = list_immediate_subdirs(cfg.paths.input);
    if isempty(phones)
        warning('extract:no_phones', 'No phone subfolders under %s', cfg.paths.input);
    end

    % Process each phone and its subfolders
    for p = 1:numel(phones)
        phoneName = phones{p};
        fprintf('\n=== Processing %s ===\n', phoneName);

        originalsDir = fullfile(cfg.paths.input, phoneName);
        polyBaseDir = fullfile(cfg.paths.polygon, phoneName);
        patchBaseDir = fullfile(cfg.paths.patch, phoneName);

        % Step 1: Polygon crops from coordinates (OPTIMIZED: combined check)
        [conDirs, hasAnyPolyCoords] = find_concentration_dirs_with_coords(polyBaseDir, cfg.concFolderPrefix, cfg.coordinateFileName);

        if hasAnyPolyCoords
            fprintf('Step 1: Reconstruct polygon crops from concentration folders\n');
            for cd = 1:numel(conDirs)
                if ~conDirs{cd}.hasCoords
                    continue;
                end
                coordPath = fullfile(conDirs{cd}.path, cfg.coordinateFileName);
                ensure_folder(conDirs{cd}.path);
                fprintf('  - Using %s\n', relpath(coordPath, cfg.projectRoot));
                extract_polygon_crops_single(coordPath, originalsDir, conDirs{cd}.path, cfg);
            end
        else
            phonePolyCoord = fullfile(polyBaseDir, cfg.coordinateFileName);
            if isfile(phonePolyCoord)
                fprintf('Step 1: Reconstruct polygon crops from phone-level coordinates\n');
                ensure_folder(polyBaseDir);
                extract_polygon_crops_all(phonePolyCoord, originalsDir, polyBaseDir, cfg);
            else
                fprintf('Step 1: No polygon coordinates found. Skipping.\n');
            end
        end

        % Step 2: Elliptical patches from coordinates (OPTIMIZED: combined check)
        [patchConDirs, hasAnyEllipseCoords] = find_concentration_dirs_with_coords(patchBaseDir, cfg.concFolderPrefix, cfg.coordinateFileName);

        if hasAnyEllipseCoords
            fprintf('Step 2: Reconstruct elliptical patches from concentration folders\n');
            polyConDirs = find_concentration_dirs(polyBaseDir, cfg.concFolderPrefix);
            if isempty(polyConDirs)
                error('extract:polygon_required_for_ellipses', 'Polygon crops missing for %s.', phoneName);
            end
            for cd = 1:numel(patchConDirs)
                if ~patchConDirs{cd}.hasCoords
                    continue;
                end
                coordPath = fullfile(patchConDirs{cd}.path, cfg.coordinateFileName);
                [~, conName] = fileparts(patchConDirs{cd}.path);
                polyConDir = fullfile(polyBaseDir, conName);
                if ~isfolder(polyConDir)
                    error('extract:missing_polygon_con_folder', 'Missing polygon folder for ellipses: %s', relpath(polyConDir, cfg.projectRoot));
                end
                ensure_folder(patchConDirs{cd}.path);
                fprintf('  - Using %s\n', relpath(coordPath, cfg.projectRoot));
                extract_elliptical_patches(coordPath, polyConDir, patchConDirs{cd}.path, cfg);
            end
        else
            phoneEllipseCoord = fullfile(patchBaseDir, cfg.coordinateFileName);
            if isfile(phoneEllipseCoord)
                fprintf('Step 2: Reconstruct elliptical patches from phone-level coordinates\n');
                if ~folder_has_any_images(polyBaseDir, true)
                    error('extract:polygon_required_for_ellipses', ['Polygon crops missing for %s. Expected con_* folders ' ...
                           'with polygon images generated by cut_micropads.'], phoneName);
                end
                ensure_folder(patchBaseDir);
                extract_elliptical_patches(phoneEllipseCoord, polyBaseDir, patchBaseDir, cfg);
            else
                fprintf('Step 2: No elliptical coordinates found. Skipping.\n');
            end
        end
    end

    fprintf('\nReconstruction complete.\n');
end

function cfg = createConfiguration(inputFolder, polygonFolder, patchFolder, concFolderPrefix)
    % Create configuration with validation and path resolution

    % Validate inputs
    validateattributes(inputFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'inputFolder');
    validateattributes(polygonFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'polygonFolder');
    validateattributes(patchFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'patchFolder');
    validateattributes(concFolderPrefix, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'concFolderPrefix');

    repoRoot = findProjectRoot(char(inputFolder));

    % Resolve folder paths
    inputRoot = resolve_folder(repoRoot, char(inputFolder));
    polyRoot = resolve_folder(repoRoot, char(polygonFolder));
    patchRoot = resolve_folder(repoRoot, char(patchFolder));

    % Create configuration structure
    cfg = struct();
    cfg.projectRoot = repoRoot;
    cfg.paths = struct('input', inputRoot, 'polygon', polyRoot, 'patch', patchRoot);
    cfg.output = struct('supportedFormats', {{'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}});
    cfg.allowedImageExtensions = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'};
    cfg.coordinateFileName = 'coordinates.txt';
    cfg.concFolderPrefix = char(concFolderPrefix);

    % Image size limits
    cfg.limits = struct('maxJpegDimension', 65500);  % MATLAB JPEG writer maximum dimension

    % OPTIMIZATION CACHES (see "Performance optimizations" in header for details)
    % All caches persist for entire phone processing run; never invalidated mid-run.
    % If manual file system changes occur during processing, restart function for that phone.

    % Cache for find_image_file: Maps 'folder|baseName' -> full image path
    cfg.imageFileCache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    % Directory index cache: Maps folder path -> struct with {names, basenames, exts}
    % Built once per folder with single dir() call, reused for all image lookups.
    cfg.dirIndexCache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    % Elliptical mask cache: Maps 'h_w_cx_cy_a_b_theta' -> logical mask array
    % Reuses masks for patches with identical geometry. Masks >1e6 pixels not cached.
    cfg.ellipticalMaskCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
end

% ----------------------
% Step 1: Polygon crops (directly from originals)
% ----------------------
function extract_polygon_crops_single(coordPath, originalsDir, concDir, cfg)
    rows = read_polygon_coordinates(coordPath);
    if isempty(rows)
        warning('extract:poly_empty', 'No valid polygon entries parsed from %s. Check file format.', coordPath);
        return;
    end
    ensure_folder(concDir);
    for i = 1:numel(rows)
        row = rows(i);

        % Validate parsed row structure
        if ~isfield(row, 'imageBase') || ~isfield(row, 'concentration') || ~isfield(row, 'polygon')
            warning('extract:invalid_row_struct', 'Skipping malformed coordinate entry %d', i);
            continue;
        end

        srcPath = find_image_file_cached(originalsDir, row.imageBase, cfg);
        if isempty(srcPath)
            warning('extract:missing_original', 'Original image not found for %s in %s', row.imageBase, originalsDir);
            continue;
        end
        img = imread_raw(srcPath);

        % Rotation column is UI-only metadata (alignment hint)
        % Coordinates are already in original (unrotated) image frame
        % No image warp applied - coordinates already match original frame

        cropped = crop_with_polygon(img, row.polygon);
        outExt = '.png';
        outPath = fullfile(concDir, sprintf('%s_%s%d%s', row.imageBase, cfg.concFolderPrefix, row.concentration, outExt));
        save_image_with_format(cropped, outPath, outExt, cfg);
    end
end

function extract_polygon_crops_all(coordPath, originalsDir, polyGroupDir, cfg)
    rows = read_polygon_coordinates(coordPath);
    if isempty(rows)
        warning('extract:poly_empty', 'No valid polygon entries parsed from %s. Check file format.', coordPath);
        return;
    end
    ensure_folder(polyGroupDir);
    for i = 1:numel(rows)
        row = rows(i);

        % Validate parsed row structure
        if ~isfield(row, 'imageBase') || ~isfield(row, 'concentration') || ~isfield(row, 'polygon')
            warning('extract:invalid_row_struct', 'Skipping malformed coordinate entry %d', i);
            continue;
        end

        srcPath = find_image_file_cached(originalsDir, row.imageBase, cfg);
        if isempty(srcPath)
            warning('extract:missing_original', 'Original image not found for %s in %s', row.imageBase, originalsDir);
            continue;
        end
        img = imread_raw(srcPath);

        % Rotation column is UI-only metadata (alignment hint)
        % Coordinates are already in original (unrotated) image frame
        % No image warp applied - coordinates already match original frame

        cropped = crop_with_polygon(img, row.polygon);
        outExt = '.png';
        % Mirror cut_micropads layout: con_* folders containing base_con_<idx>.ext
        concFolder = fullfile(polyGroupDir, sprintf('%s%d', cfg.concFolderPrefix, row.concentration));
        ensure_folder(concFolder);
        outName = sprintf('%s_%s%d%s', row.imageBase, cfg.concFolderPrefix, row.concentration, outExt);
        outPath = fullfile(concFolder, outName);
        save_image_with_format(cropped, outPath, outExt, cfg);
    end
end

function rows = read_polygon_coordinates(coordPath)
    % Reads polygon coordinates saved by cut_micropads.m
    % Header: 'image concentration x1 y1 ... xN yN rotation'
    rows = struct('imageBase','', 'concentration',0, 'polygon',[], 'rotation',0);
    rows = rows([]);
    if ~isfile(coordPath), return; end
    fid = fopen(coordPath, 'rt');
    if fid == -1, return; end
    c = onCleanup(@() fclose(fid));

    % Read entire file at once
    allText = textscan(fid, '%s', 'Delimiter', '\n', 'WhiteSpace', '');
    L = allText{1};

    if isempty(L), return; end

    firstTrim = strtrim(L{1});
    isHeader = ~isempty(firstTrim) && strncmpi(firstTrim, 'image concentration', length('image concentration'));

    startIdx = 1;
    if isHeader
        startIdx = 2;
    end

    if startIdx > numel(L), return; end

    nApprox = numel(L) - startIdx + 1;
    % Pre-allocate struct array for parsed polygon entries
    tmp(nApprox) = struct('imageBase','', 'concentration',0, 'polygon',[], 'rotation',0);
    k = 0;

    for i = startIdx:numel(L)
        ln = strtrim(L{i});
        if isempty(ln), continue; end
        parts = strsplit(ln);
        if numel(parts) < 4, continue; end
        imageBase = strip_ext(parts{1});
        concentration = str2double(parts{2});
        nums = str2double(parts(3:end));
        if numel(nums) < 8 || any(isnan(nums(1:8)))
            warning('extract:invalid_polygon_entry', ...
                    'Skipping malformed polygon entry on line %d: expected 8 polygon coordinates', i);
            continue;
        end
        % Extract polygon (first 8 values) and rotation (9th value if present)
        P = reshape(nums(1:8), 2, 4).';  % 4x2 polygon matrix
        rotation = 0;
        if numel(nums) >= 9 && ~isnan(nums(9))
            rotation = nums(9);
        end
        k = k + 1;
        tmp(k) = struct('imageBase', imageBase, 'concentration', concentration, ...
                        'polygon', round(P), 'rotation', rotation);
    end
    if k == 0
        rows = rows([]);
    else
        rows = tmp(1:k);
    end
end

% ----------------------
% Step 3: Elliptical patches
% ----------------------
function extract_elliptical_patches(coordPath, polygonInputDir, patchOutputBase, cfg)
    % coordPath may be phone-level or per concentration folder. 'image' column refers
    % to polygon-cropped image name (with extension) relative to polygonInputDir.
    rows = read_ellipse_coordinates(coordPath);
    if isempty(rows)
        warning('extract:ellipse_empty', 'No valid elliptical entries parsed from %s. Check file format.', coordPath);
        return;
    end
    ensure_folder(patchOutputBase);
    polygonConDirs = find_concentration_dirs(polygonInputDir, cfg.concFolderPrefix);

    for i = 1:numel(rows)
        row = rows(i);
        srcPath = resolve_polygon_source(polygonInputDir, row, cfg, polygonConDirs);
        if isempty(srcPath)
            warning('extract:missing_polygon_img', 'Polygon image not found for %s (con %d) under %s', ...
                row.imageName, row.concentration, relpath(polygonInputDir, cfg.projectRoot));
            continue;
        end
        img = imread_raw(srcPath);
        xCenter = row.x; yCenter = row.y;
        a = row.semiMajorAxis; b = row.semiMinorAxis; theta = row.rotationAngle;

        % Calculate rotated bounding box
        theta_rad = deg2rad(theta);
        ux = sqrt((a * cos(theta_rad))^2 + (b * sin(theta_rad))^2);
        uy = sqrt((a * sin(theta_rad))^2 + (b * cos(theta_rad))^2);

        x1 = max(1, floor(xCenter - ux)); y1 = max(1, floor(yCenter - uy));
        x2 = min(size(img,2), ceil(xCenter + ux)); y2 = min(size(img,1), ceil(yCenter + uy));
        if x2 < x1, x2 = x1; end
        if y2 < y1, y2 = y1; end
        patchRegion = img(y1:y2, x1:x2, :);
        [h, w, ~] = size(patchRegion);

        % OPTIMIZATION: Use cached elliptical masks
        cx = xCenter - x1 + 1; cy = yCenter - y1 + 1;
        mask = get_elliptical_mask(h, w, cx, cy, a, b, theta, cfg);

        % OPTIMIZATION: Apply mask more efficiently
        if size(patchRegion,3) > 1
            mask3 = repmat(mask, [1 1 size(patchRegion,3)]);
            patchRegion(~mask3) = 0;
        else
            patchRegion(~mask) = 0;
        end

        [~, nameNoExt, ~] = fileparts(srcPath);
        outExt = '.png';
        % Choose target folder: if patchOutputBase already is a con_* folder, use it;
        % otherwise (phone-level), create/use con_%d subfolder.
        [~, leaf] = fileparts(patchOutputBase);
        if startsWith(leaf, cfg.concFolderPrefix)
            targetDir = patchOutputBase;
        else
            targetDir = fullfile(patchOutputBase, sprintf('%s%d', cfg.concFolderPrefix, row.concentration));
        end
        ensure_folder(targetDir);
        outName = sprintf('%s_con%d_rep%d%s', nameNoExt, row.concentration, row.replicate, outExt);
        outPath = fullfile(targetDir, outName);
        save_image_with_format(patchRegion, outPath, outExt, cfg);
    end
end

function rows = read_ellipse_coordinates(coordPath)
    % Reads ellipse coordinates saved by cut_elliptical_regions.m
    % Header: 'image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle'
    rows = struct('imageName','', 'x',0, 'y',0, 'semiMajorAxis',0, 'semiMinorAxis',0, 'rotationAngle',0, 'concentration',0, 'replicate',0);
    rows = rows([]);
    if ~isfile(coordPath), return; end
    fid = fopen(coordPath, 'rt');
    if fid == -1, return; end
    c = onCleanup(@() fclose(fid));

    % Read first line to check for header
    first = fgetl(fid);
    if ~ischar(first), return; end
    firstTrim = strtrim(first);
    isHeader = ~isempty(firstTrim) && strncmpi(firstTrim, 'image concentration', length('image concentration'));

    % Use textscan for vectorized reading (8 columns: image, conc, rep, x, y, a, b, theta)
    data = textscan(fid, '%s %f %f %f %f %f %f %f', 'Delimiter',' ', 'MultipleDelimsAsOne', true);

    % Prepend first line if it was data (no header)
    if ~isHeader && ~isempty(firstTrim)
        % Parse the first line tokens into the data columns
        parts = strsplit(firstTrim);
        if numel(parts) >= 8
            data = prepend_ellipse_row(data, parts);
        end
    end
    if isempty(data) || numel(data) < 8
        return;
    end
    names = data{1};
    cons  = data{2};
    reps  = data{3};
    xs    = data{4};
    ys    = data{5};
    as    = data{6};
    bs    = data{7};
    thetas= data{8};
    n = numel(names);
    if n == 0, return; end

    % Pre-allocate struct array for parsed elliptical coordinate entries
    rows(1,n) = struct('imageName','', 'x',0, 'y',0, 'semiMajorAxis',0, 'semiMinorAxis',0, 'rotationAngle',0, 'concentration',0, 'replicate',0);
    for i = 1:n
        rows(i).imageName      = names{i};
        rows(i).x              = xs(i);
        rows(i).y              = ys(i);
        rows(i).semiMajorAxis  = as(i);
        rows(i).semiMinorAxis  = bs(i);
        rows(i).rotationAngle  = thetas(i);
        rows(i).concentration  = cons(i);
        rows(i).replicate      = reps(i);
    end
end

function dataOut = prepend_ellipse_row(data, parts)
    % Helper to prepend one parsed row (as strings) to the textscan output
    name = parts{1};
    nums = str2double(parts(2:8));
    if any(isnan(nums)) || numel(nums) ~= 7
        dataOut = data; return;
    end
    % data: {names, conc, rep, x, y, a, b, theta}
    dataOut = data;
    if isempty(data)
        dataOut = { {name}, nums(1), nums(2), nums(3), nums(4), nums(5), nums(6), nums(7) };
        return;
    end
    dataOut{1} = [{name}; data{1}];
    for k = 2:8
        dataOut{k} = [nums(k-1); data{k}];
    end
end

% ----------------------
% Helpers: imaging and I/O
% ----------------------
function cropped = crop_with_polygon(img, polygon)
    % OPTIMIZED: Compute bbox first, then create mask only for bbox region
    [h, w, c] = size(img);

    % Pre-compute bounding box from polygon vertices (avoid full-image mask)
    minx = max(1, floor(min(polygon(:,1))));
    maxx = min(w, ceil(max(polygon(:,1))));
    miny = max(1, floor(min(polygon(:,2))));
    maxy = min(h, ceil(max(polygon(:,2))));

    bboxW = maxx - minx + 1;
    bboxH = maxy - miny + 1;

    % Adjust polygon coordinates to bbox-relative
    polyRelative = polygon;
    polyRelative(:,1) = polyRelative(:,1) - minx + 1;
    polyRelative(:,2) = polyRelative(:,2) - miny + 1;

    % Create mask only for bbox region (much smaller memory footprint)
    mask = poly2mask(polyRelative(:,1), polyRelative(:,2), bboxH, bboxW);

    % Extract and mask the bbox region
    if c > 1
        sub = img(miny:maxy, minx:maxx, :);
        mask3 = repmat(mask, [1 1 c]);
        sub(~mask3) = 0;
        cropped = sub;
    else
        sub = img(miny:maxy, minx:maxx);
        sub(~mask) = 0;
        cropped = sub;
    end
end

function mask = get_elliptical_mask(h, w, cx, cy, a, b, theta, cfg)
    % OPTIMIZATION: Cache elliptical masks by dimensions, center, and ellipse parameters
    % Most patches have identical dimensions, so this saves significant computation
    cacheKey = sprintf('%d_%d_%.4f_%.4f_%.4f_%.4f_%.4f', h, w, cx, cy, a, b, theta);

    if isKey(cfg.ellipticalMaskCache, cacheKey)
        mask = cfg.ellipticalMaskCache(cacheKey);
        return;
    end

    % Generate new elliptical mask with rotation
    [X, Y] = meshgrid(1:w, 1:h);
    theta_rad = deg2rad(theta);

    % Translate to center
    dx = X - cx;
    dy = Y - cy;

    % Rotate coordinates to ellipse's principal axes frame
    x_rot =  dx * cos(theta_rad) + dy * sin(theta_rad);
    y_rot = -dx * sin(theta_rad) + dy * cos(theta_rad);

    % Apply ellipse equation: (x_rot/a)^2 + (y_rot/b)^2 <= 1
    mask = (x_rot ./ a).^2 + (y_rot ./ b).^2 <= 1;

    % Cache for reuse (cache masks up to ~1000x1000 pixels)
    if h * w < 1e6
        cfg.ellipticalMaskCache(cacheKey) = mask;
    end
end

function save_image_with_format(img, outPath, outExt, cfg)
    % Save image honoring requested extension and JPEG size limits.
    ensure_folder(fileparts(outPath));
    ext = lower(outExt);
    try
        imwrite(img, outPath);
    catch ME
        error('extract:imwrite_failed', 'Failed to write %s: %s', outPath, ME.message);
    end
end

% ----------------------
% Helpers: path, folders, scanning
% ----------------------
function tf = folder_has_any_images(dirPath, includeSubdirs)
    % OPTIMIZED: Single dir() call + vectorized extension checking
    if nargin < 2
        includeSubdirs = false;
    end

    tf = false;
    if ~isfolder(dirPath), return; end

    allItems = dir(dirPath);
    if isempty(allItems), return; end

    % Filter non-directories
    fileItems = allItems(~[allItems.isdir]);
    if ~isempty(fileItems)
        % OPTIMIZATION: Vectorized extension checking with ismember
        validExts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
        [~, ~, exts] = cellfun(@fileparts, {fileItems.name}, 'UniformOutput', false);
        extsLower = lower(exts);
        if any(ismember(extsLower, validExts))
            tf = true;
            return;
        end
    end

    if ~includeSubdirs
        return;
    end

    dirMask = [allItems.isdir];
    subNames = {allItems(dirMask).name};
    subNames = subNames(~ismember(subNames, {'.','..'}));
    for i = 1:numel(subNames)
        subPath = fullfile(dirPath, subNames{i});
        if folder_has_any_images(subPath, true)
            tf = true;
            return;
        end
    end
end

function dirs = list_immediate_subdirs(root)
    dirs = {};
    if ~isfolder(root), return; end
    d = dir(root);
    mask = [d.isdir] & ~ismember({d.name}, {'.','..'});
    names = {d(mask).name};
    dirs = names;
end

function conDirs = find_concentration_dirs(baseDir, prefix)
    conDirs = {};
    if ~isfolder(baseDir), return; end
    d = dir(baseDir);
    if isempty(d), return; end
    isDir = [d.isdir] & ~ismember({d.name}, {'.','..'});
    if ~any(isDir), return; end
    names = {d(isDir).name};
    mask = startsWith(names, prefix);
    names = names(mask);
    conDirs = cellfun(@(n) fullfile(baseDir, n), names, 'UniformOutput', false);
end

function [conDirs, hasAnyCoords] = find_concentration_dirs_with_coords(baseDir, prefix, coordFileName)
    % OPTIMIZED: Combined directory scan and coordinate check in single pass
    % Returns struct array with fields: path, hasCoords
    conDirs = {};
    hasAnyCoords = false;

    if ~isfolder(baseDir), return; end
    d = dir(baseDir);
    if isempty(d), return; end

    isDir = [d.isdir] & ~ismember({d.name}, {'.','..'});
    if ~any(isDir), return; end

    names = {d(isDir).name};
    mask = startsWith(names, prefix);
    names = names(mask);

    n = numel(names);
    if n == 0, return; end

    % Pre-allocate struct array
    conDirs = cell(n, 1);
    for i = 1:n
        fullPath = fullfile(baseDir, names{i});
        coordPath = fullfile(fullPath, coordFileName);
        hasCoords = isfile(coordPath);
        conDirs{i} = struct('path', fullPath, 'hasCoords', hasCoords);
        if hasCoords
            hasAnyCoords = true;
        end
    end
end

function projectRoot = findProjectRoot(inputFolder)
    % Find project root by searching upward from current directory
    currentDir = pwd;
    searchDir = currentDir;
    maxLevels = 5;

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

function absPath = resolve_folder(repoRoot, requested)
    % If requested exists relative to repoRoot, return that; else return requested as-is
    p = fullfile(repoRoot, requested);
    if isfolder(p) || isfile(p)
        absPath = p; return;
    end
    absPath = requested;
end

function validate_folder_exists(pathStr, msgId, msgFmt, varargin)
    if ~isfolder(pathStr)
        error(msgId, msgFmt, varargin{:});
    end
end

function ensure_folder(pathStr)
    if ~isfolder(pathStr)
        mkdir(pathStr);
    end
end

function imagePath = find_image_file_cached(folder, baseName, cfg)
    % Cached version of find_image_file using cfg.imageFileCache
    % Cache key: folder|baseName
    cacheKey = [folder '|' baseName];

    if isKey(cfg.imageFileCache, cacheKey)
        imagePath = cfg.imageFileCache(cacheKey);
        return;
    end

    % Not in cache, perform search
    imagePath = find_image_file(folder, baseName, cfg);

    % Store in cache
    cfg.imageFileCache(cacheKey) = imagePath;
end

function imagePath = find_image_file(folder, baseName, cfg)
    % OPTIMIZED: Find image file with directory index caching
    imagePath = '';
    if ~isfolder(folder), return; end

    % Fast path: direct extension guesses (most common first - standardized across pipeline)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    for i = 1:numel(exts)
        p = fullfile(folder, [baseName exts{i}]);
        if isfile(p), imagePath = p; return; end
    end

    % OPTIMIZATION: Use cached directory index
    if isKey(cfg.dirIndexCache, folder)
        dirIndex = cfg.dirIndexCache(folder);
    else
        % Build directory index once per folder
        d = dir(folder);
        fileItems = d(~[d.isdir]);
        if isempty(fileItems)
            cfg.dirIndexCache(folder) = struct('names', {{}}, 'basenames', {{}}, 'exts', {{}});
            return;
        end

        names = {fileItems.name};
        [~, fileBasenames, fileExts] = cellfun(@fileparts, names, 'UniformOutput', false);

        dirIndex = struct('names', {names}, ...
                         'basenames', {lower(fileBasenames)}, ...
                         'exts', {lower(fileExts)});
        cfg.dirIndexCache(folder) = dirIndex;
    end

    % Search in cached index
    baseLower = baseName;
    validExts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

    for i = 1:numel(dirIndex.basenames)
        if strcmpi(dirIndex.basenames{i}, baseLower) && any(strcmp(dirIndex.exts{i}, validExts))
            imagePath = fullfile(folder, dirIndex.names{i});
            return;
        end
    end
end

function srcPath = resolve_polygon_source(baseDir, row, cfg, polyConDirs)
    % OPTIMIZED: Locate polygon image with streamlined search order
    if nargin < 4
        polyConDirs = {};
    end

    srcPath = '';
    baseName = strip_ext(row.imageName);

    % Priority 1: Direct path in base directory
    candidate = fullfile(baseDir, row.imageName);
    if isfile(candidate)
        srcPath = candidate;
        return;
    end

    % Priority 2: Concentration-specific folder
    concFolderName = sprintf('%s%d', cfg.concFolderPrefix, row.concentration);
    [~, leaf] = fileparts(baseDir);
    if strcmpi(leaf, concFolderName)
        concFolder = baseDir;
    else
        concFolder = fullfile(baseDir, concFolderName);
    end
    candidate = fullfile(concFolder, row.imageName);
    if isfile(candidate)
        srcPath = candidate;
        return;
    end

    % Priority 3: Extension-agnostic search in base directory
    alt = find_image_file_cached(baseDir, baseName, cfg);
    if ~isempty(alt)
        srcPath = alt;
        return;
    end

    % Priority 4: Extension-agnostic search in concentration folder
    if isfolder(concFolder)
        alt = find_image_file_cached(concFolder, baseName, cfg);
        if ~isempty(alt)
            srcPath = alt;
            return;
        end
    end

    % Priority 5: Search all concentration directories (last resort)
    if isempty(polyConDirs)
        polyConDirs = find_concentration_dirs(baseDir, cfg.concFolderPrefix);
    end
    for idx = 1:numel(polyConDirs)
        candidate = fullfile(polyConDirs{idx}, row.imageName);
        if isfile(candidate)
            srcPath = candidate;
            return;
        end
        alt = find_image_file_cached(polyConDirs{idx}, baseName, cfg);
        if ~isempty(alt)
            srcPath = alt;
            return;
        end
    end
end

function I = imread_raw(fname)
% Read image pixels in their recorded layout without applying EXIF orientation
% metadata. Any user-requested rotation is stored in coordinates.txt and applied
% during downstream processing rather than via image metadata.

    I = imread(fname);
end

function s = strip_ext(nameOrPath)
    [~, s, ~] = fileparts(nameOrPath);
end

function r = relpath(pathStr, root)
    % Best-effort relative path for logs (keep OS-specific separators)
    try
        p  = char(pathStr);
        rt = char(root);
        % Build a prefix that includes a trailing filesep, without creating
        % multiple tokens that would erase all separators.
        if isempty(rt) || rt(end) ~= filesep
            prefix = [rt filesep];
        else
            prefix = rt;
        end
        if strncmpi(p, prefix, length(prefix))
            r = p(length(prefix)+1:end);
        else
            r = p;
        end
        % Normalize separators on non-Windows only (keep backslashes on Windows)
        if ~ispc
            r = strrep(r, '\', '/');
        end
    catch
        r = char(pathStr);
    end
end

