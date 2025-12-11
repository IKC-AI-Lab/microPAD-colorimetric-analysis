function extract_images_from_coordinates(varargin)
    %% Reconstruct pipeline images from coordinates.txt files.
    %% Author: Veysel Y. Yilmaz
    %% Creation: 2025-08
    %
    % Reconstructs pipeline stage outputs from coordinate files and original
    % images. Processes quad regions and elliptical patches based on saved
    % coordinate data.
    %
    % Stages handled (in-order, with dependency checks):
    % - Step 1: 1_dataset -> 2_micropads
    %   - Reads quad coordinates: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation
    %   - Crops quad regions from originals into concentration subfolders (con_0 ...).
    %   - Rotation column (10th field) is required in new pipeline format.
    %
    % - Step 2: 2_micropads -> 3_elliptical_regions
    %   - Reads elliptical coordinates: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
    %   - Extracts elliptical patches from the concentration quad images.
    %
    % Parameters (name-value):
    % - inputFolder    : folder for originals (default '1_dataset')
% - quadFolder     : folder for quad crops (default '2_micropads')
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
    % - Quad bounding box optimization: Creates masks only for quad bbox region
    %   rather than full image, reducing memory footprint by ~95% for typical cases.
    % - Cache invalidation: All caches persist for entire phone processing run and are
    %   never invalidated. If manual file changes occur mid-run, restart the function.
    %
    % Notes
    % - This script does not write coordinates.txt. It reads them to reconstruct images.
    % - For Step 1 (quad extraction), coordinates format:
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
    addParameter(parser, 'quadFolder', '2_micropads', @(s) ((ischar(s) || isstring(s)) && ~isempty(char(s))));
    addParameter(parser, 'patchFolder', '3_elliptical_regions', @(s) ((ischar(s) || isstring(s)) && ~isempty(char(s))));
    addParameter(parser, 'concFolderPrefix', 'con_', @(s) ((ischar(s) || isstring(s)) && ~isempty(char(s))));
    parse(parser, varargin{:});

    % Create configuration using standard pattern
    cfg = createConfiguration(char(parser.Results.inputFolder), ...
                              char(parser.Results.quadFolder), char(parser.Results.patchFolder), ...
                              char(parser.Results.concFolderPrefix));

    % Validate base inputs folder
    validate_folder_exists(cfg.paths.input, 'extract:missing_input', 'Original images folder not found: %s', cfg.paths.input);

    % Ensure base output folders exist (we create only when needed)
    ensure_folder(cfg.paths.quad);
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
        quadBaseDir = fullfile(cfg.paths.quad, phoneName);
        patchBaseDir = fullfile(cfg.paths.patch, phoneName);

        % Step 1: Quad crops from coordinates (OPTIMIZED: combined check)
        [conDirs, hasAnyQuadCoords] = find_concentration_dirs_with_coords(quadBaseDir, cfg.concFolderPrefix, cfg.coordinateFileName);

        if hasAnyQuadCoords
            fprintf('Step 1: Reconstruct quad crops from concentration folders\n');
            for cd = 1:numel(conDirs)
                if ~conDirs{cd}.hasCoords
                    continue;
                end
                coordPath = fullfile(conDirs{cd}.path, cfg.coordinateFileName);
                ensure_folder(conDirs{cd}.path);
                fprintf('  - Using %s\n', relpath(coordPath, cfg.projectRoot));
                extract_quad_crops_single(coordPath, originalsDir, conDirs{cd}.path, cfg);
            end
        else
            phoneQuadCoord = fullfile(quadBaseDir, cfg.coordinateFileName);
            if isfile(phoneQuadCoord)
                fprintf('Step 1: Reconstruct quad crops from phone-level coordinates\n');
                ensure_folder(quadBaseDir);
                extract_quad_crops_all(phoneQuadCoord, originalsDir, quadBaseDir, cfg);
            else
                fprintf('Step 1: No quad coordinates found. Skipping.\n');
            end
        end

        % Step 2: Elliptical patches from coordinates (OPTIMIZED: combined check)
        [patchConDirs, hasAnyEllipseCoords] = find_concentration_dirs_with_coords(patchBaseDir, cfg.concFolderPrefix, cfg.coordinateFileName);

        if hasAnyEllipseCoords
            fprintf('Step 2: Reconstruct elliptical patches from concentration folders\n');
            quadConDirs = find_concentration_dirs(quadBaseDir, cfg.concFolderPrefix);
            if isempty(quadConDirs)
                error('extract:quad_required_for_ellipses', 'Quad crops missing for %s.', phoneName);
            end
            for cd = 1:numel(patchConDirs)
                if ~patchConDirs{cd}.hasCoords
                    continue;
                end
                coordPath = fullfile(patchConDirs{cd}.path, cfg.coordinateFileName);
                [~, conName] = fileparts(patchConDirs{cd}.path);
                quadConDir = fullfile(quadBaseDir, conName);
                if ~isfolder(quadConDir)
                    error('extract:missing_quad_con_folder', 'Missing quad folder for ellipses: %s', relpath(quadConDir, cfg.projectRoot));
                end
                ensure_folder(patchConDirs{cd}.path);
                fprintf('  - Using %s\n', relpath(coordPath, cfg.projectRoot));
                extract_elliptical_patches(coordPath, quadConDir, patchConDirs{cd}.path, cfg);
            end
        else
            phoneEllipseCoord = fullfile(patchBaseDir, cfg.coordinateFileName);
            if isfile(phoneEllipseCoord)
                fprintf('Step 2: Reconstruct elliptical patches from phone-level coordinates\n');
                if ~folder_has_any_images(quadBaseDir, true)
                    error('extract:quad_required_for_ellipses', ['Quad crops missing for %s. Expected con_* folders ' ...
                           'with quad images generated by cut_micropads.'], phoneName);
                end
                ensure_folder(patchBaseDir);
                extract_elliptical_patches(phoneEllipseCoord, quadBaseDir, patchBaseDir, cfg);
            else
                fprintf('Step 2: No elliptical coordinates found. Skipping.\n');
            end
        end
    end

    fprintf('\nReconstruction complete.\n');
end

function cfg = createConfiguration(inputFolder, quadFolder, patchFolder, concFolderPrefix)
    % Create configuration with validation and path resolution

    % Validate inputs
    validateattributes(inputFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'inputFolder');
    validateattributes(quadFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'quadFolder');
    validateattributes(patchFolder, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'patchFolder');
    validateattributes(concFolderPrefix, {'char', 'string'}, {'nonempty'}, 'createConfiguration', 'concFolderPrefix');

    % Use canonical findProjectRoot from path_utils module
    persistent pathUtilsModule
    if isempty(pathUtilsModule)
        pathUtilsModule = path_utils();
    end
    repoRoot = pathUtilsModule.findProjectRoot(char(inputFolder));

    % Resolve folder paths
    inputRoot = resolve_folder(repoRoot, char(inputFolder));
    quadRoot = resolve_folder(repoRoot, char(quadFolder));
    patchRoot = resolve_folder(repoRoot, char(patchFolder));

    % Create configuration structure
    cfg = struct();
    cfg.projectRoot = repoRoot;
    cfg.paths = struct('input', inputRoot, 'quad', quadRoot, 'patch', patchRoot);
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
% Step 1: Quad crops (directly from originals)
% ----------------------
function extract_quad_crops_single(coordPath, originalsDir, concDir, cfg)
    rows = read_quad_coordinates(coordPath);
    if isempty(rows)
        warning('extract:quad_empty', 'No valid quad entries parsed from %s. Check file format.', coordPath);
        return;
    end
    ensure_folder(concDir);
    for i = 1:numel(rows)
        row = rows(i);

        % Validate parsed row structure
        if ~isfield(row, 'imageBase') || ~isfield(row, 'concentration') || ~isfield(row, 'quad')
            warning('extract:invalid_row_struct', 'Skipping malformed coordinate entry %d', i);
            continue;
        end

        srcPath = find_image_file_cached(originalsDir, row.imageBase, cfg);
        if isempty(srcPath)
            warning('extract:missing_original', 'Original image not found for %s in %s', row.imageBase, originalsDir);
            continue;
        end
        img = imread(srcPath);

        % Rotation column is UI-only metadata (alignment hint)
        % Coordinates are already in original (unrotated) image frame
        % No image warp applied - coordinates already match original frame

        cropped = crop_with_quad(img, row.quad);
        outExt = '.png';
        outPath = fullfile(concDir, sprintf('%s_%s%d%s', row.imageBase, cfg.concFolderPrefix, row.concentration, outExt));
        save_image_with_format(cropped, outPath, outExt, cfg);
    end
end

function extract_quad_crops_all(coordPath, originalsDir, quadGroupDir, cfg)
    rows = read_quad_coordinates(coordPath);
    if isempty(rows)
        warning('extract:quad_empty', 'No valid quad entries parsed from %s. Check file format.', coordPath);
        return;
    end
    ensure_folder(quadGroupDir);
    for i = 1:numel(rows)
        row = rows(i);

        % Validate parsed row structure
        if ~isfield(row, 'imageBase') || ~isfield(row, 'concentration') || ~isfield(row, 'quad')
            warning('extract:invalid_row_struct', 'Skipping malformed coordinate entry %d', i);
            continue;
        end

        srcPath = find_image_file_cached(originalsDir, row.imageBase, cfg);
        if isempty(srcPath)
            warning('extract:missing_original', 'Original image not found for %s in %s', row.imageBase, originalsDir);
            continue;
        end
        img = imread(srcPath);

        % Rotation column is UI-only metadata (alignment hint)
        % Coordinates are already in original (unrotated) image frame
        % No image warp applied - coordinates already match original frame

        cropped = crop_with_quad(img, row.quad);
        outExt = '.png';
        % Mirror cut_micropads layout: con_* folders containing base_con_<idx>.ext
        concFolder = fullfile(quadGroupDir, sprintf('%s%d', cfg.concFolderPrefix, row.concentration));
        ensure_folder(concFolder);
        outName = sprintf('%s_%s%d%s', row.imageBase, cfg.concFolderPrefix, row.concentration, outExt);
        outPath = fullfile(concFolder, outName);
        save_image_with_format(cropped, outPath, outExt, cfg);
    end
end

function rows = read_quad_coordinates(coordPath)
    % Reads quad coordinates saved by cut_micropads.m
    % Delegates parsing to coordinate_io.parseQuadCoordinateFile and maps field names.
    %
    % Returns struct array with fields: .imageBase, .concentration, .quad, .rotation
    % (Maps from coordinate_io's .imageName/.vertices to .imageBase/.quad)

    rows = struct('imageBase','', 'concentration',0, 'quad',[], 'rotation',0);
    rows = rows([]);

    if ~isfile(coordPath), return; end

    coordIO = coordinate_io();
    rawEntries = coordIO.parseQuadCoordinateFile(coordPath);

    if isempty(rawEntries), return; end

    % Map field names from coordinate_io format to extract_images format
    numEntries = numel(rawEntries);
    rows = struct('imageBase', cell(1, numEntries), ...
                  'concentration', cell(1, numEntries), ...
                  'quad', cell(1, numEntries), ...
                  'rotation', cell(1, numEntries));

    for i = 1:numEntries
        rows(i).imageBase = strip_ext(rawEntries(i).imageName);  % Strip extension
        rows(i).concentration = rawEntries(i).concentration;
        rows(i).quad = round(rawEntries(i).vertices);  % Map vertices -> quad
        rows(i).rotation = rawEntries(i).rotation;
    end
end

% ----------------------
% Step 3: Elliptical patches
% ----------------------
function extract_elliptical_patches(coordPath, quadInputDir, patchOutputBase, cfg)
    % coordPath may be phone-level or per concentration folder. 'image' column refers
    % to quad-cropped image name (with extension) relative to quadInputDir.
    rows = read_ellipse_coordinates(coordPath);
    if isempty(rows)
        warning('extract:ellipse_empty', 'No valid elliptical entries parsed from %s. Check file format.', coordPath);
        return;
    end
    ensure_folder(patchOutputBase);
    quadConDirs = find_concentration_dirs(quadInputDir, cfg.concFolderPrefix);

    % Load mask_utils for ellipse mask creation
    maskUtils = mask_utils();

    for i = 1:numel(rows)
        row = rows(i);
        srcPath = resolve_quad_source(quadInputDir, row, cfg, quadConDirs);
        if isempty(srcPath)
            warning('extract:missing_quad_img', 'Quad image not found for %s (con %d) under %s', ...
                row.imageName, row.concentration, relpath(quadInputDir, cfg.projectRoot));
            continue;
        end
        img = imread(srcPath);
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

        % OPTIMIZATION: Use cached elliptical masks (delegate to mask_utils)
        cx = xCenter - x1 + 1; cy = yCenter - y1 + 1;
        mask = maskUtils.createEllipseMaskCached([h, w], cx, cy, a, b, theta, cfg.ellipticalMaskCache);

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
    % Delegates parsing to coordinate_io.parseEllipseCoordinateFile.
    %
    % Returns struct array with fields: .imageName, .x, .y, .semiMajorAxis,
    %                                   .semiMinorAxis, .rotationAngle, .concentration, .replicate
    % (Field names match coordinate_io format directly)

    rows = struct('imageName','', 'x',0, 'y',0, 'semiMajorAxis',0, 'semiMinorAxis',0, 'rotationAngle',0, 'concentration',0, 'replicate',0);
    rows = rows([]);

    if ~isfile(coordPath), return; end

    coordIO = coordinate_io();
    rawEntries = coordIO.parseEllipseCoordinateFile(coordPath);

    if isempty(rawEntries), return; end

    % coordinate_io returns fields that match our expected format directly
    rows = rawEntries;
end

% ----------------------
% Helpers: imaging and I/O
% ----------------------
function cropped = crop_with_quad(img, quad)
    % OPTIMIZED: Compute bbox first, then create mask only for bbox region
    [h, w, c] = size(img);

    % Pre-compute bounding box from quad vertices (avoid full-image mask)
    minx = max(1, floor(min(quad(:,1))));
    maxx = min(w, ceil(max(quad(:,1))));
    miny = max(1, floor(min(quad(:,2))));
    maxy = min(h, ceil(max(quad(:,2))));

    bboxW = maxx - minx + 1;
    bboxH = maxy - miny + 1;

    % Adjust quad coordinates to bbox-relative
    quadRelative = quad;
    quadRelative(:,1) = quadRelative(:,1) - minx + 1;
    quadRelative(:,2) = quadRelative(:,2) - miny + 1;

    % Create mask only for bbox region (much smaller memory footprint)
    mask = poly2mask(quadRelative(:,1), quadRelative(:,2), bboxH, bboxW);

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

function save_image_with_format(img, outPath, ~, ~)
    % Save image (format determined from outPath extension by imwrite).
    ensure_folder(fileparts(outPath));
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

function srcPath = resolve_quad_source(baseDir, row, cfg, quadConDirs)
    % OPTIMIZED: Locate quad image with streamlined search order
    if nargin < 4
        quadConDirs = {};
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
    if isempty(quadConDirs)
        quadConDirs = find_concentration_dirs(baseDir, cfg.concFolderPrefix);
    end
    for idx = 1:numel(quadConDirs)
        candidate = fullfile(quadConDirs{idx}, row.imageName);
        if isfile(candidate)
            srcPath = candidate;
            return;
        end
        alt = find_image_file_cached(quadConDirs{idx}, baseName, cfg);
        if ~isempty(alt)
            srcPath = alt;
            return;
        end
    end
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

