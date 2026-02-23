function io = image_io()
    %% IMAGE_IO Returns a struct of function handles for image I/O operations
    %
    % This utility module provides functions for loading, saving, and finding
    % image files in the microPAD processing pipeline. Includes caching
    % infrastructure for performance optimization.
    %
    % Usage:
    %   io = image_io();
    %   img = io.loadImage(filepath);
    %   io.saveImage(img, filepath);
    %   imagePath = io.findImageFile(folder, baseName, cache);
    %
    % Caching:
    %   The module provides caching infrastructure for efficient batch processing:
    %   - Image file cache: Maps 'folder|baseName' -> full image path
    %   - Directory index cache: Maps folder -> struct with names, basenames, exts
    %
    %   Create caches at the start of batch processing:
    %     cache = io.createCaches();
    %     for each image:
    %         path = io.findImageFile(folder, name, cache);
    %
    % See also: coordinate_io, mask_utils, path_utils

    %% Public API
    % Image loading/saving
    io.loadImage = @loadImage;
    io.loadImageRaw = @loadImageRaw;
    io.saveImage = @saveImage;

    % Image file discovery
    io.findImageFile = @findImageFile;
    io.findImageFileCached = @findImageFileCached;
    io.listImageFiles = @listImageFiles;

    % Caching infrastructure
    io.createCaches = @createCaches;
    io.createImageCache = @createImageCache;
    io.createDirIndexCache = @createDirIndexCache;

    % Constants
    io.getSupportedExtensions = @getSupportedExtensions;
    io.getGlobPatterns = @getGlobPatterns;

    % Utilities
    io.stripExtension = @stripExtension;
    io.clampUint8 = @clampUint8;

    % Image Processing - Augmentation Support
    io.applyMotionBlur = @apply_motion_blur;
    io.featherQuadEdges = @feather_quad_edges;
end

%% =========================================================================
%% CONSTANTS
%% =========================================================================

function exts = getSupportedExtensions()
    % Get list of supported image file extensions (lowercase, with dots)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
end

function patterns = getGlobPatterns()
    % Get glob patterns for image files (for use with dir())
    patterns = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};
end

%% =========================================================================
%% IMAGE LOADING
%% =========================================================================

function [img, isValid] = loadImage(imagePath)
    % Load image from file without applying EXIF orientation
    %
    % INPUTS:
    %   imagePath - Full path to image file
    %
    % OUTPUTS:
    %   img     - Image matrix (H x W x C), uint8
    %   isValid - Boolean indicating successful load
    %
    % Note: Any rotation is handled via coordinates.txt rotation column,
    % not via EXIF metadata. This ensures consistent behavior across
    % different image sources.

    isValid = false;
    img = [];

    if ~isfile(imagePath)
        warning('image_io:missing_file', 'Image file not found: %s', imagePath);
        return;
    end

    try
        % Read image without EXIF orientation
        img = imread(imagePath);
        isValid = true;
    catch ME
        warning('image_io:read_error', 'Failed to read image %s: %s', imagePath, ME.message);
    end
end

function img = loadImageRaw(imagePath)
    % Load image without EXIF rotation (simple version)
    %
    % INPUTS:
    %   imagePath - Full path to image file
    %
    % OUTPUTS:
    %   img - Image matrix (H x W x C)
    %
    % This is a simple wrapper around imread() that reads pixels in their
    % recorded layout without applying EXIF orientation metadata.
    % User-requested rotation is stored in coordinates.txt and applied
    % during downstream processing rather than via image metadata.
    %
    % See also: loadImage (with validation and error handling)

    img = imread(imagePath);
end

%% =========================================================================
%% IMAGE SAVING
%% =========================================================================

function saveImage(img, outPath)
    % Save image to file (PNG format - pipeline standard)
    %
    % INPUTS:
    %   img     - Image matrix to save
    %   outPath - Output file path (should end in .png)
    %
    % Note: The microPAD pipeline standardizes on PNG for all intermediate
    % outputs to avoid JPEG compression artifacts and EXIF rotation issues.

    % Ensure output directory exists
    outDir = fileparts(outPath);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
    end

    try
        imwrite(img, outPath);
    catch ME
        error('image_io:write_failed', 'Failed to write image %s: %s', outPath, ME.message);
    end
end

%% =========================================================================
%% IMAGE FILE DISCOVERY
%% =========================================================================

function imagePath = findImageFile(folder, baseName, cache)
    % Find image file by base name, trying all supported extensions
    %
    % INPUTS:
    %   folder   - Directory to search in
    %   baseName - Image base name (without extension)
    %   cache    - (Optional) Cache struct from createCaches() for performance
    %
    % OUTPUTS:
    %   imagePath - Full path to found image, or '' if not found
    %
    % The function first tries direct extension matches (fast path),
    % then falls back to case-insensitive search using directory index.

    if nargin < 3
        cache = [];
    end

    % Use cached version if cache provided
    if ~isempty(cache) && isfield(cache, 'imageFileCache')
        imagePath = findImageFileCached(folder, baseName, cache);
        return;
    end

    imagePath = '';
    if ~isfolder(folder)
        return;
    end

    % Fast path: direct extension guesses (most common first)
    exts = getSupportedExtensions();
    for i = 1:numel(exts)
        p = fullfile(folder, [baseName exts{i}]);
        if isfile(p)
            imagePath = p;
            return;
        end
        % Also try uppercase extension
        p = fullfile(folder, [baseName upper(exts{i})]);
        if isfile(p)
            imagePath = p;
            return;
        end
    end

    % Slow path: case-insensitive search
    d = dir(folder);
    fileItems = d(~[d.isdir]);
    if isempty(fileItems)
        return;
    end

    validExts = getSupportedExtensions();

    for i = 1:numel(fileItems)
        [~, fileBase, fileExt] = fileparts(fileItems(i).name);
        if strcmpi(fileBase, baseName) && any(strcmpi(fileExt, validExts))
            imagePath = fullfile(folder, fileItems(i).name);
            return;
        end
    end
end

function imagePath = findImageFileCached(folder, baseName, cache)
    % Find image file using cache for performance
    %
    % INPUTS:
    %   folder   - Directory to search in
    %   baseName - Image base name (without extension)
    %   cache    - Cache struct with imageFileCache and dirIndexCache fields
    %
    % OUTPUTS:
    %   imagePath - Full path to found image, or '' if not found

    % Check image file cache first
    cacheKey = [folder '|' baseName];

    if isKey(cache.imageFileCache, cacheKey)
        imagePath = cache.imageFileCache(cacheKey);
        return;
    end

    % Not in cache, perform search
    imagePath = findImageFileWithDirIndex(folder, baseName, cache);

    % Store result in cache
    cache.imageFileCache(cacheKey) = imagePath;
end

function imagePath = findImageFileWithDirIndex(folder, baseName, cache)
    % Internal: Find image using directory index cache

    imagePath = '';
    if ~isfolder(folder)
        return;
    end

    % Fast path: direct extension guesses
    exts = getSupportedExtensions();
    for i = 1:numel(exts)
        p = fullfile(folder, [baseName exts{i}]);
        if isfile(p)
            imagePath = p;
            return;
        end
    end

    % Use cached directory index
    if isKey(cache.dirIndexCache, folder)
        dirIndex = cache.dirIndexCache(folder);
    else
        % Build directory index once per folder
        d = dir(folder);
        fileItems = d(~[d.isdir]);
        if isempty(fileItems)
            cache.dirIndexCache(folder) = struct('names', {{}}, 'basenames', {{}}, 'exts', {{}});
            return;
        end

        names = {fileItems.name};
        [~, fileBasenames, fileExts] = cellfun(@fileparts, names, 'UniformOutput', false);

        dirIndex = struct('names', {names}, ...
                         'basenames', {lower(fileBasenames)}, ...
                         'exts', {lower(fileExts)});
        cache.dirIndexCache(folder) = dirIndex;
    end

    % Search in cached index
    validExts = getSupportedExtensions();

    for i = 1:numel(dirIndex.basenames)
        if strcmpi(dirIndex.basenames{i}, baseName) && any(strcmpi(dirIndex.exts{i}, validExts))
            imagePath = fullfile(folder, dirIndex.names{i});
            return;
        end
    end
end

function files = listImageFiles(dirPath, extensions)
    % List all image files in a directory
    %
    % INPUTS:
    %   dirPath    - Directory path to search
    %   extensions - (Optional) Cell array of glob patterns (default: all supported)
    %
    % OUTPUTS:
    %   files - Cell array of image file names (without path)

    if nargin < 2 || isempty(extensions)
        extensions = getGlobPatterns();
    end

    files = {};

    if ~isfolder(dirPath)
        return;
    end

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

%% =========================================================================
%% CACHING INFRASTRUCTURE
%% =========================================================================

function cache = createCaches()
    % Create cache structure for batch image processing
    %
    % OUTPUTS:
    %   cache - Struct with:
    %           .imageFileCache - containers.Map for image path lookups
    %           .dirIndexCache  - containers.Map for directory contents
    %
    % Usage:
    %   cache = io.createCaches();
    %   for i = 1:numImages
    %       path = io.findImageFile(folder, names{i}, cache);
    %       % ... process image
    %   end
    %
    % Caches should NOT be reused across different processing runs as they
    % do not invalidate on file system changes.

    cache = struct();
    cache.imageFileCache = createImageCache();
    cache.dirIndexCache = createDirIndexCache();
end

function cache = createImageCache()
    % Create cache for image file path lookups
    %
    % OUTPUTS:
    %   cache - containers.Map with 'folder|baseName' keys
    %
    % The cache maps 'folder|baseName' strings to full image paths.
    % Typical hit rate >95% for datasets with repeated folder lookups.

    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
end

function cache = createDirIndexCache()
    % Create cache for directory index lookups
    %
    % OUTPUTS:
    %   cache - containers.Map with folder path keys
    %
    % Each entry contains struct with:
    %   .names     - Cell array of original filenames
    %   .basenames - Cell array of lowercase base names
    %   .exts      - Cell array of lowercase extensions
    %
    % Built once per folder with single dir() call, reused for all lookups.

    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
end

%% =========================================================================
%% UTILITIES
%% =========================================================================

function baseName = stripExtension(nameOrPath)
    % Strip extension from filename or path
    %
    % INPUTS:
    %   nameOrPath - Filename or full path
    %
    % OUTPUTS:
    %   baseName - Base name without extension

    [~, baseName, ~] = fileparts(nameOrPath);
end

function img = clampUint8(img)
    % Clamp image values to [0, 255] and convert to uint8
    %
    % INPUTS:
    %   img - Image matrix (any numeric type, values in [0, 255] range)
    %
    % OUTPUTS:
    %   img - Image as uint8 with values clamped to valid range
    %
    % Note: Input must be in [0, 255] range (not normalized [0, 1]).

    img = uint8(min(255, max(0, img)));
end

%% =========================================================================
%% IMAGE PROCESSING - AUGMENTATION SUPPORT
%% =========================================================================

function img = apply_motion_blur(img)
    % Apply slight motion blur with cached PSFs to avoid redundant kernel generation
    %
    % INPUTS:
    %   img - RGB image (uint8)
    %
    % OUTPUTS:
    %   img - Motion-blurred image (uint8)
    %
    % The blur simulates camera shake during capture with random direction.
    % PSF kernels are cached by length and angle for performance.

    persistent psf_cache
    if isempty(psf_cache)
        psf_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end

    len = 4 + randi(4);            % 5-8 px
    ang = rand() * 180;            % degrees
    ang_rounded = round(ang);
    cache_key = sprintf('%d_%d', len, ang_rounded);

    if isKey(psf_cache, cache_key)
        psf = psf_cache(cache_key);
    else
        psf = fspecial('motion', len, ang_rounded);
        psf_cache(cache_key) = psf;
    end

    img = imfilter(img, psf, 'replicate');
end

function [featheredImg, featheredAlpha] = feather_quad_edges(quadImg, alpha, featherWidth, sigma)
    % Apply edge feathering to quad image and alpha, maintaining pre-multiplied consistency
    %
    % Uses erosion followed by Gaussian blur to create smooth edge transitions.
    % Critically, this function adjusts RGB values to match the new alpha,
    % preserving the pre-multiplied relationship required for correct compositing.
    %
    % INPUTS:
    %   quadImg - RGB image (uint8), pre-multiplied by original alpha
    %   alpha - Single-precision alpha mask (0-1) from the same transform
    %   featherWidth - Erosion amount in pixels
    %   sigma - Gaussian blur sigma (auto-computed if empty)
    %
    % OUTPUTS:
    %   featheredImg - RGB image with adjusted values for new alpha (uint8)
    %   featheredAlpha - Alpha mask with soft edges (single, 0-1)

    featheredImg = quadImg;
    origAlpha = single(alpha);  % Defensive conversion + store for RGB adjustment
    featheredAlpha = origAlpha;

    if featherWidth <= 0
        return;
    end

    % Convert to binary for morphological operations
    binaryAlpha = origAlpha > 0.5;

    % Adaptive featherWidth clamping: prevent erosion from eliminating small ROIs
    % Clamp to 1/4 of minimum ROI dimension to ensure core region survives
    [rows, cols] = find(binaryAlpha);
    if isempty(rows)
        return;  % No content to feather
    end
    roiHeight = max(rows) - min(rows) + 1;
    roiWidth = max(cols) - min(cols) + 1;
    minDim = min(roiHeight, roiWidth);
    effectiveFeatherWidth = min(featherWidth, floor(minDim / 4));

    if effectiveFeatherWidth <= 0
        return;  % ROI too small for any feathering
    end

    % Auto-compute sigma if not provided (based on effective width)
    if isempty(sigma) || ~isfinite(sigma)
        sigma = max(0.5, effectiveFeatherWidth / 2);
    else
        % Scale sigma proportionally if featherWidth was clamped
        sigma = sigma * (effectiveFeatherWidth / featherWidth);
    end

    % Erode to shrink boundary inward (disk SE for isotropic erosion)
    se = strel('disk', effectiveFeatherWidth, 0);
    erodedAlpha = imerode(binaryAlpha, se);

    % Guard: if erosion still eliminates all content, skip feathering
    if ~any(erodedAlpha(:))
        return;
    end

    % Apply Gaussian blur to create smooth falloff from eroded boundary
    blurredAlpha = imgaussfilt(single(erodedAlpha), sigma);

    % Preserve core: use max of blurred and eroded
    featheredAlpha = max(blurredAlpha, single(erodedAlpha));

    % Constrain feathered alpha to not exceed original alpha
    % This prevents alpha from extending beyond the original content boundary
    featheredAlpha = min(featheredAlpha, origAlpha);

    % Clamp to valid range
    featheredAlpha = max(0, min(1, featheredAlpha));

    % Adjust RGB to maintain pre-multiplied consistency
    % Original RGB = true_color * origAlpha (pre-multiplied)
    % Need: new RGB = true_color * featheredAlpha
    % Therefore: new RGB = original RGB * (featheredAlpha / origAlpha)
    %
    % This ensures compositing formula (F + B*(1-alpha)) produces correct results

    % Only adjust where alpha changed and original alpha is significant
    needsAdjustment = (origAlpha > 0.01) & (abs(featheredAlpha - origAlpha) > 0.001);

    if any(needsAdjustment(:))
        % Compute adjustment ratio - always <= 1 since featheredAlpha <= origAlpha
        % (no amplification occurs, so no clamping needed unlike un-premultiply)
        ratio = ones(size(origAlpha), 'single');
        ratio(needsAdjustment) = featheredAlpha(needsAdjustment) ./ origAlpha(needsAdjustment);

        % Apply ratio to each RGB channel
        featheredImg = double(quadImg);
        for c = 1:size(featheredImg, 3)
            channel = featheredImg(:,:,c);
            channel = channel .* double(ratio);
            featheredImg(:,:,c) = channel;
        end
        featheredImg = uint8(min(255, max(0, featheredImg)));
    end
end
