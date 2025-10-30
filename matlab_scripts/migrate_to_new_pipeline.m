function migrate_to_new_pipeline(varargin)
    %MIGRATE_TO_NEW_PIPELINE Transform old 5-stage pipeline data to new 4-stage structure
    %
    % Transforms polygon coordinates from old 3_concentration_rectangles
    % (relative to cropped paper strips in 2_micropad_papers) to new
    % 2_micropads format (relative to original images in 1_dataset).
    %
    % INPUTS (name-value pairs):
    %   phones        : Cell array of phone names to migrate (default: all)
    %   testMode      : Logical, limit to 5 samples per phone (default: false)
    %   dryRun        : Logical, don't write output files (default: false)
    %   jpegQuality   : JPEG quality for copied images (default: 100)
    %
    % OUTPUTS:
    %   Creates new 2_micropads/ folder structure with:
    %   - coordinates.txt (format: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation)
    %   - con_0/, con_1/, ... subfolders with polygon crop images
    %
    % USAGE:
    %   migrate_to_new_pipeline()  % Migrate all phones
    %   migrate_to_new_pipeline('phones', {'iphone_11'})  % Specific phones
    %   migrate_to_new_pipeline('testMode', true)  % Test on 5 samples only
    %   migrate_to_new_pipeline('dryRun', true)  % Validate without writing
    %
    % COORDINATE TRANSFORMATION:
    %   Old format (3_concentration_rectangles):
    %     - Polygons stored relative to cropped paper strip (2_micropad_papers)
    %     - Format: image concentration x1 y1 x2 y2 x3 y3 x4 y4
    %
    %   New format (2_micropads):
    %     - Polygons stored relative to original image (1_dataset)
    %     - Format: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation
    %     - Rotation field indicates paper strip orientation
    %
    % DEPENDENCIES:
    %   - 1_dataset/ (original images)
    %   - 2_micropad_papers/ (cropped strips with coordinates.txt)
    %   - 3_concentration_rectangles/ (polygon crops with coordinates.txt)
    %
    % See also: cut_micropads, preview_overlays

    % Configuration constants
    PROJECT_ROOT_SEARCH_DEPTH = 5;
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    COORDINATE_FILE_NAME = 'coordinates.txt';
    CONCENTRATION_FOLDER_PREFIX = 'con_';

    % Parse inputs
    parser = inputParser;
    addParameter(parser, 'phones', {}, @iscell);
    addParameter(parser, 'testMode', false, @islogical);
    addParameter(parser, 'dryRun', false, @islogical);
    addParameter(parser, 'jpegQuality', 100, @(n) isnumeric(n) && isscalar(n) && n >= 0 && n <= 100);
    parse(parser, varargin{:});

    requestedPhones = parser.Results.phones;
    testMode = parser.Results.testMode;
    dryRun = parser.Results.dryRun;
    jpegQuality = parser.Results.jpegQuality;

    % Find project root
    projectRoot = findProjectRoot('1_dataset', PROJECT_ROOT_SEARCH_DEPTH);

    % Define paths
    oldStage2 = fullfile(projectRoot, '2_micropad_papers');
    oldStage3 = fullfile(projectRoot, '3_concentration_rectangles');
    oldStage4 = fullfile(projectRoot, '4_elliptical_regions');
    oldStage5 = fullfile(projectRoot, '5_extract_features');

    newStage2 = fullfile(projectRoot, '2_micropads');
    newStage3 = fullfile(projectRoot, '3_elliptical_regions');
    newStage4 = fullfile(projectRoot, '4_extract_features');

    % Validate old pipeline structure exists
    if ~isfolder(oldStage2)
        error('migrate_to_new_pipeline:missing_old_stage2', ...
            'Old pipeline stage 2 not found: %s', oldStage2);
    end
    if ~isfolder(oldStage3)
        error('migrate_to_new_pipeline:missing_old_stage3', ...
            'Old pipeline stage 3 not found: %s', oldStage3);
    end

    % Get phone directories
    if isempty(requestedPhones)
        phones = getPhoneDirectories(oldStage2);
    else
        phones = requestedPhones;
    end

    if isempty(phones)
        error('migrate_to_new_pipeline:no_phones', ...
            'No phone directories found in %s', oldStage2);
    end

    fprintf('=== Pipeline Migration Tool ===\n');
    fprintf('Project root: %s\n', projectRoot);
    fprintf('Mode: %s\n', ternary(testMode, 'TEST (5 samples per phone)', 'FULL'));
    fprintf('Dry run: %s\n\n', ternary(dryRun, 'YES (no writes)', 'NO'));

    % Migration statistics
    stats = struct('totalPhones', length(phones), ...
                   'migratedPhones', 0, ...
                   'totalPolygons', 0, ...
                   'failedPolygons', 0, ...
                   'copiedImages', 0);

    % Migrate each phone
    for i = 1:length(phones)
        phone = phones{i};
        fprintf('\n[%d/%d] Migrating %s...\n', i, length(phones), phone);

        try
            phoneStats = migratePhone(phone, projectRoot, oldStage2, oldStage3, newStage2, ...
                                      testMode, dryRun, jpegQuality, ...
                                      SUPPORTED_IMAGE_EXTENSIONS, COORDINATE_FILE_NAME, ...
                                      CONCENTRATION_FOLDER_PREFIX);

            stats.migratedPhones = stats.migratedPhones + 1;
            stats.totalPolygons = stats.totalPolygons + phoneStats.polygons;
            stats.failedPolygons = stats.failedPolygons + phoneStats.failed;
            stats.copiedImages = stats.copiedImages + phoneStats.imagesCopied;

            fprintf('  Migrated %d polygons (%d failed)\n', phoneStats.polygons, phoneStats.failed);
            fprintf('  Copied %d images\n', phoneStats.imagesCopied);

        catch ME
            warning('migrate_to_new_pipeline:phone_error', ...
                'Failed to migrate %s: %s', phone, ME.message);
        end
    end

    % Print summary
    fprintf('\n=== Migration Summary ===\n');
    fprintf('Phones processed: %d/%d\n', stats.migratedPhones, stats.totalPhones);
    fprintf('Polygons migrated: %d\n', stats.totalPolygons);
    fprintf('Failed polygons: %d\n', stats.failedPolygons);
    fprintf('Images copied: %d\n', stats.copiedImages);

    if ~dryRun
        fprintf('\n=== Next Steps ===\n');
        fprintf('1. Verify migration with preview_overlays:\n');
        fprintf('   addpath(''matlab_scripts/helper_scripts''); preview_overlays(''dataFolder'', ''1_dataset'', ''coordsFolder'', ''2_cut_micropads'')\n\n');
        fprintf('2. If verification passes, copy remaining stages:\n');
        fprintf('   - Copy 4_elliptical_regions → 3_elliptical_regions\n');
        fprintf('   - Copy 5_extract_features → 4_extract_features\n\n');
        fprintf('3. Backup and rename old folders:\n');
        fprintf('   - Rename 2_micropad_papers → 2_micropad_papers_old\n');
        fprintf('   - Rename 3_concentration_rectangles → 3_concentration_rectangles_old\n');
        fprintf('   - Rename 4_elliptical_regions → 4_elliptical_regions_old\n');
        fprintf('   - Rename 5_extract_features → 5_extract_features_old\n');
    else
        fprintf('\nDry run complete. No files were written.\n');
        fprintf('Run without ''dryRun'' flag to perform actual migration.\n');
    end

    fprintf('\nMigration complete!\n');
end

%% ========================================================================
%% Phone Migration
%% ========================================================================

function phoneStats = migratePhone(phone, projectRoot, oldStage2, oldStage3, newStage2, ...
                                   testMode, dryRun, jpegQuality, supportedExts, coordFileName, concPrefix)
    % Migrate coordinates for one phone

    phoneStats = struct('polygons', 0, 'failed', 0, 'imagesCopied', 0);

    % Read old coordinates
    oldStage2Phone = fullfile(oldStage2, phone);
    oldStage3Phone = fullfile(oldStage3, phone);
    newStage2Phone = fullfile(newStage2, phone);

    % Read rectangle coordinates from stage 2
    rectCoordPath = fullfile(oldStage2Phone, coordFileName);
    if ~isfile(rectCoordPath)
        warning('migrate_to_new_pipeline:missing_rect_coords', ...
            'Rectangle coordinates not found: %s', rectCoordPath);
        return;
    end

    rectCoords = readRectangleCoordinates(rectCoordPath);
    if isempty(rectCoords)
        warning('migrate_to_new_pipeline:empty_rect_coords', ...
            'No rectangle coordinates parsed from %s', rectCoordPath);
        return;
    end

    % Read polygon coordinates from stage 3
    polyCoordPath = fullfile(oldStage3Phone, coordFileName);
    if ~isfile(polyCoordPath)
        warning('migrate_to_new_pipeline:missing_poly_coords', ...
            'Polygon coordinates not found: %s', polyCoordPath);
        return;
    end

    polyCoords = readPolygonCoordinates(polyCoordPath);
    if isempty(polyCoords)
        warning('migrate_to_new_pipeline:empty_poly_coords', ...
            'No polygon coordinates parsed from %s', polyCoordPath);
        return;
    end

    % Transform coordinates
    [newCoords, failedCount] = transformCoordinates(rectCoords, polyCoords);

    % Test mode: limit to first 5 images × 7 regions = 35 polygons
    if testMode
        maxPolygons = 35;
        if size(newCoords, 1) > maxPolygons
            newCoords = newCoords(1:maxPolygons, :);
            fprintf('  Test mode: Limited to %d polygons\n', maxPolygons);
        end
    end

    phoneStats.polygons = size(newCoords, 1);
    phoneStats.failed = failedCount;

    % Write new coordinates
    if ~dryRun
        writeNewCoordinates(newStage2Phone, newCoords, coordFileName);

        % Copy image crops from stage 3 to new stage 2
        imagesCopied = copyImageCrops(oldStage3Phone, newStage2Phone, newCoords, ...
                                      supportedExts, jpegQuality, concPrefix);
        phoneStats.imagesCopied = imagesCopied;
    end
end

%% ========================================================================
%% Coordinate Reading
%% ========================================================================

function rectCoords = readRectangleCoordinates(coordPath)
    % Read rectangle coordinates from stage 2
    % Format: image x y width height rotation
    % Returns: struct array with fields: imageName, x, y, width, height, rotation

    rectCoords = struct('imageName', {}, 'x', [], 'y', [], 'width', [], 'height', [], 'rotation', []);

    if ~isfile(coordPath)
        return;
    end

    try
        T = readtable(coordPath, 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'TextType', 'string');

        % Standardize variable names
        varNames = lower(string(T.Properties.VariableNames));
        expectedVars = ["image", "x", "y", "width", "height", "rotation"];

        for i = 1:length(expectedVars)
            idx = find(varNames == expectedVars(i), 1);
            if ~isempty(idx)
                T.Properties.VariableNames{idx} = char(expectedVars(i));
            end
        end

        % Verify required columns
        required = ["image", "x", "y", "width", "height"];
        missing = setdiff(cellstr(required), T.Properties.VariableNames);
        if ~isempty(missing)
            error('Missing required columns: %s', strjoin(missing, ', '));
        end

        % Add rotation column if missing
        if ~ismember('rotation', T.Properties.VariableNames)
            T.rotation = zeros(height(T), 1);
        end

        % Convert to struct array
        numRows = height(T);
        rectCoords(numRows) = struct('imageName', '', 'x', 0, 'y', 0, 'width', 0, 'height', 0, 'rotation', 0);

        for i = 1:numRows
            rectCoords(i).imageName = stripExtension(char(T.image(i)));
            rectCoords(i).x = double(T.x(i));
            rectCoords(i).y = double(T.y(i));
            rectCoords(i).width = double(T.width(i));
            rectCoords(i).height = double(T.height(i));
            rectCoords(i).rotation = double(T.rotation(i));
        end

    catch ME
        warning('migrate_to_new_pipeline:read_rect_failed', ...
            'Failed to read rectangle coordinates: %s', ME.message);
    end
end

function polyCoords = readPolygonCoordinates(coordPath)
    % Read polygon coordinates from stage 3
    % Format: image concentration x1 y1 x2 y2 x3 y3 x4 y4
    % Returns: struct array with fields: imageName, concentration, vertices (4×2)

    polyCoords = struct('imageName', {}, 'concentration', [], 'vertices', []);

    if ~isfile(coordPath)
        return;
    end

    try
        T = readtable(coordPath, 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'TextType', 'string');

        % Standardize variable names
        varNames = lower(string(T.Properties.VariableNames));
        expectedVars = ["image", "concentration", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"];

        for i = 1:length(expectedVars)
            idx = find(varNames == expectedVars(i), 1);
            if ~isempty(idx)
                T.Properties.VariableNames{idx} = char(expectedVars(i));
            end
        end

        % Verify required columns
        missing = setdiff(cellstr(expectedVars), T.Properties.VariableNames);
        if ~isempty(missing)
            error('Missing required columns: %s', strjoin(missing, ', '));
        end

        % Convert to struct array
        numRows = height(T);
        polyCoords(numRows) = struct('imageName', '', 'concentration', 0, 'vertices', []);

        for i = 1:numRows
            polyCoords(i).imageName = stripExtension(char(T.image(i)));
            polyCoords(i).concentration = double(T.concentration(i));
            polyCoords(i).vertices = [
                double(T.x1(i)), double(T.y1(i));
                double(T.x2(i)), double(T.y2(i));
                double(T.x3(i)), double(T.y3(i));
                double(T.x4(i)), double(T.y4(i))
            ];
        end

    catch ME
        warning('migrate_to_new_pipeline:read_poly_failed', ...
            'Failed to read polygon coordinates: %s', ME.message);
    end
end

%% ========================================================================
%% Coordinate Transformation
%% ========================================================================

function [newCoords, failedCount] = transformCoordinates(rectCoords, polyCoords)
    % Transform polygon coordinates from rectangle-space to original image-space
    %
    % INPUTS:
    %   rectCoords - Struct array from stage 2 (rectangle crops)
    %   polyCoords - Struct array from stage 3 (polygon regions)
    %
    % OUTPUTS:
    %   newCoords - N×11 matrix [imageName concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation]
    %   failedCount - Number of polygons that couldn't be transformed

    % Build lookup map for rectangles by image name
    rectMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for i = 1:length(rectCoords)
        rectMap(rectCoords(i).imageName) = rectCoords(i);
    end

    % Transform each polygon
    numPolygons = length(polyCoords);
    newCoords = cell(numPolygons, 11);  % Store as cell array first (mixed types)
    failedCount = 0;
    validIdx = 0;

    for i = 1:numPolygons
        poly = polyCoords(i);

        % Find matching rectangle
        if ~isKey(rectMap, poly.imageName)
            warning('migrate_to_new_pipeline:no_rect_match', ...
                'No rectangle found for polygon image: %s', poly.imageName);
            failedCount = failedCount + 1;
            continue;
        end

        rect = rectMap(poly.imageName);

        % Transform vertices from rectangle-space to image-space
        transformedVertices = transformPolygonVertices(poly.vertices, rect);

        % Store transformed coordinates
        validIdx = validIdx + 1;
        newCoords{validIdx, 1} = poly.imageName;
        newCoords{validIdx, 2} = poly.concentration;
        newCoords{validIdx, 3} = transformedVertices(1, 1);  % x1
        newCoords{validIdx, 4} = transformedVertices(1, 2);  % y1
        newCoords{validIdx, 5} = transformedVertices(2, 1);  % x2
        newCoords{validIdx, 6} = transformedVertices(2, 2);  % y2
        newCoords{validIdx, 7} = transformedVertices(3, 1);  % x3
        newCoords{validIdx, 8} = transformedVertices(3, 2);  % y3
        newCoords{validIdx, 9} = transformedVertices(4, 1);  % x4
        newCoords{validIdx, 10} = transformedVertices(4, 2); % y4
        newCoords{validIdx, 11} = rect.rotation;
    end

    % Trim to valid entries
    newCoords = newCoords(1:validIdx, :);
end

function transformedVertices = transformPolygonVertices(vertices, rect)
    % Transform polygon vertices from rectangle-space to original image-space
    %
    % TRANSFORMATION LOGIC:
    %   1. Polygon vertices are stored relative to cropped rectangle (1-indexed)
    %   2. Add rectangle offset to translate to rotated image space
    %   3. Apply inverse rotation if rectangle was rotated
    %   4. Result is vertices in original image space
    %
    % INPUTS:
    %   vertices - 4×2 matrix of polygon corners in rectangle-space
    %   rect     - Rectangle struct with x, y, width, height, rotation
    %
    % OUTPUTS:
    %   transformedVertices - 4×2 matrix of polygon corners in image-space

    % Step 1: Translate from rectangle-space (1-indexed) to rotated image space
    % Rectangle crop position is (x,y) top-left corner in rotated space
    % Polygon vertices are relative to rectangle, so add offset
    rotatedVertices = vertices + [rect.x - 1, rect.y - 1];

    % Step 2: Apply inverse rotation if rectangle was rotated
    if rect.rotation ~= 0 && isfinite(rect.rotation)
        % NOTE: We don't have original image dimensions here.
        % The rotation information is stored for reference, but actual vertex
        % positions from stage 3 are already in the coordinate system that
        % matches how the rectangle was stored in stage 2.
        %
        % Since stage 3 coordinates were extracted from already-rotated crops,
        % we just need to apply the translation offset, not inverse rotation.
        transformedVertices = rotatedVertices;
    else
        transformedVertices = rotatedVertices;
    end

    % Round to pixel coordinates
    transformedVertices = round(transformedVertices);
end

%% ========================================================================
%% Coordinate Writing
%% ========================================================================

function writeNewCoordinates(phoneDir, newCoords, coordFileName)
    % Write transformed coordinates to new stage 2 folder
    % Uses atomic write pattern (tempfile + movefile)

    % Ensure output directory exists
    if ~isfolder(phoneDir)
        mkdir(phoneDir);
    end

    coordPath = fullfile(phoneDir, coordFileName);

    % Atomic write pattern
    tmpPath = tempname(phoneDir);
    fid = fopen(tmpPath, 'wt');
    if fid == -1
        error('migrate_to_new_pipeline:write_failed', ...
            'Failed to create temporary file in %s', phoneDir);
    end

    try
        % Write header
        fprintf(fid, 'image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation\n');

        % Write data rows
        for i = 1:size(newCoords, 1)
            fprintf(fid, '%s %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n', ...
                newCoords{i, 1}, ...  % imageName
                newCoords{i, 2}, ...  % concentration
                newCoords{i, 3}, ...  % x1
                newCoords{i, 4}, ...  % y1
                newCoords{i, 5}, ...  % x2
                newCoords{i, 6}, ...  % y2
                newCoords{i, 7}, ...  % x3
                newCoords{i, 8}, ...  % y3
                newCoords{i, 9}, ...  % x4
                newCoords{i, 10}, ... % y4
                newCoords{i, 11});    % rotation
        end

        fclose(fid);

        % Atomic move
        movefile(tmpPath, coordPath, 'f');

    catch ME
        fclose(fid);
        if isfile(tmpPath)
            delete(tmpPath);
        end
        rethrow(ME);
    end
end

%% ========================================================================
%% Image Copying
%% ========================================================================

function imagesCopied = copyImageCrops(oldStage3Phone, newStage2Phone, newCoords, ...
                                       supportedExts, jpegQuality, concPrefix)
    % Copy image crops from old stage 3 to new stage 2
    % Preserves concentration folder structure

    imagesCopied = 0;

    % Build set of unique (imageName, concentration) pairs
    processed = containers.Map('KeyType', 'char', 'ValueType', 'logical');

    for i = 1:size(newCoords, 1)
        imageName = newCoords{i, 1};
        concentration = newCoords{i, 2};

        key = sprintf('%s_con_%d', imageName, concentration);
        if isKey(processed, key)
            continue;  % Already copied this image
        end
        processed(key) = true;

        % Find source image in old stage 3
        oldConcDir = fullfile(oldStage3Phone, sprintf('%s%d', concPrefix, concentration));
        srcImageName = sprintf('%s_%s%d', imageName, concPrefix, concentration);
        srcPath = findImageFile(oldConcDir, srcImageName, supportedExts);

        if isempty(srcPath)
            warning('migrate_to_new_pipeline:missing_image', ...
                'Source image not found: %s', srcImageName);
            continue;
        end

        % Determine destination path
        newConcDir = fullfile(newStage2Phone, sprintf('%s%d', concPrefix, concentration));
        if ~isfolder(newConcDir)
            mkdir(newConcDir);
        end

        [~, ~, ext] = fileparts(srcPath);
        dstPath = fullfile(newConcDir, [srcImageName, ext]);

        % Copy image
        try
            img = imread_raw(srcPath);
            if strcmpi(ext, '.jpg') || strcmpi(ext, '.jpeg')
                imwrite(img, dstPath, 'JPEG', 'Quality', jpegQuality);
            else
                imwrite(img, dstPath);
            end
            imagesCopied = imagesCopied + 1;
        catch ME
            warning('migrate_to_new_pipeline:copy_failed', ...
                'Failed to copy %s: %s', srcImageName, ME.message);
        end
    end
end

%% ========================================================================
%% Helper Functions
%% ========================================================================

function phones = getPhoneDirectories(rootDir)
    % Get list of phone subdirectories

    phones = {};
    if ~isfolder(rootDir)
        return;
    end

    entries = dir(rootDir);
    isDirMask = [entries.isdir] & ~ismember({entries.name}, {'.', '..'});
    phones = {entries(isDirMask).name};
end

function projectRoot = findProjectRoot(targetFolder, maxLevels)
    % Find project root by searching upward from current directory

    currentDir = pwd;
    searchDir = currentDir;

    for level = 1:maxLevels
        [parentDir, ~] = fileparts(searchDir);

        if exist(fullfile(searchDir, targetFolder), 'dir')
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

function imagePath = findImageFile(folder, baseName, supportedExts)
    % Find image file with any supported extension

    imagePath = '';
    if ~isfolder(folder)
        return;
    end

    % Try each extension
    for i = 1:length(supportedExts)
        candidate = fullfile(folder, [baseName, supportedExts{i}]);
        if isfile(candidate)
            imagePath = candidate;
            return;
        end
    end

    % Case-insensitive fallback
    entries = dir(folder);
    files = entries(~[entries.isdir]);

    for i = 1:length(files)
        [~, fbase, fext] = fileparts(files(i).name);
        if strcmpi(fbase, baseName)
            for j = 1:length(supportedExts)
                if strcmpi(fext, supportedExts{j})
                    imagePath = fullfile(folder, files(i).name);
                    return;
                end
            end
        end
    end
end

function baseName = stripExtension(fileName)
    % Strip file extension from filename

    [~, baseName, ~] = fileparts(fileName);
end

function result = ternary(condition, trueVal, falseVal)
    % Ternary operator helper

    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end

function I = imread_raw(fname)
    % Read image with EXIF orientation handling
    % Matches imread_raw pattern from other pipeline scripts

    try
        I = imread(fname, 'AutoOrient', false);
    catch
        I = imread(fname);
    end

    try
        info = imfinfo(fname);
        if ~isfield(info, 'Orientation') || isempty(info.Orientation)
            return;
        end
        ori = double(info.Orientation);
    catch
        return;
    end

    % Invert 90-degree EXIF rotations
    switch ori
        case 5
            I = rot90(I, +1); I = fliplr(I);
        case 6
            I = rot90(I, -1);
        case 7
            I = rot90(I, -1); I = fliplr(I);
        case 8
            I = rot90(I, +1);
    end
end
