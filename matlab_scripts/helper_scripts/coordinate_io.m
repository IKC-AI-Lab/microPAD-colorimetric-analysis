function io = coordinate_io()
    %% COORDINATE_IO Returns a struct of function handles for coordinate file I/O
    %
    % This utility module provides functions for reading and writing coordinate
    % files used in the microPAD processing pipeline. Supports both polygon
    % (Stage 2) and ellipse (Stage 3) coordinate formats.
    %
    % Usage:
    %   io = coordinate_io();
    %   polygons = io.loadPolygonCoordinates(filepath, imageName, numExpected);
    %   ellipses = io.loadEllipseCoordinates(filepath, imageName);
    %   io.appendPolygonCoordinates(folder, baseName, concentration, vertices, rotation);
    %   io.appendEllipseCoordinates(folder, baseName, ellipseData);
    %
    % Coordinate File Formats:
    %
    % Stage 2 (Polygon) - 2_micropads/[phone]/coordinates.txt:
    %   image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation
    %   - 11 columns: 1 string + 10 numeric
    %   - Vertices in clockwise order from top-left
    %   - Coordinates relative to ORIGINAL (unrotated) image in 1_dataset
    %
    % Stage 3 (Ellipse) - 3_elliptical_regions/[phone]/coordinates.txt:
    %   image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
    %   - 8 columns: 1 string + 7 numeric
    %   - Coordinates relative to ORIGINAL (unrotated) image in 1_dataset
    %   - rotationAngle is the GEOMETRIC orientation of the ellipse
    %
    % ROTATION SEMANTICS CONTRACT:
    %
    % Stage 2 "rotation" field (11th column):
    %   - Purpose: UI-only alignment hint (NOT a coordinate transformation)
    %   - Frame: Coordinates are always in the original unrotated image frame
    %   - Usage: Records how much the user rotated the image in cut_micropads UI
    %            to facilitate labeling. Downstream scripts MAY ignore this value.
    %   - Values: Degrees, normalized to [-180, 180] range
    %   - Consumers:
    %     * cut_micropads.m: Writes this value from UI state
    %     * extract_features.m: IGNORES this value (uses Stage 3 ellipse rotation)
    %     * augment_dataset.m: Reads and adjusts for augmentation transforms
    %
    % Stage 3 "rotationAngle" field (8th column):
    %   - Purpose: GEOMETRIC orientation of the ellipse shape
    %   - Frame: Degrees, positive = clockwise rotation from horizontal
    %   - Usage: Defines ellipse orientation for masking and feature extraction
    %   - Values: Degrees, normalized to [-180, 180] range
    %   - Note: This is NOT a UI hint - it's the actual ellipse geometry
    %
    % See also: geometry_transform, image_io, mask_utils

    %% Public API
    % Reading functions
    io.loadPolygonCoordinates = @loadPolygonCoordinates;
    io.loadEllipseCoordinates = @loadEllipseCoordinates;
    io.parsePolygonCoordinateFile = @parsePolygonCoordinateFile;
    io.parseEllipseCoordinateFile = @parseEllipseCoordinateFile;
    io.parsePolygonCoordinateFileAsTable = @parsePolygonCoordinateFileAsTable;
    io.parseEllipseCoordinateFileAsTable = @parseEllipseCoordinateFileAsTable;

    % Writing functions
    io.appendPolygonCoordinates = @appendPolygonCoordinates;
    io.appendEllipseCoordinates = @appendEllipseCoordinates;

    % Low-level utilities (exposed for advanced use)
    io.readExistingCoordinates = @readExistingCoordinates;
    io.atomicWriteCoordinates = @atomicWriteCoordinates;
    io.filterConflictingEntries = @filterConflictingEntries;

    % Constants
    io.POLYGON_HEADER = 'image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation';
    io.ELLIPSE_HEADER = 'image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle';
    io.POLYGON_NUMERIC_COUNT = 10;  % concentration + 8 coords + rotation
    io.ELLIPSE_NUMERIC_COUNT = 7;   % concentration + replicate + 5 ellipse params
    io.DEFAULT_COORDINATE_FILENAME = 'coordinates.txt';

    % Format strings for writing (matches header column order)
    % Polygon: imageName conc x1 y1 x2 y2 x3 y3 x4 y4 rotation
    io.POLYGON_WRITE_FMT = '%s %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.2f\n';
    % Ellipse: imageName conc replicate x y semiMajor semiMinor rotation
    io.ELLIPSE_WRITE_FMT = '%s %.0f %.0f %.2f %.2f %.4f %.4f %.2f\n';
end

%% =========================================================================
%% POLYGON COORDINATE FUNCTIONS
%% =========================================================================

function [polygonParams, found] = loadPolygonCoordinates(coordFile, imageName, numExpected)
    % Load polygon coordinates from coordinates.txt file for a specific image
    %
    % INPUTS:
    %   coordFile   - Full path to coordinates.txt file
    %   imageName   - Base image name to filter rows (case-insensitive)
    %   numExpected - Expected number of polygons (optional, for validation)
    %
    % OUTPUTS:
    %   polygonParams - Nx4x2 matrix of polygon vertices (N concentrations, 4 vertices, 2 coords)
    %   found         - Boolean indicating if file exists and contains data for image

    if nargin < 3
        numExpected = [];
    end

    polygonParams = [];
    found = false;

    if ~isfile(coordFile)
        return;
    end

    try
        fid = fopen(coordFile, 'rt');
        if fid == -1
            return;
        end
        cleanupObj = onCleanup(@() fclose(fid));

        % Read header
        headerLine = fgetl(fid);
        if ~ischar(headerLine)
            return;
        end

        % Read all data rows
        allRows = {};
        while ~feof(fid)
            line = fgetl(fid);
            if ischar(line) && ~isempty(strtrim(line))
                allRows{end+1} = line; %#ok<AGROW>
            end
        end

        % Parse rows matching this image
        [~, baseNameNoExt, ~] = fileparts(imageName);
        matchingRows = {};

        for i = 1:length(allRows)
            parts = strsplit(strtrim(allRows{i}));
            if length(parts) >= 11
                rowImageName = parts{1};
                [~, rowBaseNoExt, ~] = fileparts(rowImageName);

                if strcmpi(rowBaseNoExt, baseNameNoExt)
                    matchingRows{end+1} = allRows{i}; %#ok<AGROW>
                end
            end
        end

        if isempty(matchingRows)
            return;
        end

        % Parse matching rows into polygon matrix
        numRows = length(matchingRows);
        polygonParams = zeros(numRows, 4, 2);
        validCount = 0;

        for i = 1:numRows
            parts = strsplit(strtrim(matchingRows{i}));
            if length(parts) >= 11
                % Parse: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation
                concIdx = str2double(parts{2});

                % Validate concentration index
                if ~isempty(numExpected) && (concIdx < 0 || concIdx >= numExpected)
                    warning('coordinate_io:invalid_concentration', ...
                        'Invalid concentration index %.0f (expected 0-%d) for image %s - skipping row', ...
                        concIdx, numExpected - 1, imageName);
                    continue;
                end

                % Extract vertices (columns 3-10)
                coords = str2double(parts(3:10));

                % Validate coordinates are finite
                if any(~isfinite(coords))
                    warning('coordinate_io:invalid_coordinates', ...
                        'Invalid polygon coordinates for image %s, concentration %d - skipping row', ...
                        imageName, concIdx);
                    continue;
                end

                % Store as 4x2 matrix: [x1 y1; x2 y2; x3 y3; x4 y4]
                validCount = validCount + 1;
                polygonParams(validCount, :, :) = reshape(coords, 2, 4)';
            end
        end

        % Trim to valid rows
        if validCount > 0
            polygonParams = polygonParams(1:validCount, :, :);
            found = true;
        else
            polygonParams = [];
        end

        % Validate polygon count matches expected
        if found && ~isempty(numExpected) && size(polygonParams, 1) ~= numExpected
            warning('coordinate_io:polygon_count_mismatch', ...
                'Expected %d polygons, found %d for image %s', ...
                numExpected, size(polygonParams, 1), imageName);
        end

    catch ME
        warning('coordinate_io:polygon_load_error', ...
            'Failed to load polygon coordinates from %s: %s', coordFile, ME.message);
        polygonParams = [];
        found = false;
    end
end

function polygons = parsePolygonCoordinateFile(coordFile)
    % Parse entire polygon coordinate file into struct array
    %
    % INPUTS:
    %   coordFile - Full path to coordinates.txt file
    %
    % OUTPUTS:
    %   polygons - Struct array with fields:
    %              .imageName (string)
    %              .concentration (integer)
    %              .vertices (4x2 matrix)
    %              .rotation (degrees)

    polygons = struct('imageName', {}, 'concentration', {}, 'vertices', {}, 'rotation', {});

    if ~isfile(coordFile)
        return;
    end

    try
        fid = fopen(coordFile, 'rt');
        if fid == -1
            return;
        end
        cleanupObj = onCleanup(@() fclose(fid));

        % Check for header
        firstLine = fgetl(fid);
        if ~ischar(firstLine)
            return;
        end

        trimmed = strtrim(firstLine);
        isHeader = strncmpi(trimmed, 'image', 5);
        if ~isHeader
            fseek(fid, 0, 'bof');
        end

        % Read all data
        data = textscan(fid, '%s %f %f %f %f %f %f %f %f %f %f', ...
            'Delimiter', ' ', 'MultipleDelimsAsOne', true);

        if isempty(data) || isempty(data{1})
            return;
        end

        names = data{1};
        n = numel(names);

        % Pre-allocate struct array
        polygons(n).imageName = '';
        polygons(n).concentration = 0;
        polygons(n).vertices = zeros(4, 2);
        polygons(n).rotation = 0;

        validCount = 0;
        for i = 1:n
            coords = [data{3}(i), data{4}(i), data{5}(i), data{6}(i), ...
                      data{7}(i), data{8}(i), data{9}(i), data{10}(i)];

            if any(~isfinite(coords))
                warning('coordinate_io:invalid_polygon_entry', ...
                    'Skipping row %d with invalid coordinates', i);
                continue;
            end

            validCount = validCount + 1;
            [~, baseName, ~] = fileparts(names{i});
            polygons(validCount).imageName = baseName;
            polygons(validCount).concentration = data{2}(i);
            polygons(validCount).vertices = reshape(coords, 2, 4)';
            polygons(validCount).rotation = data{11}(i);
        end

        % Trim to valid entries
        polygons = polygons(1:validCount);

    catch ME
        warning('coordinate_io:parse_polygon_error', ...
            'Failed to parse polygon coordinates from %s: %s', coordFile, ME.message);
        polygons = struct('imageName', {}, 'concentration', {}, 'vertices', {}, 'rotation', {});
    end
end

function T = parsePolygonCoordinateFileAsTable(coordFile)
    % Parse polygon coordinate file and return as MATLAB table
    %
    % This function provides the same parsing as parsePolygonCoordinateFile
    % but returns a table for vectorized filtering operations in preview scripts.
    %
    % INPUTS:
    %   coordFile - Full path to coordinates.txt file
    %
    % OUTPUTS:
    %   T - Table with columns:
    %       image (string), concentration (double), x1-y4 (double), rotation (double)
    %       Returns empty table if file not found or parse fails.
    %
    % Example:
    %   coordIO = coordinate_io();
    %   T = coordIO.parsePolygonCoordinateFileAsTable('2_micropads/phone1/coordinates.txt');
    %   filtered = T(T.concentration == 3, :);
    %
    % See also: parsePolygonCoordinateFile (returns struct array)

    % Define expected column names
    expectedCols = {'image', 'concentration', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'rotation'};

    T = table('Size', [0, numel(expectedCols)], ...
              'VariableTypes', [{'string'}, repmat({'double'}, 1, 10)], ...
              'VariableNames', expectedCols);

    if ~isfile(coordFile)
        return;
    end

    try
        fid = fopen(coordFile, 'rt');
        if fid == -1
            return;
        end
        cleanupObj = onCleanup(@() fclose(fid));

        % Check for header
        firstLine = fgetl(fid);
        if ~ischar(firstLine)
            return;
        end

        trimmed = strtrim(firstLine);
        isHeader = strncmpi(trimmed, 'image', 5);
        if ~isHeader
            fseek(fid, 0, 'bof');
        end

        % Read all data using textscan
        data = textscan(fid, '%s %f %f %f %f %f %f %f %f %f %f', ...
            'Delimiter', ' ', 'MultipleDelimsAsOne', true);

        if isempty(data) || isempty(data{1})
            return;
        end

        names = data{1};
        n = numel(names);

        if n == 0
            return;
        end

        % Build table directly from parsed data
        T = table(string(names), data{2}, data{3}, data{4}, data{5}, data{6}, ...
                  data{7}, data{8}, data{9}, data{10}, data{11}, ...
                  'VariableNames', expectedCols);

        % Validate coordinates (remove rows with NaN in coordinate columns)
        coordCols = {'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'};
        invalidMask = false(height(T), 1);
        for i = 1:numel(coordCols)
            invalidMask = invalidMask | ~isfinite(T.(coordCols{i}));
        end

        if any(invalidMask)
            warning('coordinate_io:invalid_polygon_entries', ...
                'Skipping %d rows with invalid coordinates in %s', sum(invalidMask), coordFile);
            T = T(~invalidMask, :);
        end

    catch ME
        warning('coordinate_io:parse_polygon_table_error', ...
            'Failed to parse polygon coordinates as table from %s: %s', coordFile, ME.message);
        T = table('Size', [0, numel(expectedCols)], ...
                  'VariableTypes', [{'string'}, repmat({'double'}, 1, 10)], ...
                  'VariableNames', expectedCols);
    end
end

function appendPolygonCoordinates(phoneOutputDir, baseName, concentration, polygon, rotation, coordinateFileName)
    % Append polygon vertex coordinates to phone-level coordinates file with atomic write
    % Overwrites existing entry for same image/concentration combination
    %
    % INPUTS:
    %   phoneOutputDir     - Directory containing coordinates.txt (e.g., 2_micropads/phone1)
    %   baseName           - Base image name (without extension)
    %   concentration      - Concentration index (0-based)
    %   polygon            - 4x2 matrix of vertices [x1 y1; x2 y2; x3 y3; x4 y4]
    %   rotation           - Rotation angle in degrees (UI metadata)
    %   coordinateFileName - Name of coordinate file (default: 'coordinates.txt')

    if nargin < 6
        coordinateFileName = 'coordinates.txt';
    end

    coordFolder = phoneOutputDir;
    coordPath = fullfile(coordFolder, coordinateFileName);

    % Validate polygon
    if ~isnumeric(polygon) || size(polygon, 2) ~= 2
        warning('coordinate_io:invalid_polygon', ...
            'Polygon must be an Nx2 numeric array. Skipping write for %s.', baseName);
        return;
    end

    nVerts = size(polygon, 1);
    if nVerts ~= 4
        warning('coordinate_io:vertex_count', ...
            'Expected 4-vertex polygon; got %d. Proceeding may break downstream tools.', nVerts);
    end

    numericCount = 1 + 2 * nVerts + 1; % concentration, vertices, rotation

    % Build header dynamically
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

%% =========================================================================
%% ELLIPSE COORDINATE FUNCTIONS
%% =========================================================================

function [ellipseData, found] = loadEllipseCoordinates(coordFile, imageName)
    % Load ellipse coordinates from coordinates.txt file for a specific image
    %
    % INPUTS:
    %   coordFile  - Full path to coordinates.txt file
    %   imageName  - Base image name to filter rows (case-insensitive)
    %
    % OUTPUTS:
    %   ellipseData - Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
    %   found       - Boolean indicating if file exists and contains data for image

    ellipseData = [];
    found = false;

    if ~isfile(coordFile)
        return;
    end

    try
        fid = fopen(coordFile, 'rt');
        if fid == -1
            return;
        end
        cleanupObj = onCleanup(@() fclose(fid));

        % Read header
        headerLine = fgetl(fid);
        if ~ischar(headerLine)
            return;
        end

        % Read all data rows
        allRows = {};
        while ~feof(fid)
            line = fgetl(fid);
            if ischar(line) && ~isempty(strtrim(line))
                allRows{end+1} = line; %#ok<AGROW>
            end
        end

        % Parse rows matching this image
        [~, baseNameNoExt, ~] = fileparts(imageName);
        matchingRows = {};

        for i = 1:length(allRows)
            parts = strsplit(strtrim(allRows{i}));
            if length(parts) >= 8
                rowImageName = parts{1};
                [~, rowBaseNoExt, ~] = fileparts(rowImageName);

                if strcmpi(rowBaseNoExt, baseNameNoExt)
                    matchingRows{end+1} = allRows{i}; %#ok<AGROW>
                end
            end
        end

        if isempty(matchingRows)
            return;
        end

        % Parse matching rows into ellipse matrix
        numRows = length(matchingRows);
        ellipseData = zeros(numRows, 7);
        validCount = 0;

        for i = 1:numRows
            parts = strsplit(strtrim(matchingRows{i}));
            if length(parts) >= 8
                % Parse: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle
                values = str2double(parts(2:8));

                % Validate all numeric values are finite
                if any(~isfinite(values))
                    warning('coordinate_io:invalid_ellipse', ...
                        'Invalid ellipse data for image %s - skipping row', imageName);
                    continue;
                end

                % Validate ellipse geometry
                semiMajor = values(4);
                semiMinor = values(5);
                if semiMajor <= 0 || semiMinor <= 0
                    warning('coordinate_io:invalid_ellipse_axes', ...
                        'Invalid ellipse axes (semiMajor=%.2f, semiMinor=%.2f) for image %s - skipping row', ...
                        semiMajor, semiMinor, imageName);
                    continue;
                end

                validCount = validCount + 1;
                ellipseData(validCount, :) = values;
            end
        end

        % Trim to valid rows
        if validCount > 0
            ellipseData = ellipseData(1:validCount, :);
            found = true;
        else
            ellipseData = [];
        end

    catch ME
        warning('coordinate_io:ellipse_load_error', ...
            'Failed to load ellipse coordinates from %s: %s', coordFile, ME.message);
        ellipseData = [];
        found = false;
    end
end

function ellipses = parseEllipseCoordinateFile(coordFile)
    % Parse entire ellipse coordinate file into struct array
    %
    % INPUTS:
    %   coordFile - Full path to coordinates.txt file
    %
    % OUTPUTS:
    %   ellipses - Struct array with fields:
    %              .imageName (string)
    %              .concentration (integer)
    %              .replicate (integer)
    %              .x, .y (center coordinates)
    %              .semiMajorAxis, .semiMinorAxis (axis lengths)
    %              .rotationAngle (degrees)

    ellipses = struct('imageName', {}, 'concentration', {}, 'replicate', {}, ...
                      'x', {}, 'y', {}, 'semiMajorAxis', {}, 'semiMinorAxis', {}, ...
                      'rotationAngle', {});

    if ~isfile(coordFile)
        return;
    end

    try
        fid = fopen(coordFile, 'rt');
        if fid == -1
            return;
        end
        cleanupObj = onCleanup(@() fclose(fid));

        % Check for header
        firstLine = fgetl(fid);
        if ~ischar(firstLine)
            return;
        end

        trimmed = strtrim(firstLine);
        isHeader = strncmpi(trimmed, 'image', 5);
        if ~isHeader
            fseek(fid, 0, 'bof');
        end

        % Read all data
        data = textscan(fid, '%s %f %f %f %f %f %f %f', ...
            'Delimiter', ' ', 'MultipleDelimsAsOne', true);

        if isempty(data) || isempty(data{1})
            return;
        end

        names = data{1};
        n = numel(names);

        % Pre-allocate struct array
        ellipses(n).imageName = '';
        ellipses(n).concentration = 0;
        ellipses(n).replicate = 0;
        ellipses(n).x = 0;
        ellipses(n).y = 0;
        ellipses(n).semiMajorAxis = 0;
        ellipses(n).semiMinorAxis = 0;
        ellipses(n).rotationAngle = 0;

        validCount = 0;
        for i = 1:n
            x = data{4}(i);
            y = data{5}(i);
            semiMajor = data{6}(i);
            semiMinor = data{7}(i);

            if ~isfinite(x) || ~isfinite(y) || semiMajor <= 0 || semiMinor <= 0
                warning('coordinate_io:invalid_ellipse_entry', ...
                    'Skipping row %d with invalid ellipse parameters', i);
                continue;
            end

            validCount = validCount + 1;
            [~, baseName, ~] = fileparts(names{i});
            ellipses(validCount).imageName = baseName;
            ellipses(validCount).concentration = data{2}(i);
            ellipses(validCount).replicate = data{3}(i);
            ellipses(validCount).x = x;
            ellipses(validCount).y = y;
            ellipses(validCount).semiMajorAxis = semiMajor;
            ellipses(validCount).semiMinorAxis = semiMinor;
            ellipses(validCount).rotationAngle = data{8}(i);
        end

        % Trim to valid entries
        ellipses = ellipses(1:validCount);

    catch ME
        warning('coordinate_io:parse_ellipse_error', ...
            'Failed to parse ellipse coordinates from %s: %s', coordFile, ME.message);
        ellipses = struct('imageName', {}, 'concentration', {}, 'replicate', {}, ...
                          'x', {}, 'y', {}, 'semiMajorAxis', {}, 'semiMinorAxis', {}, ...
                          'rotationAngle', {});
    end
end

function T = parseEllipseCoordinateFileAsTable(coordFile)
    % Parse ellipse coordinate file and return as MATLAB table
    %
    % This function provides the same parsing as parseEllipseCoordinateFile
    % but returns a table for vectorized filtering operations in preview scripts.
    %
    % INPUTS:
    %   coordFile - Full path to coordinates.txt file
    %
    % OUTPUTS:
    %   T - Table with columns:
    %       image (string), concentration (double), replicate (double),
    %       x (double), y (double), semiMajorAxis (double),
    %       semiMinorAxis (double), rotationAngle (double)
    %       Returns empty table if file not found or parse fails.
    %
    % Example:
    %   coordIO = coordinate_io();
    %   T = coordIO.parseEllipseCoordinateFileAsTable('3_elliptical_regions/phone1/coordinates.txt');
    %   filtered = T(T.concentration == 3, :);
    %
    % See also: parseEllipseCoordinateFile (returns struct array)

    % Define expected column names
    expectedCols = {'image', 'concentration', 'replicate', 'x', 'y', 'semiMajorAxis', 'semiMinorAxis', 'rotationAngle'};

    T = table('Size', [0, numel(expectedCols)], ...
              'VariableTypes', [{'string'}, repmat({'double'}, 1, 7)], ...
              'VariableNames', expectedCols);

    if ~isfile(coordFile)
        return;
    end

    try
        fid = fopen(coordFile, 'rt');
        if fid == -1
            return;
        end
        cleanupObj = onCleanup(@() fclose(fid));

        % Check for header
        firstLine = fgetl(fid);
        if ~ischar(firstLine)
            return;
        end

        trimmed = strtrim(firstLine);
        isHeader = strncmpi(trimmed, 'image', 5);
        if ~isHeader
            fseek(fid, 0, 'bof');
        end

        % Read all data using textscan
        data = textscan(fid, '%s %f %f %f %f %f %f %f', ...
            'Delimiter', ' ', 'MultipleDelimsAsOne', true);

        if isempty(data) || isempty(data{1})
            return;
        end

        names = data{1};
        n = numel(names);

        if n == 0
            return;
        end

        % Build table directly from parsed data
        T = table(string(names), data{2}, data{3}, data{4}, data{5}, data{6}, data{7}, data{8}, ...
                  'VariableNames', expectedCols);

        % Validate ellipse parameters (remove rows with invalid values)
        invalidMask = ~isfinite(T.x) | ~isfinite(T.y) | ...
                      T.semiMajorAxis <= 0 | T.semiMinorAxis <= 0;

        if any(invalidMask)
            warning('coordinate_io:invalid_ellipse_entries', ...
                'Skipping %d rows with invalid ellipse parameters in %s', sum(invalidMask), coordFile);
            T = T(~invalidMask, :);
        end

    catch ME
        warning('coordinate_io:parse_ellipse_table_error', ...
            'Failed to parse ellipse coordinates as table from %s: %s', coordFile, ME.message);
        T = table('Size', [0, numel(expectedCols)], ...
                  'VariableTypes', [{'string'}, repmat({'double'}, 1, 7)], ...
                  'VariableNames', expectedCols);
    end
end

function appendEllipseCoordinates(phoneOutputDir, baseName, ellipseData, coordinateFileName)
    % Append ellipse coordinates to phone-level coordinates file with atomic write
    % Replaces all existing entries for the same image
    %
    % INPUTS:
    %   phoneOutputDir     - Directory containing coordinates.txt
    %   baseName           - Base image name (without extension)
    %   ellipseData        - Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
    %   coordinateFileName - Name of coordinate file (default: 'coordinates.txt')

    if nargin < 4
        coordinateFileName = 'coordinates.txt';
    end

    coordFolder = phoneOutputDir;
    coordPath = fullfile(coordFolder, coordinateFileName);

    header = 'image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle';
    numericCount = 7;

    scanFmt = ['%s' repmat(' %f', 1, numericCount)];
    writeFmt = '%s %.0f %.0f %.6f %.6f %.6f %.6f %.6f\n';

    % Read existing entries
    [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount);

    % Filter out entries for this image (remove all existing rows for same image)
    if ~isempty(existingNames)
        existingNames = existingNames(:);
        keepMask = ~strcmpi(existingNames, baseName);
        existingNames = existingNames(keepMask);
        if ~isempty(existingNums)
            existingNums = existingNums(keepMask, :);
        end
    end

    % Build new rows for this image (only valid ellipses with x > 0)
    validMask = ellipseData(:, 3) > 0;
    validData = ellipseData(validMask, :);

    if isempty(validData)
        % Still write file to remove old entries
        if ~isempty(existingNames)
            atomicWriteCoordinates(coordPath, header, existingNames, existingNums, writeFmt, coordFolder);
        end
        return;
    end

    numValid = size(validData, 1);
    newNames = cell(numValid, 1);
    newNames(:) = {baseName};
    newNums = validData;

    % Combine and write atomically
    allNames = [existingNames; newNames];
    allNums = [existingNums; newNums];

    atomicWriteCoordinates(coordPath, header, allNames, allNums, writeFmt, coordFolder);
end

%% =========================================================================
%% LOW-LEVEL I/O UTILITIES
%% =========================================================================

function [existingNames, existingNums] = readExistingCoordinates(coordPath, scanFmt, numericCount)
    % Read existing coordinates from file
    %
    % INPUTS:
    %   coordPath    - Full path to coordinates file
    %   scanFmt      - textscan format string (e.g., '%s %f %f %f ...')
    %   numericCount - Expected number of numeric columns
    %
    % OUTPUTS:
    %   existingNames - Cell array of image names
    %   existingNums  - Numeric matrix (Nx numericCount)

    existingNames = {};
    existingNums = zeros(0, numericCount);

    if ~isfile(coordPath)
        return;
    end

    fid = fopen(coordPath, 'rt');
    if fid == -1
        warning('coordinate_io:read_failed', 'Cannot open coordinates file for reading: %s', coordPath);
        return;
    end
    cleanupObj = onCleanup(@() fclose(fid));

    % Check for header line
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

    if ~isempty(data)
        if numel(data) >= 1 && ~isempty(data{1})
            existingNames = data{1};
        end
        if numel(data) >= 2 && ~isempty(data{2})
            nums = data{2};

            % Validate coordinate format
            if size(nums, 2) ~= numericCount
                error('coordinate_io:invalid_format', ...
                    ['Coordinate file has invalid format: %d numeric columns found, expected %d.\n' ...
                     'File: %s\n' ...
                     'NOTE: This project is in active development mode with no backward compatibility.\n' ...
                     'Delete the corrupted file and rerun the stage to regenerate.'], ...
                    size(nums, 2), numericCount, coordPath);
            end

            existingNums = nums;

            % Validate numeric content (skip rotation column for NaN check)
            % Polygon format (10 cols): cols 2:9 are x1,y1,x2,y2,x3,y3,x4,y4
            % Ellipse format (7 cols): cols 3:6 are x,y,semiMajor,semiMinor
            if numericCount == 10
                coordCols = 2:9;
            elseif numericCount == 7
                coordCols = 3:6;
            else
                coordCols = 2:(numericCount-1);
            end
            invalidRows = any(~isfinite(existingNums(:, coordCols)), 2);
            if any(invalidRows)
                warning('coordinate_io:corrupt_coords', ...
                    'Found %d rows with invalid coordinates in %s. Skipping corrupted entries.', ...
                    sum(invalidRows), coordPath);
                validMask = ~invalidRows;
                existingNames = existingNames(validMask);
                existingNums = existingNums(validMask, :);
            end
        end
    end

    % Ensure consistent row counts
    if ~isempty(existingNames) && ~isempty(existingNums)
        rows = min(numel(existingNames), size(existingNums, 1));
        if size(existingNums, 1) ~= numel(existingNames)
            existingNames = existingNames(1:rows);
            existingNums = existingNums(1:rows, :);
        end

        % Remove empty name entries
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
    % Remove entries with same image name AND concentration index
    %
    % INPUTS:
    %   existingNames - Cell array of existing image names
    %   existingNums  - Numeric matrix of existing data
    %   newName       - New image name to check against
    %   concentration - Concentration index to check (column 1 of existingNums)
    %
    % OUTPUTS:
    %   filteredNames - Filtered cell array
    %   filteredNums  - Filtered numeric matrix

    if isempty(existingNames)
        filteredNames = existingNames;
        filteredNums = existingNums;
        return;
    end

    existingNames = existingNames(:);
    sameImageMask = strcmpi(existingNames, newName);
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
    % Write coordinates atomically using temp file + movefile pattern
    %
    % INPUTS:
    %   coordPath   - Final destination path for coordinates file
    %   header      - Header line string
    %   names       - Cell array of image names
    %   nums        - Numeric matrix of coordinate data
    %   writeFmt    - fprintf format string for each row
    %   coordFolder - Folder for temp file creation

    % Ensure folder exists
    if ~isfolder(coordFolder)
        mkdir(coordFolder);
    end

    tmpPath = tempname(coordFolder);

    fid = fopen(tmpPath, 'wt');
    if fid == -1
        error('coordinate_io:write_failed', ...
              'Cannot open temp coordinates file for writing: %s\nCheck folder permissions.', tmpPath);
    end

    % Use onCleanup to guarantee file closure even if error occurs
    cleanupFile = onCleanup(@() safeCloseFile(fid));

    fprintf(fid, '%s\n', header);

    for j = 1:numel(names)
        rowVals = nums(j, :);
        rowVals(isnan(rowVals)) = 0;
        fprintf(fid, writeFmt, names{j}, rowVals);
    end

    fclose(fid);
    delete(cleanupFile);  % Prevent double-close

    % Atomic move
    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        warning('coordinate_io:move_failed', ...
            'Failed to move temp file to coordinates.txt: %s (%s). Attempting fallback copy.', msg, msgid);
        [copied, cmsg, ~] = copyfile(tmpPath, coordPath, 'f');
        if ~copied
            if isfile(tmpPath)
                delete(tmpPath);
            end
            error('coordinate_io:write_failed', ...
                'Cannot write coordinates to %s: movefile failed (%s), copyfile failed (%s).', ...
                coordPath, msg, cmsg);
        end
        if isfile(tmpPath)
            delete(tmpPath);
        end
    end
end

function safeCloseFile(fid)
    % Safely close file handle if still open
    %
    % Used by onCleanup to prevent double-close errors.

    if fid ~= -1
        try
            fclose(fid);
        catch
            % File already closed or invalid handle - ignore
        end
    end
end
