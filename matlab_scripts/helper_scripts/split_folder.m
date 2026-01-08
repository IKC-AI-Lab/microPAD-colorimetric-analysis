function split_folder(folderPath, numSplits, varargin)
    % SPLIT_FOLDER Split a folder into N equal subdirectories
    %
    % This function splits any folder into N numbered subfolders with equal
    % distribution of files. It can split directly or apply an existing split
    % from a reference folder.
    %
    % INPUTS:
    %   folderPath - Absolute or relative path to the folder to split
    %   numSplits  - Number of equal splits to create (must be >= 2)
    %
    % OPTIONAL NAME-VALUE PARAMETERS:
    %   'ReferenceFolder' - Path to folder with existing split to replicate
    %                       If provided, uses the same file assignments as reference
    %                       instead of creating new assignments. Useful for folders
    %                       with only subdirectories or related datasets.
    %                       Default: '' (create new split)
    %
    % BEHAVIOR:
    %   - Auto-detects all files at the root level (or uses reference assignments)
    %   - Auto-detects all subdirectories
    %   - Replicates directory structure in each split folder
    %   - Distributes files evenly across splits
    %   - Auto-detects and splits coordinates.txt if present
    %   - Creates numbered folders (1, 2, 3, ...)
    %   - Moves derived files (e.g., *_aug_*.png) with their source files
    %
    % EXAMPLES:
    %   % Direct split: Split folder with files at root level
    %   split_folder('1_dataset/lateral_flow_assay_v3i', 4)
    %
    %   % Apply existing split: Use assignments from another folder
    %   split_folder('2_micropads/lateral_flow_assay_v3i', 4, ...
    %                'ReferenceFolder', '1_dataset/lateral_flow_assay_v3i')
    %
    %   % Split any folder into 3 equal parts
    %   split_folder('C:\data\my_folder', 3)
    %
    % See also: coordinate_io

    %% Parse input arguments
    parser = inputParser();
    addRequired(parser, 'folderPath', @(x) validateattributes(x, {'char', 'string'}, {'scalartext'}));
    addRequired(parser, 'numSplits', @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>=', 2}));
    addParameter(parser, 'ReferenceFolder', '', @(x) validateattributes(x, {'char', 'string'}, {'scalartext'}));
    parse(parser, folderPath, numSplits, varargin{:});

    referenceFolder = char(parser.Results.ReferenceFolder);
    useReferenceFolder = ~isempty(referenceFolder);

    %% Input Validation
    % Validate folderPath exists
    if ~isfolder(folderPath)
        error('split_folder:folder_not_found', ...
            'Folder not found: %s', folderPath);
    end

    % Validate referenceFolder if provided
    if useReferenceFolder && ~isfolder(referenceFolder)
        error('split_folder:reference_not_found', ...
            'Reference folder not found: %s', referenceFolder);
    end

    %% Get assignments (either from reference or create new)
    coordIO = coordinate_io();

    if useReferenceFolder
        % Use existing split from reference folder
        assignments = read_existing_split(referenceFolder, numSplits, coordIO);
        fprintf('Using split assignments from reference: %s\n', referenceFolder);
        for i = 1:length(assignments)
            fprintf('  Folder %d: %d base names\n', i, length(assignments{i}));
        end
        fprintf('\n');
    else
        % Create new split assignments from root-level files
        assignments = create_new_split(folderPath, numSplits, coordIO);
    end

    %% Scan target folder structure
    allItems = dir(folderPath);
    isDirItem = [allItems.isdir];
    dirNames = {allItems(isDirItem).name};

    % Filter out numbered directories and . and ..
    subdirs = {};
    for i = 1:length(dirNames)
        name = dirNames{i};
        if ~ismember(name, {'.', '..'}) && isnan(str2double(name))
            subdirs{end+1} = name; %#ok<AGROW>
        end
    end

    % Check for coordinates.txt
    hasCoordinates = isfile(fullfile(folderPath, 'coordinates.txt'));

    fprintf('  Found %d subdirectories\n', length(subdirs));
    if hasCoordinates
        fprintf('  Found coordinates.txt - will split automatically\n');
    end
    fprintf('\n');

    %% Create split folder structure
    fprintf('Creating split folder structure...\n');

    for i = 1:numSplits
        splitDir = fullfile(folderPath, num2str(i));
        if ~isfolder(splitDir)
            mkdir(splitDir);
        end

        % Replicate subdirectory structure
        for j = 1:length(subdirs)
            subDirPath = fullfile(splitDir, subdirs{j});
            if ~isfolder(subDirPath)
                mkdir(subDirPath);
            end
        end
    end

    fprintf('  Created %d split folders with subdirectories\n\n', numSplits);

    %% Move files
    fprintf('Moving files to split folders...\n');
    movedCount = 0;

    % Move files at root level
    for i = 1:numSplits
        splitDir = fullfile(folderPath, num2str(i));

        for j = 1:length(assignments{i})
            baseName = assignments{i}{j};

            % Check for files at root matching this base name
            allFiles = dir(folderPath);
            for f = 1:length(allFiles)
                if allFiles(f).isdir
                    continue;
                end

                if strcmpi(allFiles(f).name, 'coordinates.txt')
                    continue;  % Handle coordinates separately
                end

                [~, fileBase, fileExt] = fileparts(allFiles(f).name);
                fileName = [fileBase fileExt];

                % Check if matches baseName with delimiter-aware logic
                % Note: Case-insensitive matching for Windows filesystem compatibility
                if startsWith(fileBase, baseName, 'IgnoreCase', true)
                    remainder = fileBase(length(baseName)+1:end);
                    if isempty(remainder) || remainder(1) == '_'
                        srcFile = fullfile(folderPath, fileName);
                        dstFile = fullfile(splitDir, fileName);

                        if isfile(srcFile)
                            [success, msg] = movefile(srcFile, dstFile);
                            if ~success
                                warning('split_folder:move_failed', ...
                                    'Failed to move %s: %s', fileName, msg);
                            else
                                movedCount = movedCount + 1;
                            end
                        end
                    end
                end
            end
        end
    end

    % Move files from subdirectories (including derived files)
    if ~isempty(subdirs)
        for i = 1:numSplits
            splitDir = fullfile(folderPath, num2str(i));

            for j = 1:length(assignments{i})
                baseName = assignments{i}{j};

                for k = 1:length(subdirs)
                    srcSubDir = fullfile(folderPath, subdirs{k});
                    dstSubDir = fullfile(splitDir, subdirs{k});

                    if ~isfolder(srcSubDir)
                        continue;
                    end

                    % Get all files and match with delimiter awareness
                    allFiles = dir(srcSubDir);

                    for f = 1:length(allFiles)
                        if allFiles(f).isdir
                            continue;
                        end

                        [~, fileBase, fileExt] = fileparts(allFiles(f).name);
                        fileName = [fileBase fileExt];

                        % Valid match if exact or starts with baseName_
                        % Note: Case-insensitive matching for Windows filesystem compatibility
                        if startsWith(fileBase, baseName, 'IgnoreCase', true)
                            remainder = fileBase(length(baseName)+1:end);
                            if isempty(remainder) || remainder(1) == '_'
                                srcFile = fullfile(srcSubDir, fileName);
                                dstFile = fullfile(dstSubDir, fileName);

                                if isfile(srcFile)
                                    [success, msg] = movefile(srcFile, dstFile);
                                    if ~success
                                        warning('split_folder:move_failed', ...
                                            'Failed to move %s: %s', fileName, msg);
                                    else
                                        movedCount = movedCount + 1;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    fprintf('  Moved %d files total\n\n', movedCount);

    % Warn if no files were moved (may indicate mismatch in reference mode)
    if movedCount == 0
        if useReferenceFolder
            warning('split_folder:no_files_moved', ...
                'No files were moved. Verify that base names in target folder match reference assignments.');
        else
            warning('split_folder:no_files_moved', ...
                'No files were moved. Check that files exist in the folder or subdirectories.');
        end
    end

    %% Split coordinates.txt if present
    if hasCoordinates
        fprintf('Splitting coordinates.txt...\n');
        coordFile = fullfile(folderPath, 'coordinates.txt');

        % Validate coordinate format (only quad format supported)
        fid = fopen(coordFile, 'rt');
        if fid == -1
            warning('split_folder:cannot_read_coords', ...
                'Cannot open coordinates.txt for reading - skipping coordinate split');
        else
            headerLine = fgetl(fid);
            fclose(fid);

            % Check if it's quad format (has x1, y1, x2, y2, etc.)
            if ~contains(headerLine, 'x1') || ~contains(headerLine, 'y1')
                error('split_folder:unsupported_coords', ...
                    ['This function only supports Stage 2 (quad) coordinate files.\n' ...
                     'Found ellipse-format or unknown coordinates in: %s\n' ...
                     'Quad format expected: image concentration x1 y1 x2 y2 x3 y3 x4 y4 rotation'], ...
                    coordFile);
            end
        end

        coordTable = coordIO.parseQuadCoordinateFileAsTable(coordFile);

        % Validate table structure
        if ~isempty(coordTable)
            requiredCols = {'image', 'concentration', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'rotation'};
            if ~all(ismember(requiredCols, coordTable.Properties.VariableNames))
                warning('split_folder:invalid_coords', ...
                    'Coordinate file missing required columns - skipping coordinate split');
                coordTable = [];
            end
        end

        % Split coordinates for each folder
        writeSuccess = true;
        if ~isempty(coordTable)
            for i = 1:numSplits
                % Filter coordinates with delimiter-aware matching
                mask = false(height(coordTable), 1);
                for j = 1:length(assignments{i})
                    exactMatch = strcmpi(coordTable.image, assignments{i}{j});
                    prefixPattern = strcat(assignments{i}{j}, '_');
                    prefixMatch = strncmpi(coordTable.image, prefixPattern, length(prefixPattern));
                    mask = mask | exactMatch | prefixMatch;
                end

                splitCoords = coordTable(mask, :);

                if ~isempty(splitCoords) && height(splitCoords) > 0
                    targetCoordFile = fullfile(folderPath, num2str(i), 'coordinates.txt');
                    try
                        write_quad_coordinates_table(targetCoordFile, splitCoords, coordIO);
                    catch ME
                        warning('split_folder:write_failed', ...
                            'Failed to write %s: %s', targetCoordFile, ME.message);
                        writeSuccess = false;
                    end
                end
            end

            % Remove original coordinates.txt only if all writes succeeded
            if writeSuccess
                delete(coordFile);
                fprintf('  Split coordinates into %d files\n\n', numSplits);
            else
                warning('split_folder:partial_split', ...
                    'Partial split failure - original coordinates.txt preserved');
            end
        end
    end

    %% Clean up empty subdirectories
    fprintf('Cleaning up empty subdirectories...\n');
    removedCount = 0;

    for k = 1:length(subdirs)
        dirPath = fullfile(folderPath, subdirs{k});
        if isfolder(dirPath)
            items = dir(dirPath);
            items = items(~ismember({items.name}, {'.', '..'}));
            if isempty(items)
                rmdir(dirPath);
                removedCount = removedCount + 1;
            end
        end
    end

    fprintf('  Removed %d empty subdirectories\n\n', removedCount);

    fprintf('Folder split complete!\n');
    fprintf('Created %d split folders in: %s\n', numSplits, folderPath);
end

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

function assignments = create_new_split(folderPath, numSplits, coordIO)
    % Create new split assignments from root-level files

    fprintf('Scanning folder structure: %s\n', folderPath);

    allItems = dir(folderPath);
    isFile = ~[allItems.isdir];
    fileNames = {allItems(isFile).name};

    % Filter out coordinates.txt
    coordIdx = strcmpi(fileNames, 'coordinates.txt');

    % Filter to image files only
    validExts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    isImageFile = false(size(fileNames));
    for i = 1:length(fileNames)
        [~, ~, ext] = fileparts(fileNames{i});
        isImageFile(i) = any(strcmpi(ext, validExts));
    end

    filesToSplit = fileNames(isImageFile & ~coordIdx);

    % Validate we have files to split
    if isempty(filesToSplit)
        error('split_folder:no_files', ...
            'No image files found to split in: %s', folderPath);
    end

    fprintf('  Found %d files to split\n', length(filesToSplit));

    % Warn if fewer files than splits
    if length(filesToSplit) < numSplits
        warning('split_folder:fewer_files_than_splits', ...
            'Only %d files to split across %d folders. Some folders will be empty.', ...
            length(filesToSplit), numSplits);
    end

    % Distribute files evenly
    sortedFiles = sort(filesToSplit);
    assignments = distribute_files(sortedFiles, numSplits, coordIO);

    % Display distribution
    fprintf('File distribution:\n');
    for i = 1:numSplits
        fprintf('  Folder %d: %d files\n', i, length(assignments{i}));
    end
    fprintf('\n');
end

function assignments = read_existing_split(referenceFolder, numSplits, coordIO)
    % Read split assignments from an existing reference folder

    fprintf('Reading split assignments from: %s\n', referenceFolder);

    refItems = dir(referenceFolder);
    isDirItem = [refItems.isdir];
    dirNames = {refItems(isDirItem).name};

    % Find numbered directories
    splitDirs = {};
    for i = 1:length(dirNames)
        if ~ismember(dirNames{i}, {'.', '..'}) && ~isnan(str2double(dirNames{i}))
            splitDirs{end+1} = dirNames{i}; %#ok<AGROW>
        end
    end

    if isempty(splitDirs)
        error('split_folder:no_splits_in_reference', ...
            'No numbered split folders found in reference: %s', referenceFolder);
    end

    % Sort split directories numerically
    splitNums = cellfun(@str2double, splitDirs);
    [~, sortIdx] = sort(splitNums);
    splitDirs = splitDirs(sortIdx);
    actualNumSplits = length(splitDirs);

    if actualNumSplits ~= numSplits
        error('split_folder:split_count_mismatch', ...
            'Reference has %d splits but requested %d splits', actualNumSplits, numSplits);
    end

    fprintf('  Found %d split folders: %s\n', numSplits, strjoin(splitDirs, ', '));

    % Extract base names from each split
    assignments = cell(1, numSplits);

    for i = 1:numSplits
        splitDir = fullfile(referenceFolder, splitDirs{i});
        files = dir(splitDir);

        baseNames = {};
        for j = 1:length(files)
            if ~files(j).isdir
                baseName = coordIO.strip_image_extension(files(j).name);
                baseNames{end+1} = baseName; %#ok<AGROW>
            end
        end

        assignments{i} = unique(baseNames);
    end
end

function assignments = distribute_files(fileNames, numSplits, coordIO)
    % Distribute files evenly across splits and return base names
    %
    % INPUTS:
    %   fileNames - Cell array of filenames
    %   numSplits - Number of target splits
    %   coordIO   - coordinate_io module for extension stripping
    %
    % OUTPUTS:
    %   assignments - Cell array where assignments{i} contains base names for split i

    numFiles = length(fileNames);
    baseCount = floor(numFiles / numSplits);
    extras = mod(numFiles, numSplits);

    assignments = cell(1, numSplits);
    startIdx = 1;

    for i = 1:numSplits
        % First 'extras' splits get one extra file
        count = baseCount;
        if i <= extras
            count = count + 1;
        end

        endIdx = startIdx + count - 1;
        files = fileNames(startIdx:endIdx);

        % Convert to base names (strip extensions)
        baseNames = cellfun(@(x) coordIO.strip_image_extension(x), files, 'UniformOutput', false);
        assignments{i} = baseNames;

        startIdx = endIdx + 1;
    end
end

function write_quad_coordinates_table(targetFile, coordTable, coordIO)
    % Write quad coordinates from table to file using atomic write
    %
    % INPUTS:
    %   targetFile - Full path to target coordinates.txt file
    %   coordTable - Table with columns: image, concentration, x1-y4, rotation
    %   coordIO    - coordinate_io() module instance

    % Extract data from table
    names = cellstr(coordTable.image);
    nums = [coordTable.concentration, ...
            coordTable.x1, coordTable.y1, ...
            coordTable.x2, coordTable.y2, ...
            coordTable.x3, coordTable.y3, ...
            coordTable.x4, coordTable.y4, ...
            coordTable.rotation];

    % Use atomic write from coordinate_io
    targetFolder = fileparts(targetFile);
    coordIO.atomicWriteCoordinates(targetFile, coordIO.QUAD_HEADER, names, nums, ...
                                   coordIO.QUAD_WRITE_FMT, targetFolder);
end
