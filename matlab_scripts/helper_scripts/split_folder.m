function split_folder(folderPath, numSplits, varargin)
    % SPLIT_FOLDER Split subfolders into N equal parts with linked folder support
    %
    % Takes a folder containing subfolders with files, pools all files together,
    % and distributes them into N new split folders. Can also split linked folders
    % (e.g., 2_micropads, 3_elliptical_regions) using the same base name assignments.
    %
    % INPUTS:
    %   folderPath - Path to the primary folder containing subfolders to split
    %   numSplits  - Number of equal splits to create (must be >= 2)
    %
    % OPTIONAL NAME-VALUE PARAMETERS:
    %   'Shuffle'         - If true, randomly shuffle files before distributing
    %                       Default: false (alphabetical order)
    %
    %   'ShuffleSeed'     - Random seed for reproducible shuffles (requires Shuffle=true)
    %                       Default: [] (non-reproducible)
    %
    %   'LinkedFolders'   - Cell array of folders to split using same assignments
    %                       Default: {} (no linked folders)
    %
    % BEHAVIOR:
    %   - Collects all image files from all subfolders in primary folder
    %   - Prefixes files with subfolder name to prevent collisions
    %     (e.g., iphone_15/IMG_0001.jpg -> iphone_15_IMG_0001.jpg)
    %   - Creates split folders INSIDE the parent folder with shared timestamp
    %   - Distributes files evenly across splits
    %   - Removes original subfolders after successful split
    %   - Unmatched coordinates saved to unmatched_coordinates.txt
    %   - For linked folders:
    %     * Prefixes all files and coordinate entries with phone subfolder
    %     * Splits coordinates.txt by prefixed base name matching
    %     * Splits con_N/ subfolders by prefixed base name matching
    %     * Preserves con_N/ folder structure inside each split
    %     * Saves unmatched images to unmatched_images.txt
    %
    % EXAMPLES:
    %   % Simple split
    %   split_folder('1_dataset', 4, 'Shuffle', true)
    %
    %   % Split with linked folders (coordinates follow image assignments)
    %   split_folder('1_dataset', 4, 'Shuffle', true, 'ShuffleSeed', 42, ...
    %                'LinkedFolders', {'2_micropads', '3_elliptical_regions'})
    %
    % LINKED FOLDER FORMATS:
    %   - 2_micropads: quad coordinates, con_N/ folders with cropped quads
    %   - 3_elliptical_regions: ellipse coordinates, con_N/ folders with patches
    %   - Prefixed base name linking:
    %       iphone_15/IMG_4754.jpg -> iphone_15_IMG_4754.jpg
    %       coordinates: iphone_15_IMG_4754 (prefixed)
    %       con_0/: iphone_15_IMG_4754_con_0.png (prefixed)
    %
    % REQUIREMENTS:
    %   - Phone subfolder names must be IDENTICAL across all folders
    %     (e.g., if 1_dataset has 'iphone_15', linked folders must also have 'iphone_15')
    %   - Files with same base name but different extensions will trigger a warning
    %     and may be split inconsistently
    %
    % See also: coordinate_io

    %% Parse input arguments
    parser = inputParser();
    addRequired(parser, 'folderPath', @(x) ischar(x) || isstring(x));
    addRequired(parser, 'numSplits', @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer', '>=', 2}));
    addParameter(parser, 'Shuffle', false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
    addParameter(parser, 'ShuffleSeed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isfinite(x)));
    addParameter(parser, 'LinkedFolders', {}, @(x) iscell(x) || isempty(x));
    parse(parser, folderPath, numSplits, varargin{:});

    folderPath = char(folderPath);
    shuffleFiles = logical(parser.Results.Shuffle);
    shuffleSeed = parser.Results.ShuffleSeed;
    linkedFolders = parser.Results.LinkedFolders;

    % Ensure linkedFolders is cell array
    if ischar(linkedFolders) || isstring(linkedFolders)
        linkedFolders = {char(linkedFolders)};
    elseif ~iscell(linkedFolders)
        linkedFolders = {};
    end

    %% Input Validation
    if ~isfolder(folderPath)
        error('split_folder:folder_not_found', 'Folder not found: %s', folderPath);
    end

    if ~isempty(shuffleSeed) && ~shuffleFiles
        warning('split_folder:seed_without_shuffle', 'ShuffleSeed ignored because Shuffle is false');
    end

    % Filter linked folders - skip non-existent ones with warning
    validLinkedFolders = {};
    for i = 1:length(linkedFolders)
        if isfolder(linkedFolders{i})
            validLinkedFolders{end+1} = linkedFolders{i}; %#ok<AGROW>
        else
            warning('split_folder:linked_folder_not_found', ...
                    'Linked folder not found, skipping: %s', linkedFolders{i});
        end
    end
    linkedFolders = validLinkedFolders;

    %% Analyze primary folder structure
    coordIO = coordinate_io();

    % Find subfolders
    allItems = dir(folderPath);
    subdirs = {};
    for i = 1:length(allItems)
        if allItems(i).isdir && ~ismember(allItems(i).name, {'.', '..'})
            % Skip any existing split folders
            if ~startsWith(allItems(i).name, 'split_')
                subdirs{end+1} = allItems(i).name; %#ok<AGROW>
            end
        end
    end

    if isempty(subdirs)
        error('split_folder:no_subfolders', 'No subfolders found to split in: %s', folderPath);
    end

    fprintf('=== SPLIT FOLDER ===\n');
    fprintf('Primary folder: %s\n', folderPath);
    fprintf('  Subfolders: %s\n', strjoin(subdirs, ', '));
    if ~isempty(linkedFolders)
        fprintf('  Linked folders: %s\n', strjoin(linkedFolders, ', '));
    end

    %% Collect all files from primary folder subfolders
    % Files are prefixed with phone subfolder name to prevent collisions
    % e.g., iphone_15/IMG_0001.jpg -> iphone_15_IMG_0001.jpg
    allFiles = {};  % {fullpath, originalName, prefixedName, prefixedBaseName, phonePrefix}
    for s = 1:length(subdirs)
        subdir = subdirs{s};
        srcSubDir = fullfile(folderPath, subdir);
        files = get_image_files(srcSubDir);

        for f = 1:length(files)
            originalName = files{f};
            [~, nameNoExt, ext] = fileparts(originalName);
            prefixedName = sprintf('%s_%s%s', subdir, nameNoExt, ext);
            prefixedBaseName = sprintf('%s_%s', subdir, nameNoExt);

            allFiles{end+1} = struct('path', fullfile(srcSubDir, originalName), ...
                                     'originalName', originalName, ...
                                     'prefixedName', prefixedName, ...
                                     'prefixedBaseName', prefixedBaseName, ...
                                     'phonePrefix', subdir); %#ok<AGROW>
        end
        fprintf('  %s: %d files\n', subdir, length(files));
    end

    totalFiles = length(allFiles);
    fprintf('  Total files: %d\n', totalFiles);

    if totalFiles == 0
        error('split_folder:no_files', 'No image files found in subfolders');
    end

    %% Generate split folder names (shared timestamp)
    splitNames = generate_split_names(numSplits);
    fprintf('\n');

    %% Shuffle/sort and distribute files
    if shuffleFiles
        oldRng = rng;
        if ~isempty(shuffleSeed)
            rng(shuffleSeed, 'twister');
            fprintf('Shuffling files (seed=%d)...\n', shuffleSeed);
        else
            rng('shuffle');
            fprintf('Shuffling files (random)...\n');
        end
        shuffleIdx = randperm(length(allFiles));
        allFiles = allFiles(shuffleIdx);
        rng(oldRng);
    else
        % Sort by prefixed filename
        fileNames = cellfun(@(x) x.prefixedName, allFiles, 'UniformOutput', false);
        [~, sortIdx] = sort(fileNames);
        allFiles = allFiles(sortIdx);
    end

    %% Check for duplicate base names (same name, different extension)
    allBaseNames = cellfun(@(x) x.prefixedBaseName, allFiles, 'UniformOutput', false);
    [uniqueBaseNames, ~, ic] = unique(allBaseNames);
    if length(uniqueBaseNames) < length(allFiles)
        % Find duplicates
        counts = accumarray(ic, 1);
        duplicateIdx = find(counts > 1);
        duplicateNames = uniqueBaseNames(duplicateIdx);
        warning('split_folder:duplicate_base_names', ...
                '%d base names have multiple files (different extensions). ' + ...
                'These may be split inconsistently: %s', ...
                length(duplicateNames), strjoin(duplicateNames(1:min(3,end)), ', '));
    end

    %% Build prefixed base name -> split index mapping
    % Uses prefixed names (e.g., 'iphone_15_IMG_0001') to avoid collisions
    % Note: If duplicate base names exist, last assignment wins
    assignments = distribute_to_splits(length(allFiles), numSplits);
    baseNameToSplit = containers.Map('KeyType', 'char', 'ValueType', 'double');

    for i = 1:numSplits
        for j = assignments{i}
            prefixedBaseName = allFiles{j}.prefixedBaseName;
            baseNameToSplit(prefixedBaseName) = i;
        end
    end

    fprintf('Built split assignments for %d unique prefixed base names\n', length(baseNameToSplit));

    % Show sample base names for verification
    keys = baseNameToSplit.keys();
    if length(keys) > 3
        fprintf('  Samples: %s, %s, %s, ...\n\n', keys{1}, keys{2}, keys{3});
    elseif ~isempty(keys)
        fprintf('  All: %s\n\n', strjoin(keys, ', '));
    else
        fprintf('\n');
    end

    %% Process primary folder
    fprintf('Processing primary folder: %s\n', folderPath);
    process_primary_folder(folderPath, splitNames, allFiles, assignments, subdirs, coordIO);

    %% Process linked folders
    if ~isempty(linkedFolders)
        fprintf('\n');
        for i = 1:length(linkedFolders)
            linkedFolder = linkedFolders{i};
            fprintf('Processing linked folder: %s\n', linkedFolder);
            process_linked_folder(linkedFolder, splitNames, baseNameToSplit, coordIO);
        end
    end

    %% Clean up original subfolders
    fprintf('\nCleaning up original subfolders...\n');
    for s = 1:length(subdirs)
        subdir = subdirs{s};
        subPath = fullfile(folderPath, subdir);
        cleanup_folder(subPath);
    end

    fprintf('\n=== Split complete! ===\n');
    fprintf('Split %d files into %d folders\n', totalFiles, numSplits);
    if ~isempty(linkedFolders)
        fprintf('Processed %d linked folders\n', length(linkedFolders));
    end
end

%% =========================================================================
%% PROCESSING FUNCTIONS
%% =========================================================================

function process_primary_folder(folderPath, splitNames, allFiles, assignments, phoneSubdirs, coordIO)
    % Process primary folder: create splits, move files with prefixed names, split coordinates

    numSplits = length(splitNames);

    % Create split folders
    fprintf('  Creating split folders...\n');
    for i = 1:numSplits
        splitDir = fullfile(folderPath, splitNames{i});
        if ~isfolder(splitDir)
            mkdir(splitDir);
        end
    end

    % Distribute files to splits with prefixed names
    fprintf('  Distributing files (with phone prefix)...\n');
    totalMoved = 0;
    for i = 1:numSplits
        splitDir = fullfile(folderPath, splitNames{i});
        count = 0;

        for j = assignments{i}
            srcFile = allFiles{j}.path;
            % Use prefixed name to prevent collisions
            dstFile = fullfile(splitDir, allFiles{j}.prefixedName);

            if isfile(srcFile)
                movefile(srcFile, dstFile);
                count = count + 1;
                totalMoved = totalMoved + 1;
            end
        end

        fprintf('    %s: %d files\n', splitNames{i}, count);
    end

    % Split and merge coordinates.txt from all phone subfolders
    split_coordinates_primary_merged(folderPath, phoneSubdirs, splitNames, coordIO);
end

function process_linked_folder(linkedFolder, splitNames, baseNameToSplit, coordIO)
    % Process linked folder: split coordinates and con_N/ folders
    % Structure mirrors primary folder - flat splits, no phone subfolders
    % All files and coordinates are prefixed with phone subfolder name

    numSplits = length(splitNames);

    % Find phone subfolders
    allItems = dir(linkedFolder);
    phoneSubdirs = {};
    for i = 1:length(allItems)
        if allItems(i).isdir && ~ismember(allItems(i).name, {'.', '..'})
            if ~startsWith(allItems(i).name, 'split_')
                phoneSubdirs{end+1} = allItems(i).name; %#ok<AGROW>
            end
        end
    end

    if isempty(phoneSubdirs)
        fprintf('  No phone subfolders found\n');
        return;
    end

    fprintf('  Phone subfolders: %s\n', strjoin(phoneSubdirs, ', '));

    % Create split folders
    for i = 1:numSplits
        splitDir = fullfile(linkedFolder, splitNames{i});
        if ~isfolder(splitDir)
            mkdir(splitDir);
        end
    end

    % Collect ALL coordinates from ALL phone subfolders with phone prefix
    allCoordTables = {};
    isQuadFormat = [];
    for p = 1:length(phoneSubdirs)
        phoneDir = fullfile(linkedFolder, phoneSubdirs{p});
        coordFile = fullfile(phoneDir, 'coordinates.txt');

        if isfile(coordFile)
            [coordTable, isQuad] = parse_coordinate_file(coordFile, coordIO);
            if ~isempty(coordTable)
                % Prefix image names with phone subfolder
                prefixedImages = cell(height(coordTable), 1);
                for i = 1:height(coordTable)
                    prefixedImages{i} = sprintf('%s_%s', phoneSubdirs{p}, char(coordTable.image(i)));
                end
                coordTable.image = prefixedImages;

                allCoordTables{end+1} = coordTable; %#ok<AGROW>
                isQuadFormat(end+1) = isQuad; %#ok<AGROW>
                fprintf('    %s: %d coordinate entries (prefixed)\n', phoneSubdirs{p}, height(coordTable));
            end
        end
    end

    % Merge and split coordinates
    if ~isempty(allCoordTables)
        % All should be same format
        isQuad = isQuadFormat(1);
        mergedCoords = vertcat(allCoordTables{:});
        split_merged_coordinates(linkedFolder, mergedCoords, isQuad, splitNames, baseNameToSplit, coordIO);

        % Delete original coordinate files
        for p = 1:length(phoneSubdirs)
            coordFile = fullfile(linkedFolder, phoneSubdirs{p}, 'coordinates.txt');
            if isfile(coordFile)
                delete(coordFile);
            end
        end
    end

    % Collect ALL con_N folders from ALL phone subfolders with phone info
    % Structure: conName -> {struct(path, phonePrefix), ...}
    allConFolders = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for p = 1:length(phoneSubdirs)
        phoneDir = fullfile(linkedFolder, phoneSubdirs{p});
        conFolders = find_con_folders(phoneDir);

        for c = 1:length(conFolders)
            conName = conFolders{c};
            conPath = fullfile(phoneDir, conName);

            if ~allConFolders.isKey(conName)
                allConFolders(conName) = {};
            end
            entries = allConFolders(conName);
            entries{end+1} = struct('path', conPath, 'phonePrefix', phoneSubdirs{p}); %#ok<AGROW>
            allConFolders(conName) = entries;
        end
    end

    % Split con_N folders (merge images from all phones into flat structure with prefix)
    if allConFolders.Count > 0
        split_merged_con_folders(linkedFolder, allConFolders, splitNames, baseNameToSplit);
    end

    % Clean up original phone folders
    for p = 1:length(phoneSubdirs)
        phoneDir = fullfile(linkedFolder, phoneSubdirs{p});
        cleanup_folder(phoneDir);
    end
end

%% =========================================================================
%% COORDINATE SPLITTING
%% =========================================================================

function split_coordinates_primary_merged(folderPath, phoneSubdirs, splitNames, coordIO)
    % Split and merge coordinates.txt from all phone subfolders
    % Prefixes image names with phone subfolder name to match prefixed files

    numSplits = length(splitNames);

    % Collect all coordinates from all phone subfolders
    allCoordTables = {};
    isQuadFormat = [];

    for p = 1:length(phoneSubdirs)
        phoneDir = fullfile(folderPath, phoneSubdirs{p});
        coordFile = fullfile(phoneDir, 'coordinates.txt');

        if ~isfile(coordFile)
            continue;
        end

        [coordTable, isQuad] = parse_coordinate_file(coordFile, coordIO);
        if isempty(coordTable)
            continue;
        end

        % Prefix image names with phone subfolder
        prefixedImages = cell(height(coordTable), 1);
        for i = 1:height(coordTable)
            prefixedImages{i} = sprintf('%s_%s', phoneSubdirs{p}, char(coordTable.image(i)));
        end
        coordTable.image = prefixedImages;

        allCoordTables{end+1} = coordTable; %#ok<AGROW>
        isQuadFormat(end+1) = isQuad; %#ok<AGROW>

        fprintf('    %s: %d coordinate entries (prefixed)\n', phoneSubdirs{p}, height(coordTable));
    end

    if isempty(allCoordTables)
        fprintf('  No coordinate files found in phone subfolders\n');
        return;
    end

    % All should be same format
    isQuad = isQuadFormat(1);
    mergedCoords = vertcat(allCoordTables{:});
    fprintf('  Total merged entries: %d\n', height(mergedCoords));

    % Build set of prefixed base names in each split (image files only)
    splitBaseNames = cell(1, numSplits);
    for i = 1:numSplits
        splitDir = fullfile(folderPath, splitNames{i});
        imgFiles = get_image_files(splitDir);

        baseNames = {};
        for f = 1:length(imgFiles)
            baseNames{end+1} = coordIO.strip_image_extension(imgFiles{f}); %#ok<AGROW>
        end
        splitBaseNames{i} = unique(baseNames);
    end

    % Assign each coordinate entry to appropriate split
    splitTables = cell(1, numSplits);
    unmatchedTable = mergedCoords(1:0, :);  % Empty with same schema
    for i = 1:numSplits
        splitTables{i} = mergedCoords(1:0, :);
    end

    for i = 1:height(mergedCoords)
        imageName = char(mergedCoords.image(i));
        matched = false;

        for s = 1:numSplits
            % Check exact match or prefix match
            for j = 1:length(splitBaseNames{s})
                baseName = splitBaseNames{s}{j};
                if strcmpi(imageName, baseName) || ...
                   startsWith(imageName, [baseName '_'], 'IgnoreCase', true)
                    splitTables{s} = [splitTables{s}; mergedCoords(i, :)];
                    matched = true;
                    break;
                end
            end
            if matched
                break;
            end
        end

        if ~matched
            unmatchedTable = [unmatchedTable; mergedCoords(i, :)]; %#ok<AGROW>
        end
    end

    % Write split coordinate files
    filesWritten = 0;
    for i = 1:numSplits
        if height(splitTables{i}) > 0
            splitDir = fullfile(folderPath, splitNames{i});
            targetFile = fullfile(splitDir, 'coordinates.txt');

            if isQuad
                write_quad_coordinates(targetFile, splitTables{i}, coordIO);
            else
                write_ellipse_coordinates(targetFile, splitTables{i}, coordIO);
            end
            filesWritten = filesWritten + 1;
            fprintf('    %s: %d entries\n', splitNames{i}, height(splitTables{i}));
        end
    end

    % Write unmatched coordinates if any
    if height(unmatchedTable) > 0
        unmatchedFile = fullfile(folderPath, 'unmatched_coordinates.txt');
        if isQuad
            write_quad_coordinates(unmatchedFile, unmatchedTable, coordIO);
        else
            write_ellipse_coordinates(unmatchedFile, unmatchedTable, coordIO);
        end
        fprintf('  WARNING: %d unmatched entries saved to unmatched_coordinates.txt\n', height(unmatchedTable));
    end

    % Delete original phone coordinate files
    for p = 1:length(phoneSubdirs)
        coordFile = fullfile(folderPath, phoneSubdirs{p}, 'coordinates.txt');
        if isfile(coordFile)
            delete(coordFile);
        end
    end

    fprintf('  Split coordinates into %d files\n', filesWritten);
end

function [coordTable, isQuad] = parse_coordinate_file(coordFile, coordIO)
    % Parse coordinate file and return table with format type

    coordTable = [];
    isQuad = false;

    fid = fopen(coordFile, 'rt');
    if fid == -1
        return;
    end
    headerLine = fgetl(fid);
    fclose(fid);

    % Detect format and parse
    if contains(headerLine, 'semiMajorAxis')
        coordTable = coordIO.parseEllipseCoordinateFileAsTable(coordFile);
        isQuad = false;
    elseif contains(headerLine, 'x1') && contains(headerLine, 'y1')
        coordTable = coordIO.parseQuadCoordinateFileAsTable(coordFile);
        isQuad = true;
    end
end

function split_merged_coordinates(linkedFolder, coordTable, isQuad, splitNames, baseNameToSplit, coordIO)
    % Split merged coordinates (already prefixed) into flat split folders
    % Unmatched entries are saved to unmatched_coordinates.txt

    numSplits = length(splitNames);

    if isempty(coordTable) || height(coordTable) == 0
        fprintf('  No coordinate entries to split\n');
        return;
    end

    % Build split tables and unmatched table
    splitTables = cell(1, numSplits);
    unmatchedTable = coordTable(1:0, :);  % Empty with same schema
    for i = 1:numSplits
        splitTables{i} = coordTable(1:0, :);
    end

    % Assign each row to a split based on prefixed base name
    totalMatched = 0;

    for i = 1:height(coordTable)
        imageName = char(coordTable.image(i));
        % Image name is already prefixed (e.g., 'iphone_15_IMG_4754_con_0')
        % Extract prefixed base name for matching
        baseName = extract_prefixed_base_name_from_derived(imageName);

        if baseNameToSplit.isKey(baseName)
            splitIdx = baseNameToSplit(baseName);
            splitTables{splitIdx} = [splitTables{splitIdx}; coordTable(i, :)];
            totalMatched = totalMatched + 1;
        else
            unmatchedTable = [unmatchedTable; coordTable(i, :)]; %#ok<AGROW>
        end
    end

    % Write split coordinate files (flat, at split folder level)
    filesWritten = 0;
    for i = 1:numSplits
        if height(splitTables{i}) > 0
            splitDir = fullfile(linkedFolder, splitNames{i});
            targetFile = fullfile(splitDir, 'coordinates.txt');

            if isQuad
                write_quad_coordinates(targetFile, splitTables{i}, coordIO);
            else
                write_ellipse_coordinates(targetFile, splitTables{i}, coordIO);
            end
            filesWritten = filesWritten + 1;
            fprintf('    %s: %d coordinate entries\n', splitNames{i}, height(splitTables{i}));
        end
    end

    fprintf('  Split %d/%d coordinate entries into %d files\n', ...
            totalMatched, height(coordTable), filesWritten);

    % Save unmatched entries to file
    if height(unmatchedTable) > 0
        unmatchedFile = fullfile(linkedFolder, 'unmatched_coordinates.txt');
        if isQuad
            write_quad_coordinates(unmatchedFile, unmatchedTable, coordIO);
        else
            write_ellipse_coordinates(unmatchedFile, unmatchedTable, coordIO);
        end
        fprintf('  WARNING: %d unmatched entries saved to unmatched_coordinates.txt\n', height(unmatchedTable));
    end
end

%% =========================================================================
%% CON_N FOLDER SPLITTING
%% =========================================================================

function conFolders = find_con_folders(phoneDir)
    % Find con_N/ subfolders in phone directory

    allItems = dir(phoneDir);
    conFolders = {};

    for i = 1:length(allItems)
        if allItems(i).isdir && ~ismember(allItems(i).name, {'.', '..'})
            if startsWith(allItems(i).name, 'con_')
                conFolders{end+1} = allItems(i).name; %#ok<AGROW>
            end
        end
    end
end

function split_merged_con_folders(linkedFolder, allConFolders, splitNames, baseNameToSplit)
    % Split merged con_N folders from all phones into flat split folders
    % Files are prefixed with phone name to prevent collisions
    % Unmatched images are logged to unmatched_images.txt

    numSplits = length(splitNames);
    totalMoved = 0;
    unmatchedImages = {};  % {prefixedName, conName, sourcePath}

    conNames = allConFolders.keys();

    for c = 1:length(conNames)
        conName = conNames{c};
        conEntries = allConFolders(conName);  % Cell array of structs with path and phonePrefix

        % Create con_N/ folders in each split
        for i = 1:numSplits
            splitConDir = fullfile(linkedFolder, splitNames{i}, conName);
            if ~isfolder(splitConDir)
                mkdir(splitConDir);
            end
        end

        % Process images from all phone con_N folders
        for p = 1:length(conEntries)
            conPath = conEntries{p}.path;
            phonePrefix = conEntries{p}.phonePrefix;
            imgFiles = get_image_files(conPath);

            for f = 1:length(imgFiles)
                originalName = imgFiles{f};
                [~, nameNoExt, ext] = fileparts(originalName);

                % Create prefixed name and base name
                prefixedName = sprintf('%s_%s%s', phonePrefix, nameNoExt, ext);
                prefixedBaseName = sprintf('%s_%s', phonePrefix, ...
                    extract_base_name_from_derived(originalName));

                if baseNameToSplit.isKey(prefixedBaseName)
                    splitIdx = baseNameToSplit(prefixedBaseName);
                    srcFile = fullfile(conPath, originalName);
                    dstFile = fullfile(linkedFolder, splitNames{splitIdx}, conName, prefixedName);

                    if isfile(srcFile)
                        movefile(srcFile, dstFile);
                        totalMoved = totalMoved + 1;
                    end
                else
                    unmatchedImages{end+1} = struct('name', prefixedName, ...
                                                    'conFolder', conName, ...
                                                    'source', conPath); %#ok<AGROW>
                end
            end

            % Clean up source con_N/ folder
            cleanup_folder(conPath);
        end
    end

    % Report results
    if totalMoved > 0
        fprintf('  Moved %d images from %d con_N folder types\n', totalMoved, length(conNames));
    end

    % Save unmatched images to file
    if ~isempty(unmatchedImages)
        unmatchedFile = fullfile(linkedFolder, 'unmatched_images.txt');
        fid = fopen(unmatchedFile, 'wt');
        if fid ~= -1
            fprintf(fid, 'prefixed_name\tcon_folder\tsource_path\n');
            for i = 1:length(unmatchedImages)
                fprintf(fid, '%s\t%s\t%s\n', ...
                    unmatchedImages{i}.name, ...
                    unmatchedImages{i}.conFolder, ...
                    unmatchedImages{i}.source);
            end
            fclose(fid);
        end
        fprintf('  WARNING: %d unmatched images saved to unmatched_images.txt\n', length(unmatchedImages));
    end
end

%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

function splitNames = generate_split_names(numSplits)
    % Generate split folder names with shared timestamp
    % Format: split_YYYYMMDD_HHMMSS_nnn

    splitNames = cell(1, numSplits);
    baseTimeStr = datestr(now, 'yyyymmdd_HHMMSS');

    for i = 1:numSplits
        splitNames{i} = sprintf('split_%s_%03d', baseTimeStr, i-1);
    end
end

function files = get_image_files(folderPath)
    % Get list of image files in a folder

    validExts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    allItems = dir(folderPath);
    files = {};

    for i = 1:length(allItems)
        if allItems(i).isdir
            continue;
        end
        [~, ~, ext] = fileparts(allItems(i).name);
        if any(strcmpi(ext, validExts))
            files{end+1} = allItems(i).name; %#ok<AGROW>
        end
    end
end

function assignments = distribute_to_splits(numFiles, numSplits)
    % Return cell array where assignments{i} contains indices for split i

    baseCount = floor(numFiles / numSplits);
    extras = mod(numFiles, numSplits);

    assignments = cell(1, numSplits);
    startIdx = 1;

    for i = 1:numSplits
        count = baseCount + (i <= extras);
        endIdx = startIdx + count - 1;
        assignments{i} = startIdx:endIdx;
        startIdx = endIdx + 1;
    end
end

function baseName = extract_base_name_from_derived(fileName)
    % Extract base name from derived filenames (unprefixed)
    % Examples:
    %   IMG_4754_con_3_rep_0.png -> IMG_4754  (3_elliptical_regions format)
    %   IMG_4754_con_0.png -> IMG_4754        (2_micropads format)
    %   IMG_4754.jpg -> IMG_4754              (source image)
    %   IMG_4754 -> IMG_4754                  (coordinates entry)

    % Strip extension first
    coordIO = coordinate_io();
    noExt = coordIO.strip_image_extension(fileName);

    % Try patterns in order of specificity (most specific first)
    % Pattern 1: _con_N_rep_M (3_elliptical_regions)
    % Pattern 2: _con_N (2_micropads)
    patterns = {'_con_\d+_rep_\d+$', '_con_\d+$'};

    baseName = noExt;
    for i = 1:length(patterns)
        result = regexprep(noExt, patterns{i}, '');
        if ~strcmp(result, noExt)
            baseName = result;
            break;
        end
    end

    % Safety check: never return empty
    if isempty(baseName)
        baseName = noExt;
    end
end

function prefixedBaseName = extract_prefixed_base_name_from_derived(fileName)
    % Extract prefixed base name from derived filenames
    % Examples (with phone prefix):
    %   iphone_15_IMG_4754_con_3_rep_0.png -> iphone_15_IMG_4754
    %   iphone_15_IMG_4754_con_0.png -> iphone_15_IMG_4754
    %   iphone_15_IMG_4754 -> iphone_15_IMG_4754
    %
    % This works the same as extract_base_name_from_derived but preserves
    % any phone prefix that was added during collection.

    % Strip extension first
    coordIO = coordinate_io();
    noExt = coordIO.strip_image_extension(fileName);

    % Try patterns in order of specificity (most specific first)
    % Pattern 1: _con_N_rep_M (3_elliptical_regions)
    % Pattern 2: _con_N (2_micropads)
    patterns = {'_con_\d+_rep_\d+$', '_con_\d+$'};

    prefixedBaseName = noExt;
    for i = 1:length(patterns)
        result = regexprep(noExt, patterns{i}, '');
        if ~strcmp(result, noExt)
            prefixedBaseName = result;
            break;
        end
    end

    % Safety check: never return empty
    if isempty(prefixedBaseName)
        prefixedBaseName = noExt;
    end
end

function write_quad_coordinates(targetFile, coordTable, coordIO)
    % Write quad coordinates table to file

    names = cellstr(coordTable.image);
    nums = [coordTable.concentration, ...
            coordTable.x1, coordTable.y1, ...
            coordTable.x2, coordTable.y2, ...
            coordTable.x3, coordTable.y3, ...
            coordTable.x4, coordTable.y4, ...
            coordTable.rotation];

    targetFolder = fileparts(targetFile);
    coordIO.atomicWriteCoordinates(targetFile, coordIO.QUAD_HEADER, names, nums, ...
                                   coordIO.QUAD_WRITE_FMT, targetFolder);
end

function write_ellipse_coordinates(targetFile, coordTable, coordIO)
    % Write ellipse coordinates table to file

    names = cellstr(coordTable.image);
    nums = [coordTable.concentration, ...
            coordTable.replicate, ...
            coordTable.x, coordTable.y, ...
            coordTable.semiMajorAxis, ...
            coordTable.semiMinorAxis, ...
            coordTable.rotationAngle];

    targetFolder = fileparts(targetFile);
    coordIO.atomicWriteCoordinates(targetFile, coordIO.ELLIPSE_HEADER, names, nums, ...
                                   coordIO.ELLIPSE_WRITE_FMT, targetFolder);
end

function cleanup_folder(folderPath)
    % Remove folder if empty (recursively checks subfolders), otherwise warn

    if ~isfolder(folderPath)
        return;
    end

    items = dir(folderPath);
    items = items(~ismember({items.name}, {'.', '..'}));

    % First, recursively cleanup any subdirectories
    for i = 1:length(items)
        if items(i).isdir
            cleanup_folder(fullfile(folderPath, items(i).name));
        end
    end

    % Re-check after cleaning subdirectories
    items = dir(folderPath);
    items = items(~ismember({items.name}, {'.', '..'}));

    if isempty(items)
        rmdir(folderPath);
        fprintf('  Removed: %s\n', folderPath);
    else
        % Only show files, not empty subdirectories
        fileItems = items(~[items.isdir]);
        if ~isempty(fileItems)
            remaining = {fileItems.name};
            fprintf('  Kept (not empty): %s [%d files: %s]\n', folderPath, ...
                    length(remaining), strjoin(remaining(1:min(3,end)), ', '));
        else
            % Has subdirectories that couldn't be removed
            dirItems = items([items.isdir]);
            remaining = {dirItems.name};
            fprintf('  Kept (subdirs not empty): %s [%s]\n', folderPath, strjoin(remaining, ', '));
        end
    end
end
