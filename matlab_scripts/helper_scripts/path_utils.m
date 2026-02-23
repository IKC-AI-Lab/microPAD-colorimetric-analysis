function utils = path_utils()
    %% PATH_UTILS Returns a struct of function handles for path operations
    %
    % This utility module provides functions for path resolution, folder
    % management, and project structure navigation in the microPAD pipeline.
    %
    % Usage:
    %   paths = path_utils();
    %   root = paths.findProjectRoot('1_dataset');
    %   fullPath = paths.resolvePath(root, '2_micropads');
    %   paths.ensureFolder(outputDir);
    %
    % Project Structure:
    %   The microPAD project uses a standard folder structure:
    %     project_root/
    %     ├── 1_dataset/[phone]/          - Original images
    %     ├── 2_micropads/[phone]/        - Polygon crops
    %     ├── 3_elliptical_regions/[phone]/ - Ellipse patches
    %     ├── 4_extract_features/         - Feature outputs
    %     ├── matlab_scripts/             - MATLAB code
    %     │   └── helper_scripts/         - Helper modules
    %     └── models/                     - AI models
    %
    % See also: coordinate_io, image_io

    %% Public API
    % Project root detection
    utils.findProjectRoot = @findProjectRoot;
    utils.resolvePath = @resolvePath;

    % Folder management
    utils.ensureFolder = @ensureFolder;
    utils.validateFolderExists = @validateFolderExists;

    % Directory listing
    utils.listSubfolders = @listSubfolders;
    utils.findConcentrationFolders = @findConcentrationFolders;
    utils.findConcentrationFoldersWithCoords = @findConcentrationFoldersWithCoords;

    % Path utilities
    utils.addHelperScriptsToPath = @addHelperScriptsToPath;
    utils.relativePath = @relativePath;
    utils.stripExtension = @stripExtension;
    utils.executeInFolder = @executeInFolder;

    % Constants
    utils.DEFAULT_CONC_PREFIX = 'con_';
    utils.DEFAULT_COORD_FILENAME = 'coordinates.txt';
    utils.MAX_SEARCH_DEPTH = 5;
end

%% =========================================================================
%% PROJECT ROOT DETECTION
%% =========================================================================

function projectRoot = findProjectRoot(markerFolder, maxLevels)
    % Find project root by searching upward for a marker folder
    %
    % INPUTS:
    %   markerFolder - Folder name to search for (e.g., '1_dataset', 'matlab_scripts')
    %   maxLevels    - (Optional) Maximum levels to search upward (default: 5)
    %
    % OUTPUTS:
    %   projectRoot - Full path to project root, or pwd if not found
    %
    % The function starts from the current directory and searches upward
    % for the specified marker folder. This allows scripts to work regardless
    % of whether they're run from project root or matlab_scripts/.

    if nargin < 2
        maxLevels = 5;
    end

    currentDir = pwd;
    searchDir = currentDir;

    for level = 1:maxLevels
        % Check if marker folder exists at this level
        if isfolder(fullfile(searchDir, markerFolder))
            projectRoot = searchDir;
            return;
        end

        % Move up one level
        [parentDir, ~] = fileparts(searchDir);

        % Stop if we've reached the filesystem root
        if strcmp(searchDir, parentDir)
            break;
        end

        searchDir = parentDir;
    end

    % Fallback to current directory
    projectRoot = currentDir;
end

function absPath = resolvePath(root, relativePath)
    % Resolve a path relative to root, with fallback to absolute path
    %
    % INPUTS:
    %   root         - Project root directory
    %   relativePath - Relative path from root (or absolute path)
    %
    % OUTPUTS:
    %   absPath - Resolved absolute path
    %
    % If relativePath exists relative to root, returns that path.
    % Otherwise returns relativePath as-is (for absolute paths).

    if isempty(relativePath)
        absPath = root;
        return;
    end

    % Try relative to root first
    candidate = fullfile(root, relativePath);
    if isfolder(candidate) || isfile(candidate)
        absPath = candidate;
        return;
    end

    % Return as-is (might be absolute or not yet created)
    absPath = relativePath;
end

%% =========================================================================
%% FOLDER MANAGEMENT
%% =========================================================================

function ensureFolder(pathStr)
    % Create folder if it doesn't exist
    %
    % INPUTS:
    %   pathStr - Path to folder to create
    %
    % Silently succeeds if folder already exists.

    if ~isfolder(pathStr)
        mkdir(pathStr);
    end
end

function validateFolderExists(pathStr, errorId, messageFormat, varargin)
    % Validate that a folder exists, throw error if not
    %
    % INPUTS:
    %   pathStr       - Path to validate
    %   errorId       - Error identifier (e.g., 'script:missing_folder')
    %   messageFormat - Error message format string
    %   varargin      - Additional arguments for sprintf
    %
    % Example:
    %   paths.validateFolderExists(inputDir, 'cut:missing_input', ...
    %       'Input folder not found: %s', inputDir);

    if ~isfolder(pathStr)
        error(errorId, messageFormat, varargin{:});
    end
end

%% =========================================================================
%% DIRECTORY LISTING
%% =========================================================================

function folders = listSubfolders(dirPath)
    % List immediate subdirectories of a folder
    %
    % INPUTS:
    %   dirPath - Directory to list
    %
    % OUTPUTS:
    %   folders - Cell array of folder names (not full paths)
    %
    % Excludes '.' and '..' entries.

    folders = {};

    if ~isfolder(dirPath)
        return;
    end

    items = dir(dirPath);
    mask = [items.isdir] & ~ismember({items.name}, {'.', '..'});
    folders = {items(mask).name};
end

function conDirs = findConcentrationFolders(baseDir, prefix)
    % Find concentration folders (con_0, con_1, etc.) in a directory
    %
    % INPUTS:
    %   baseDir - Directory to search
    %   prefix  - (Optional) Concentration folder prefix (default: 'con_')
    %
    % OUTPUTS:
    %   conDirs - Cell array of full paths to concentration folders

    if nargin < 2
        prefix = 'con_';
    end

    conDirs = {};

    if ~isfolder(baseDir)
        return;
    end

    d = dir(baseDir);
    if isempty(d)
        return;
    end

    isDir = [d.isdir] & ~ismember({d.name}, {'.', '..'});
    if ~any(isDir)
        return;
    end

    names = {d(isDir).name};
    mask = startsWith(names, prefix);
    matchingNames = names(mask);

    conDirs = cellfun(@(n) fullfile(baseDir, n), matchingNames, 'UniformOutput', false);
end

function [conDirs, hasAnyCoords] = findConcentrationFoldersWithCoords(baseDir, prefix, coordFileName)
    % Find concentration folders and check which have coordinate files
    %
    % INPUTS:
    %   baseDir       - Directory to search
    %   prefix        - (Optional) Concentration folder prefix (default: 'con_')
    %   coordFileName - (Optional) Coordinate filename (default: 'coordinates.txt')
    %
    % OUTPUTS:
    %   conDirs      - Cell array of structs with fields:
    %                  .path      - Full path to folder
    %                  .hasCoords - Boolean if coordinates.txt exists
    %   hasAnyCoords - Boolean if any folder has coordinates

    if nargin < 2
        prefix = 'con_';
    end
    if nargin < 3
        coordFileName = 'coordinates.txt';
    end

    conDirs = {};
    hasAnyCoords = false;

    if ~isfolder(baseDir)
        return;
    end

    d = dir(baseDir);
    if isempty(d)
        return;
    end

    isDir = [d.isdir] & ~ismember({d.name}, {'.', '..'});
    if ~any(isDir)
        return;
    end

    names = {d(isDir).name};
    mask = startsWith(names, prefix);
    matchingNames = names(mask);

    n = numel(matchingNames);
    if n == 0
        return;
    end

    % Pre-allocate struct array
    conDirs = cell(n, 1);
    for i = 1:n
        fullPath = fullfile(baseDir, matchingNames{i});
        coordPath = fullfile(fullPath, coordFileName);
        hasCoords = isfile(coordPath);
        conDirs{i} = struct('path', fullPath, 'hasCoords', hasCoords);
        if hasCoords
            hasAnyCoords = true;
        end
    end
end

%% =========================================================================
%% PATH UTILITIES
%% =========================================================================

function addHelperScriptsToPath()
    % Add helper_scripts folder to MATLAB path
    %
    % Detects the helper_scripts folder relative to the calling script
    % and adds it to the path if not already present.
    %
    % Usage (at top of main scripts):
    %   paths = path_utils();
    %   paths.addHelperScriptsToPath();

    % Get the directory of the calling script
    st = dbstack('-completenames');
    if numel(st) >= 2
        callerPath = st(2).file;
        callerDir = fileparts(callerPath);
    else
        callerDir = pwd;
    end

    % Try to find helper_scripts relative to caller
    helperDir = fullfile(callerDir, 'helper_scripts');

    if ~isfolder(helperDir)
        % Caller might be in helper_scripts already
        [parentDir, folderName] = fileparts(callerDir);
        if strcmp(folderName, 'helper_scripts')
            helperDir = callerDir;
        else
            % Try parent/helper_scripts
            helperDir = fullfile(parentDir, 'helper_scripts');
        end
    end

    % Add to path if found and not already present
    if isfolder(helperDir) && ~contains(path, helperDir)
        addpath(helperDir);
    end
end

function relPath = relativePath(fullPath, rootPath)
    % Convert absolute path to relative path from root
    %
    % INPUTS:
    %   fullPath - Full absolute path
    %   rootPath - Root path to make relative to
    %
    % OUTPUTS:
    %   relPath - Relative path, or original path if not under root
    %
    % Preserves OS-specific separators (backslashes on Windows).

    try
        p = char(fullPath);
        rt = char(rootPath);

        % Ensure root has trailing separator
        if isempty(rt) || rt(end) ~= filesep
            prefix = [rt filesep];
        else
            prefix = rt;
        end

        % Check if path starts with root
        if strncmpi(p, prefix, length(prefix))
            relPath = p(length(prefix)+1:end);
        else
            relPath = p;
        end

        % Normalize separators on non-Windows only
        if ~ispc
            relPath = strrep(relPath, '\', '/');
        end
    catch
        relPath = char(fullPath);
    end
end

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

function executeInFolder(folder, func)
    % Execute function in specified folder, restoring original directory on exit
    %
    % INPUTS:
    %   folder - Directory path to change to
    %   func   - Function handle to execute (no arguments)
    %
    % Uses onCleanup to guarantee directory restoration even if error occurs.

    origDir = pwd;
    cleanupObj = onCleanup(@() cd(origDir));
    cd(folder);
    func();
end
