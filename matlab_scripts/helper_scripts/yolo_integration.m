function yolo = yolo_integration()
    %% YOLO_INTEGRATION Returns a struct of function handles for YOLO detection
    %
    % This utility module provides Python subprocess communication for
    % YOLO-based microPAD detection.
    %
    % Usage:
    %   yolo = yolo_integration();
    %   yolo.ensurePythonSetup(pythonPath);
    %   [quads, confs, outputFile, imgPath] = yolo.detectQuadsYOLO(img, cfg);
    %   [done, quads, confs, err] = yolo.checkDetectionComplete(outputFile, img);
    %
    % See also: cut_micropads

    %% Public API
    % Python environment
    yolo.ensurePythonSetup = @ensurePythonSetup;
    yolo.searchForPython = @searchForPython;

    % Detection
    yolo.detectQuadsYOLO = @detectQuadsYOLO;
    yolo.checkDetectionComplete = @checkDetectionComplete;
    yolo.parseDetectionOutput = @parseDetectionOutput;

    % Utilities
    yolo.cleanupTempFile = @cleanupTempFile;
end

%% =========================================================================
%% PYTHON ENVIRONMENT SETUP
%% =========================================================================

function ensurePythonSetup(pythonPath)
    % Ensure Python environment is properly configured
    %
    % Validates and caches Python path for subsequent detection calls.
    % Searches common locations if path not explicitly provided.
    %
    % Input:
    %   pythonPath - Path to Python executable (can be empty to auto-detect)
    %
    % Errors:
    %   yolo_integration:python_not_configured - No Python found
    %   yolo_integration:python_missing - Specified path doesn't exist

    persistent setupComplete
    if ~isempty(setupComplete) && setupComplete
        return;
    end

    try
        pythonPath = char(pythonPath);

        % Check environment variable first
        envPath = getenv('MICROPAD_PYTHON');
        if ~isempty(envPath)
            pythonPath = envPath;
        end

        % If still empty, try platform-specific search paths
        if isempty(pythonPath)
            pythonPath = searchForPython();
        end

        % Validate Python path is provided
        if isempty(pythonPath)
            error('yolo_integration:python_not_configured', ...
                ['Python path not configured! Options:\n', ...
                 '  1. Set MICROPAD_PYTHON environment variable\n', ...
                 '  2. Pass pythonPath parameter: cut_micropads(''pythonPath'', ''path/to/python'')\n', ...
                 '  3. Ensure Python is in system PATH']);
        end

        if ~isfile(pythonPath)
            error('yolo_integration:python_missing', ...
                'Python executable not found at: %s', pythonPath);
        end

        fprintf('Python configured: %s\n', pythonPath);
        setupComplete = true;
    catch ME
        setupComplete = [];
        rethrow(ME);
    end
end

function pythonPath = searchForPython()
    % Search for Python executable in common platform-specific locations
    %
    % Output:
    %   pythonPath - Path to found Python executable, empty if not found

    pythonPath = '';

    % First, try to find project root and check for local .conda_env
    projectCondaPath = findProjectCondaEnv();

    if ispc()
        % Windows: check local .conda_env first, then common conda/miniconda locations
        commonPaths = {
            projectCondaPath
            fullfile(getenv('USERPROFILE'), 'miniconda3', 'envs', 'microPAD-python-env', 'python.exe')
            fullfile(getenv('USERPROFILE'), 'anaconda3', 'envs', 'microPAD-python-env', 'python.exe')
        };
    elseif ismac()
        % macOS: check local .conda_env first, then common conda/homebrew locations
        commonPaths = {
            projectCondaPath
            fullfile(getenv('HOME'), 'miniconda3', 'envs', 'microPAD-python-env', 'bin', 'python')
            fullfile(getenv('HOME'), 'anaconda3', 'envs', 'microPAD-python-env', 'bin', 'python')
            '/usr/local/bin/python3'
            '/opt/homebrew/bin/python3'
        };
    else
        % Linux: check local .conda_env first, then common conda/system locations
        commonPaths = {
            projectCondaPath
            fullfile(getenv('HOME'), 'miniconda3', 'envs', 'microPAD-python-env', 'bin', 'python')
            fullfile(getenv('HOME'), 'anaconda3', 'envs', 'microPAD-python-env', 'bin', 'python')
            '/usr/bin/python3'
        };
    end

    % Filter out empty paths (e.g., when findProjectCondaEnv returns empty)
    commonPaths = commonPaths(~cellfun(@isempty, commonPaths));

    % Check each path
    for i = 1:numel(commonPaths)
        if isfile(commonPaths{i})
            pythonPath = commonPaths{i};
            return;
        end
    end

    % Try system PATH as fallback
    if ispc()
        [status, result] = system('where python');
    else
        [status, result] = system('which python3');
    end

    if status == 0
        lines = strsplit(strtrim(result), newline);
        if ~isempty(lines)
            pythonPath = char(lines{1});
        end
    end
end

%% =========================================================================
%% YOLO DETECTION
%% =========================================================================

function [quads, confidences, outputFile, imgPath] = detectQuadsYOLO(img, cfg, varargin)
    % Run YOLO detection via Python helper script (subprocess interface)
    %
    % Inputs:
    %   img - input image array
    %   cfg - configuration struct with fields:
    %         .pythonPath - path to Python executable
    %         .pythonScriptPath - path to detect_quads.py
    %         .detectionModel - path to YOLO model
    %         .minConfidence - minimum detection confidence
    %         .inferenceSize - inference image size
    %   varargin - optional name-value pairs:
    %     'async' - if true, launch non-blocking and return immediately
    %               returns empty quads/confidences, non-empty outputFile
    %
    % Outputs:
    %   quads - detected quadrilaterals (Nx4x2 array) or [] if async
    %   confidences - detection confidences (Nx1 array) or [] if async
    %   outputFile - path to output file for async mode, empty otherwise
    %   imgPath - path to temp image file (for caller cleanup in async mode)

    % Extract image dimensions for validation
    [imageHeight, imageWidth, ~] = size(img);

    p = inputParser;
    addParameter(p, 'async', false, @islogical);
    parse(p, varargin{:});
    asyncMode = p.Results.async;

    % Save image to temporary file
    tmpDir = tempdir;
    [~, tmpName] = fileparts(tempname);
    tmpImgPath = fullfile(tmpDir, sprintf('%s_micropad_detect.png', tmpName));
    imwrite(img, tmpImgPath);

    % Ensure cleanup even if error occurs (only in blocking mode)
    if ~asyncMode
        cleanupObj = onCleanup(@() cleanupTempFile(tmpImgPath));
    end

    % Initialize output variables
    outputFile = '';
    imgPath = '';

    % Create output file for async mode
    if asyncMode
        outputFile = fullfile(tmpDir, sprintf('%s_detection_output.txt', tmpName));
    end

    % Build command with platform-specific syntax
    if asyncMode
        % Platform-specific background execution
        if ispc
            % Windows: Build command with proper quote handling for cmd /c
            innerCmd = sprintf('"%s" "%s" "%s" "%s" --conf %.2f --imgsz %d', ...
                cfg.pythonPath, cfg.pythonScriptPath, tmpImgPath, cfg.detectionModel, ...
                cfg.minConfidence, cfg.inferenceSize);

            % Escape quotes with double-quote for cmd /c context
            escapedOutput = strrep(outputFile, '"', '""');

            % Construct command: double-quotes work inside cmd /c "..."
            cmd = sprintf('start /B "" cmd /c "%s > ""%s"" 2>&1"', innerCmd, escapedOutput);
        else
            % Unix/macOS: Use '&' suffix for background execution
            cmd = sprintf('"%s" "%s" "%s" "%s" --conf %.2f --imgsz %d > "%s" 2>&1 &', ...
                cfg.pythonPath, cfg.pythonScriptPath, tmpImgPath, cfg.detectionModel, ...
                cfg.minConfidence, cfg.inferenceSize, outputFile);
        end
    else
        % Blocking mode: redirect stderr to stdout to capture all output
        cmd = sprintf('"%s" "%s" "%s" "%s" --conf %.2f --imgsz %d 2>&1', ...
            cfg.pythonPath, cfg.pythonScriptPath, tmpImgPath, cfg.detectionModel, ...
            cfg.minConfidence, cfg.inferenceSize);
    end

    % Run detection
    if asyncMode
        % Launch background process and return immediately
        system(cmd);
        quads = [];
        confidences = [];
        imgPath = tmpImgPath;  % Return temp image path for caller cleanup
        return;
    end

    % Blocking mode: continue with original code
    [status, output] = system(cmd);

    if status ~= 0
        error('yolo_integration:detection_failed', 'Python detection failed (exit code %d): %s', status, output);
    end

    % Parse output (split by newlines - R2019b compatible)
    lines = strsplit(output, {'\n', '\r\n', '\r'}, 'CollapseDelimiters', false);
    lines = lines(~cellfun(@isempty, lines));
    [quads, confidences] = parseDetectionOutput(lines, imageHeight, imageWidth);
end

function [isComplete, quads, confidences, errorMsg] = checkDetectionComplete(outputFile, img)
    % Check if async detection has completed and parse results
    %
    % Inputs:
    %   outputFile - path to detection output file
    %   img - input image for dimension validation
    %
    % Outputs:
    %   isComplete - true if detection finished (success or error)
    %   quads - detected quadrilaterals (Nx4x2) or [] if not complete/failed
    %   confidences - detection confidences (Nx1) or [] if not complete/failed
    %   errorMsg - error message string if parsing failed, empty otherwise

    isComplete = false;
    quads = [];
    confidences = [];
    errorMsg = '';

    % Extract image dimensions for validation
    [imageHeight, imageWidth, ~] = size(img);

    % Check if output file exists and has content
    if ~exist(outputFile, 'file')
        return;
    end

    % Try to read the file
    try
        fid = fopen(outputFile, 'rt');
        if fid < 0
            return;
        end
        cleanupObj = onCleanup(@() fclose(fid));

        % Read all content
        content = fread(fid, '*char')';

        % Empty file means process hasn't written yet
        if isempty(content)
            return;
        end

        % Check for error patterns using single regexp
        isError = ~isempty(regexp(content, '(ERROR:|Traceback|Exception|FileNotFoundError)', 'once', 'ignorecase'));

        if isError
            % Extract first 200 characters for error context (simple and reliable)
            errorMsg = content;
            if length(errorMsg) > 200
                errorMsg = [errorMsg(1:200) '...'];
            end
            isComplete = true;
            return;
        end

        % Parse output
        lines = strsplit(content, {'\n', '\r\n', '\r'}, 'CollapseDelimiters', false);
        lines = lines(~cellfun(@isempty, lines));

        % Check for incomplete writes (detection count > 0 but no valid detections parsed)
        if ~isempty(lines)
            numDetections = str2double(lines{1});
            if numDetections > 0 && length(lines) < numDetections + 1
                % Incomplete write - Python still writing output
                return;
            end
        end

        [quads, confidences] = parseDetectionOutput(lines, imageHeight, imageWidth);

        % Additional check: if count > 0 but parsing returned nothing, might be incomplete
        if ~isempty(lines)
            numDetections = str2double(lines{1});
            if numDetections > 0 && isempty(quads)
                return;
            end
        end

        % Successfully parsed - mark complete
        isComplete = true;

    catch ME
        % Error reading or parsing - consider complete but failed
        isComplete = true;
        errorMsg = sprintf('Failed to parse detection output: %s', ME.message);
        warning('yolo_integration:detection_parse_error', '%s', errorMsg);
    end
end

function [quads, confidences] = parseDetectionOutput(lines, imageHeight, imageWidth)
    % Parse YOLO detection output format into polygon arrays
    %
    % Inputs:
    %   lines - cell array of text lines (first line = count, rest = detections)
    %   imageHeight - image height for bounds validation
    %   imageWidth - image width for bounds validation
    %
    % Outputs:
    %   quads - detected quadrilaterals [N x 4 x 2]
    %   confidences - detection confidence scores [N x 1]
    %
    % Format: Each detection line contains 9 space-separated values:
    %   x1 y1 x2 y2 x3 y3 x4 y4 confidence
    %   (0-based coordinates from Python, converted to 1-based for MATLAB)

    quads = [];
    confidences = [];

    if isempty(lines)
        return;
    end

    numDetections = str2double(lines{1});

    if numDetections == 0 || isnan(numDetections)
        return;
    end

    quads = zeros(numDetections, 4, 2);
    confidences = zeros(numDetections, 1);

    for i = 1:numDetections
        if i+1 > length(lines)
            break;
        end

        parts = str2double(split(lines{i+1}));
        if length(parts) < 9 || any(isnan(parts)) || any(isinf(parts))
            warning('yolo_integration:invalid_detection', ...
                    'Skipping detection %d: invalid numeric data', i);
            continue;
        end

        % Parse: x1 y1 x2 y2 x3 y3 x4 y4 confidence (0-based from Python)
        % Convert to MATLAB 1-based indexing
        vertices = parts(1:8) + 1;

        % Validate vertices are within reasonable bounds (2× image size for rotations)
        % After 1-based conversion, minimum valid value is 1
        if any(vertices < 1) || any(vertices > max([imageHeight, imageWidth]) * 2)
            warning('yolo_integration:out_of_bounds', ...
                    'Skipping detection %d: vertices out of bounds', i);
            continue;
        end

        quad = reshape(vertices, 2, 4)';  % Reshape to 4x2 matrix
        quads(i, :, :) = quad;
        confidences(i) = parts(9);
    end

    % Filter out empty detections (where confidence = 0)
    validMask = confidences > 0;
    quads = quads(validMask, :, :);
    confidences = confidences(validMask);

    % NOTE: Quad ordering is handled by sortQuadArrayByX in caller
    % (adaptive orientation-aware sorting for horizontal vs vertical strips)
end

%% =========================================================================
%% UTILITIES
%% =========================================================================

function cleanupTempFile(tmpPath)
    % Helper to clean up temporary detection image file
    %
    % Input:
    %   tmpPath - path to temporary file to delete

    if isfile(tmpPath)
        try
            delete(tmpPath);
        catch
            % Silently ignore cleanup errors
        end
    end
end

function pythonPath = findProjectCondaEnv()
    % Find Python executable in project-local .conda_env directory
    %
    % Searches up from this script's location to find the project root
    % (identified by CLAUDE.md), then checks for .conda_env/bin/python.
    %
    % Output:
    %   pythonPath - Path to Python executable, empty if not found

    pythonPath = '';

    % Start from this script's directory and search upwards
    currentDir = fileparts(mfilename('fullpath'));

    % Search up to 5 levels (script is at helper_scripts/, 2 levels below project root)
    for i = 1:5
        % Check for project root marker (CLAUDE.md)
        if isfile(fullfile(currentDir, 'CLAUDE.md'))
            % Found project root, check for .conda_env
            if ispc()
                condaPython = fullfile(currentDir, '.conda_env', 'python.exe');
            else
                condaPython = fullfile(currentDir, '.conda_env', 'bin', 'python');
            end

            if isfile(condaPython)
                pythonPath = condaPython;
                return;
            end
        end

        % Move up one directory
        parentDir = fileparts(currentDir);
        if strcmp(parentDir, currentDir)
            % Reached filesystem root
            break;
        end
        currentDir = parentDir;
    end
end
