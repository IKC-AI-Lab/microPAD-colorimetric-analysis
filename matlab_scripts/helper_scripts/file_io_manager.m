function fileIO = file_io_manager()
    %% FILE_IO_MANAGER Returns a struct of function handles for file I/O operations
    %
    % This utility module provides coordinate file management, image saving,
    % and directory operations for the cut_micropads pipeline.
    %
    % COORDINATE I/O DELEGATION:
    %   All coordinate I/O functions delegate to coordinate_io.m, which is the
    %   authoritative source for coordinate file parsing and writing. This module
    %   provides thin wrappers that adapt the API (e.g., extracting coordinateFileName
    %   from cfg struct) while using the canonical implementation.
    %
    % Usage:
    %   fileIO = file_io_manager();
    %   fileIO.appendQuadCoordinates(phoneOutputDir, baseName, concentration, quad, cfg, rotation);
    %   [quadParams, found, rotation] = fileIO.loadQuadCoordinates(coordFile, imageName, numExpected);
    %
    % See also: coordinate_io, cut_micropads

    %% Load coordinate_io module (authoritative source for coordinate I/O)
    coordIO = coordinate_io();

    %% Public API
    % Coordinate I/O (delegated to coordinate_io.m)
    fileIO.appendQuadCoordinates = @(varargin) appendQuadCoordinatesWrapper(coordIO, varargin{:});
    fileIO.appendEllipseCoordinates = @(varargin) appendEllipseCoordinatesWrapper(coordIO, varargin{:});
    fileIO.loadQuadCoordinates = coordIO.loadQuadCoordinates;
    fileIO.loadEllipseCoordinates = coordIO.loadEllipseCoordinates;
    fileIO.readExistingCoordinates = coordIO.readExistingCoordinates;
    fileIO.filterConflictingEntries = coordIO.filterConflictingEntries;
    fileIO.atomicWriteCoordinates = coordIO.atomicWriteCoordinates;

    % Image saving
    fileIO.saveCroppedRegions = @saveCroppedRegions;
    fileIO.saveEllipseData = @saveEllipseData;
    fileIO.saveEllipticalPatches = @saveEllipticalPatches;
    fileIO.cropImageWithQuad = @cropImageWithQuad;
    fileIO.saveImageWithFormat = @saveImageWithFormat;

    % Directory management
    fileIO.createOutputDirectory = @createOutputDirectory;
    fileIO.getImageFiles = @getImageFiles;
end

%% =========================================================================
%% COORDINATE I/O WRAPPERS (delegate to coordinate_io.m)
%% =========================================================================

function appendQuadCoordinatesWrapper(coordIO, phoneOutputDir, baseName, concentration, quad, cfg, rotation)
    % Wrapper that extracts coordinateFileName from cfg and delegates to coordinate_io
    %
    % This wrapper exists because cut_micropads.m passes a cfg struct with
    % coordinateFileName field, while coordinate_io.m expects it as a direct parameter.

    if isfield(cfg, 'coordinateFileName')
        coordinateFileName = cfg.coordinateFileName;
    else
        coordinateFileName = 'coordinates.txt';
    end

    coordIO.appendQuadCoordinates(phoneOutputDir, baseName, concentration, quad, rotation, coordinateFileName);
end

function appendEllipseCoordinatesWrapper(coordIO, phoneOutputDir, baseName, ellipseData, cfg)
    % Wrapper that extracts coordinateFileName from cfg and delegates to coordinate_io
    %
    % This wrapper exists because cut_micropads.m passes a cfg struct with
    % coordinateFileName field, while coordinate_io.m expects it as a direct parameter.

    if isfield(cfg, 'coordinateFileName')
        coordinateFileName = cfg.coordinateFileName;
    else
        coordinateFileName = 'coordinates.txt';
    end

    coordIO.appendEllipseCoordinates(phoneOutputDir, baseName, ellipseData, coordinateFileName);
end

%% =========================================================================
%% IMAGE SAVING - CROPPED REGIONS
%% =========================================================================

function saveCroppedRegions(img, imageName, quads, outputDir, cfg, rotation, fileIOMgr)
    [~, baseName, ~] = fileparts(imageName);
    outExt = '.png';

    numRegions = size(quads, 1);

    for concentration = 0:(numRegions - 1)
        quad = squeeze(quads(concentration + 1, :, :));

        croppedImg = cropImageWithQuad(img, quad);

        concFolder = sprintf('%s%d', cfg.concFolderPrefix, concentration);
        concPath = fullfile(outputDir, concFolder);

        outputName = sprintf('%s_con_%d%s', baseName, concentration, outExt);
        outputPath = fullfile(concPath, outputName);

        saveImageWithFormat(croppedImg, outputPath);

        if cfg.output.saveCoordinates
            if nargin >= 7 && ~isempty(fileIOMgr)
                fileIOMgr.appendQuadCoordinates(outputDir, baseName, concentration, quad, cfg, rotation);
            else
                % Fallback: load coordinate_io directly
                coordIO = coordinate_io();
                coordinateFileName = 'coordinates.txt';
                if isfield(cfg, 'coordinateFileName')
                    coordinateFileName = cfg.coordinateFileName;
                end
                coordIO.appendQuadCoordinates(outputDir, baseName, concentration, quad, rotation, coordinateFileName);
            end
        end
    end
end

%% =========================================================================
%% IMAGE SAVING - ELLIPSE DATA
%% =========================================================================

function saveEllipseData(img, imageName, ~, ellipseData, outputDir, cfg, fileIOMgr)
    % Save ellipse patches and coordinates to 3_elliptical_regions/
    % ellipseData: Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]

    [~, baseName, ~] = fileparts(imageName);

    % Save elliptical patches
    saveEllipticalPatches(img, baseName, ellipseData, outputDir, cfg);

    % Save ellipse coordinates to phone-level coordinates.txt
    if cfg.output.saveCoordinates
        if nargin >= 7 && ~isempty(fileIOMgr)
            fileIOMgr.appendEllipseCoordinates(outputDir, baseName, ellipseData, cfg);
        else
            % Fallback: load coordinate_io directly
            coordIO = coordinate_io();
            coordinateFileName = 'coordinates.txt';
            if isfield(cfg, 'coordinateFileName')
                coordinateFileName = cfg.coordinateFileName;
            end
            coordIO.appendEllipseCoordinates(outputDir, baseName, ellipseData, coordinateFileName);
        end
    end
end

%% =========================================================================
%% IMAGE SAVING - ELLIPTICAL PATCHES
%% =========================================================================

function saveEllipticalPatches(img, baseName, ellipseData, outputDir, cfg)
    % Extract and save elliptical patches from original image
    % ellipseData: Nx7 matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
    %
    % Delegates ellipse mask creation to mask_utils.createEllipseMask for
    % consistent masking across all scripts.
    %
    % See also: mask_utils.createEllipseMask

    persistent masks
    if isempty(masks)
        masks = mask_utils();
    end

    [imgH, imgW, numChannels] = size(img);

    for i = 1:size(ellipseData, 1)
        if ellipseData(i, 3) > 0
            concIdx = ellipseData(i, 1);
            repIdx = ellipseData(i, 2);
            x = ellipseData(i, 3);
            y = ellipseData(i, 4);
            a = ellipseData(i, 5);
            b = ellipseData(i, 6);
            theta = ellipseData(i, 7);

            % Calculate axis-aligned bounding box (delegate to mask_utils for
            % consistent rotation convention handling)
            [x1, y1, x2, y2] = masks.computeEllipseBoundingBox(x, y, a, b, theta, imgW, imgH);

            % Extract region
            patchRegion = img(y1:y2, x1:x2, :);
            [patchH, patchW, ~] = size(patchRegion);

            % Create elliptical mask using mask_utils (center relative to patch)
            centerX_patch = x - x1 + 1;
            centerY_patch = y - y1 + 1;
            ellipseMask = masks.createEllipseMask([patchH, patchW], centerX_patch, centerY_patch, a, b, theta);

            % Apply mask (zero out pixels outside ellipse)
            ellipticalPatch = patchRegion;
            inverseMask3D = repmat(~ellipseMask, [1, 1, numChannels]);
            ellipticalPatch(inverseMask3D) = 0;

            % Save patch
            concFolder = fullfile(outputDir, sprintf('%s%d', cfg.concFolderPrefix, concIdx));
            if ~exist(concFolder, 'dir')
                mkdir(concFolder);
            end

            patchFileName = sprintf('%s_con%d_rep%d.png', baseName, concIdx, repIdx);
            patchPath = fullfile(concFolder, patchFileName);
            imwrite(ellipticalPatch, patchPath);
        end
    end
end

%% =========================================================================
%% IMAGE SAVING - UTILITIES
%% =========================================================================

function croppedImg = cropImageWithQuad(img, quadVertices)
    % Crop image to quadrilateral region with mask applied
    %
    % Delegates to mask_utils.cropWithQuadMask for consistent masking
    % across all scripts. See mask_utils.m for authoritative implementation.
    %
    % See also: mask_utils.cropWithQuadMask

    persistent masks
    if isempty(masks)
        masks = mask_utils();
    end

    [croppedImg, ~] = masks.cropWithQuadMask(img, quadVertices);
end

function saveImageWithFormat(img, outPath)
    % Save image to file
    %
    % Delegates to image_io.saveImage for consistent saving with automatic
    % directory creation. See image_io.m for authoritative implementation.
    %
    % See also: image_io.saveImage

    persistent imageIO
    if isempty(imageIO)
        imageIO = image_io();
    end

    imageIO.saveImage(img, outPath);
end

%% =========================================================================
%% DIRECTORY MANAGEMENT
%% =========================================================================

function outputDirs = createOutputDirectory(basePathQuads, basePathEllipses, phoneName, numConcentrations, concFolderPrefix)
    % Create quad output directories
    phoneOutputDirQuads = fullfile(basePathQuads, phoneName);
    if ~isfolder(phoneOutputDirQuads)
        mkdir(phoneOutputDirQuads);
    end

    for i = 0:(numConcentrations - 1)
        concFolder = sprintf('%s%d', concFolderPrefix, i);
        concPath = fullfile(phoneOutputDirQuads, concFolder);
        if ~isfolder(concPath)
            mkdir(concPath);
        end
    end

    % Create ellipse output directories
    phoneOutputDirEllipses = fullfile(basePathEllipses, phoneName);
    if ~isfolder(phoneOutputDirEllipses)
        mkdir(phoneOutputDirEllipses);
    end

    for i = 0:(numConcentrations - 1)
        concFolder = sprintf('%s%d', concFolderPrefix, i);
        concPath = fullfile(phoneOutputDirEllipses, concFolder);
        if ~isfolder(concPath)
            mkdir(concPath);
        end
    end

    % Return both directories as struct
    outputDirs = struct();
    outputDirs.quadDir = phoneOutputDirQuads;
    outputDirs.ellipseDir = phoneOutputDirEllipses;
end

function files = getImageFiles(dirPath, extensions)
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
