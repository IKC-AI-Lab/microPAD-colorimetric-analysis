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
    %   fileIO.appendPolygonCoordinates(phoneOutputDir, baseName, concentration, polygon, cfg, rotation);
    %   [polygonParams, found] = fileIO.loadPolygonCoordinates(coordFile, imageName, numExpected);
    %
    % See also: coordinate_io, cut_micropads

    %% Load coordinate_io module (authoritative source for coordinate I/O)
    coordIO = coordinate_io();

    %% Public API
    % Coordinate I/O (delegated to coordinate_io.m)
    fileIO.appendPolygonCoordinates = @(varargin) appendPolygonCoordinatesWrapper(coordIO, varargin{:});
    fileIO.appendEllipseCoordinates = @(varargin) appendEllipseCoordinatesWrapper(coordIO, varargin{:});
    fileIO.loadPolygonCoordinates = coordIO.loadPolygonCoordinates;
    fileIO.loadEllipseCoordinates = coordIO.loadEllipseCoordinates;
    fileIO.readExistingCoordinates = coordIO.readExistingCoordinates;
    fileIO.filterConflictingEntries = coordIO.filterConflictingEntries;
    fileIO.atomicWriteCoordinates = coordIO.atomicWriteCoordinates;

    % Image saving
    fileIO.saveCroppedRegions = @saveCroppedRegions;
    fileIO.saveEllipseData = @saveEllipseData;
    fileIO.saveEllipticalPatches = @saveEllipticalPatches;
    fileIO.cropImageWithPolygon = @cropImageWithPolygon;
    fileIO.saveImageWithFormat = @saveImageWithFormat;

    % Directory management
    fileIO.createOutputDirectory = @createOutputDirectory;
    fileIO.getImageFiles = @getImageFiles;
end

%% =========================================================================
%% COORDINATE I/O WRAPPERS (delegate to coordinate_io.m)
%% =========================================================================

function appendPolygonCoordinatesWrapper(coordIO, phoneOutputDir, baseName, concentration, polygon, cfg, rotation)
    % Wrapper that extracts coordinateFileName from cfg and delegates to coordinate_io
    %
    % This wrapper exists because cut_micropads.m passes a cfg struct with
    % coordinateFileName field, while coordinate_io.m expects it as a direct parameter.

    if isfield(cfg, 'coordinateFileName')
        coordinateFileName = cfg.coordinateFileName;
    else
        coordinateFileName = 'coordinates.txt';
    end

    coordIO.appendPolygonCoordinates(phoneOutputDir, baseName, concentration, polygon, rotation, coordinateFileName);
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

function saveCroppedRegions(img, imageName, polygons, outputDir, cfg, rotation, fileIOMgr)
    [~, baseName, ~] = fileparts(imageName);
    outExt = '.png';

    numRegions = size(polygons, 1);

    for concentration = 0:(numRegions - 1)
        polygon = squeeze(polygons(concentration + 1, :, :));

        croppedImg = cropImageWithPolygon(img, polygon);

        concFolder = sprintf('%s%d', cfg.concFolderPrefix, concentration);
        concPath = fullfile(outputDir, concFolder);

        outputName = sprintf('%s_con_%d%s', baseName, concentration, outExt);
        outputPath = fullfile(concPath, outputName);

        saveImageWithFormat(croppedImg, outputPath);

        if cfg.output.saveCoordinates
            if nargin >= 7 && ~isempty(fileIOMgr)
                fileIOMgr.appendPolygonCoordinates(outputDir, baseName, concentration, polygon, cfg, rotation);
            else
                % Fallback: load coordinate_io directly
                coordIO = coordinate_io();
                coordinateFileName = 'coordinates.txt';
                if isfield(cfg, 'coordinateFileName')
                    coordinateFileName = cfg.coordinateFileName;
                end
                coordIO.appendPolygonCoordinates(outputDir, baseName, concentration, polygon, rotation, coordinateFileName);
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

            % Calculate axis-aligned bounding box
            % Note: sign of theta doesn't matter for bounding box since we square sin/cos
            theta_rad = deg2rad(theta);
            ux = sqrt((a * cos(theta_rad))^2 + (b * sin(theta_rad))^2);
            uy = sqrt((a * sin(theta_rad))^2 + (b * cos(theta_rad))^2);

            x1 = max(1, floor(x - ux));
            y1 = max(1, floor(y - uy));
            x2 = min(imgW, ceil(x + ux));
            y2 = min(imgH, ceil(y + uy));

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

function croppedImg = cropImageWithPolygon(img, polygonVertices)
    % Crop image to polygon region with mask applied
    %
    % Delegates to mask_utils.cropWithPolygonMask for consistent masking
    % across all scripts. See mask_utils.m for authoritative implementation.
    %
    % See also: mask_utils.cropWithPolygonMask

    persistent masks
    if isempty(masks)
        masks = mask_utils();
    end

    [croppedImg, ~] = masks.cropWithPolygonMask(img, polygonVertices);
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

function outputDirs = createOutputDirectory(basePathPolygons, basePathEllipses, phoneName, numConcentrations, concFolderPrefix)
    % Create polygon output directories
    phoneOutputDirPolygons = fullfile(basePathPolygons, phoneName);
    if ~isfolder(phoneOutputDirPolygons)
        mkdir(phoneOutputDirPolygons);
    end

    for i = 0:(numConcentrations - 1)
        concFolder = sprintf('%s%d', concFolderPrefix, i);
        concPath = fullfile(phoneOutputDirPolygons, concFolder);
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
    outputDirs.polygonDir = phoneOutputDirPolygons;
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
