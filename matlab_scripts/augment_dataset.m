function augment_dataset(varargin)
    %% microPAD Colorimetric Analysis — Dataset Augmentation Tool
    %% Generates synthetic training datasets from microPAD paper images for polygon detection
    %% Author: Veysel Y. Yilmaz
    %
    % FEATURES:
    % - Procedural textured backgrounds (uniform, speckled, laminate, skin)
    % - Perspective and rotation transformations
    % - Random spatial placement with uniform distribution
    % - Independent rotation per concentration region
    % - Collision detection to prevent overlap
    % - Optional photometric augmentation, white-balance jitter, and blur
    % - Optional thin occlusions over polygons (hair/strap-like) for robustness
    % - Variable-density distractor artifacts (1-20 per image, unconstrained placement)
    %
    % Generates synthetic training data by applying geometric and photometric
    % transformations to microPAD paper images and their labeled concentration regions.
    %
    % PIPELINE:
    % 1. Read microPAD paper images from 2_micropad_papers/
    % 2. Load polygon coordinates from 3_concentration_rectangles/
    % 3. Generate 1 original + N augmented versions per paper (N+1 total)
    % 4. Write outputs to augmented_* directories
    %
    % TRANSFORMATION ORDER (applied to each concentration region):
    %   a) Shared perspective transformation (same for all regions from one paper)
    %   b) Shared rotation (same for all regions from one paper)
    %   c) Independent rotation (unique per region)
    %   d) Random spatial translation (Gaussian-distributed, center-biased)
    %   e) Composite onto procedural background
    %
    % OUTPUT STRUCTURE:
    %   augmented_1_dataset/[phone]/           - Full synthetic scenes
    %   augmented_2_concentration_rectangles/  - Polygon crops + coordinates.txt
    %
    % Parameters (Name-Value):
    % - 'numAugmentations' (positive integer, default 3): augmented versions per paper
    %   Note: Total output is numAugmentations + 1 (includes original at aug_000)
    % - 'rngSeed' (numeric, optional): for reproducibility
    % - 'phones' (cellstr/string array): subset of phones to process
    % - 'backgroundWidth' (positive integer, default 4000): background width
    % - 'backgroundHeight' (positive integer, default 3000): background height
    % - 'scenePrefix' (char/string, default 'synthetic'): output naming prefix
    % - 'photometricAugmentation' (logical, default true): enable color/lighting variation
    % - 'blurProbability' (0-1, default 0.25): fraction of samples with slight blur
    % - 'papersPerScene' (positive integer, default 1): papers combined per scene (reserved for future use)
    %
    % Examples:
    % augment_dataset('numAugmentations', 5, 'rngSeed', 42)  % Generates 6 versions total (1 original + 5 augmented)
    % augment_dataset('phones', {'iphone_11'}, 'photometricAugmentation', false)

    %% =====================================================================
    %% CONFIGURATION CONSTANTS
    %% =====================================================================
    DEFAULT_INPUT_STAGE2 = '2_micropad_papers';
    DEFAULT_INPUT_STAGE3_COORDS = '3_concentration_rectangles';
    DEFAULT_INPUT_STAGE4_COORDS = '4_elliptical_regions';
    DEFAULT_OUTPUT_STAGE1 = 'augmented_1_dataset';
    DEFAULT_OUTPUT_STAGE2 = 'augmented_2_concentration_rectangles';
    DEFAULT_OUTPUT_STAGE3 = 'augmented_3_elliptical_regions';

    COORDINATE_FILENAME = 'coordinates.txt';
    CONCENTRATION_PREFIX = 'con_';
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    JPEG_QUALITY = 95;
    MIN_VALID_POLYGON_AREA = 100;

    % Camera/transformation parameters
    CAMERA = struct( ...
        'maxAngleDeg', 60, ...
        'xRange', [-0.8, 0.8], ...
        'yRange', [-0.8, 0.8], ...
        'zRange', [1.2, 3.0], ...
        'coverageCenter', 0.97, ...
        'coverageOffcenter', 0.90);

    ROTATION_RANGE = [0, 180];

    % Background generation parameters
    TEXTURE = struct( ...
        'woodBaseRGB', [180, 170, 150], ...
        'woodVariation', 20, ...
        'speckleHighFreq', 35, ...
        'speckleLowFreq', 20, ...
        'perlinGridScale', 80, ...
        'perlinAmplitude', 50, ...
        'uniformBaseRGB', [220, 218, 215], ...
        'uniformVariation', 15, ...
        'uniformNoiseRange', [10, 25], ...
        'tileBaseRGB', [200, 195, 185], ...
        'tileVariation', 20, ...
        'tileSpacingRange', [100, 200], ...
        'groutWidthRange', [2, 5]);

    % Artifact generation parameters
    ARTIFACTS = struct( ...
        'countRange', [5, 30], ...
        'cornerProximityBias', 0.3, ...
        'cornerExclusionRadius', 8, ...
        'sizeRangePercent', [0.01, 1.0], ...
        'probability', 1.0, ...
        'minSizePixels', 3, ...
        'overhangMargin', 0.5, ...
        'lineWidthRatio', 0.02, ...
        'lineRotationPadding', 10, ...
        'ellipseRadiusARange', [0.4, 0.7], ...
        'ellipseRadiusBRange', [0.3, 0.6], ...
        'ellipseBlurSigma', 1.5, ...
        'rectangleSizeRange', [0.5, 0.9], ...
        'rectangleBlurSigma', 2.0, ...
        'quadSizeRange', [0.5, 0.9], ...
        'quadPerturbation', 0.15, ...
        'quadBlurSigma', 1.5, ...
        'triangleSizeRange', [0.6, 0.9], ...
        'triangleBlurSigma', 1.2, ...
        'lineBlurSigma', 0.8, ...
        'lineIntensityRange', [-80, -40], ...
        'blobDarkIntensityRange', [-60, -30], ...
        'blobLightIntensityRange', [20, 50]);

    % === CORNER ROBUSTNESS AUGMENTATION ===
    CORNER_OCCLUSION = struct( ...
        'probability', 0.15, ...
        'occlusionTypes', {{'finger', 'shadow', 'small_object'}}, ...
        'sizeRange', [15, 40], ...
        'maxCornersPerPolygon', 2, ...
        'intensityRange', [-80, -30]);

    EDGE_DEGRADATION = struct( ...
        'probability', 0.25, ...
        'blurTypes', {{'gaussian', 'motion'}}, ...
        'blurRadiusRange', [1.5, 4.0], ...
        'affectsEdgesOnly', true, ...
        'edgeWidth', 10);

    % Polygon placement parameters
    PLACEMENT = struct( ...
        'margin', 50, ...
        'minSpacing', 30, ...
        'maxAttempts', 50, ...
        'expandFactor', 1.3);

    %% =====================================================================
    %% INPUT PARSING
    %% =====================================================================
    parser = inputParser();
    parser.FunctionName = mfilename;

    addParameter(parser, 'numAugmentations', 3, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>=',1}));
    addParameter(parser, 'rngSeed', [], @(n) isempty(n) || isnumeric(n));
    addParameter(parser, 'phones', {}, @(c) iscellstr(c) || isstring(c));
    addParameter(parser, 'backgroundWidth', 4000, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>',0}));
    addParameter(parser, 'backgroundHeight', 3000, @(n) validateattributes(n, {'numeric'}, {'scalar','integer','>',0}));
    addParameter(parser, 'scenePrefix', 'synthetic', @(s) validateattributes(s, {'char','string'}, {'scalartext'}));
    addParameter(parser, 'photometricAugmentation', true, @islogical);
    addParameter(parser, 'blurProbability', 0.25, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'motionBlurProbability', 0.15, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'occlusionProbability', 0.0, @(n) validateattributes(n, {'numeric'}, {'scalar','>=',0,'<=',1}));
    addParameter(parser, 'independentRotation', false, @islogical);
    addParameter(parser, 'multiScale', true, @islogical);
    addParameter(parser, 'scales', [640, 800, 1024], @(x) validateattributes(x, {'numeric'}, {'vector', 'positive', 'integer'}));
    addParameter(parser, 'extremeCasesProbability', 0.10, @(x) validateattributes(x, {'numeric'}, {'scalar', '>=', 0, '<=', 1}));

    parse(parser, varargin{:});
    opts = parser.Results;

    % Set random seed
    if isempty(opts.rngSeed)
        rng('shuffle');
    else
        rng(opts.rngSeed);
    end

    % Build configuration
    cfg = struct();
    cfg.numAugmentations = opts.numAugmentations;
    cfg.backgroundSize = [opts.backgroundWidth, opts.backgroundHeight];
    cfg.scenePrefix = char(opts.scenePrefix); 
    cfg.photometricAugmentation = opts.photometricAugmentation;
    cfg.blurProbability = opts.blurProbability;
    cfg.motionBlurProbability = opts.motionBlurProbability;
    cfg.occlusionProbability = opts.occlusionProbability;
    cfg.independentRotation = opts.independentRotation;
    cfg.files = struct('coordinates', COORDINATE_FILENAME);
    cfg.concPrefix = CONCENTRATION_PREFIX;
    cfg.supportedFormats = SUPPORTED_FORMATS;
    cfg.camera = CAMERA;
    cfg.rotationRange = ROTATION_RANGE;
    cfg.jpegQuality = JPEG_QUALITY;
    cfg.minValidPolygonArea = MIN_VALID_POLYGON_AREA;
    cfg.texture = TEXTURE;
    cfg.artifacts = ARTIFACTS;
    cfg.placement = PLACEMENT;
    cfg.cornerOcclusion = CORNER_OCCLUSION;
    cfg.edgeDegradation = EDGE_DEGRADATION;
    cfg.multiScale = opts.multiScale;
    cfg.scales = opts.scales;
    cfg.extremeCasesProbability = opts.extremeCasesProbability;

    % Resolve paths
    projectRoot = find_project_root(DEFAULT_INPUT_STAGE2);
    cfg.projectRoot = projectRoot;
    cfg.paths = struct( ...
        'stage2Input', fullfile(projectRoot, DEFAULT_INPUT_STAGE2), ...
        'stage3Coords', fullfile(projectRoot, DEFAULT_INPUT_STAGE3_COORDS), ...
        'stage4Coords', fullfile(projectRoot, DEFAULT_INPUT_STAGE4_COORDS), ...
        'stage1Output', DEFAULT_OUTPUT_STAGE1, ...
        'stage2Output', DEFAULT_OUTPUT_STAGE2, ...
        'stage3Output', DEFAULT_OUTPUT_STAGE3);

    % Validate inputs exist
    if ~isfolder(cfg.paths.stage2Input)
        error('augmentDataset:missingInput', 'Stage 2 input not found: %s', cfg.paths.stage2Input);
    end
    if ~isfolder(cfg.paths.stage3Coords)
        error('augmentDataset:missingCoords', 'Stage 3 coordinates folder not found: %s', cfg.paths.stage3Coords);
    end
    if ~isfolder(cfg.paths.stage4Coords)
        warning('augmentDataset:missingStage4', ...
                'Stage 4 coordinates folder not found: %s\nEllipse processing will be skipped.', ...
                cfg.paths.stage4Coords);
    end

    % Get phone list
    requestedPhones = string(opts.phones);
    requestedPhones = requestedPhones(requestedPhones ~= "");

    phoneList = list_phones(cfg.paths.stage2Input);
    if isempty(phoneList)
        error('augmentDataset:noPhones', 'No phone folders found in %s', cfg.paths.stage2Input);
    end

    % Validate configuration consistency
    if cfg.independentRotation && cfg.extremeCasesProbability > 0.5
        warning('augmentDataset:config', ...
            'Independent rotation + high extreme cases may generate too-difficult samples');
    end

    % Process each phone
    fprintf('\n=== Augmentation Configuration ===\n');
    fprintf('Camera perspective: %.0f° max angle, X=[%.1f,%.1f], Y=[%.1f,%.1f], Z=[%.1f,%.1f]\n', ...
        cfg.camera.maxAngleDeg, cfg.camera.xRange, cfg.camera.yRange, cfg.camera.zRange);
    fprintf('Corner occlusion: %.0f%% probability, max %d corners/polygon\n', ...
        cfg.cornerOcclusion.probability*100, cfg.cornerOcclusion.maxCornersPerPolygon);
    fprintf('Edge degradation: %.0f%% probability\n', cfg.edgeDegradation.probability*100);
    fprintf('Multi-scale: %s (scales: %s)\n', ...
        string(cfg.multiScale), strjoin(string(cfg.scales), ', '));
    fprintf('Artifacts: %d-%d per image, %.0f%% near corners\n', ...
        cfg.artifacts.countRange(1), cfg.artifacts.countRange(2), ...
        cfg.artifacts.cornerProximityBias*100);
    fprintf('Extreme cases: %.0f%% probability\n', cfg.extremeCasesProbability*100);
    fprintf('Augmentations per paper: %d\n', cfg.numAugmentations);
    fprintf('Background size: %dx%d\n', cfg.backgroundSize(1), cfg.backgroundSize(2));
    fprintf('Backgrounds: 4 types (uniform, speckled, laminate, skin)\n');
    fprintf('Photometric augmentation: %s\n', char(string(cfg.photometricAugmentation)));
    fprintf('Blur probability: %.1f%%\n', cfg.blurProbability * 100);
    fprintf('==================================\n');

    for i = 1:numel(phoneList)
        phoneName = phoneList{i};
        if ~isempty(requestedPhones) && ~any(strcmpi(requestedPhones, phoneName))
            continue;
        end
        fprintf('\n=== Processing phone: %s ===\n', phoneName);
        augment_phone(phoneName, cfg);
    end

    fprintf('\n=== Augmentation Complete ===\n');
end

%% -------------------------------------------------------------------------
function augment_phone(phoneName, cfg)
    % Main processing loop for a single phone
    % Strategy: For each paper, generate N augmented versions

    % Define phone-specific paths
    stage2PhoneDir = fullfile(cfg.paths.stage2Input, phoneName);
    stage3PhoneCoords = fullfile(cfg.paths.stage3Coords, phoneName, cfg.files.coordinates);
    stage4PhoneCoords = fullfile(cfg.paths.stage4Coords, phoneName, cfg.files.coordinates);

    % Validate stage 2 images exist
    if ~isfolder(stage2PhoneDir)
        warning('augmentDataset:missingPhone', 'Stage 2 folder not found for %s', phoneName);
        return;
    end

    % Load polygon coordinates (required)
    if ~isfile(stage3PhoneCoords)
        warning('augmentDataset:noPolygonCoords', 'No polygon coordinates for %s. Skipping.', phoneName);
        return;
    end

    polygonEntries = read_polygon_coordinates(stage3PhoneCoords);
    if isempty(polygonEntries)
        warning('augmentDataset:emptyPolygons', 'No valid polygon entries for %s', phoneName);
        return;
    end

    % Load ellipse coordinates (optional)
    ellipseEntries = struct('image', {}, 'concentration', {}, 'replicate', {}, ...
                            'center', {}, 'semiMajor', {}, 'semiMinor', {}, 'rotation', {});
    hasEllipses = false;
    if isfile(stage4PhoneCoords)
        ellipseEntries = read_ellipse_coordinates(stage4PhoneCoords);
        hasEllipses = ~isempty(ellipseEntries);
    else
        fprintf('  [INFO] No ellipse coordinates. Will process polygons only.\n');
    end

    % Build lookup structures
    ellipseMap = group_ellipses_by_parent(ellipseEntries, hasEllipses);
    paperGroups = group_polygons_by_image(polygonEntries);

    % Create output directories
    stage1PhoneOut = fullfile(cfg.projectRoot, cfg.paths.stage1Output, phoneName);
    stage2PhoneOut = fullfile(cfg.projectRoot, cfg.paths.stage2Output, phoneName);
    stage3PhoneOut = fullfile(cfg.projectRoot, cfg.paths.stage3Output, phoneName);
    ensure_folder(stage1PhoneOut);
    ensure_folder(stage2PhoneOut);
    ensure_folder(stage3PhoneOut);

    % Get unique paper names
    paperNames = keys(paperGroups);
    fprintf('  Total papers: %d\n', numel(paperNames));
    fprintf('  Total polygons: %d\n', numel(polygonEntries));
    fprintf('  Total ellipses: %d\n', numel(ellipseEntries));

    % Process each paper
    for paperIdx = 1:numel(paperNames)
        paperBase = paperNames{paperIdx};
        fprintf('  -> Paper %d/%d: %s\n', paperIdx, numel(paperNames), paperBase);

        % Find stage 2 image
        imgPath = find_stage2_image(stage2PhoneDir, paperBase, cfg.supportedFormats);
        if isempty(imgPath)
            warning('augmentDataset:missingImage', 'Stage 2 image not found for %s', paperBase);
            continue;
        end

        % Load image once (using imread_raw to handle EXIF orientation)
        stage2Img = imread_raw(imgPath);

        % Convert grayscale to RGB (synthetic backgrounds are always RGB)
        if size(stage2Img, 3) == 1
            stage2Img = repmat(stage2Img, [1, 1, 3]);
        end

        [~, ~, imgExt] = fileparts(imgPath);

        % Get all polygons from this paper
        polygons = paperGroups(paperBase);

        % Generate N augmented versions + 1 original (augIdx=0)
        for augIdx = 0:cfg.numAugmentations
            augment_single_paper(paperBase, imgExt, stage2Img, polygons, ellipseMap, ...
                                 hasEllipses, augIdx, stage1PhoneOut, stage2PhoneOut, ...
                                 stage3PhoneOut, cfg);
        end
    end
end

%% -------------------------------------------------------------------------
function augment_single_paper(paperBase, imgExt, stage2Img, polygons, ellipseMap, ...
                               hasEllipses, augIdx, stage1PhoneOut, stage2PhoneOut, ...
                               stage3PhoneOut, cfg)
    % Generate one augmented version of a paper with all its concentration regions

    % Sample transformation (same for all regions in this augmentation)
    % Special case: augIdx=0 uses identity transformations (original, non-transformed)
    if augIdx == 0
        % Identity transformations: no perspective distortion, no rotation
        tformPersp = affine2d(eye(3));
        tformRot = affine2d(eye(3));
    else
        % Standard random transformations for augmented versions
        % Phase 1.7: Extreme edge cases
        if rand() < cfg.extremeCasesProbability
            % Extreme camera viewpoint
            extremeCamera = cfg.camera;
            extremeCamera.maxAngleDeg = 75;
            extremeCamera.zRange = [0.8, 4.0];
            viewParams = sample_viewpoint(extremeCamera);
        else
            % Normal camera viewpoint
            viewParams = sample_viewpoint(cfg.camera);
        end
        tformPersp = compute_homography(size(stage2Img), viewParams, cfg.camera);
        rotAngle = rand_range(cfg.rotationRange);
        tformRot = centered_rotation_tform(size(stage2Img), rotAngle);
    end

    % Pre-allocate coordinate accumulators
    % Stage 2: one entry per polygon (upper bound = validCount after validation)
    % Stage 3: multiple entries per polygon (ellipses)
    maxPolygons = numel(polygons);
    stage2Coords = cell(maxPolygons, 1);
    stage3Coords = cell(maxPolygons * 10, 1);  % generous estimate
    s2Count = 0;
    s3Count = 0;

    % Temporary storage for transformed polygon crops and their properties
    transformedRegions = cell(numel(polygons), 1);
    validCount = 0;

    % Transform all polygons and extract crops
    for polyIdx = 1:numel(polygons)
        polyEntry = polygons(polyIdx);
        concentration = polyEntry.concentration;
        origVertices = polyEntry.vertices;

        % Apply shared perspective transformation
        augVertices = transform_polygon(origVertices, tformPersp);
        augVertices = transform_polygon(augVertices, tformRot);

        % Apply independent rotation per polygon if enabled
        if cfg.independentRotation && augIdx ~= 0
            independentRotAngle = rand_range(cfg.rotationRange);
            tformIndepRot = centered_rotation_tform(size(stage2Img), independentRotAngle);
            augVertices = transform_polygon(augVertices, tformIndepRot);
        else
            independentRotAngle = 0;
        end

        % Validate transformed polygon
        if ~is_valid_polygon(augVertices, cfg.minValidPolygonArea)
            warning('augmentDataset:degeneratePolygon', ...
                    '  ! Polygon %s con %d degenerate after transform. Skipping.', ...
                    paperBase, concentration);
            continue;
        end

        % Extract polygon content with masking
        [polygonContent, contentBbox] = extract_polygon_masked(stage2Img, origVertices);

        % Transform extracted content to match augmented shape
        augPolygonImg = transform_polygon_content(polygonContent, ...
                                                  origVertices, augVertices, contentBbox);

        % Store transformed region for later composition
        validCount = validCount + 1;
        transformedRegions{validCount} = struct( ...
            'concentration', concentration, ...
            'augVertices', augVertices, ...
            'augPolygonImg', augPolygonImg, ...
            'contentBbox', contentBbox, ...
            'origVertices', origVertices, ...
            'independentRotAngle', independentRotAngle);
    end

    % Trim to valid regions
    transformedRegions = transformedRegions(1:validCount);

    if validCount == 0
        warning('augmentDataset:noValidRegions', '  ! No valid regions for %s aug %d', paperBase, augIdx);
        return;
    end

    % Compute individual polygon bounding boxes for random placement
    polygonBboxes = cell(validCount, 1);
    for i = 1:validCount
        verts = transformedRegions{i}.augVertices;
        polygonBboxes{i} = struct( ...
            'minX', min(verts(:,1)), ...
            'maxX', max(verts(:,1)), ...
            'minY', min(verts(:,2)), ...
            'maxY', max(verts(:,2)), ...
            'width', max(verts(:,1)) - min(verts(:,1)), ...
            'height', max(verts(:,2)) - min(verts(:,2)));
    end

    % Start with default background size
    bgWidth = double(cfg.backgroundSize(1));
    bgHeight = double(cfg.backgroundSize(2));

    % Place polygons at random non-overlapping positions
    randomPositions = place_polygons_nonoverlapping(polygonBboxes, ...
                                                     bgWidth, bgHeight, ...
                                                     cfg.placement.margin, ...
                                                     cfg.placement.minSpacing, ...
                                                     cfg.placement.maxAttempts);

    if isempty(randomPositions)
        % Expand background and retry once
        bgWidth = round(bgWidth * cfg.placement.expandFactor);
        bgHeight = round(bgHeight * cfg.placement.expandFactor);
        fprintf('     Expanding background to %dx%d\n', bgWidth, bgHeight);

        randomPositions = place_polygons_nonoverlapping(polygonBboxes, ...
                                                         bgWidth, bgHeight, ...
                                                         cfg.placement.margin, ...
                                                         cfg.placement.minSpacing, ...
                                                         cfg.placement.maxAttempts);

        if isempty(randomPositions)
            warning('augmentDataset:positioningFailed', ...
                    '  ! Could not place polygons for %s aug %d. Skipping.', ...
                    paperBase, augIdx);
            return;
        end
    end

    % Generate realistic background with final size
    background = generate_realistic_lab_surface(bgWidth, bgHeight, cfg.texture, cfg.artifacts);

    % Composite each region onto background and save outputs
    sceneName = sprintf('%s_aug_%03d', paperBase, augIdx);
    scenePolygons = {};

    for i = 1:validCount
        region = transformedRegions{i};
        concentration = region.concentration;
        augVertices = region.augVertices;
        augPolygonImg = region.augPolygonImg;

        % Get random position for this polygon
        pos = randomPositions{i};

        % Compute offset to move polygon from its current position to random position
        % Random position specifies the top-left corner of the bounding box
        currentMinX = polygonBboxes{i}.minX;
        currentMinY = polygonBboxes{i}.minY;
        offsetX = pos.x - currentMinX;
        offsetY = pos.y - currentMinY;

        % Translate vertices to background coordinates
        sceneVertices = augVertices + [offsetX, offsetY];

        % Composite onto background
        background = composite_to_background(background, augPolygonImg, sceneVertices);

        % Paper lies flat on surface; no shadows needed

        % Save polygon crop (stage 2 output)
        concDirOut = fullfile(stage2PhoneOut, sprintf('%s%d', cfg.concPrefix, concentration));
        ensure_folder(concDirOut);

        polygonFileName = sprintf('%s_%s%d%s', sceneName, cfg.concPrefix, concentration, imgExt);
        polygonOutPath = fullfile(concDirOut, polygonFileName);
        imwrite(augPolygonImg, polygonOutPath);

        % Record stage 2 coordinates (polygon in scene)
        s2Count = s2Count + 1;
        stage2Coords{s2Count} = struct( ...
            'image', polygonFileName, ...
            'concentration', concentration, ...
            'vertices', sceneVertices);

        % Track polygon in scene space for optional occlusions
        scenePolygons{end+1} = sceneVertices; %#ok<AGROW>

        % Process ellipses for this concentration (stage 3)
        ellipseKey = sprintf('%s#%d', paperBase, concentration);
        if hasEllipses && isKey(ellipseMap, ellipseKey)
            ellipseList = ellipseMap(ellipseKey);

            for eIdx = 1:numel(ellipseList)
                ellipseIn = ellipseList(eIdx);

                % Map ellipse from crop space to image space
                ellipseInImageSpace = map_ellipse_crop_to_image(ellipseIn, region.contentBbox);

                % Validate ellipse is inside original polygon (before transforms)
                if ~inpolygon(ellipseInImageSpace.center(1), ellipseInImageSpace.center(2), ...
                              region.origVertices(:,1), region.origVertices(:,2))
                    warning('augmentDataset:ellipseOutsideOriginal', ...
                            '  ! Ellipse %s con %d rep %d outside original polygon. Skipping.', ...
                            paperBase, concentration, ellipseIn.replicate);
                    continue;
                end

                % Transform ellipse with same transforms as the parent polygon
                ellipseAug = transform_ellipse(ellipseInImageSpace, tformPersp);
                if ~ellipseAug.valid
                    warning('augmentDataset:ellipseInvalid1', ...
                            '  ! Ellipse %s con %d rep %d invalid after perspective. Skipping.', ...
                            paperBase, concentration, ellipseIn.replicate);
                    continue;
                end

                ellipseAug = transform_ellipse(ellipseAug, tformRot);
                if ~ellipseAug.valid
                    warning('augmentDataset:ellipseInvalid2', ...
                            '  ! Ellipse %s con %d rep %d invalid after shared rotation. Skipping.', ...
                            paperBase, concentration, ellipseIn.replicate);
                    continue;
                end

                % Apply same independent rotation as the parent polygon
                tformIndepRot = centered_rotation_tform(size(stage2Img), region.independentRotAngle);
                ellipseAug = transform_ellipse(ellipseAug, tformIndepRot);
                if ~ellipseAug.valid
                    warning('augmentDataset:ellipseInvalid3', ...
                            '  ! Ellipse %s con %d rep %d invalid after independent rotation. Skipping.', ...
                            paperBase, concentration, ellipseIn.replicate);
                    continue;
                end

                % Validate ellipse is inside the transformed polygon (pre-translation)
                if ~inpolygon(ellipseAug.center(1), ellipseAug.center(2), ...
                              augVertices(:,1), augVertices(:,2))
                    warning('augmentDataset:ellipseOutside', ...
                            '  ! Ellipse %s con %d rep %d outside transformed polygon. Skipping.', ...
                            paperBase, concentration, ellipseIn.replicate);
                    continue;
                end

                % Convert ellipse to polygon-crop coordinate space
                minXCrop = min(augVertices(:,1));
                minYCrop = min(augVertices(:,2));
                ellipseCrop = ellipseAug;
                ellipseCrop.center = ellipseAug.center - [minXCrop, minYCrop];

                % Validate and normalize ellipse geometry
                if ~isfinite(ellipseCrop.semiMajor) || ~isfinite(ellipseCrop.semiMinor) || ...
                   ellipseCrop.semiMajor <= 0 || ellipseCrop.semiMinor <= 0
                    warning('augmentDataset:invalidAxes', ...
                            '  ! Ellipse %s con %d rep %d has invalid axes (major=%.4f, minor=%.4f). Skipping.', ...
                            paperBase, concentration, ellipseIn.replicate, ...
                            ellipseCrop.semiMajor, ellipseCrop.semiMinor);
                    continue;
                end

                % Enforce semiMajor >= semiMinor convention
                if ellipseCrop.semiMajor < ellipseCrop.semiMinor
                    tmp = ellipseCrop.semiMajor;
                    ellipseCrop.semiMajor = ellipseCrop.semiMinor;
                    ellipseCrop.semiMinor = tmp;
                    ellipseCrop.rotation = ellipseCrop.rotation + 90;
                end

                % Normalize rotation angle to [-180, 180]
                ellipseCrop.rotation = mod(ellipseCrop.rotation + 180, 360) - 180;

                % Extract ellipse patch
                [patchImg, patchValid] = crop_ellipse_patch(augPolygonImg, ellipseCrop);
                if ~patchValid
                    warning('augmentDataset:patchInvalid', ...
                            '  ! Ellipse patch %s con %d rep %d invalid. Skipping.', ...
                            paperBase, concentration, ellipseIn.replicate);
                    continue;
                end

                % Save ellipse patch (stage 3 output)
                ellipseConcDir = fullfile(stage3PhoneOut, sprintf('%s%d', cfg.concPrefix, concentration));
                ensure_folder(ellipseConcDir);
                patchFileName = sprintf('%s_%s%d_rep%d%s', sceneName, cfg.concPrefix, ...
                                        concentration, ellipseIn.replicate, imgExt);
                patchOutPath = fullfile(ellipseConcDir, patchFileName);
                imwrite(patchImg, patchOutPath);

                % Record stage 3 coordinates (ellipse in polygon-crop space)
                s3Count = s3Count + 1;
                if s3Count > numel(stage3Coords)
                    stage3Coords{end + maxPolygons} = [];
                end
                stage3Coords{s3Count} = struct( ...
                    'image', polygonFileName, ...
                    'concentration', concentration, ...
                    'replicate', ellipseIn.replicate, ...
                    'center', ellipseCrop.center, ...
                    'semiMajor', ellipseCrop.semiMajor, ...
                    'semiMinor', ellipseCrop.semiMinor, ...
                    'rotation', ellipseCrop.rotation);
            end
        end
    end

    % Optional: add thin occlusions (e.g., hair/strap-like) over polygons
    if cfg.occlusionProbability > 0 && ~isempty(scenePolygons)
        background = add_polygon_occlusions(background, scenePolygons, cfg.occlusionProbability);
    end

    % Apply photometric augmentation and non-overlapping blur before saving
    if cfg.photometricAugmentation
        % Phase 1.7: Extreme photometric conditions (low lighting)
        if rand() < cfg.extremeCasesProbability
            background = apply_photometric_augmentation(background, 'extreme');
        else
            background = apply_photometric_augmentation(background, 'subtle');
        end
    end

    % Ensure at most one blur type is applied to avoid double-softening
    blurApplied = false;
    if cfg.motionBlurProbability > 0 && rand() < cfg.motionBlurProbability
        background = apply_motion_blur(background);
        blurApplied = true;
    end
    if ~blurApplied && cfg.blurProbability > 0 && rand() < cfg.blurProbability
        blurSigma = 0.25 + rand() * 0.40;  % [0.25, 0.65] pixels - very subtle
        background = imgaussfilt(background, blurSigma);
    end

    % Save synthetic scene (stage 1 output)
    sceneFileName = sprintf('%s%s', sceneName, '.jpg');
    sceneOutPath = fullfile(stage1PhoneOut, sceneFileName);
    imwrite(background, sceneOutPath, 'JPEG', 'Quality', cfg.jpegQuality);

    % Export corner keypoint labels (Phase 1.4)
    export_corner_labels(stage1PhoneOut, sceneName, scenePolygons, size(background));

    % Multi-scale scene generation (Phase 1.3)
    if cfg.multiScale && numel(cfg.scales) > 0
        [origH, origW, ~] = size(background);
        for scaleIdx = 1:numel(cfg.scales)
            targetSize = cfg.scales(scaleIdx);

            % Resize scene to target scale
            scaleFactor = targetSize / max(origH, origW);
            scaledScene = imresize(background, scaleFactor);

            % Scale polygon coordinates proportionally
            scaledPolygons = cell(size(scenePolygons));
            for i = 1:numel(scenePolygons)
                scaledPolygons{i} = scenePolygons{i} * scaleFactor;
            end

            % Save with scale suffix
            scaleSceneName = sprintf('%s_scale%d', sceneName, targetSize);
            scaleFileName = sprintf('%s%s', scaleSceneName, '.jpg');
            scaleOutPath = fullfile(stage1PhoneOut, scaleFileName);
            imwrite(scaledScene, scaleOutPath, 'JPEG', 'Quality', cfg.jpegQuality);

            % Export labels for this scale
            export_corner_labels(stage1PhoneOut, scaleSceneName, scaledPolygons, size(scaledScene));
        end
    end

    % Trim coordinate arrays to actual size
    stage2Coords = stage2Coords(1:s2Count);
    stage3Coords = stage3Coords(1:s3Count);

    % Write coordinates
    write_stage2_coordinates(stage2Coords, stage2PhoneOut, cfg.files.coordinates);
    if s3Count > 0
        write_stage3_coordinates(stage3Coords, stage3PhoneOut, cfg.files.coordinates);
    end

    if augIdx == 0
        fprintf('     Generated: %s [ORIGINAL] (%d polygons, %d ellipses)\n', ...
                sceneFileName, numel(stage2Coords), numel(stage3Coords));
    else
        fprintf('     Generated: %s (%d polygons, %d ellipses)\n', ...
                sceneFileName, numel(stage2Coords), numel(stage3Coords));
    end
end

%% =========================================================================
%% CORE PROCESSING FUNCTIONS
%% =========================================================================

function [content, bbox] = extract_polygon_masked(img, vertices)
    % Extract polygon region with masking to avoid black pixels
    [h, w, ~] = size(img);

    % Compute polygon mask in full image coordinates
    maskFull = poly2mask(vertices(:,1), vertices(:,2), h, w);
    cols = any(maskFull, 1);
    rows = any(maskFull, 2);
    if ~any(cols) || ~any(rows)
        content = zeros(0, 0, size(img, 3), 'like', img);
        bbox = [1, 1, 0, 0];
        return;
    end

    % Bounding box (inclusive indices)
    minX = find(cols, 1, 'first');
    maxX = find(cols, 1, 'last');
    minY = find(rows, 1, 'first');
    maxY = find(rows, 1, 'last');

    bbox = [minX, minY, maxX - minX + 1, maxY - minY + 1];

    % Extract bbox region
    bboxContent = img(minY:maxY, minX:maxX, :);

    % Slice polygon mask relative to bbox
    mask = maskFull(minY:maxY, minX:maxX);

    % Apply mask using input image type
    if size(bboxContent, 3) == 3
        mask3d = repmat(mask, [1, 1, 3]);
        content = bboxContent .* cast(mask3d, 'like', bboxContent);
    else
        content = bboxContent .* cast(mask, 'like', bboxContent);
    end
end

function augImg = transform_polygon_content(content, origVerts, augVerts, bbox)
    % Transform polygon content to match target geometry

    % Convert vertices to bbox-relative coordinates
    origVertsRel = origVerts - [bbox(1) - 1, bbox(2) - 1];
    minX = min(augVerts(:,1));
    minY = min(augVerts(:,2));
    augVertsRel = augVerts - [minX, minY];

    % Compute projective transformation
    tform = fitgeotrans(origVertsRel, augVertsRel, 'projective');

    % Determine output dimensions
    outWidth = ceil(max(augVertsRel(:,1)) - min(augVertsRel(:,1)) + 1);
    outHeight = ceil(max(augVertsRel(:,2)) - min(augVertsRel(:,2)) + 1);
    outRef = imref2d([outHeight, outWidth]);

    % Apply transformation
    augImg = imwarp(content, tform, 'OutputView', outRef, ...
                    'InterpolationMethod', 'linear', 'FillValues', 0);
end

function bg = composite_to_background(bg, polygonImg, sceneVerts)
    % Composite transformed polygon onto background using per-channel alpha blending

    % Compute target region in background
    minX = max(1, floor(min(sceneVerts(:,1))));
    maxX = min(size(bg, 2), ceil(max(sceneVerts(:,1))));
    minY = max(1, floor(min(sceneVerts(:,2))));
    maxY = min(size(bg, 1), ceil(max(sceneVerts(:,2))));

    targetWidth = maxX - minX + 1;
    targetHeight = maxY - minY + 1;

    % Guard: degenerate target (outside image or zero area)
    if targetWidth < 1 || targetHeight < 1
        return;
    end

    % Resize polygon to target size only when necessary. In the common case the
    % warped patch already matches the target bbox; skip extra resampling to
    % preserve edges and save time.
    [patchH, patchW, ~] = size(polygonImg);
    if patchH == targetHeight && patchW == targetWidth
        resized = polygonImg;
    else
        % Use nearest-neighbor to prevent color bleeding across masked boundaries
        resized = imresize(polygonImg, [targetHeight, targetWidth], 'nearest');
    end

    % Create mask for target region
    vertsTarget = sceneVerts - [minX - 1, minY - 1];
    targetMask = poly2mask(vertsTarget(:,1), vertsTarget(:,2), targetHeight, targetWidth);

    % If mask is empty, nothing to composite
    if ~any(targetMask(:))
        return;
    end

    % Composite per-channel using arithmetic (no logical linearization)
    bgRegion = bg(minY:maxY, minX:maxX, :);

    % Prepare alpha in double for stable math
    alpha = double(targetMask);

    % All images are RGB at this point (converted on load)
    for c = 1:3
        R = double(bgRegion(:,:,c));
        F = double(resized(:,:,c));
        bgRegion(:,:,c) = uint8(R .* (1 - alpha) + F .* alpha);
    end

    bg(minY:maxY, minX:maxX, :) = bgRegion;
end

function [patchImg, isValid] = crop_ellipse_patch(polygonImg, ellipse)
    % Crop elliptical patch from polygon image
    [imgHeight, imgWidth, ~] = size(polygonImg);
    bbox = ellipse_bounding_box(ellipse);

    x1 = max(1, floor(bbox(1)));
    y1 = max(1, floor(bbox(2)));
    x2 = min(imgWidth, ceil(bbox(3)));
    y2 = min(imgHeight, ceil(bbox(4)));

    if x1 > x2 || y1 > y2
        patchImg = [];
        isValid = false;
        return;
    end

    patchImg = polygonImg(y1:y2, x1:x2, :);
    [h, w, ~] = size(patchImg);

    % Create ellipse mask
    [X, Y] = meshgrid(1:w, 1:h);
    cx = ellipse.center(1) - x1 + 1;
    cy = ellipse.center(2) - y1 + 1;

    theta = deg2rad(ellipse.rotation);
    dx = X - cx;
    dy = Y - cy;
    xRot =  dx * cos(theta) + dy * sin(theta);
    yRot = -dx * sin(theta) + dy * cos(theta);
    mask = (xRot / ellipse.semiMajor).^2 + (yRot / ellipse.semiMinor).^2 <= 1;

    if ~any(mask(:))
        patchImg = [];
        isValid = false;
        return;
    end

    % Zero out pixels outside the ellipse per-channel (all images are RGB)
    for c = 1:3
        plane = patchImg(:,:,c);
        plane(~mask) = 0;
        patchImg(:,:,c) = plane;
    end
    isValid = true;
end

function bbox = ellipse_bounding_box(ellipse)
    % Compute axis-aligned bounding box for rotated ellipse
    theta = deg2rad(ellipse.rotation);
    a = ellipse.semiMajor;
    b = ellipse.semiMinor;
    dx = sqrt((a * cos(theta))^2 + (b * sin(theta))^2);
    dy = sqrt((a * sin(theta))^2 + (b * cos(theta))^2);
    xc = ellipse.center(1);
    yc = ellipse.center(2);
    bbox = [xc - dx, yc - dy, xc + dx, yc + dy];
end

%% =========================================================================
%% BACKGROUND GENERATION (PROCEDURAL TEXTURES)
%% =========================================================================

function bg = generate_realistic_lab_surface(width, height, textureCfg, artifactCfg)
    % Generate realistic lab surface backgrounds with procedural textures
    % Prevents background overfitting by creating organic patterns

    width = max(1, round(double(width)));
    height = max(1, round(double(height)));

    surfaceType = randi(4);

    if surfaceType == 1
        % Uniform surface
        baseRGB = textureCfg.uniformBaseRGB + randi([-textureCfg.uniformVariation, textureCfg.uniformVariation], [1, 3]);
        texture = randn(height, width) * (textureCfg.uniformNoiseRange(1) + rand() * diff(textureCfg.uniformNoiseRange));

    elseif surfaceType == 2
        % Speckled surface
        baseGray = 160 + randi([-25, 25]);
        baseRGB = [baseGray, baseGray, baseGray] + randi([-5, 5], [1, 3]);
        highFreqNoise = randn(height, width) * textureCfg.speckleHighFreq;
        lowFreqNoise = imgaussfilt(randn(height, width), 8) * textureCfg.speckleLowFreq;
        texture = highFreqNoise + lowFreqNoise;

    elseif surfaceType == 3
        % Laminate surface
        if rand() < 0.5
            baseRGB = [245, 245, 245] + randi([-5, 5], [1, 3]);
        else
            baseRGB = [30, 30, 30] + randi([-5, 5], [1, 3]);
        end
        texture = generate_laminate_texture(width, height, textureCfg);

    else
        % Skin texture
        h = 0.03 + rand() * 0.07;
        s = 0.25 + rand() * 0.35;
        v = 0.55 + rand() * 0.35;
        baseRGB = round(255 * hsv2rgb([h, s, v]));
        texture = generate_skin_texture(width, height);
    end

    baseRGB = max(100, min(230, baseRGB));

    % Create RGB background with base color
    bg = repmat(reshape(uint8(baseRGB), [1, 1, 3]), [height, width, 1]);

    % Apply texture to all channels
    for c = 1:3
        plane = double(bg(:,:,c)) + texture;
        bg(:,:,c) = clamp_uint8(plane);
    end

    % Add lighting gradients (60% of samples) - simulates directional lighting and vignetting
    if rand() < 0.60
        bg = add_lighting_gradient(bg, width, height);
    end

    % Add polygon artifacts (100% always) - high-density distractors for robust detection training
    bg = add_sparse_artifacts(bg, width, height, artifactCfg);
end

function texture = generate_laminate_texture(width, height, ~)
    % Generate high-contrast laminate surface with subtle noise.
    % Note: white/black base color selection is handled by caller.

    width = max(1, round(double(width)));
    height = max(1, round(double(height)));

    % Subtle grain/noise common to both white and black laminate bases
    texture = randn(height, width) * 5;
end

function texture = generate_skin_texture(width, height)
    % Generate subtle skin-like microtexture: low-frequency tone + fine noise
    width = max(1, round(double(width)));
    height = max(1, round(double(height)));

    % Low-frequency shading and pores
    lowFreq = imgaussfilt(randn(height, width), 12) * 6;   % soft variation
    midFreq = imgaussfilt(randn(height, width), 3) * 2;    % mild pores
    highFreq = randn(height, width) * 1.0;                 % fine grain

    texture = lowFreq + midFreq + highFreq;
end


function bg = add_lighting_gradient(bg, width, height)
    % Add simple linear lighting gradient to simulate directional lighting

    % Quick guard for tiny backgrounds
    width = max(1, round(double(width)));
    height = max(1, round(double(height)));
    if width < 50 || height < 50
        return;
    end

    % Create simple linear gradient in random direction
    lightAngle = rand() * 2 * pi;
    [X, Y] = meshgrid(1:width, 1:height);

    % Project along light direction (normalized)
    xNorm = (X - 1) / (width - 1);
    yNorm = (Y - 1) / (height - 1);
    projection = xNorm * cos(lightAngle) + yNorm * sin(lightAngle);

    % Subtle gradient: 5-10% variation
    gradientStrength = 0.05 + rand() * 0.05;
    gradient = 1 - gradientStrength/2 + projection * gradientStrength;
    gradient = max(0.90, min(1.10, gradient));

    % Apply to all channels
    for c = 1:3
        bg(:,:,c) = clamp_uint8(double(bg(:,:,c)) .* gradient);
    end
end

function bg = add_sparse_artifacts(bg, width, height, artifactCfg)
    % Add variable-density artifacts anywhere on background for robust detection training
    % Artifacts: rectangles, quadrilaterals, triangles, ellipses, lines
    % Count: 1-100 (variable complexity per image)
    % Size: 1-100% of image diagonal (allows artifacts larger than frame)
    % Placement: unconstrained (artifacts can extend beyond boundaries for uniform spatial distribution)

    % Quick guard for tiny backgrounds
    width = max(1, round(double(width)));
    height = max(1, round(double(height)));
    if width < 8 || height < 8
        return;
    end

    % Number of artifacts: variable (1-100 by default)
    numArtifacts = artifactCfg.countRange(1) + randi(diff(artifactCfg.countRange) + 1);

    % Image diagonal for relative sizing (allows artifacts larger than image dimensions)
    diagSize = sqrt(width^2 + height^2);

    for i = 1:numArtifacts
        % Select artifact type (equal probability)
        artifactTypeRand = rand();
        if artifactTypeRand < 0.20
            artifactType = 'rectangle';
        elseif artifactTypeRand < 0.40
            artifactType = 'quadrilateral';
        elseif artifactTypeRand < 0.60
            artifactType = 'triangle';
        elseif artifactTypeRand < 0.80
            artifactType = 'ellipse';
        else
            artifactType = 'line';
        end

        % Uniform size: 1-100% of image diagonal (allows artifacts larger than frame)
        artifactSize = round(diagSize * (artifactCfg.sizeRangePercent(1) + rand() * diff(artifactCfg.sizeRangePercent)));
        artifactSize = max(artifactCfg.minSizePixels, artifactSize);

        % Lines: use artifactSize as length, add smaller width
        if strcmp(artifactType, 'line')
            lineLength = artifactSize;
            lineWidth = max(1, round(artifactSize * artifactCfg.lineWidthRatio));
            artifactSize = lineLength + artifactCfg.lineRotationPadding;
        end

        % Unconstrained random placement (artifacts can extend beyond frame boundaries)
        % Overhang margin creates partial artifacts at edges and uniform spatial distribution
        margin = round(artifactSize * artifactCfg.overhangMargin);
        xMin = 1 - margin;
        xMax = width + margin;
        yMin = 1 - margin;
        yMax = height + margin;

        x = randi([xMin, xMax]);
        y = randi([yMin, yMax]);

        % Create artifact mask based on type
        if strcmp(artifactType, 'ellipse')
            % Elliptical blob
            [X, Y] = meshgrid(1:artifactSize, 1:artifactSize);
            centerX = artifactSize / 2;
            centerY = artifactSize / 2;
            radiusA = artifactSize / 2 * (artifactCfg.ellipseRadiusARange(1) + rand() * diff(artifactCfg.ellipseRadiusARange));
            radiusB = artifactSize / 2 * (artifactCfg.ellipseRadiusBRange(1) + rand() * diff(artifactCfg.ellipseRadiusBRange));
            angle = rand() * pi;
            xRot = (X - centerX) * cos(angle) - (Y - centerY) * sin(angle);
            yRot = (X - centerX) * sin(angle) + (Y - centerY) * cos(angle);
            mask = (xRot / radiusA).^2 + (yRot / radiusB).^2 <= 1;
            mask = imgaussfilt(double(mask), artifactCfg.ellipseBlurSigma);

        elseif strcmp(artifactType, 'rectangle')
            % Rectangular blob (paper scrap, label)
            [X, Y] = meshgrid(1:artifactSize, 1:artifactSize);
            centerX = artifactSize / 2;
            centerY = artifactSize / 2;
            rectWidth = artifactSize * (artifactCfg.rectangleSizeRange(1) + rand() * diff(artifactCfg.rectangleSizeRange));
            rectHeight = artifactSize * (artifactCfg.rectangleSizeRange(1) + rand() * diff(artifactCfg.rectangleSizeRange));
            angle = rand() * pi;  % Random rotation

            % Rotate coordinates
            xRot = (X - centerX) * cos(angle) - (Y - centerY) * sin(angle);
            yRot = (X - centerX) * sin(angle) + (Y - centerY) * cos(angle);
            mask = (abs(xRot) <= rectWidth/2) & (abs(yRot) <= rectHeight/2);
            mask = imgaussfilt(double(mask), artifactCfg.rectangleBlurSigma);

        elseif strcmp(artifactType, 'quadrilateral')
            % Irregular quadrilateral (distorted rectangle - similar to perspective-transformed rectangles)
            centerX = artifactSize / 2;
            centerY = artifactSize / 2;
            baseWidth = artifactSize * (artifactCfg.quadSizeRange(1) + rand() * diff(artifactCfg.quadSizeRange));
            baseHeight = artifactSize * (artifactCfg.quadSizeRange(1) + rand() * diff(artifactCfg.quadSizeRange));

            % Create irregular quadrilateral by perturbing rectangle corners
            perturbation = artifactCfg.quadPerturbation * artifactSize;
            vertices = [
                centerX - baseWidth/2 + (rand()-0.5)*perturbation, centerY - baseHeight/2 + (rand()-0.5)*perturbation;
                centerX + baseWidth/2 + (rand()-0.5)*perturbation, centerY - baseHeight/2 + (rand()-0.5)*perturbation;
                centerX + baseWidth/2 + (rand()-0.5)*perturbation, centerY + baseHeight/2 + (rand()-0.5)*perturbation;
                centerX - baseWidth/2 + (rand()-0.5)*perturbation, centerY + baseHeight/2 + (rand()-0.5)*perturbation
            ];

            % Clamp vertices to valid range
            vertices = max(1, min(artifactSize, vertices));

            % Create polygon mask
            mask = poly2mask(vertices(:,1), vertices(:,2), artifactSize, artifactSize);

            % Skip if polygon is degenerate (empty mask)
            if ~any(mask(:))
                continue;
            end

            mask = imgaussfilt(double(mask), artifactCfg.quadBlurSigma);

        elseif strcmp(artifactType, 'triangle')
            % Triangle (paper corner, folded edge)
            centerX = artifactSize / 2;
            centerY = artifactSize / 2;
            baseSize = artifactSize * (artifactCfg.triangleSizeRange(1) + rand() * diff(artifactCfg.triangleSizeRange));

            % Random triangle orientation
            angle = rand() * 2 * pi;
            vertices = [
                centerX + baseSize * cos(angle), centerY + baseSize * sin(angle);
                centerX + baseSize * cos(angle + 2*pi/3), centerY + baseSize * sin(angle + 2*pi/3);
                centerX + baseSize * cos(angle + 4*pi/3), centerY + baseSize * sin(angle + 4*pi/3)
            ];

            % Clamp vertices to valid range
            vertices = max(1, min(artifactSize, vertices));

            % Create polygon mask
            mask = poly2mask(vertices(:,1), vertices(:,2), artifactSize, artifactSize);

            % Skip if polygon is degenerate (empty mask)
            if ~any(mask(:))
                continue;
            end

            mask = imgaussfilt(double(mask), artifactCfg.triangleBlurSigma);

        else  % 'line'
            % Thin line (scratch, pen mark, table edge) via distance-to-line mask
            centerX = artifactSize / 2;
            centerY = artifactSize / 2;
            angle = rand() * pi;  % Random orientation

            [Xg, Yg] = meshgrid(1:artifactSize, 1:artifactSize);
            % Coordinates relative to line center
            dx = Xg - centerX;
            dy = Yg - centerY;
            % Projection along line direction and perpendicular distance
            tproj =  dx * cos(angle) + dy * sin(angle);
            dperp = abs(-dx * sin(angle) + dy * cos(angle));
            % Keep points within the finite segment and within half-width
            lineCore = (abs(tproj) <= lineLength/2) & (dperp <= lineWidth);
            mask = imgaussfilt(double(lineCore), artifactCfg.lineBlurSigma);
        end

        % Random intensity: darker or lighter
        % Lines tend to be darker; blobs can be either
        if strcmp(artifactType, 'line')
            intensity = artifactCfg.lineIntensityRange(1) + randi(diff(artifactCfg.lineIntensityRange) + 1);
        else
            if rand() < 0.5
                intensity = artifactCfg.blobDarkIntensityRange(1) + randi(diff(artifactCfg.blobDarkIntensityRange) + 1);
            else
                intensity = artifactCfg.blobLightIntensityRange(1) + randi(diff(artifactCfg.blobLightIntensityRange) + 1);
            end
        end

        % Blend into background, handling artifacts that extend beyond frame boundaries
        % Compute valid intersection between artifact bbox and image bounds
        xStart = max(1, x);
        yStart = max(1, y);
        xEnd = min(width, x + artifactSize - 1);
        yEnd = min(height, y + artifactSize - 1);

        % Validate intersection exists
        if xEnd < xStart || yEnd < yStart
            continue;  % Artifact completely outside bounds
        end

        % Compute corresponding mask region (offset if artifact starts outside frame)
        maskXStart = max(1, 2 - x);  % Offset into mask if x < 1
        maskYStart = max(1, 2 - y);  % Offset into mask if y < 1
        maskXEnd = maskXStart + (xEnd - xStart);
        maskYEnd = maskYStart + (yEnd - yStart);

        % Blend artifact into background
        for c = 1:3
            region = double(bg(yStart:yEnd, xStart:xEnd, c));
            maskRegion = mask(maskYStart:maskYEnd, maskXStart:maskXEnd);
            region = region + maskRegion * intensity;
            bg(yStart:yEnd, xStart:xEnd, c) = clamp_uint8(region);
        end
    end
end

function bg = add_polygon_occlusions(bg, scenePolygons, probability)
    % Draw thin occlusions (e.g., hair/strap-like) across polygons with some probability.
    % Each occlusion is a soft line that slightly darkens or lightens the image beneath.

    if probability <= 0
        return;
    end

    [imgH, imgW, ~] = size(bg);

    for i = 1:numel(scenePolygons)
        if rand() >= probability
            continue;
        end

        verts = scenePolygons{i};
        if isempty(verts) || any(~isfinite(verts(:)))
            continue;
        end

        minX = max(1, floor(min(verts(:,1))));
        maxX = min(imgW, ceil(max(verts(:,1))));
        minY = max(1, floor(min(verts(:,2))));
        maxY = min(imgH, ceil(max(verts(:,2))));
        if maxX <= minX || maxY <= minY
            continue;
        end

        % Build local grid for the polygon bbox
        [X, Y] = meshgrid(minX:maxX, minY:maxY);

        % Polygon mask for clipping
        polyMask = poly2mask(verts(:,1) - (minX - 1), verts(:,2) - (minY - 1), maxY - minY + 1, maxX - minX + 1);
        if ~any(polyMask(:))
            continue;
        end

        % Choose line params centered near polygon centroid
        cx = mean(verts(:,1));
        cy = mean(verts(:,2));
        angle = rand() * 2 * pi;      % random orientation
        halfWidth = 1 + rand() * 2;   % ~2-3 px thick

        % Distance to a line through (cx,cy) with normal [sin, -cos]
        d = abs((X - cx) * sin(angle) - (Y - cy) * cos(angle));
        lineMask = double(d <= halfWidth);
        lineMask = imgaussfilt(lineMask, 0.8);

        % Clip to polygon region
        lineMask = lineMask .* double(polyMask);
        if ~any(lineMask(:))
            continue;
        end

        % Random intensity: slight darken or lighten
        if rand() < 0.5
            delta = -(20 + randi(20));
        else
            delta =  (20 + randi(20));
        end

        % Blend into background
        region = bg(minY:maxY, minX:maxX, :);
        for c = 1:3
            plane = double(region(:,:,c));
            plane = plane + lineMask * double(delta);
            region(:,:,c) = clamp_uint8(plane);
        end
        bg(minY:maxY, minX:maxX, :) = region;
    end
end


%% =========================================================================
%% PHOTOMETRIC AUGMENTATION (NEW IN V3)
%% =========================================================================

function img = apply_photometric_augmentation(img, mode)
    % Apply color-safe photometric augmentation to entire scene
    % Preserves relative color relationships between concentration regions
    %
    % Inputs:
    %   img - RGB image (uint8)
    %   mode - 'subtle' (default), 'moderate', or 'extreme' (Phase 1.7)

    if nargin < 2
        mode = 'subtle';
    end

    % Convert to double in [0,1] for processing
    imgDouble = im2double(img);

    % 1. Global brightness adjustment
    if strcmp(mode, 'subtle')
        brightRange = [0.95, 1.05];  % ±5% (reduced from ±10%)
    elseif strcmp(mode, 'extreme')
        brightRange = [0.40, 0.60];  % Very low lighting (Phase 1.7)
    else
        brightRange = [0.90, 1.10];  % ±10% (reduced from ±15%)
    end
    brightFactor = brightRange(1) + rand() * diff(brightRange);
    imgDouble = imgDouble * brightFactor;

    % 2. Global contrast adjustment (around image mean)
    if strcmp(mode, 'subtle')
        contrastRange = [0.96, 1.04];  % ±4% (reduced from ±8%)
    else
        contrastRange = [0.92, 1.08];  % ±8% (reduced from ±12%)
    end
    contrastFactor = contrastRange(1) + rand() * diff(contrastRange);
    imgMean = mean(imgDouble(:));
    imgDouble = (imgDouble - imgMean) * contrastFactor + imgMean;

    % 3. White balance jitter (per-channel gain), 60% probability
    if rand() < 0.60
        gains = [0.92 + rand() * 0.16, 0.92 + rand() * 0.16, 0.92 + rand() * 0.16];
        for c = 1:3
            imgDouble(:,:,c) = imgDouble(:,:,c) * gains(c);
        end
    end

    % 4. Subtle saturation adjustment (preserve hue) - 60% of augmented samples
    if rand() < 0.6
        % Clamp before color-space conversion to avoid numeric spill
        imgDouble = min(1, max(0, imgDouble));
        imgHSV = rgb2hsv(imgDouble);
        satFactor = 0.94 + rand() * 0.12;  % [0.94, 1.06]
        imgHSV(:,:,2) = min(1, max(0, imgHSV(:,:,2) * satFactor));
        imgDouble = hsv2rgb(imgHSV);
    end

    % 5. Gamma correction (exposure simulation) - 40% of augmented samples
    %    Ensure input is within [0,1] before exponentiation to avoid
    %    negative^fraction -> complex results.
    if rand() < 0.4
        gamma = 0.92 + rand() * 0.16;  % [0.92, 1.08]
        imgDouble = min(1, max(0, imgDouble));
        imgDouble = imgDouble .^ gamma;
    end

    % Final clamp and convert back to uint8
    img = im2uint8(min(1, max(0, imgDouble)));
end

%% =========================================================================
%% TRANSFORMATION FUNCTIONS
%% =========================================================================

function viewParams = sample_viewpoint(cameraCfg)
    % Sample camera viewpoint with uniform distribution

    % Simple uniform sampling over camera range
    vx = cameraCfg.xRange(1) + rand() * diff(cameraCfg.xRange);
    vy = cameraCfg.yRange(1) + rand() * diff(cameraCfg.yRange);
    vz = cameraCfg.zRange(1) + rand() * diff(cameraCfg.zRange);

    viewParams = struct('vx', vx, 'vy', vy, 'vz', vz);
end

function tform = compute_homography(imageSize, viewParams, cameraCfg)
    imgHeight = imageSize(1);
    imgWidth = imageSize(2);
    corners = [1 1; imgWidth 1; imgWidth imgHeight; 1 imgHeight];

    yawDeg = normalize_to_angle(viewParams.vx, cameraCfg.xRange, cameraCfg.maxAngleDeg);
    pitchDeg = normalize_to_angle(viewParams.vy, cameraCfg.yRange, cameraCfg.maxAngleDeg);

    projected = project_corners(imgWidth, imgHeight, yawDeg, pitchDeg, viewParams.vz);
    coverage = cameraCfg.coverageOffcenter;
    if abs(viewParams.vx) < 1e-3 && abs(viewParams.vy) < 1e-3
        coverage = cameraCfg.coverageCenter;
    end
    aligned = fit_points_to_frame(projected, imgWidth, imgHeight, coverage);

    tform = fitgeotrans(corners, aligned, 'projective');
end

function projected = project_corners(imgWidth, imgHeight, yawDeg, pitchDeg, viewZ)
    corners = [1 1; imgWidth 1; imgWidth imgHeight; 1 imgHeight];
    cx = (imgWidth + 1) / 2;
    cy = (imgHeight + 1) / 2;
    scale = max(imgWidth, imgHeight);

    pts = [(corners(:,1) - cx) / scale, (corners(:,2) - cy) / scale, zeros(4,1)];
    yaw = deg2rad(yawDeg);
    pitch = deg2rad(pitchDeg);
    Ry = [cos(yaw) 0 sin(yaw); 0 1 0; -sin(yaw) 0 cos(yaw)];
    Rx = [1 0 0; 0 cos(pitch) -sin(pitch); 0 sin(pitch) cos(pitch)];
    R = Rx * Ry;

    rotated = (R * pts')';
    rotated(:,3) = rotated(:,3) + viewZ;

    f = viewZ;
    u = f * rotated(:,1) ./ rotated(:,3);
    v = f * rotated(:,2) ./ rotated(:,3);

    projected = [u * scale + cx, v * scale + cy];
end

function aligned = fit_points_to_frame(projected, imgWidth, imgHeight, coverage)
    minX = min(projected(:,1));
    maxX = max(projected(:,1));
    minY = min(projected(:,2));
    maxY = max(projected(:,2));
    width = maxX - minX;
    height = maxY - minY;
    if width < eps || height < eps
        aligned = projected;
        return;
    end

    scale = coverage * min((imgWidth - 1) / width, (imgHeight - 1) / height);
    center = [(maxX + minX) / 2, (maxY + minY) / 2];
    targetCenter = [(imgWidth + 1) / 2, (imgHeight + 1) / 2];

    aligned = (projected - center) * scale + targetCenter;
end

function polygonOut = transform_polygon(vertices, tform)
    [x, y] = transformPointsForward(tform, vertices(:,1), vertices(:,2));
    polygonOut = [x, y];
end

function ellipseOut = transform_ellipse(ellipseIn, tform)
    conic = ellipse_to_conic(ellipseIn);

    % Check if conic is degenerate (from invalid input ellipse)
    if all(conic(:) == 0)
        ellipseOut = invalid_ellipse();
        return;
    end

    H = tform.T';

    % Validate transformation matrix is not singular before inversion
    if abs(det(H)) < 1e-10
        ellipseOut = invalid_ellipse();
        return;
    end

    Hinv = inv(H);
    transformedConic = Hinv' * conic * Hinv;
    ellipseOut = conic_to_ellipse(transformedConic);
end

function ellipseImageSpace = map_ellipse_crop_to_image(ellipseCrop, cropBbox)
    % Map ellipse from crop space to image space

    validateattributes(cropBbox, {'numeric'}, {'vector','numel',4}, mfilename, 'cropBbox');

    xOffset = double(cropBbox(1) - 1);
    yOffset = double(cropBbox(2) - 1);

    ellipseImageSpace = ellipseCrop;
    ellipseImageSpace.center = double(ellipseCrop.center) + [xOffset, yOffset];
    ellipseImageSpace.semiMajor = double(ellipseCrop.semiMajor);
    ellipseImageSpace.semiMinor = double(ellipseCrop.semiMinor);
    ellipseImageSpace.rotation = double(ellipseCrop.rotation);
    ellipseImageSpace.valid = true;
end

function conic = ellipse_to_conic(ellipse)
    xc = ellipse.center(1);
    yc = ellipse.center(2);
    a = ellipse.semiMajor;
    b = ellipse.semiMinor;
    theta = deg2rad(ellipse.rotation);

    % Validate axes are positive to prevent division by zero
    % Minimum threshold prevents numerical instability
    MIN_AXIS = 0.1;  % pixels
    if a < MIN_AXIS || b < MIN_AXIS || ~isfinite(a) || ~isfinite(b)
        % Return degenerate conic that will be detected by conic_to_ellipse
        conic = zeros(3, 3);
        return;
    end

    c = cos(theta);
    s = sin(theta);
    R = [c -s; s c];
    D = diag([1/a^2, 1/b^2]);
    Q = R * D * R';
    center = [xc; yc];

    conic = [Q, -Q*center; -center'*Q, center'*Q*center - 1];
end

function ellipse = conic_to_ellipse(C)
    % Validate C(3,3) is not zero before normalization to prevent division by zero
    if abs(C(3,3)) < 1e-10
        ellipse = invalid_ellipse();
        return;
    end

    C = C ./ C(3,3);
    A = C(1,1);
    B = 2*C(1,2);
    Cc = C(2,2);
    D = 2*C(1,3);
    E = 2*C(2,3);
    F = C(3,3);

    denom = B^2 - 4*A*Cc;
    if denom >= 0
        ellipse = invalid_ellipse();
        return;
    end

    xc = (2*Cc*D - B*E) / denom;
    yc = (2*A*E - B*D) / denom;

    theta = 0.5 * atan2(B, A - Cc);
    cosT = cos(theta);
    sinT = sin(theta);

    A1 = A*cosT^2 + B*cosT*sinT + Cc*sinT^2;
    C1 = A*sinT^2 - B*cosT*sinT + Cc*cosT^2;

    F0 = F + D*xc + E*yc + A*xc^2 + B*xc*yc + Cc*yc^2;

    % Check finiteness first (catches NaN/Inf from upstream operations)
    if ~all(isfinite([xc, yc, A1, C1, F0]))
        ellipse = invalid_ellipse();
        return;
    end

    % Validate ellipse exists (F0 < 0 and positive diagonal elements)
    if F0 >= 0 || A1 <= 0 || C1 <= 0
        ellipse = invalid_ellipse();
        return;
    end

    % Compute axes (arguments guaranteed positive by above checks)
    a = sqrt(-F0 / A1);
    b = sqrt(-F0 / C1);

    if a < b
        tmp = a;
        a = b;
        b = tmp;
        theta = theta + pi/2;
    end

    ellipse = struct( ...
        'center', [xc, yc], ...
        'semiMajor', a, ...
        'semiMinor', b, ...
        'rotation', rad2deg(theta), ...
        'valid', true);
end

function ellipse = invalid_ellipse()
    ellipse = struct('center', [NaN, NaN], 'semiMajor', NaN, 'semiMinor', NaN, ...
                     'rotation', NaN, 'valid', false);
end

function tform = centered_rotation_tform(imageSize, angleDeg)
    height = imageSize(1);
    width = imageSize(2);
    cx = (width + 1) / 2;
    cy = (height + 1) / 2;

    cosA = cosd(angleDeg);
    sinA = sind(angleDeg);

    translateToOrigin = [1 0 0; 0 1 0; -cx -cy 1];
    rotation = [cosA -sinA 0; sinA cosA 0; 0 0 1];
    translateBack = [1 0 0; 0 1 0; cx cy 1];

    matrix = translateToOrigin * rotation * translateBack;
    tform = affine2d(matrix);
end

function angleDeg = normalize_to_angle(value, range, maxAngle)
    mid = mean(range);
    span = (range(2) - range(1)) / 2;
    if span <= 0
        angleDeg = 0;
        return;
    end
    normalized = (value - mid) / span;
    angleDeg = normalized * maxAngle;
end

function val = rand_range(range)
    val = range(1) + (range(2) - range(1)) * rand();
end

%% =========================================================================
%% COORDINATE FILE I/O
%% =========================================================================

function write_stage2_coordinates(coords, outputDir, filename)
    % Atomically write stage 2 coordinates (deduplicated by image+concentration)
    % Format: image concentration x1 y1 x2 y2 x3 y3 x4 y4

    coordFolder = outputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, filename);

    % Load existing entries (if any)
    existing = read_polygon_coordinates(coordPath);
    map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    if ~isempty(existing)
        for k = 1:numel(existing)
            e = existing(k);
            key = sprintf('%s|%d', char(e.image), e.concentration);
            map(key) = e;
        end
    end

    % Merge/override with new rows
    for i = 1:numel(coords)
        c = coords{i};
        key = sprintf('%s|%d', char(c.image), c.concentration);
        map(key) = c;
    end

    % Write atomically to a temp file then move over
    header = 'image concentration x1 y1 x2 y2 x3 y3 x4 y4';
    tmpPath = tempname(coordFolder);
    fid = fopen(tmpPath, 'wt');
    if fid == -1
        error('augmentDataset:coordWrite', 'Cannot open temp coordinates file for writing: %s', tmpPath);
    end
    fprintf(fid, '%s\n', header);

    keysArr = map.keys;
    for i = 1:numel(keysArr)
        e = map(keysArr{i});
        verts = round(e.vertices);
        fprintf(fid, '%s %d %d %d %d %d %d %d %d %d\n', ...
                e.image, e.concentration, ...
                verts(1,1), verts(1,2), verts(2,1), verts(2,2), ...
                verts(3,1), verts(3,2), verts(4,1), verts(4,2));
    end
    fclose(fid);

    % Atomic write using movefile (handles same-volume and cross-volume cases)
    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        % Fallback for cross-volume operations where movefile fails
        warning('augmentDataset:coordMove', ...
                'movefile failed (%s: %s). Attempting copyfile (cross-volume or locked file).', ...
                msgid, msg);
        [copied, cmsg] = copyfile(tmpPath, coordPath, 'f');
        if ~copied
            if exist(tmpPath, 'file') == 2, delete(tmpPath); end
            error('augmentDataset:coordWriteFail', ...
                  'Cannot write coordinates to %s: copyfile failed (%s).', coordPath, cmsg);
        end
        if exist(tmpPath, 'file') == 2, delete(tmpPath); end
    end
end

function write_stage3_coordinates(coords, outputDir, filename)
    % Atomically write stage 3 coordinates (dedup by image+concentration+replicate)
    % Format: image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle

    coordFolder = outputDir;
    if ~exist(coordFolder, 'dir')
        mkdir(coordFolder);
    end
    coordPath = fullfile(coordFolder, filename);

    % Load existing entries (if any)
    existing = read_ellipse_coordinates(coordPath);
    map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    if ~isempty(existing)
        for k = 1:numel(existing)
            e = existing(k);
            key = sprintf('%s|%d|%d', char(e.image), e.concentration, e.replicate);
            map(key) = e;
        end
    end

    % Merge/override with new rows
    for i = 1:numel(coords)
        c = coords{i};
        key = sprintf('%s|%d|%d', char(c.image), c.concentration, c.replicate);
        map(key) = c;
    end

    % Write atomically to a temp file then move over
    header = 'image concentration replicate x y semiMajorAxis semiMinorAxis rotationAngle';
    tmpPath = tempname(coordFolder);
    fid = fopen(tmpPath, 'wt');
    if fid == -1
        error('augmentDataset:coordWrite', 'Cannot open temp coordinates file for writing: %s', tmpPath);
    end
    fprintf(fid, '%s\n', header);

    keysArr = map.keys;
    for i = 1:numel(keysArr)
        e = map(keysArr{i});
        fprintf(fid, '%s %d %d %.2f %.2f %.4f %.4f %.2f\n', ...
                e.image, e.concentration, e.replicate, ...
                e.center(1), e.center(2), e.semiMajor, e.semiMinor, e.rotation);
    end
    fclose(fid);

    % Atomic write using movefile (handles same-volume and cross-volume cases)
    [ok, msg, msgid] = movefile(tmpPath, coordPath, 'f');
    if ~ok
        % Fallback for cross-volume operations where movefile fails
        warning('augmentDataset:coordMove', ...
                'movefile failed (%s: %s). Attempting copyfile (cross-volume or locked file).', ...
                msgid, msg);
        [copied, cmsg] = copyfile(tmpPath, coordPath, 'f');
        if ~copied
            if exist(tmpPath, 'file') == 2, delete(tmpPath); end
            error('augmentDataset:coordWriteFail', ...
                  'Cannot write coordinates to %s: copyfile failed (%s).', coordPath, cmsg);
        end
        if exist(tmpPath, 'file') == 2, delete(tmpPath); end
    end
end

function entries = read_polygon_coordinates(coordPath)
    % Read polygon coordinates from stage 3
    entries = struct('image', {}, 'concentration', {}, 'vertices', {});

    if ~isfile(coordPath)
        return;
    end

    fid = fopen(coordPath, 'rt');
    if fid == -1
        return;
    end
    cleaner = onCleanup(@() fclose(fid));

    % Skip header
    headerLine = fgetl(fid);
    if ~ischar(headerLine) || ~contains(lower(headerLine), 'image concentration')
        fseek(fid, 0, 'bof');
    end

    % Read data with pre-allocation
    maxEntries = 10000;
    entries = struct('image', {}, 'concentration', {}, 'vertices', {});
    entries(maxEntries).image = '';
    count = 0;

    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        trimmed = strtrim(line);
        if isempty(trimmed)
            continue;
        end

        parts = strsplit(trimmed);
        if numel(parts) < 10
            continue;
        end

        imgName = parts{1};
        concentration = str2double(parts{2});
        coords = str2double(parts(3:10));

        if any(isnan([concentration, coords]))
            continue;
        end

        vertices = reshape(coords, [2, 4])';

        count = count + 1;
        if count > maxEntries
            entries(count).image = imgName;
            entries(count).concentration = concentration;
            entries(count).vertices = vertices;
        else
            entries(count) = struct('image', imgName, ...
                                    'concentration', concentration, ...
                                    'vertices', vertices);
        end
    end

    entries = entries(1:count);
end

function entries = read_ellipse_coordinates(coordPath)
    % Read ellipse coordinates from stage 4
    entries = struct('image', {}, 'concentration', {}, 'replicate', {}, ...
                     'center', {}, 'semiMajor', {}, 'semiMinor', {}, 'rotation', {});

    if ~isfile(coordPath)
        return;
    end

    fid = fopen(coordPath, 'rt');
    if fid == -1
        return;
    end
    cleaner = onCleanup(@() fclose(fid));

    % Skip header
    headerLine = fgetl(fid);
    if ~ischar(headerLine) || ~contains(lower(headerLine), 'image concentration')
        fseek(fid, 0, 'bof');
    end

    % Read data with pre-allocation
    maxEntries = 10000;
    entries = struct('image', {}, 'concentration', {}, 'replicate', {}, ...
                     'center', {}, 'semiMajor', {}, 'semiMinor', {}, 'rotation', {});
    entries(maxEntries).image = '';
    count = 0;

    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        trimmed = strtrim(line);
        if isempty(trimmed)
            continue;
        end

        parts = strsplit(trimmed);
        if numel(parts) < 8
            continue;
        end

        imgName = parts{1};
        nums = str2double(parts(2:8));

        if any(isnan(nums))
            continue;
        end

        count = count + 1;
        if count > maxEntries
            entries(count).image = imgName;
            entries(count).concentration = nums(1);
            entries(count).replicate = nums(2);
            entries(count).center = nums(3:4);
            entries(count).semiMajor = nums(5);
            entries(count).semiMinor = nums(6);
            entries(count).rotation = nums(7);
        else
            entries(count) = struct('image', imgName, ...
                                    'concentration', nums(1), ...
                                    'replicate', nums(2), ...
                                    'center', nums(3:4), ...
                                    'semiMajor', nums(5), ...
                                    'semiMinor', nums(6), ...
                                    'rotation', nums(7));
        end
    end

    entries = entries(1:count);
end

%% =========================================================================
%% GROUPING AND LOOKUP
%% =========================================================================

function ellipseMap = group_ellipses_by_parent(ellipseEntries, hasEllipses)
    % Group ellipses by parent image and concentration
    ellipseMap = containers.Map('KeyType', 'char', 'ValueType', 'any');

    if ~hasEllipses
        return;
    end

    for i = 1:numel(ellipseEntries)
        e = ellipseEntries(i);
        % Extract base image name
        imgName = e.image;
        tokens = regexp(imgName, '(.+)_con_\d+', 'tokens');
        if ~isempty(tokens)
            baseName = tokens{1}{1};
        else
            [~, baseName, ~] = fileparts(imgName);
        end

        key = sprintf('%s#%d', baseName, e.concentration);

        if ~isKey(ellipseMap, key)
            ellipseMap(key) = e;
        else
            buf = ellipseMap(key);
            ellipseMap(key) = [buf, e];
        end
    end
end

function paperGroups = group_polygons_by_image(polygonEntries)
    % Group polygons by source image name
    paperGroups = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:numel(polygonEntries)
        p = polygonEntries(i);
        [~, imgBase, ~] = fileparts(p.image);

        if ~isKey(paperGroups, imgBase)
            paperGroups(imgBase) = p;
        else
            buf = paperGroups(imgBase);
            paperGroups(imgBase) = [buf, p];
        end
    end
end

%% =========================================================================
%% VALIDATION AND UTILITIES
%% =========================================================================

function valid = is_valid_polygon(vertices, minArea)
    % Check if polygon is valid (non-degenerate)
    area = polyarea(vertices(:,1), vertices(:,2));
    valid = area > minArea;
end

function positions = place_polygons_nonoverlapping(polygonBboxes, bgWidth, bgHeight, margin, minSpacing, maxAttempts)
    % Place polygons at random non-overlapping positions using spatial grid acceleration
    % Returns cell array of position structs {x, y} or empty if placement fails

    numPolygons = numel(polygonBboxes);
    positions = cell(numPolygons, 1);

    % Sort by area (largest first)
    areas = zeros(numPolygons, 1);
    for i = 1:numPolygons
        areas(i) = polygonBboxes{i}.width * polygonBboxes{i}.height;
    end
    [~, sortOrder] = sort(areas, 'descend');

    % Spatial grid for fast neighbor lookup
    cellSize = max(50, minSpacing);
    gridWidth = ceil(bgWidth / cellSize);
    gridHeight = ceil(bgHeight / cellSize);
    grid = cell(gridHeight, gridWidth);

    placedBboxes = zeros(numPolygons, 4);

    for idx = 1:numPolygons
        i = sortOrder(idx);
        bbox = polygonBboxes{i};
        placed = false;

        for attempt = 1:maxAttempts
            % Uniform random position
            x = margin + rand() * max(1, bgWidth - bbox.width - 2*margin);
            y = margin + rand() * max(1, bgHeight - bbox.height - 2*margin);

            candidateBbox = [x, y, x + bbox.width, y + bbox.height];

            % Check collision using grid
            minCellX = max(1, floor(x / cellSize));
            maxCellX = min(gridWidth, ceil((x + bbox.width) / cellSize));
            minCellY = max(1, floor(y / cellSize));
            maxCellY = min(gridHeight, ceil((y + bbox.height) / cellSize));

            collision = false;
            for cy = minCellY:maxCellY
                for cx = minCellX:maxCellX
                    neighborIndices = grid{cy, cx};
                    for j = 1:numel(neighborIndices)
                        if bboxes_overlap(candidateBbox, placedBboxes(neighborIndices(j),:), minSpacing)
                            collision = true;
                            break;
                        end
                    end
                    if collision, break; end
                end
                if collision, break; end
            end

            if ~collision
                % Place polygon and update grid
                positions{i} = struct('x', x, 'y', y);
                placedBboxes(i, :) = candidateBbox;

                for cy = minCellY:maxCellY
                    for cx = minCellX:maxCellX
                        grid{cy, cx}(end+1) = i;
                    end
                end

                placed = true;
                break;
            end
        end

        if ~placed
            positions = [];
            return;
        end
    end
end

function overlap = bboxes_overlap(bbox1, bbox2, minSpacing)
    % Check if two axis-aligned bounding boxes overlap with minimum spacing
    % bbox format: [x1, y1, x2, y2]

    % Expand bboxes by minSpacing/2 on all sides
    bbox1_expanded = [bbox1(1) - minSpacing/2, bbox1(2) - minSpacing/2, ...
                      bbox1(3) + minSpacing/2, bbox1(4) + minSpacing/2];
    bbox2_expanded = [bbox2(1) - minSpacing/2, bbox2(2) - minSpacing/2, ...
                      bbox2(3) + minSpacing/2, bbox2(4) + minSpacing/2];

    % Check for overlap
    overlap = ~(bbox1_expanded(3) < bbox2_expanded(1) || ...  % bbox1 is left of bbox2
                bbox1_expanded(1) > bbox2_expanded(3) || ...  % bbox1 is right of bbox2
                bbox1_expanded(4) < bbox2_expanded(2) || ...  % bbox1 is above bbox2
                bbox1_expanded(2) > bbox2_expanded(4));       % bbox1 is below bbox2
end

function imgPath = find_stage2_image(folder, baseName, supportedFormats)
    % Find stage 2 image by base name
    imgPath = '';
    for i = 1:numel(supportedFormats)
        candidate = fullfile(folder, [baseName supportedFormats{i}]);
        if isfile(candidate)
            imgPath = candidate;
            return;
        end
    end
end

function phoneDirs = list_phones(stage2Root)
    if ~isfolder(stage2Root)
        phoneDirs = {};
        return;
    end
    d = dir(stage2Root);
    mask = [d.isdir] & ~ismember({d.name}, {'.', '..'});
    phoneDirs = {d(mask).name};
end

function img = clamp_uint8(img)
    % Clamp image values to [0, 255] and convert to uint8
    img = uint8(min(255, max(0, img)));
end

function img = apply_motion_blur(img)
    % Apply slight motion blur with random length and angle
    len = 4 + randi(4);            % 5-8 px
    ang = rand() * 180;            % degrees
    psf = fspecial('motion', len, ang);
    img = imfilter(img, psf, 'replicate');
end

function I = imread_raw(fname)
    % Read image with EXIF orientation handling for 90-degree rotations
    % Inverts EXIF 90-degree rotation tags (5/6/7/8) to preserve raw sensor layout

    try
        I = imread(fname, 'AutoOrient', false);
    catch
        I = imread(fname);
    end

    try
        info = imfinfo(fname);
        if ~isfield(info, 'Orientation') || isempty(info.Orientation), return; end
        ori = double(info.Orientation);
    catch
        return;
    end

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

function ensure_folder(pathStr)
    if ~isempty(pathStr) && ~isfolder(pathStr)
        mkdir(pathStr);
    end
end

function projectRoot = find_project_root(inputFolder)
    % Find project root by searching up directory tree
    currentDir = pwd;
    searchDir = currentDir;
    maxLevels = 5;

    for level = 1:maxLevels
        [parentDir, ~] = fileparts(searchDir);

        if exist(fullfile(searchDir, inputFolder), 'dir')
            projectRoot = searchDir;
            return;
        end

        if strcmp(searchDir, parentDir)
            break;
        end
        searchDir = parentDir;
    end

    warning('augmentDataset:pathResolution', ...
            'Could not find input folder "%s". Using current directory.', inputFolder);
    projectRoot = currentDir;
end

%% =========================================================================
%% CORNER KEYPOINT LABEL EXPORT FUNCTIONS (Phase 1.4)
%% =========================================================================

function export_corner_labels(outputDir, imageName, polygons, imageSize)
    % Export corner labels in keypoint detection format
    % Format: JSON with corner heatmap targets + sub-pixel offsets + embeddings

    labelDir = fullfile(outputDir, 'labels');
    if ~isfolder(labelDir), mkdir(labelDir); end

    labelPath = fullfile(labelDir, [imageName '.json']);

    labels = struct();
    labels.image_size = imageSize;
    labels.image_name = imageName;
    labels.quads = [];

    for i = 1:numel(polygons)
        quad = polygons{i};  % Extract 4x2 vertices from cell array

        % Order corners: TL, TR, BR, BL (clockwise from top-left)
        quad = order_corners_clockwise(quad);

        % Generate Gaussian heatmap targets (sigma=3 for sub-pixel accuracy)
        heatmaps = generate_gaussian_targets(quad, imageSize, 3);

        % Compute sub-pixel offsets (CRITICAL for <3px accuracy)
        offsets = compute_subpixel_offsets(quad, imageSize);

        % Embedding ID for grouping (each quad gets unique ID)
        embeddingID = i;

        quadStruct = struct( ...
            'quad_id', i, ...
            'corners', quad, ...
            'corners_normalized', quad ./ [imageSize(2), imageSize(1)], ...
            'heatmaps', heatmaps, ...
            'offsets', offsets, ...
            'embedding_id', embeddingID);

        if isempty(labels.quads)
            labels.quads = quadStruct;
        else
            labels.quads(end+1) = quadStruct;
        end
    end

    % Write JSON (atomic write pattern)
    tmpPath = tempname(labelDir);
    fid = fopen(tmpPath, 'w');
    if fid < 0
        error('augmentDataset:jsonWrite', 'Cannot write label file: %s', labelPath);
    end
    jsonStr = jsonencode(labels, 'PrettyPrint', true);
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
    movefile(tmpPath, labelPath, 'f');
end

function quad_ordered = order_corners_clockwise(quad)
    % Order vertices: TL, TR, BR, BL (clockwise from top-left)
    % Method: Sort by angle from centroid, then rotate to start from top-left

    centroid = mean(quad, 1);
    angles = atan2(quad(:,2) - centroid(2), quad(:,1) - centroid(1));
    [~, order] = sort(angles);
    quad_ordered = quad(order, :);

    % Ensure top-left corner is first (minimum distance from origin)
    [~, topLeftIdx] = min(sum(quad_ordered.^2, 2));
    quad_ordered = circshift(quad_ordered, -topLeftIdx + 1, 1);
end

function heatmaps = generate_gaussian_targets(quad, imageSize, sigma)
    % Generate 4 separate heatmaps (one per corner type: TL, TR, BR, BL)
    % Output resolution: imageSize / 4 (downsampled for efficiency)

    H = round(imageSize(1) / 4);
    W = round(imageSize(2) / 4);
    heatmaps = zeros(4, H, W, 'single');

    for i = 1:4
        % Normalize corner to downsampled space
        cx = quad(i, 1) * W / imageSize(2);
        cy = quad(i, 2) * H / imageSize(1);

        % Generate 2D Gaussian centered at (cx, cy)
        [X, Y] = meshgrid(1:W, 1:H);
        gaussian = exp(-((X - cx).^2 + (Y - cy).^2) / (2 * sigma^2));

        % Normalize to [0, 1]
        if max(gaussian(:)) > 0
            heatmaps(i, :, :) = gaussian / max(gaussian(:));
        end
    end
end

function offsets = compute_subpixel_offsets(quad, imageSize)
    % Compute fractional pixel offsets for sub-pixel accuracy
    % These offsets allow the model to predict corners with <1px precision

    H = round(imageSize(1) / 4);
    W = round(imageSize(2) / 4);
    offsets = zeros(4, 2, 'single');

    for i = 1:4
        % Normalize coordinates to downsampled space
        cx = quad(i, 1) * W / imageSize(2);
        cy = quad(i, 2) * H / imageSize(1);

        % Separate integer and fractional parts
        cx_int = floor(cx);
        cy_int = floor(cy);

        % Fractional offsets (0 to 1)
        offsets(i, 1) = cx - cx_int;  % dx
        offsets(i, 2) = cy - cy_int;  % dy
    end
end
