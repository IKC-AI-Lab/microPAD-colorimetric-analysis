function augSynth = augmentation_synthesis()
    %% AUGMENTATION_SYNTHESIS Returns a struct of function handles for synthetic data generation
    %
    % This utility module consolidates all synthetic generation for the augmentation
    % pipeline: background textures, sparse artifacts, and polygon distractors.
    %
    % Usage:
    %   augSynth = augmentation_synthesis();
    %
    %   % Background generation
    %   bg = augSynth.bg.generate(width, height, textureCfg);
    %   texture = augSynth.bg.borrowTexture(surfaceType, width, height, textureCfg);
    %
    %   % Sparse artifacts
    %   bg = augSynth.artifacts.addSparse(bg, width, height, artifactCfg);
    %   mask = augSynth.artifacts.generatePolygonMask(verticesPix, targetSize);
    %
    %   % Polygon distractors
    %   [bg, count] = augSynth.distractors.addPolygon(bg, regions, bboxes, occupied, cfg, funcs);
    %
    % See also: augment_dataset

    %% Load image_io module for clampUint8 utility
    persistent imageIOModule
    if isempty(imageIOModule)
        imageIOModule = image_io();
    end

    %% Public API - Background synthesis
    augSynth.bg.generate = @generateRealisticLabSurface;
    augSynth.bg.generateRaw = @generateRealisticLabSurfaceRaw;
    augSynth.bg.borrowTexture = @borrowBackgroundTexture;
    augSynth.bg.initializePool = @initializeBackgroundTexturePool;
    augSynth.bg.poolConfigChanged = @texturePoolConfigChanged;
    augSynth.bg.generateTextureBase = @generateSurfaceTextureBase;
    augSynth.bg.applyTextureJitter = @applyTexturePoolJitter;
    augSynth.bg.generateLaminateTexture = @generateLaminateTexture;
    augSynth.bg.generateSkinTexture = @generateSkinTexture;
    augSynth.bg.addLightingGradient = @addLightingGradient;

    %% Public API - Sparse artifacts
    augSynth.artifacts.addSparse = @addSparseArtifacts;
    augSynth.artifacts.generatePolygonMask = @generatePolygonMask;

    %% Public API - Polygon distractors
    augSynth.distractors.addPolygon = @addPolygonDistractors;
    augSynth.distractors.sampleType = @sampleDistractorType;
    augSynth.distractors.synthesizePatch = @synthesizeDistractorPatch;
    augSynth.distractors.finalizePatch = @finalizeDistractorPatch;
    augSynth.distractors.jitterPatch = @jitterPolygonPatch;
    augSynth.distractors.scalePatch = @scaleDistractorPatch;
    augSynth.distractors.sampleColor = @sampleDistractorColor;
    augSynth.distractors.synthesizeTexture = @synthesizeDistractorTexture;
    augSynth.distractors.computeOutlineMask = @computeOutlineMask;
    augSynth.distractors.sampleOutlineWidth = @sampleOutlineWidth;

    %% Public API - Utilities (shared)
    augSynth.clampUint8 = imageIOModule.clampUint8;
    augSynth.resolveRange = @resolveRange;
    augSynth.sampleRangeValue = @sampleRangeValue;
    augSynth.castPatchLikeTemplate = @castPatchLikeTemplate;
end

%% =========================================================================
%% BACKGROUND SYNTHESIS - MAIN GENERATION
%% =========================================================================

function bg = generateRealisticLabSurface(width, height, textureCfg)
    %% Generate realistic lab surface backgrounds with pooled single-precision textures
    %
    % This function generates a background and clamps to uint8, but does NOT
    % add sparse artifacts. Use generateBackgroundRaw() for single-precision output.
    %
    % Inputs:
    %   width - Background width in pixels
    %   height - Background height in pixels
    %   textureCfg - Texture configuration struct
    %
    % Output:
    %   bg - uint8 RGB background image [height x width x 3]

    bgSingle = generateRealisticLabSurfaceRaw(width, height, textureCfg);
    bg = uint8(min(255, max(0, bgSingle)));
end

function bg = generateRealisticLabSurfaceRaw(width, height, textureCfg)
    %% Generate realistic lab surface backgrounds (single-precision output)
    %
    % Returns single-precision output for further processing before clamping.
    %
    % Inputs:
    %   width - Background width in pixels
    %   height - Background height in pixels
    %   textureCfg - Texture configuration struct
    %
    % Output:
    %   bg - single RGB background image [height x width x 3]

    width = max(1, round(width));
    height = max(1, round(height));

    surfaceType = randi(4);
    texture = borrowBackgroundTexture(surfaceType, width, height, textureCfg);

    switch surfaceType
        case 1  % Uniform surface
            baseRGB = textureCfg.uniformBaseRGB + randi([-textureCfg.uniformVariation, textureCfg.uniformVariation], [1, 3]);
            noiseAmplitude = textureCfg.uniformNoiseRange(1) + rand() * diff(textureCfg.uniformNoiseRange);
            texture = texture .* single(noiseAmplitude);
        case 2  % Speckled surface
            baseGray = 160 + randi([-25, 25]);
            baseRGB = [baseGray, baseGray, baseGray] + randi([-5, 5], [1, 3]);
        case 3  % Laminate surface
            if rand() < 0.5
                baseRGB = [245, 245, 245] + randi([-5, 5], [1, 3]);
            else
                baseRGB = [30, 30, 30] + randi([-5, 5], [1, 3]);
            end
        otherwise  % Skin texture
            h = 0.03 + rand() * 0.07;
            s = 0.25 + rand() * 0.35;
            v = 0.55 + rand() * 0.35;
            baseRGB = round(255 * hsv2rgb([h, s, v]));
    end

    baseRGB = max(100, min(230, baseRGB));

    numChannels = 3;  % RGB backgrounds only
    bg = repmat(reshape(single(baseRGB), [1, 1, numChannels]), [height, width, 1]);
    for c = 1:numChannels
        bg(:,:,c) = bg(:,:,c) + texture;
    end

    if rand() < 0.60
        bg = addLightingGradient(bg, width, height);
    end
end

%% =========================================================================
%% BACKGROUND SYNTHESIS - TEXTURE POOL MANAGEMENT
%% =========================================================================

function texture = borrowBackgroundTexture(surfaceType, width, height, textureCfg)
    %% Borrow a texture from the pool with lazy initialization and jitter
    persistent poolState

    if isempty(poolState) || texturePoolConfigChanged(poolState, width, height, textureCfg)
        if ~isempty(poolState)
            oldWidth = poolState.width;
            oldHeight = poolState.height;
            widthDiff = abs(width - oldWidth) / max(oldWidth, 1);
            heightDiff = abs(height - oldHeight) / max(oldHeight, 1);
            if widthDiff > 0.01 || heightDiff > 0.01
                warning('augmentation_synthesis:poolDimensionChange', ...
                    'Background dimensions changed from %dx%d to %dx%d. Texture pool reset.', ...
                    oldWidth, oldHeight, width, height);
            end
        end
        poolState = initializeBackgroundTexturePool(width, height, textureCfg);
    end

    entry = poolState.surface(surfaceType);
    if entry.cursor > entry.poolSize
        entry.order = randperm(entry.poolSize);
        entry.cursor = 1;
    end

    slot = entry.order(entry.cursor);
    entry.cursor = entry.cursor + 1;

    baseTexture = entry.textures{slot};
    if isempty(baseTexture)
        baseTexture = generateSurfaceTextureBase(surfaceType, width, height, textureCfg);
        entry.textures{slot} = baseTexture;
    end

    texture = applyTexturePoolJitter(baseTexture, poolState);

    entry.usage(slot) = entry.usage(slot) + 1;
    if entry.usage(slot) >= poolState.refreshInterval
        entry.textures{slot} = [];
        entry.usage(slot) = 0;
    end

    poolState.surface(surfaceType) = entry;
end

function poolState = initializeBackgroundTexturePool(width, height, textureCfg)
    %% Initialize texture pool state
    requestedPoolSize = max(1, round(textureCfg.poolSize));
    bytesPerTexture = max(1, double(width) * double(height) * 4);
    surfaces = 4;
    maxPoolBytes = textureCfg.poolMaxMemoryMB * 1024 * 1024;
    maxPerSurface = max(1, floor((maxPoolBytes / surfaces) / bytesPerTexture));
    poolSize = min(requestedPoolSize, maxPerSurface);
    refreshInterval = max(1, round(textureCfg.poolRefreshInterval));

    surfaceTemplate = struct( ...
        'textures', {cell(poolSize, 1)}, ...
        'usage', zeros(poolSize, 1, 'uint32'), ...
        'order', randperm(poolSize), ...
        'cursor', 1, ...
        'poolSize', poolSize);

    poolState = struct();
    poolState.width = width;
    poolState.height = height;
    poolState.cfgSnapshot = textureCfg;
    poolState.poolSize = poolSize;
    poolState.refreshInterval = refreshInterval;
    poolState.shiftPixels = max(0, round(textureCfg.poolShiftPixels));
    poolState.scaleRange = sort(textureCfg.poolScaleRange);
    if numel(poolState.scaleRange) ~= 2 || any(~isfinite(poolState.scaleRange))
        poolState.scaleRange = [1, 1];
    end
    poolState.flipProb = max(0, min(1, textureCfg.poolFlipProbability));
    poolState.surface = repmat(surfaceTemplate, 1, 4);
    for st = 1:4
        entry = poolState.surface(st);
        entry.order = randperm(entry.poolSize);
        entry.cursor = 1;
        entry.usage(:) = 0;
        poolState.surface(st) = entry;
    end
end

function changed = texturePoolConfigChanged(poolState, width, height, textureCfg)
    %% Check if pool configuration has changed
    changed = poolState.width ~= width || poolState.height ~= height || ~isequal(poolState.cfgSnapshot, textureCfg);
end

%% =========================================================================
%% BACKGROUND SYNTHESIS - TEXTURE GENERATORS
%% =========================================================================

function texture = generateSurfaceTextureBase(surfaceType, width, height, textureCfg)
    %% Generate base texture for a surface type
    height = max(1, round(double(height)));
    width = max(1, round(double(width)));

    persistent randBuffer1 randBuffer2 bufferSize
    if isempty(bufferSize) || any(bufferSize ~= [height, width])
        randBuffer1 = zeros(height, width, 'single');
        randBuffer2 = zeros(height, width, 'single');
        bufferSize = [height, width];
    end

    switch surfaceType
        case 1  % Uniform noise baseline (scaled per sample)
            randBuffer1(:) = single(randn(height, width));
            texture = randBuffer1;
        case 2  % Speckled surface (high + low frequency noise)
            randBuffer1(:) = single(randn(height, width));
            randBuffer1 = randBuffer1 .* single(textureCfg.speckleHighFreq);

            randBuffer2(:) = single(randn(height, width));
            randBuffer2 = imgaussfilt(randBuffer2, 8);
            randBuffer2 = randBuffer2 .* single(textureCfg.speckleLowFreq);

            texture = randBuffer1 + randBuffer2;
        case 3  % Laminate grain
            texture = generateLaminateTexture(width, height, textureCfg);
        case 4  % Skin-like microtexture
            texture = generateSkinTexture(width, height, textureCfg);
        otherwise
            randBuffer1(:) = single(randn(height, width));
            texture = randBuffer1;
    end
end

function texture = applyTexturePoolJitter(baseTexture, poolState)
    %% Apply random jitter to pooled texture
    texture = baseTexture;

    if poolState.shiftPixels > 0
        shiftX = randi([-poolState.shiftPixels, poolState.shiftPixels]);
        shiftY = randi([-poolState.shiftPixels, poolState.shiftPixels]);
        if shiftX ~= 0 || shiftY ~= 0
            texture = circshift(texture, [shiftY, shiftX]);
        end
    end

    if poolState.flipProb > 0
        if rand() < poolState.flipProb
            texture = flip(texture, 2);
        end
        if rand() < poolState.flipProb
            texture = flip(texture, 1);
        end
    end

    scaleRange = poolState.scaleRange;
    if numel(scaleRange) == 2 && scaleRange(2) > scaleRange(1)
        scale = scaleRange(1) + rand() * (scaleRange(2) - scaleRange(1));
        texture = texture .* single(scale);
    end
end

function texture = generateLaminateTexture(width, height, textureCfg)
    %% Generate high-contrast laminate surface with subtle noise (single precision)
    width = max(1, round(width));
    height = max(1, round(height));
    texture = single(randn(height, width)) .* single(textureCfg.laminateNoiseStrength);
end

function texture = generateSkinTexture(width, height, textureCfg)
    %% Generate subtle skin-like microtexture (single precision)
    width = max(1, round(width));
    height = max(1, round(height));

    lowFreq = imgaussfilt(single(randn(height, width)), 12) .* single(textureCfg.skinLowFreqStrength);
    midFreq = imgaussfilt(single(randn(height, width)), 3) .* single(textureCfg.skinMidFreqStrength);
    highFreq = single(randn(height, width)) .* single(textureCfg.skinHighFreqStrength);

    texture = lowFreq + midFreq + highFreq;
end

%% =========================================================================
%% BACKGROUND SYNTHESIS - EFFECTS
%% =========================================================================

function bg = addLightingGradient(bg, width, height)
    %% Add simple linear lighting gradient to simulate directional lighting
    width = max(1, round(width));
    height = max(1, round(height));
    if width < 50 || height < 50
        return;
    end

    lightAngle = rand() * 2 * pi;
    xAxis = single(0:(width - 1));
    yAxis = single(0:(height - 1));
    if width > 1
        xAxis = xAxis / single(width - 1);
    else
        xAxis = zeros(size(xAxis), 'single');
    end
    if height > 1
        yAxis = yAxis / single(height - 1);
    else
        yAxis = zeros(size(yAxis), 'single');
    end

    [Ygrid, Xgrid] = ndgrid(yAxis, xAxis);
    projection = Xgrid .* single(cos(lightAngle)) + Ygrid .* single(sin(lightAngle));

    gradientStrength = single(0.05 + rand() * 0.05);
    gradient = single(1) - gradientStrength/2 + projection .* gradientStrength;
    gradient = max(single(0.90), min(single(1.10), gradient));

    for c = 1:size(bg, 3)
        bg(:,:,c) = bg(:,:,c) .* gradient;
    end
end

%% =========================================================================
%% SPARSE ARTIFACTS - MAIN FUNCTION
%% =========================================================================

function bg = addSparseArtifacts(bg, width, height, artifactCfg)
    %% Add variable-density artifacts anywhere on background for robust detection training
    %
    % OPTIMIZATION: Ellipses/lines use unit-square normalization (default 64x64 defined
    % in artifactCfg.unitMaskSize) to avoid large meshgrid allocations. Polygonal artifacts
    % render directly at target resolution so corner geometry remains crisp.
    %
    % Inputs:
    %   bg - Background image [height x width x channels] (uint8 or single)
    %   width - Image width in pixels
    %   height - Image height in pixels
    %   artifactCfg - Artifact configuration struct with fields:
    %       .countRange - [min, max] number of artifacts
    %       .sizeRangePercent - [min, max] size as fraction of diagonal
    %       .minSizePixels - Minimum artifact size
    %       .unitMaskSize - Resolution for ellipse/line unit masks
    %       .overhangMargin - Fraction of size for boundary overhang
    %       .lineWidthRatio - Line width as fraction of length
    %       .lineRotationPadding - Extra padding for line rotation
    %       .ellipseRadiusARange - [min, max] ellipse semi-major axis fraction
    %       .ellipseRadiusBRange - [min, max] ellipse semi-minor axis fraction
    %       .rectangleSizeRange - [min, max] rectangle size fraction
    %       .quadSizeRange - [min, max] quadrilateral size fraction
    %       .quadPerturbation - Vertex perturbation for quads
    %       .triangleSizeRange - [min, max] triangle size fraction
    %       .lineIntensityRange - [min, max] line intensity offset
    %       .blobDarkIntensityRange - [min, max] dark blob intensity
    %       .blobLightIntensityRange - [min, max] light blob intensity
    %
    % Output:
    %   bg - Background image with artifacts added
    %
    % Artifacts: rectangles, quadrilaterals, triangles, ellipses, lines
    % Count: configurable via artifactCfg.countRange (default 5-30)
    % Size: 1-100% of image diagonal (allows artifacts larger than frame)
    % Placement: unconstrained (artifacts can extend beyond boundaries for uniform spatial distribution)

    % Quick guard for tiny backgrounds
    width = max(1, round(width));
    height = max(1, round(height));
    if width < 8 || height < 8
        return;
    end

    % Number of artifacts: variable (1-100 by default)
    numArtifacts = randi([artifactCfg.countRange(1), artifactCfg.countRange(2)]);

    % Image diagonal for relative sizing (allows artifacts larger than image dimensions)
    diagSize = sqrt(width^2 + height^2);

    if isfield(artifactCfg, 'unitMaskSize') && ~isempty(artifactCfg.unitMaskSize)
        unitMaskSize = max(8, round(double(artifactCfg.unitMaskSize)));
    else
        unitMaskSize = 64;
    end
    unitCoords = linspace(0, 1, unitMaskSize);
    [unitGridX, unitGridY] = meshgrid(unitCoords, unitCoords);
    unitCenteredX = unitGridX - 0.5;
    unitCenteredY = unitGridY - 0.5;

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

        % Create artifact mask; polygons draw directly at target resolution to keep sharp edges
        mask = [];
        unitMask = [];
        switch artifactType
            case 'ellipse'
                radiusAFraction = 0.5 * (artifactCfg.ellipseRadiusARange(1) + rand() * diff(artifactCfg.ellipseRadiusARange));
                radiusBFraction = 0.5 * (artifactCfg.ellipseRadiusBRange(1) + rand() * diff(artifactCfg.ellipseRadiusBRange));
                radiusAFraction = max(radiusAFraction, 1e-3);
                radiusBFraction = max(radiusBFraction, 1e-3);
                angle = rand() * pi;
                cosTheta = cos(angle);
                sinTheta = sin(angle);
                xRot = unitCenteredX * cosTheta - unitCenteredY * sinTheta;
                yRot = unitCenteredX * sinTheta + unitCenteredY * cosTheta;
                unitMask = single((xRot / radiusAFraction).^2 + (yRot / radiusBFraction).^2 <= 1);

            case 'rectangle'
                rectWidthFraction = artifactCfg.rectangleSizeRange(1) + rand() * diff(artifactCfg.rectangleSizeRange);
                rectHeightFraction = artifactCfg.rectangleSizeRange(1) + rand() * diff(artifactCfg.rectangleSizeRange);
                rectHalfWidth = max(rectWidthFraction * artifactSize / 2, 0.5);
                rectHalfHeight = max(rectHeightFraction * artifactSize / 2, 0.5);
                angle = rand() * pi;
                cosTheta = cos(angle);
                sinTheta = sin(angle);
                baseVerts = [
                    -rectHalfWidth, -rectHalfHeight;
                    rectHalfWidth, -rectHalfHeight;
                    rectHalfWidth,  rectHalfHeight;
                    -rectHalfWidth,  rectHalfHeight];
                rotMatrix = [cosTheta, -sinTheta; sinTheta, cosTheta];
                rotatedVerts = baseVerts * rotMatrix';
                centerPix = [(artifactSize + 1) / 2, (artifactSize + 1) / 2];
                verticesPix = rotatedVerts + centerPix;
                mask = generatePolygonMask(verticesPix, artifactSize);

            case 'quadrilateral'
                baseWidthFraction = artifactCfg.quadSizeRange(1) + rand() * diff(artifactCfg.quadSizeRange);
                baseHeightFraction = artifactCfg.quadSizeRange(1) + rand() * diff(artifactCfg.quadSizeRange);
                perturbFraction = artifactCfg.quadPerturbation;
                halfWidth = max(baseWidthFraction / 2, 1e-3);
                halfHeight = max(baseHeightFraction / 2, 1e-3);
                verticesNorm = [
                    0.5 - halfWidth + (rand()-0.5) * perturbFraction, 0.5 - halfHeight + (rand()-0.5) * perturbFraction;
                    0.5 + halfWidth + (rand()-0.5) * perturbFraction, 0.5 - halfHeight + (rand()-0.5) * perturbFraction;
                    0.5 + halfWidth + (rand()-0.5) * perturbFraction, 0.5 + halfHeight + (rand()-0.5) * perturbFraction;
                    0.5 - halfWidth + (rand()-0.5) * perturbFraction, 0.5 + halfHeight + (rand()-0.5) * perturbFraction
                ];
                centeredVerts = (verticesNorm - 0.5) * (artifactSize - 1);
                centerPix = [(artifactSize + 1) / 2, (artifactSize + 1) / 2];
                verticesPix = centeredVerts + centerPix;
                mask = generatePolygonMask(verticesPix, artifactSize);

            case 'triangle'
                baseSizeFraction = artifactCfg.triangleSizeRange(1) + rand() * diff(artifactCfg.triangleSizeRange);
                radius = max(baseSizeFraction * (artifactSize - 1) / 2, 0.5);
                angle = rand() * 2 * pi;
                verticesNorm = [
                    cos(angle),           sin(angle);
                    cos(angle + 2*pi/3),  sin(angle + 2*pi/3);
                    cos(angle + 4*pi/3),  sin(angle + 4*pi/3)
                ];
                centeredVerts = radius * verticesNorm;
                centerPix = [(artifactSize + 1) / 2, (artifactSize + 1) / 2];
                verticesPix = centeredVerts + centerPix;
                mask = generatePolygonMask(verticesPix, artifactSize);

            otherwise  % 'line'
                angle = rand() * pi;
                cosTheta = cos(angle);
                sinTheta = sin(angle);
                lengthNorm = min(1, lineLength / artifactSize);
                halfLengthNorm = max(lengthNorm / 2, 1e-3);
                halfWidthNorm = max(lineWidth / artifactSize, 1 / artifactSize);
                xRot = unitCenteredX * cosTheta - unitCenteredY * sinTheta;
                yRot = unitCenteredX * sinTheta + unitCenteredY * cosTheta;
                lineCore = (abs(xRot) <= halfLengthNorm) & (abs(yRot) <= halfWidthNorm);
                unitMask = single(lineCore);
        end

        if isempty(mask) && isempty(unitMask)
            continue;
        end

        if isempty(mask)
            mask = imresize(unitMask, [artifactSize, artifactSize], 'nearest');
            mask = max(mask, single(0));
            mask = min(mask, single(1));
        end
        if ~any(mask(:))
            continue;
        end

        % Random intensity: darker or lighter
        % Lines tend to be darker; blobs can be either
        if strcmp(artifactType, 'line')
            intensity = randi([artifactCfg.lineIntensityRange(1), artifactCfg.lineIntensityRange(2)]);
        else
            if rand() < 0.5
                intensity = randi([artifactCfg.blobDarkIntensityRange(1), artifactCfg.blobDarkIntensityRange(2)]);
            else
                intensity = randi([artifactCfg.blobLightIntensityRange(1), artifactCfg.blobLightIntensityRange(2)]);
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
        maskRegion = single(mask(maskYStart:maskYEnd, maskXStart:maskXEnd));
        intensitySingle = single(intensity);
        numChannels = size(bg, 3);
        for c = 1:numChannels
            region = single(bg(yStart:yEnd, xStart:xEnd, c));
            region = region + maskRegion .* intensitySingle;
            bg(yStart:yEnd, xStart:xEnd, c) = uint8(min(255, max(0, region)));
        end
    end
end

function mask = generatePolygonMask(verticesPix, targetSize)
    %% Rasterize polygon vertices expressed in pixel coordinates into a binary mask
    %
    % Inputs:
    %   verticesPix - Nx2 matrix of [x, y] vertex coordinates in pixels
    %   targetSize - Output mask dimension (creates targetSize x targetSize mask)
    %
    % Output:
    %   mask - Single-precision binary mask, or empty if invalid

    if isempty(verticesPix) || size(verticesPix, 2) ~= 2
        mask = [];
        return;
    end

    verticesPix = double(verticesPix);
    polyMask = poly2mask(verticesPix(:,1), verticesPix(:,2), targetSize, targetSize);
    if ~any(polyMask(:))
        mask = [];
        return;
    end

    mask = single(polyMask);
end

%% =========================================================================
%% POLYGON DISTRACTORS - MAIN GENERATION
%% =========================================================================

function [bg, placedCount] = addPolygonDistractors(bg, regions, polygonBboxes, occupiedBboxes, cfg, placementFuncs)
    % Inject additional polygon-shaped distractors matching source geometry statistics.
    %
    % Inputs:
    %   bg - Background image to composite distractors onto
    %   regions - Cell array of source region structures
    %   polygonBboxes - Cell array of bbox info structs for each region
    %   occupiedBboxes - Nx4 matrix of already-occupied bounding boxes
    %   cfg - Configuration struct with .distractors, .texture, .placement fields
    %   placementFuncs - Struct with function handles:
    %       .randomTopLeft - @(bbox, margin, bgW, bgH) -> [x, y]
    %       .bboxesOverlap - @(bbox1, bbox2, spacing) -> bool
    %       .compositeToBackground - @(bg, patch, verts) -> bg
    %
    % Outputs:
    %   bg - Updated background with distractors composited
    %   placedCount - Number of distractors successfully placed

    distractorCfg = cfg.distractors;
    if isempty(regions) || ~distractorCfg.enabled
        placedCount = 0;
        return;
    end

    targetCount = randi([distractorCfg.minCount, distractorCfg.maxCount]);
    if targetCount <= 0
        placedCount = 0;
        return;
    end

    [bgHeight, bgWidth, ~] = size(bg);
    if bgHeight < 1 || bgWidth < 1
        placedCount = 0;
        return;
    end

    numSource = numel(regions);
    maxBboxes = targetCount + size(occupiedBboxes, 1);
    allBboxes = zeros(maxBboxes, 4);
    bboxCount = 0;

    if ~isempty(occupiedBboxes)
        bboxCount = size(occupiedBboxes, 1);
        allBboxes(1:bboxCount, :) = double(occupiedBboxes);
    end

    placedCount = 0;
    for k = 1:targetCount
        srcIdx = randi(numSource);
        region = regions{srcIdx};
        bboxInfo = polygonBboxes{srcIdx};
        templatePatch = region.augPolygonImg;

        if isempty(templatePatch) || bboxInfo.width <= 0 || bboxInfo.height <= 0
            continue;
        end

        patchType = sampleDistractorType(distractorCfg);
        patch = synthesizeDistractorPatch(templatePatch, cfg.texture, distractorCfg, patchType);
        if isempty(patch)
            continue;
        end

        patch = jitterPolygonPatch(patch, distractorCfg);

        % Apply random uniform scaling to distractor
        scaleRange = distractorCfg.sizeScaleRange;
        scaleFactor = scaleRange(1) + rand() * diff(scaleRange);
        [patch, localVerts] = scaleDistractorPatch(patch, region.augVertices, bboxInfo, scaleFactor);

        if isempty(patch)
            continue;
        end

        % Compute scaled bbox dimensions
        scaledWidth = round(bboxInfo.width * scaleFactor);
        scaledHeight = round(bboxInfo.height * scaleFactor);
        if scaledWidth <= 0 || scaledHeight <= 0
            continue;
        end

        bboxStruct = struct('width', scaledWidth, 'height', scaledHeight);
        for attempt = 1:distractorCfg.maxPlacementAttempts
            [xCandidate, yCandidate] = placementFuncs.randomTopLeft(bboxStruct, cfg.placement.margin, bgWidth, bgHeight);
            if ~isfinite(xCandidate) || ~isfinite(yCandidate)
                continue;
            end

            candidateBbox = [xCandidate, yCandidate, xCandidate + scaledWidth, yCandidate + scaledHeight];

            % Check collision with all placed bboxes
            hasConflict = false;
            for j = 1:bboxCount
                if placementFuncs.bboxesOverlap(candidateBbox, allBboxes(j, :), cfg.placement.minSpacing)
                    hasConflict = true;
                    break;
                end
            end
            if hasConflict
                continue;
            end

            sceneVerts = localVerts + [xCandidate, yCandidate];

            bg = placementFuncs.compositeToBackground(bg, patch, sceneVerts);
            bboxCount = bboxCount + 1;
            allBboxes(bboxCount, :) = candidateBbox;
            placedCount = placedCount + 1;
            break;
        end
    end
end

%% =========================================================================
%% POLYGON DISTRACTORS - PATCH SYNTHESIS
%% =========================================================================

function patchType = sampleDistractorType(distractorCfg)
    % Sample distractor rendering style using configured weights.
    %
    % Types: 1=Outline, 2=Solid fill, 3=Textured fill

    weights = [1, 1, 1];
    if isfield(distractorCfg, 'typeWeights')
        candidate = double(distractorCfg.typeWeights(:)');
        candidate = candidate(isfinite(candidate) & candidate >= 0);
        if ~isempty(candidate)
            limit = min(3, numel(candidate));
            weights(1:limit) = candidate(1:limit);
        end
    end

    totalWeight = sum(weights);
    if totalWeight <= 0
        weights = [1, 1, 1];
        totalWeight = 3;
    end

    cumulative = cumsum(weights);
    r = rand() * totalWeight;
    patchType = find(r <= cumulative, 1, 'first');
    if isempty(patchType)
        patchType = 2;
    end
end

function patch = synthesizeDistractorPatch(templatePatch, textureCfg, distractorCfg, patchType)
    % Create a synthetic distractor polygon using the original mask as a template.

    if isempty(templatePatch)
        patch = templatePatch;
        return;
    end

    mask = any(templatePatch > 0, 3);
    if ~any(mask(:))
        patch = [];
        return;
    end

    numChannels = size(templatePatch, 3);
    baseColor = sampleDistractorColor(textureCfg, numChannels);
    if nargin < 4 || isempty(patchType) || ~ismember(patchType, 1:3)
        patchType = 2;
    end

    maskFloat = single(mask);
    baseColorNorm = single(baseColor) / 255;
    [height, width, ~] = size(templatePatch);
    patchFloat = zeros(height, width, numChannels, 'single');

    switch patchType
        case 1  % Outline only
            outlineMask = computeOutlineMask(mask, distractorCfg);
            if ~any(outlineMask(:))
                patch = [];
                return;
            end
            outlineFloat = single(outlineMask);
            strokeScale = 1 + 0.12 * (single(rand(1, numChannels)) - 0.5);
            strokeScale = max(0.6, min(1.4, strokeScale));
            for c = 1:numChannels
                patchFloat(:,:,c) = outlineFloat * (baseColorNorm(c) * strokeScale(c));
            end
            activeMask = outlineMask;

        case 3  % Textured fill
            texture = synthesizeDistractorTexture(mask, textureCfg, distractorCfg);
            channelScale = 1 + 0.10 * (single(rand(1, numChannels)) - 0.5);
            channelScale = max(0.7, min(1.3, channelScale));
            for c = 1:numChannels
                modulation = texture * channelScale(c);
                patchFloat(:,:,c) = (baseColorNorm(c) + modulation) .* maskFloat;
            end
            activeMask = mask;

        otherwise  % Solid fill
            for c = 1:numChannels
                patchFloat(:,:,c) = maskFloat * baseColorNorm(c);
            end
            activeMask = mask;
    end

    patch = finalizeDistractorPatch(patchFloat, activeMask, templatePatch);
end

function patch = finalizeDistractorPatch(patchFloat, activeMask, templatePatch)
    % Finalize distractor patch by clamping and converting to template format.

    if isempty(activeMask) || ~any(activeMask(:))
        patch = [];
        return;
    end

    mask3 = repmat(single(activeMask), [1, 1, size(patchFloat, 3)]);
    patchFloat = min(1, max(0, patchFloat .* mask3));

    patch = castPatchLikeTemplate(patchFloat, templatePatch);
end

function jittered = jitterPolygonPatch(patch, distractorCfg)
    % Apply lightweight photometric jitter while preserving mask boundaries.

    if isempty(patch)
        jittered = patch;
        return;
    end

    mask = any(patch > 0, 3);
    if ~any(mask(:))
        jittered = patch;
        return;
    end

    patchFloat = im2single(patch);

    contrastRange = resolveRange(distractorCfg, 'contrastScaleRange', [1, 1], 0);
    contrastScale = sampleRangeValue(contrastRange);

    brightnessRange = resolveRange(distractorCfg, 'brightnessOffsetRange', [0, 0]);
    brightnessOffset = sampleRangeValue(brightnessRange) / 255;

    patchFloat = (patchFloat - 0.5) * contrastScale + 0.5 + brightnessOffset;

    if isfield(distractorCfg, 'noiseStd') && distractorCfg.noiseStd > 0
        sigma = distractorCfg.noiseStd / 255;
        patchFloat = patchFloat + sigma * randn(size(patchFloat), 'like', patchFloat);
    end

    mask3 = repmat(single(mask), [1, 1, size(patchFloat, 3)]);
    patchFloat = min(1, max(0, patchFloat .* mask3));

    jittered = castPatchLikeTemplate(patchFloat, patch);
end

function [scaledPatch, scaledLocalVerts] = scaleDistractorPatch(patch, vertices, bboxInfo, scaleFactor)
    % Apply uniform scaling to distractor patch and vertices.

    if isempty(patch) || scaleFactor <= 0
        scaledPatch = [];
        scaledLocalVerts = [];
        return;
    end

    % Resize patch image
    [origHeight, origWidth, ~] = size(patch);
    newHeight = round(origHeight * scaleFactor);
    newWidth = round(origWidth * scaleFactor);

    if newHeight < 1 || newWidth < 1
        scaledPatch = [];
        scaledLocalVerts = [];
        return;
    end

    scaledPatch = imresize(patch, [newHeight, newWidth], 'nearest');

    % Scale vertices relative to bbox origin
    localVerts = vertices - [bboxInfo.minX, bboxInfo.minY];
    scaledLocalVerts = localVerts * scaleFactor;
end

%% =========================================================================
%% POLYGON DISTRACTORS - COLOR AND TEXTURE
%% =========================================================================

function baseColor = sampleDistractorColor(textureCfg, numChannels)
    % Sample a base color for distractor from texture configuration.

    if nargin < 2 || isempty(numChannels)
        numChannels = 3;
    end

    switch randi(4)
        case 1  % Uniform surface
            baseRGB = textureCfg.uniformBaseRGB + randi([-textureCfg.uniformVariation, textureCfg.uniformVariation], [1, 3]);
        case 2  % Speckled surface
            baseGray = 160 + randi([-25, 25]);
            baseRGB = [baseGray, baseGray, baseGray] + randi([-5, 5], [1, 3]);
        case 3  % Laminate surface
            if rand() < 0.5
                baseRGB = [245, 245, 245] + randi([-5, 5], [1, 3]);
            else
                baseRGB = [30, 30, 30] + randi([-5, 5], [1, 3]);
            end
        otherwise  % Skin-like hues
            hsvVal = [0.03 + rand() * 0.07, 0.25 + rand() * 0.35, 0.55 + rand() * 0.35];
            baseRGB = round(255 * hsv2rgb(hsvVal));
    end

    baseRGB = max(80, min(220, baseRGB));

    if numChannels ~= numel(baseRGB)
        if numChannels < numel(baseRGB)
            baseColor = baseRGB(1:numChannels);
        else
            baseColor = repmat(baseRGB(end), 1, numChannels);
            baseColor(1:numel(baseRGB)) = baseRGB;
        end
    else
        baseColor = baseRGB;
    end
end

function texture = synthesizeDistractorTexture(mask, textureCfg, distractorCfg)
    % Generate texture for distractor fill.

    [height, width] = size(mask);

    surfaceTypes = 1:4;
    if isfield(distractorCfg, 'textureSurfaceTypes')
        candidate = unique(round(double(distractorCfg.textureSurfaceTypes(:)')));
        candidate = candidate(isfinite(candidate) & candidate >= 1 & candidate <= 4);
        if ~isempty(candidate)
            surfaceTypes = candidate;
        end
    end
    surfaceType = surfaceTypes(randi(numel(surfaceTypes)));

    % Use internal texture generation
    texture = generateSurfaceTextureBase(surfaceType, width, height, textureCfg);
    texture = single(texture);

    activeVals = texture(mask);
    if isempty(activeVals)
        activeVals = single(randn(height * width, 1));
    end
    textureMean = mean(activeVals);
    textureStd = std(activeVals);
    if ~isfinite(textureStd) || textureStd < eps
        textureStd = 1;
    end

    texture = (texture - single(textureMean)) / single(textureStd);

    gainRange = resolveRange(distractorCfg, 'textureGainRange', [0.06, 0.18], 0);
    gain = sampleRangeValue(gainRange);

    texture = texture * single(gain);
end

%% =========================================================================
%% POLYGON DISTRACTORS - OUTLINE UTILITIES
%% =========================================================================

function outlineMask = computeOutlineMask(mask, distractorCfg)
    % Compute an outline mask from the filled polygon mask.

    thickness = sampleOutlineWidth(distractorCfg);
    outlineMask = bwperim(mask);
    if thickness > 1
        radius = max(0, thickness - 1);
        se = strel('disk', radius, 0);
        outlineMask = imdilate(outlineMask, se);
        outlineMask = outlineMask & mask;
    end

    if ~any(outlineMask(:))
        outlineMask = mask;
    end
end

function thickness = sampleOutlineWidth(distractorCfg)
    % Sample outline stroke thickness in pixels.

    range = resolveRange(distractorCfg, 'outlineWidthRange', [1.5, 4.0], 1);
    widthVal = sampleRangeValue(range);
    thickness = max(1, round(widthVal));
end

%% =========================================================================
%% SHARED UTILITIES
%% =========================================================================

function range = resolveRange(cfg, fieldName, defaultRange, minValue)
    % Resolve a range parameter from config with default and minimum.

    if nargin < 4
        minValue = -inf;
    end

    range = defaultRange;
    if isfield(cfg, fieldName)
        values = double(cfg.(fieldName)(:).');
        values = values(isfinite(values));
        if isempty(values)
            range = defaultRange;
        elseif numel(values) >= 2
            range = sort(values(1:2));
        else
            range = [values(1), values(1)];
        end
    end

    range = max(minValue, range);
    if numel(range) < 2
        range = [range(1), range(1)];
    elseif range(1) > range(2)
        range(2) = range(1);
    end
end

function value = sampleRangeValue(range)
    % Sample a value uniformly from a range.

    range = range(:).';
    if isempty(range)
        value = 0;
        return;
    end

    if isscalar(range) || range(2) <= range(1)
        value = range(1);
    else
        value = range(1) + rand() * (range(2) - range(1));
    end
end

function patch = castPatchLikeTemplate(patchFloat, templatePatch)
    % Convert float patch to same type as template.

    if isa(templatePatch, 'uint8')
        patch = im2uint8(patchFloat);
    elseif isa(templatePatch, 'uint16')
        patch = im2uint16(patchFloat);
    elseif isa(templatePatch, 'single')
        patch = patchFloat;
    else
        patch = cast(patchFloat, 'like', templatePatch);
    end
end
