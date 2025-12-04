function augSynth = augmentation_synthesis()
    %% AUGMENTATION_SYNTHESIS Returns a struct of function handles for synthetic data generation
    %
    % This utility module consolidates all synthetic generation for the augmentation
    % pipeline: background textures, sparse artifacts, and quadrilateral distractors.
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
    %   mask = augSynth.artifacts.generateQuadMask(verticesPix, targetSize);
    %
    %   % Quadrilateral distractors
    %   [bg, count] = augSynth.distractors.addQuad(bg, regions, bboxes, occupied, cfg, funcs);
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
    augSynth.artifacts.generateQuadMask = @generateQuadMask;

    %% Public API - Quadrilateral distractors
    augSynth.distractors.addQuad = @addQuadDistractors;
    augSynth.distractors.sampleType = @sampleDistractorType;
    augSynth.distractors.synthesizePatch = @synthesizeDistractorPatch;
    augSynth.distractors.finalizePatch = @finalizeDistractorPatch;
    augSynth.distractors.jitterPatch = @jitterQuadPatch;
    augSynth.distractors.scalePatch = @scaleDistractorPatch;
    augSynth.distractors.sampleColor = @sampleDistractorColor;
    augSynth.distractors.synthesizeTexture = @synthesizeDistractorTexture;
    augSynth.distractors.computeOutlineMask = @computeOutlineMask;
    augSynth.distractors.sampleOutlineWidth = @sampleOutlineWidth;

    %% Public API - Paper Damage
    augSynth.damage.apply = @apply_paper_damage;

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
    % in artifactCfg.unitMaskSize) to avoid large meshgrid allocations. Quadrilateral artifacts
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
    %       .quadPerturbation - Vertex perturbation for quadrilaterals
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

        % Create artifact mask; quadrilaterals draw directly at target resolution to keep sharp edges
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
                mask = generateQuadMask(verticesPix, artifactSize);

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
                mask = generateQuadMask(verticesPix, artifactSize);

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
                mask = generateQuadMask(verticesPix, artifactSize);

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

function mask = generateQuadMask(verticesPix, targetSize)
    %% Rasterize quadrilateral vertices expressed in pixel coordinates into a binary mask
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
    quadMask = poly2mask(verticesPix(:,1), verticesPix(:,2), targetSize, targetSize);
    if ~any(quadMask(:))
        mask = [];
        return;
    end

    mask = single(quadMask);
end

%% =========================================================================
%% QUADRILATERAL DISTRACTORS - MAIN GENERATION
%% =========================================================================

function [bg, placedCount] = addQuadDistractors(bg, regions, quadBboxes, occupiedBboxes, cfg, placementFuncs)
    % Inject additional quadrilateral-shaped distractors matching source geometry statistics.
    %
    % Inputs:
    %   bg - Background image to composite distractors onto
    %   regions - Cell array of source region structures
    %   quadBboxes - Cell array of bbox info structs for each region
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
        bboxInfo = quadBboxes{srcIdx};
        templatePatch = region.augQuadImg;

        if isempty(templatePatch) || bboxInfo.width <= 0 || bboxInfo.height <= 0
            continue;
        end

        patchType = sampleDistractorType(distractorCfg);
        patch = synthesizeDistractorPatch(templatePatch, cfg.texture, distractorCfg, patchType);
        if isempty(patch)
            continue;
        end

        patch = jitterQuadPatch(patch, distractorCfg);

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
%% QUADRILATERAL DISTRACTORS - PATCH SYNTHESIS
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
    % Create a synthetic distractor quadrilateral using the original mask as a template.

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

function jittered = jitterQuadPatch(patch, distractorCfg)
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
%% QUADRILATERAL DISTRACTORS - COLOR AND TEXTURE
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
%% QUADRILATERAL DISTRACTORS - OUTLINE UTILITIES
%% =========================================================================

function outlineMask = computeOutlineMask(mask, distractorCfg)
    % Compute an outline mask from the filled quadrilateral mask.

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

%% =========================================================================
%% PAPER DAMAGE AUGMENTATION
%% =========================================================================

function [damagedRGB, damagedAlpha, maskEditable, maskProtected, damageCorners] = apply_paper_damage(quadRGB, quadAlpha, quadCorners, ellipseList, damageCfg, rngState)
    % Apply paper defect augmentation pipeline to quad patch
    %
    % INPUTS:
    %   quadRGB - [H x W x 3] uint8 color content
    %   quadAlpha - [H x W] logical mask
    %   quadCorners - [4 x 2] vertex coordinates
    %   ellipseList - struct array with fields: center, semiMajor, semiMinor, rotation
    %   damageCfg - cfg.damage struct
    %   rngState - RNG state for reproducibility
    %
    % OUTPUTS:
    %   damagedRGB - [H x W x 3] uint8 damaged color content
    %   damagedAlpha - [H x W] logical damaged mask
    %   maskEditable - [H x W] logical editable mask
    %   maskProtected - [H x W] logical protected mask
    %   damageCorners - [4 x 2] quad vertices after warp/shear (crop space)

    % Save/restore RNG only when determinism is requested so random draws do
    % not replay for every quad.
    savedRngState = [];
    if ~isempty(rngState)
        savedRngState = rng();
        rng(rngState);
    end

    % Initialize outputs
    damagedRGB = quadRGB;
    damagedAlpha = quadAlpha;
    damageCorners = double(quadCorners);

    % Check if damage should be applied
    if rand() > damageCfg.probability
        maskEditable = false(size(quadAlpha));
        maskProtected = false(size(quadAlpha));
        if ~isempty(savedRngState)
            rng(savedRngState);
        end
        return;
    end

    % Sample damage profile using precomputed cumulative weights
    rVal = rand();
    profileIdx = find(rVal <= damageCfg.profileCumWeights, 1, 'first');
    if isempty(profileIdx)
        profileIdx = numel(damageCfg.profileNames);
    end
    selectedProfile = damageCfg.profileNames{profileIdx};

    % Phase 0: Prepare masks for damage operations
    [H, W, ~] = size(quadRGB);
    maskQuad = damagedAlpha;
    maskProtected = false(H, W);
    maskPreCuts = damagedAlpha;
    hasEllipses = ~isempty(ellipseList) && numel(ellipseList) > 0;
    originalArea = sum(maskQuad(:));

    guardCenters = zeros(max(1, numel(ellipseList)), 2);
    guardCenterCount = 0;
    ellipseProtectionActive = false;

    maxRemovalFraction = [];
    if isfield(damageCfg, 'maxAreaRemovalFraction') && ~isempty(damageCfg.maxAreaRemovalFraction)
        maxRemovalFraction = min(max(damageCfg.maxAreaRemovalFraction, 0), 1);
    end

    if hasEllipses
        % Compute bounding box for all ellipses to minimize meshgrid size
        ellipseBBox = compute_ellipse_bbox_union(ellipseList, W, H);
        
        if ~isempty(ellipseBBox)
            % Create meshgrid only for ROI containing all ellipses
            [xx, yy] = meshgrid(ellipseBBox.xMin:ellipseBBox.xMax, ellipseBBox.yMin:ellipseBBox.yMax);

            keepFraction = 1;
            if ~isempty(maxRemovalFraction)
                keepFraction = max(0, 1 - maxRemovalFraction);
            end
            ellipseAxisScale = sqrt(keepFraction);
            
            for eIdx = 1:numel(ellipseList)
                ellipse = ellipseList(eIdx);

                if ~isfield(ellipse, 'center') || ~isfield(ellipse, 'semiMajor') || ...
                   ~isfield(ellipse, 'semiMinor') || ~isfield(ellipse, 'rotation')
                    continue;
                end

                cx = ellipse.center(1);
                cy = ellipse.center(2);
                a = ellipse.semiMajor;
                b = ellipse.semiMinor;
                theta = ellipse.rotation * pi / 180;

                if a <= 0 || b <= 0
                    continue;
                end

                if cx < 1 || cx > W || cy < 1 || cy > H
                    continue;
                end

                guardCenterCount = guardCenterCount + 1;
                guardCenters(guardCenterCount, :) = [cx, cy];

                % Compute ellipse mask in ROI coordinates
                xx_rot = (xx - cx) * cos(theta) + (yy - cy) * sin(theta);
                yy_rot = -(xx - cx) * sin(theta) + (yy - cy) * cos(theta);

                ellipseCoreMask = false(size(xx_rot));
                if keepFraction > 0 && ellipseAxisScale > 0
                    coreA = a * ellipseAxisScale;
                    coreB = b * ellipseAxisScale;
                    if coreA > 0 && coreB > 0
                        ellipseCoreMask = (xx_rot .^ 2) / (coreA ^ 2) + (yy_rot .^ 2) / (coreB ^ 2) <= 1;
                    end
                end

                % Map ROI mask back to full image coordinates
                if isempty(maskProtected) || ~any(maskProtected(:))
                    maskProtected = false(H, W);
                end
                if any(ellipseCoreMask(:))
                    maskProtected(ellipseBBox.yMin:ellipseBBox.yMax, ellipseBBox.xMin:ellipseBBox.xMax) = ...
                        maskProtected(ellipseBBox.yMin:ellipseBBox.yMax, ellipseBBox.xMin:ellipseBBox.xMax) | ellipseCoreMask;
                    ellipseProtectionActive = true;
                end
            end

            maskProtected = maskProtected & maskQuad;
        end
    end

    if guardCenterCount > 0 && ellipseProtectionActive
        guardCenters = guardCenters(1:guardCenterCount, :);
        bridgeMask = build_guard_bridge_mask(maskQuad, guardCenters, damageCorners);
        if ~isempty(bridgeMask)
            maskProtected = maskProtected | bridgeMask;
        end
    end

    minAllowedAreaPixels = [];
    if ~hasEllipses && ~isempty(maxRemovalFraction)
        coreGuardScale = 1 - maxRemovalFraction;
        if coreGuardScale > 0
            coreMask = build_core_guard_mask(damageCorners, H, W, coreGuardScale);
            if ~isempty(coreMask)
                maskProtected = maskProtected | (coreMask & maskQuad);
            end
        end
        minAllowedAreaPixels = originalArea * (1 - maxRemovalFraction);
    end

    maskEditable = maskQuad & ~maskProtected;
    hasProtectedRegions = any(maskProtected(:));

    if sum(maskEditable(:)) == 0
        maskEditable = false(size(maskProtected));
        if ~isempty(savedRngState)
            rng(savedRngState);
        end
        return;
    end

    % Phase 1: Base Warp & Shear

    % 1.1 Projective jitter (all profiles)
    imgCorners = [1, 1; W, 1; W, H; 1, H];  % TL, TR, BR, BL
    jitterScale = 0.03;
    rotJitter = 2.0 * pi / 180;

    jitteredCorners = imgCorners;
    for i = 1:4
        % Translation jitter: +/- 3% of image dimensions
        jitteredCorners(i, 1) = imgCorners(i, 1) + (rand() * 2 - 1) * jitterScale * W;
        jitteredCorners(i, 2) = imgCorners(i, 2) + (rand() * 2 - 1) * jitterScale * H;

        % Rotation jitter: +/- 2° around corner
        angle = (rand() * 2 - 1) * rotJitter;
        cx = imgCorners(i, 1);
        cy = imgCorners(i, 2);
        dx = jitteredCorners(i, 1) - cx;
        dy = jitteredCorners(i, 2) - cy;
        jitteredCorners(i, 1) = cx + dx * cos(angle) - dy * sin(angle);
        jitteredCorners(i, 2) = cy + dx * sin(angle) + dy * cos(angle);
    end

    % Clamp to image bounds
    jitteredCorners(:, 1) = max(1, min(W, jitteredCorners(:, 1)));
    jitteredCorners(:, 2) = max(1, min(H, jitteredCorners(:, 2)));

    try
        tform = fitgeotrans(imgCorners, jitteredCorners, 'projective');
        outputView = imref2d([H, W]);

        damagedRGB = imwarp(damagedRGB, tform, 'OutputView', outputView, ...
                           'InterpolationMethod', 'linear', 'FillValues', 0);

        alphaDouble = double(damagedAlpha);
        warpedAlpha = imwarp(alphaDouble, tform, 'OutputView', outputView, ...
                            'InterpolationMethod', 'linear', 'FillValues', 0);
        damagedAlphaCand = warpedAlpha > 0.5;

        if hasProtectedRegions
            maskProtectedDouble = double(maskProtected);
            warpedProtected = imwarp(maskProtectedDouble, tform, 'OutputView', outputView, ...
                                    'InterpolationMethod', 'nearest', 'FillValues', 0);
            maskProtected = warpedProtected > 0.5;
            hasProtectedRegions = any(maskProtected(:));
        end

        if hasProtectedRegions
            damagedAlpha = damagedAlphaCand | maskProtected;
            maskEditable = damagedAlpha & ~maskProtected;
        else
            damagedAlpha = damagedAlphaCand;
            maskEditable = damagedAlphaCand;
        end
        [damageCorners(:,1), damageCorners(:,2)] = transformPointsForward(tform, damageCorners(:,1), damageCorners(:,2));
        maskPreCuts = damagedAlpha;
    catch
        % fitgeotrans can fail with near-singular matrices from extreme jitter
        % Continue with undamaged quad - no action needed
    end

    % 1.2 Nonlinear edge bending (minimalWarp and sideCollapse only)
    % Skip for small quads where warp overhead exceeds benefit
    minDim = min(H, W);
    applyNonlinearWarp = minDim > 200;
    
    if applyNonlinearWarp && (strcmp(selectedProfile, 'minimalWarp') || strcmp(selectedProfile, 'sideCollapse'))
        % Control points at edge midpoints
        controlPoints = [
            W/2, 1;      % Top edge midpoint
            W, H/2;      % Right edge midpoint
            W/2, H;      % Bottom edge midpoint
            1, H/2       % Left edge midpoint
        ];

        % Add corners as fixed anchor points
        movingPts = [controlPoints; [1, 1; W, 1; W, H; 1, H]];
        fixedPts = movingPts;

        % Offset edge midpoints perpendicular to edge
        ampRange = damageCfg.edgeWaveAmplitudeRange * minDim;
        for i = 1:4
            offset = ampRange(1) + rand() * (ampRange(2) - ampRange(1));
            offset = offset * (2 * (rand() > 0.5) - 1);  % Random sign

            if i == 1  % Top edge
                fixedPts(i, 2) = fixedPts(i, 2) + offset;
            elseif i == 2  % Right edge
                fixedPts(i, 1) = fixedPts(i, 1) + offset;
            elseif i == 3  % Bottom edge
                fixedPts(i, 2) = fixedPts(i, 2) + offset;
            else  % Left edge
                fixedPts(i, 1) = fixedPts(i, 1) + offset;
            end
        end

        % Clamp to valid region
        fixedPts(:, 1) = max(1, min(W, fixedPts(:, 1)));
        fixedPts(:, 2) = max(1, min(H, fixedPts(:, 2)));

        try
            tform = fitgeotrans(movingPts, fixedPts, 'lwm', 8);
            outputView = imref2d([H, W]);

            damagedRGB = imwarp(damagedRGB, tform, 'OutputView', outputView, ...
                               'InterpolationMethod', 'linear', 'FillValues', 0);

            alphaDouble = double(damagedAlpha);
            warpedAlpha = imwarp(alphaDouble, tform, 'OutputView', outputView, ...
                                'InterpolationMethod', 'linear', 'FillValues', 0);
            damagedAlphaCand = warpedAlpha > 0.5;

            if hasProtectedRegions
                maskProtectedDouble = double(maskProtected);
                warpedProtected = imwarp(maskProtectedDouble, tform, 'OutputView', outputView, ...
                                        'InterpolationMethod', 'nearest', 'FillValues', 0);
                maskProtected = warpedProtected > 0.5;
                hasProtectedRegions = any(maskProtected(:));
            end

            damagedAlpha = damagedAlphaCand;
            if hasProtectedRegions
                damagedAlpha = damagedAlpha | maskProtected;
                maskEditable = damagedAlpha & ~maskProtected;
            else
                maskEditable = damagedAlpha;
            end
            [damageCorners(:,1), damageCorners(:,2)] = transformPointsForward(tform, damageCorners(:,1), damageCorners(:,2));
            maskPreCuts = damagedAlpha;
        catch
            % fitgeotrans (lwm) can fail with degenerate control points
            % Continue with undamaged quad - no action needed
        end
    end

    % Phase 2: Structural Cuts (Material Removal)
    if strcmp(selectedProfile, 'cornerChew') || strcmp(selectedProfile, 'sideCollapse')
        if strcmp(selectedProfile, 'cornerChew')
            operationPool = {'cornerClip', 'cornerTear', 'sideBite', 'taperedSide'};
            operationWeights = [0.35, 0.25, 0.25, 0.15];
        else
            operationPool = {'cornerClip', 'cornerTear', 'sideBite', 'taperedSide'};
            operationWeights = [0.10, 0.10, 0.50, 0.30];
        end

        numOps = randi([1, damageCfg.maxOperations]);

        prevOp = '';
        for opIdx = 1:numOps
            weights = operationWeights;
            if ~isempty(prevOp)
                weights(strcmp(operationPool, prevOp)) = 0;
            end
            if sum(weights) == 0
                weights = operationWeights;
            end
            weights = weights / sum(weights);
            cumWeights = cumsum(weights);
            opType = operationPool{find(rand() <= cumWeights, 1, 'first')};

            maskBeforeOp = maskEditable;
            switch opType
                case 'cornerClip'
                    maskEditable = apply_corner_clip(maskEditable, damageCorners, damageCfg);
                case 'cornerTear'
                    maskEditable = apply_corner_tear(maskEditable, damageCorners, damageCfg);
                case 'sideBite'
                    maskEditable = apply_side_bite(maskEditable, damageCorners, damageCfg, maskProtected, hasProtectedRegions);
                case 'taperedSide'
                    maskEditable = apply_tapered_side(maskEditable, damageCorners, damageCfg);
            end

            currentArea = sum(maskEditable(:)) + sum(maskProtected(:));
            if ~isempty(minAllowedAreaPixels) && currentArea < minAllowedAreaPixels
                maskEditable = maskBeforeOp;
                break;
            end

            prevOp = opType;
        end

        damagedAlpha = maskEditable;
        if hasProtectedRegions
            damagedAlpha = damagedAlpha | maskProtected;
        end

        for c = 1:3
            channel = damagedRGB(:,:,c);
            channel(~damagedAlpha) = 0;
            damagedRGB(:,:,c) = channel;
        end
    end

    maskEditablePreWear = maskEditable;
    damagedAlphaPreWear = damagedAlpha;
    damagedRGBPreWear = damagedRGB;

    % Phase 3: Edge Wear & Thickness Cues
    if strcmp(selectedProfile, 'cornerChew') || strcmp(selectedProfile, 'sideCollapse')
        maskEditable = apply_edge_wave_noise(maskEditable, damageCfg);

        if rand() < 0.5
            maskEditable = apply_edge_fray(maskEditable, maskProtected);
        end

        damagedAlpha = maskEditable;
        if hasProtectedRegions
            damagedAlpha = damagedAlpha | maskProtected;
        end

        damagedRGB = apply_thickness_shadows(damagedRGB, damagedAlpha, maskPreCuts);

        for c = 1:3
            channel = damagedRGB(:,:,c);
            channel(~damagedAlpha) = 0;
            damagedRGB(:,:,c) = channel;
        end
    elseif strcmp(selectedProfile, 'minimalWarp')
        damagedAlpha = maskEditable;
        if hasProtectedRegions
            damagedAlpha = damagedAlpha | maskProtected;
        end

        for c = 1:3
            channel = damagedRGB(:,:,c);
            channel(~damagedAlpha) = 0;
            damagedRGB(:,:,c) = channel;
        end
    end

    if ~isempty(minAllowedAreaPixels)
        finalArea = sum(maskEditable(:)) + sum(maskProtected(:));
        if finalArea < minAllowedAreaPixels
            maskEditable = maskEditablePreWear;
            damagedAlpha = damagedAlphaPreWear;
            damagedRGB = damagedRGBPreWear;
        end
    end

    if ~isempty(savedRngState)
        rng(savedRngState);
    end
end

function coreMask = build_core_guard_mask(quadCorners, height, width, scaleFactor)
    % Construct a scaled version of the quad that remains untouched
    if isempty(quadCorners) || size(quadCorners, 2) ~= 2 || scaleFactor <= 0
        coreMask = [];
        return;
    end

    scaleFactor = min(scaleFactor, 1);

    centroid = mean(quadCorners, 1, 'omitnan');
    if any(isnan(centroid))
        coreMask = [];
        return;
    end

    scaledCorners = (quadCorners - centroid) * scaleFactor + centroid;
    coreMask = poly2mask(scaledCorners(:, 1), scaledCorners(:, 2), height, width);
end

function ellipseBBox = compute_ellipse_bbox_union(ellipseList, width, height)
    % Compute tight bounding box containing all ellipses for ROI-based processing
    
    if isempty(ellipseList)
        ellipseBBox = [];
        return;
    end
    
    % Collect all ellipse bounding boxes
    numEllipses = numel(ellipseList);
    xMins = zeros(numEllipses, 1);
    xMaxs = zeros(numEllipses, 1);
    yMins = zeros(numEllipses, 1);
    yMaxs = zeros(numEllipses, 1);
    validCount = 0;
    
    for i = 1:numEllipses
        ellipse = ellipseList(i);
        
        if ~isfield(ellipse, 'center') || ~isfield(ellipse, 'semiMajor') || ...
           ~isfield(ellipse, 'semiMinor') || ~isfield(ellipse, 'rotation')
            continue;
        end
        
        cx = ellipse.center(1);
        cy = ellipse.center(2);
        
        if cx < 1 || cx > width || cy < 1 || cy > height
            continue;
        end
        
        % Compute axis-aligned bounding box for rotated ellipse
        theta = ellipse.rotation * pi / 180;
        a = ellipse.semiMajor;
        b = ellipse.semiMinor;
        if a <= 0 || b <= 0
            continue;
        end
        dx = sqrt((a * cos(theta))^2 + (b * sin(theta))^2);
        dy = sqrt((a * sin(theta))^2 + (b * cos(theta))^2);
        
        validCount = validCount + 1;
        xMins(validCount) = cx - dx;
        xMaxs(validCount) = cx + dx;
        yMins(validCount) = cy - dy;
        yMaxs(validCount) = cy + dy;
    end
    
    if validCount == 0
        ellipseBBox = [];
        return;
    end
    
    xMins = xMins(1:validCount);
    xMaxs = xMaxs(1:validCount);
    yMins = yMins(1:validCount);
    yMaxs = yMaxs(1:validCount);
    
    ellipseBBox = struct();
    ellipseBBox.xMin = max(1, floor(min(xMins)));
    ellipseBBox.xMax = min(width, ceil(max(xMaxs)));
    ellipseBBox.yMin = max(1, floor(min(yMins)));
    ellipseBBox.yMax = min(height, ceil(max(yMaxs)));
end

function bridgeMask = build_guard_bridge_mask(maskQuad, guardCenters, quadCorners)
    % Add protected connectors so guarded ellipses remain attached to the strip

    bridgeMask = false(size(maskQuad));
    if isempty(guardCenters)
        return;
    end

    centroid = mean(quadCorners, 1, 'omitnan');
    if any(~isfinite(centroid))
        return;
    end

    [height, width] = size(maskQuad);
    centroid(1) = max(1, min(width, centroid(1)));
    centroid(2) = max(1, min(height, centroid(2)));

    valid = all(isfinite(guardCenters), 2);
    guardCenters = guardCenters(valid, :);
    if isempty(guardCenters)
        return;
    end

    for idx = 1:size(guardCenters, 1)
        startPt = guardCenters(idx, :);
        delta = abs(startPt - centroid);
        numSteps = max(delta);
        numSteps = max(ceil(numSteps), 1);
        xLine = round(linspace(startPt(1), centroid(1), numSteps));
        yLine = round(linspace(startPt(2), centroid(2), numSteps));
        keep = xLine >= 1 & xLine <= width & yLine >= 1 & yLine <= height;
        xLine = xLine(keep);
        yLine = yLine(keep);
        ind = sub2ind([height, width], yLine, xLine);
        bridgeMask(ind) = true;
    end

    se = strel('disk', 2, 0);
    bridgeMask = imdilate(bridgeMask, se) & maskQuad;
end

function edgeInfo = get_quad_edges(quadCorners)
    edgeInfo = struct();
    edgeInfo.corners = quadCorners;

    edgeInfo.edges = zeros(4, 2, 2);
    for i = 1:4
        nextIdx = mod(i, 4) + 1;
        edgeInfo.edges(i, :, :) = [quadCorners(i, :); quadCorners(nextIdx, :)];
    end

    edgeInfo.midpoints = zeros(4, 2);
    edgeInfo.lengths = zeros(4, 1);
    edgeInfo.directions = zeros(4, 2);
    edgeInfo.normals = zeros(4, 2);

    for i = 1:4
        startPt = squeeze(edgeInfo.edges(i, 1, :))';
        endPt = squeeze(edgeInfo.edges(i, 2, :))';

        edgeInfo.midpoints(i, :) = (startPt + endPt) / 2;

        edgeVec = endPt - startPt;
        len = norm(edgeVec);
        edgeInfo.lengths(i) = len;
        if len > eps
            edgeInfo.directions(i, :) = edgeVec / len;
            edgeInfo.normals(i, :) = [edgeVec(2), -edgeVec(1)] / len;
        else
            edgeInfo.directions(i, :) = [1, 0];
            edgeInfo.normals(i, :) = [0, -1];
        end
    end

    centroid = mean(quadCorners, 1);
    for i = 1:4
        toCenter = centroid - edgeInfo.midpoints(i, :);
        if dot(edgeInfo.normals(i, :), toCenter) < 0
            edgeInfo.normals(i, :) = -edgeInfo.normals(i, :);
        end
    end
end

function mask = apply_corner_clip(mask, quadCorners, damageCfg)
    [H, W] = size(mask);
    edgeInfo = get_quad_edges(quadCorners);
    refDim = mean(edgeInfo.lengths);

    cornerWeights = [0.3, 0.5, 0.2];
    rVal = rand();
    if rVal < cornerWeights(1)
        numCorners = 1;
    elseif rVal < cornerWeights(1) + cornerWeights(2)
        numCorners = 2;
    else
        numCorners = 3;
    end

    if numCorners == 1
        selectedCorners = randi([1, 4], 1);
    elseif numCorners == 2
        if rand() < 0.7
            startCorner = randi([1, 4]);
            selectedCorners = [startCorner, mod(startCorner, 4) + 1];
        else
            perm = randperm(4);
            selectedCorners = perm(1:2);
        end
    else
        perm = randperm(4);
        selectedCorners = perm(1:3);
    end

    for i = 1:numel(selectedCorners)
        cornerIdx = selectedCorners(i);
        cornerPt = edgeInfo.corners(cornerIdx, :);

        depthFrac = damageCfg.cornerClipRange(1) + ...
                    rand() * (damageCfg.cornerClipRange(2) - damageCfg.cornerClipRange(1));
        depth = depthFrac * refDim;

        prevCornerIdx = mod(cornerIdx - 2, 4) + 1;
        prevDir = edgeInfo.directions(prevCornerIdx, :);
        nextDir = edgeInfo.directions(cornerIdx, :);

        pt1 = cornerPt - prevDir * depth;
        pt2 = cornerPt;
        pt3 = cornerPt + nextDir * depth;

        triMask = poly2mask([pt1(1), pt2(1), pt3(1)], ...
                           [pt1(2), pt2(2), pt3(2)], H, W);
        mask = mask & ~triMask;
    end
end

function mask = apply_corner_tear(mask, quadCorners, damageCfg)
    [H, W] = size(mask);
    edgeInfo = get_quad_edges(quadCorners);
    refDim = mean(edgeInfo.lengths);

    cornerIdx = randi([1, 4]);
    cornerPt = edgeInfo.corners(cornerIdx, :);

    depthFrac = damageCfg.cornerClipRange(1) + ...
                rand() * (damageCfg.cornerClipRange(2) - damageCfg.cornerClipRange(1));
    depth = depthFrac * refDim;

    numVerts = randi([4, 6]);
    tearVerts = zeros(numVerts, 2);

    prevCornerIdx = mod(cornerIdx - 2, 4) + 1;
    prevDir = edgeInfo.directions(prevCornerIdx, :);
    nextDir = edgeInfo.directions(cornerIdx, :);

    for i = 1:numVerts
        t = (i - 1) / (numVerts - 1);
        if t < 0.5
            dir = -prevDir;
            offset = t * 2 * depth;
        else
            dir = nextDir;
            offset = (t - 0.5) * 2 * depth;
        end

        perpDir = [-dir(2), dir(1)];
        jitter = (rand() - 0.5) * depth * 0.3;

        tearVerts(i, :) = cornerPt + dir * offset + perpDir * jitter;
    end

    tearMask = poly2mask(tearVerts(:, 1), tearVerts(:, 2), H, W);

    if rand() < 0.5
        se = strel('disk', 2);
        tearMask = imdilate(tearMask, se);
    end

    mask = mask & ~tearMask;
end

function mask = apply_side_bite(mask, quadCorners, damageCfg, maskProtected, hasProtectedRegions)
    [H, W] = size(mask);
    edgeInfo = get_quad_edges(quadCorners);

    edgeIdx = randi([1, 4]);
    edgePos = 0.3 + rand() * 0.4;
    depthFrac = damageCfg.sideBiteRange(1) + ...
                rand() * (damageCfg.sideBiteRange(2) - damageCfg.sideBiteRange(1));

    edgeLength = edgeInfo.lengths(edgeIdx);
    edgeMidpoint = edgeInfo.midpoints(edgeIdx, :);
    edgeDirection = edgeInfo.directions(edgeIdx, :);
    edgeNormal = edgeInfo.normals(edgeIdx, :);

    tParam = (edgePos - 0.5) * 2;
    biteCenterOnEdge = edgeMidpoint + tParam * edgeDirection * (edgeLength / 2);
    radius = depthFrac * edgeLength;
    circleCenter = biteCenterOnEdge + edgeNormal * radius;

    [xx, yy] = meshgrid(1:W, 1:H);
    biteMask = ((xx - circleCenter(1)).^2 + (yy - circleCenter(2)).^2) <= radius^2;

    mask = mask & ~biteMask;

    if rand() < 0.4
        offsetSign = (rand() < 0.5) * 2 - 1;
        edgePos2 = edgePos + offsetSign * (0.2 + rand() * 0.2);
        edgePos2 = max(0.1, min(0.9, edgePos2));

        tParam2 = (edgePos2 - 0.5) * 2;
        biteCenterOnEdge2 = edgeMidpoint + tParam2 * edgeDirection * (edgeLength / 2);
        
        depthFrac2 = depthFrac * (0.5 + rand() * 0.2);
        radius2 = depthFrac2 * edgeLength;
        circleCenter2 = biteCenterOnEdge2 + edgeNormal * radius2;

        centerDist = norm(circleCenter2 - circleCenter);
        minSpacing = (radius + radius2) * 0.7;
        
        if centerDist >= minSpacing
            biteMask2 = ((xx - circleCenter2(1)).^2 + (yy - circleCenter2(2)).^2) <= radius2^2;
            
            if hasProtectedRegions && any(biteMask2(:) & maskProtected(:))
                return;
            end
            
            mask = mask & ~biteMask2;
        end
    end
end

function mask = apply_tapered_side(mask, quadCorners, damageCfg)
    [H, W] = size(mask);
    edgeInfo = get_quad_edges(quadCorners);

    edgeIdx = randi([1, 4]);

    taperStrength = damageCfg.taperStrengthRange(1) + ...
                    rand() * (damageCfg.taperStrengthRange(2) - damageCfg.taperStrengthRange(1));

    edgeStart = squeeze(edgeInfo.edges(edgeIdx, 1, :))';
    edgeEnd = squeeze(edgeInfo.edges(edgeIdx, 2, :))';
    edgeNormal = edgeInfo.normals(edgeIdx, :);
    edgeLength = edgeInfo.lengths(edgeIdx);

    [xx, yy] = meshgrid(1:W, 1:H);

    edgeVec = edgeEnd - edgeStart;

    px = xx(:) - edgeStart(1);
    py = yy(:) - edgeStart(2);
    t = (px * edgeVec(1) + py * edgeVec(2)) / (edgeVec(1)^2 + edgeVec(2)^2);
    t = reshape(t, H, W);
    t = max(0, min(1, t));

    taperDepth = taperStrength * edgeLength * t;

    pointOnEdgeX = edgeStart(1) + t * edgeVec(1);
    pointOnEdgeY = edgeStart(2) + t * edgeVec(2);
    signedDist = (xx - pointOnEdgeX) * edgeNormal(1) + (yy - pointOnEdgeY) * edgeNormal(2);

    gradientMask = signedDist > -taperDepth;

    mask = mask & gradientMask;
end

function mask = apply_edge_wave_noise(mask, damageCfg)
    [H, W] = size(mask);
    minDim = min(H, W);
    originalMask = mask;
    
    if minDim < 150
        return;
    end

    perim = bwperim(mask);
    if ~any(perim(:))
        return;
    end

    maxWaveDist = damageCfg.edgeWaveAmplitudeRange(2) * minDim * 2;
    signedDist = fast_signed_distance(mask, ceil(maxWaveDist));

    freq = damageCfg.edgeWaveFrequencyRange(1) + ...
           rand() * (damageCfg.edgeWaveFrequencyRange(2) - damageCfg.edgeWaveFrequencyRange(1));

    amp = damageCfg.edgeWaveAmplitudeRange(1) + ...
          rand() * (damageCfg.edgeWaveAmplitudeRange(2) - damageCfg.edgeWaveAmplitudeRange(1));
    amp = amp * minDim;

    [xx, yy] = meshgrid(1:W, 1:H);

    cx = W / 2;
    cy = H / 2;
    theta = atan2(yy - cy, xx - cx);

    wavePattern = amp * sin(freq * theta);

    noiseAmp = amp * 0.3;
    noise = randn(H, W) * noiseAmp;
    noise = imgaussfilt(noise, 3);

    modulation = wavePattern + noise;

    modulatedDist = signedDist + modulation;

    mask = (modulatedDist >= 0) & originalMask;
end

function signedDist = fast_signed_distance(mask, maxDist)
    if maxDist > 20
        D_out = bwdist(~mask);
        D_in = bwdist(mask);
        signedDist = D_out - D_in;
        return;
    end
    
    [H, W] = size(mask);
    signedDist = zeros(H, W);
    
    se = strel('disk', 1, 0);
    tempMask = mask;
    for d = 1:maxDist
        tempMask = imerode(tempMask, se);
        if ~any(tempMask(:))
            break;
        end
        signedDist = signedDist + double(tempMask);
    end
    
    tempMask = ~mask;
    for d = 1:maxDist
        tempMask = imerode(tempMask, se);
        if ~any(tempMask(:))
            break;
        end
        signedDist = signedDist - double(tempMask);
    end
end

function mask = apply_edge_fray(mask, maskProtected)
    [H, W] = size(mask);

    perim = bwperim(mask);
    [py, px] = find(perim);

    if isempty(px)
        return;
    end

    numPerimPixels = numel(px);
    numFrayPoints = max(3, round(numPerimPixels / 100 * rand() * 3));

    minSpacing = 10;
    selectedPoints = zeros(numFrayPoints, 2);
    pointCount = 0;
    attempts = 0;
    maxAttempts = numFrayPoints * 10;

    while pointCount < numFrayPoints && attempts < maxAttempts
        idx = randi(numPerimPixels);
        candidate = [px(idx), py(idx)];

        if pointCount == 0
            pointCount = 1;
            selectedPoints(pointCount, :) = candidate;
        else
            dists = sqrt(sum((selectedPoints(1:pointCount, :) - candidate).^2, 2));
            if all(dists >= minSpacing)
                pointCount = pointCount + 1;
                selectedPoints(pointCount, :) = candidate;
            end
        end
        attempts = attempts + 1;
    end
    selectedPoints = selectedPoints(1:pointCount, :);

    frayMask = false(H, W);
    [xx, yy] = meshgrid(1:W, 1:H);
    for i = 1:size(selectedPoints, 1)
        radius = randi([1, 3]);

        distFromPoint = sqrt((xx - selectedPoints(i, 1)).^2 + (yy - selectedPoints(i, 2)).^2);
        blob = distFromPoint <= radius;

        blob = blob & ~maskProtected;
        frayMask = frayMask | blob;
    end

    mask = mask & ~frayMask;
end

function img = apply_thickness_shadows(img, damagedMask, originalMask)
    removedRegion = originalMask & ~damagedMask;

    if ~any(removedRegion(:))
        return;
    end

    distFromRemoved = bwdist(removedRegion);

    shadowWidth = 8;
    shadowRamp = max(0, 1 - distFromRemoved / shadowWidth);
    shadowRamp = shadowRamp .* double(damagedMask);

    shadowStrength = 0.15;
    darkening = 1 - shadowStrength * shadowRamp;

    for c = 1:3
        channel = double(img(:,:,c));
        channel = channel .* darkening;
        img(:,:,c) = uint8(channel);
    end
end

