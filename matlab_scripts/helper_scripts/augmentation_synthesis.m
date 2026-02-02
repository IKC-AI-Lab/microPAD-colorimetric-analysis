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

    %% Public API - Shadows
    augSynth.shadows.generateDropShadow = @generateDropShadow;
    augSynth.shadows.sampleTieredShadow = @sampleTieredShadow;

    %% Public API - Paper Stains
    augSynth.stains.apply = @apply_paper_stains;
    augSynth.stains.generateIrregular = @generateIrregularStain;
    augSynth.stains.sampleColor = @sampleRealisticStainColor;
    augSynth.stains.blend = @blendStain;
    augSynth.stains.shrinkMaskToCoverage = @shrinkMaskToCoverage;

    %% Public API - Specular Highlights
    augSynth.specular.apply = @apply_specular_highlight;

    %% Public API - Scene-space conversions
    augSynth.sceneSpace.getSceneSpaceWidth = @getSceneSpaceWidth;
    augSynth.sceneSpace.getLocalScale = @getLocalScale;
    augSynth.sceneSpace.computeCoreDiagonal = @computeCoreDiagonal;

    %% Public API - Utilities (shared)
    augSynth.clampUint8 = imageIOModule.clampUint8;
    augSynth.resolveRange = @resolveRange;
    augSynth.sampleRangeValue = @sampleRangeValue;
    augSynth.castPatchLikeTemplate = @castPatchLikeTemplate;
end

%% =========================================================================
%% BACKGROUND SYNTHESIS - MAIN GENERATION
%% =========================================================================

function [bg, bgType] = generateRealisticLabSurface(width, height, textureCfg)
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
    % Outputs:
    %   bg - uint8 RGB background image [height x width x 3]
    %   bgType - Background surface type (1-5)

    [bgSingle, bgType] = generateRealisticLabSurfaceRaw(width, height, textureCfg);
    bg = uint8(min(255, max(0, bgSingle)));
end

function [bg, bgType] = generateRealisticLabSurfaceRaw(width, height, textureCfg)
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

    % Background type weights: [uniform, speckled, laminate, skin-tone, white+shadows]
    BACKGROUND_WEIGHTS = [0.35, 0.25, 0.20, 0.12, 0.08];

    % Validate weights
    if numel(BACKGROUND_WEIGHTS) ~= 5
        error('augmentation_synthesis:invalidWeights', ...
            'BACKGROUND_WEIGHTS must have exactly 5 elements');
    end
    if any(BACKGROUND_WEIGHTS < 0)
        error('augmentation_synthesis:invalidWeights', ...
            'BACKGROUND_WEIGHTS cannot contain negative values');
    end
    weightSum = sum(BACKGROUND_WEIGHTS);
    if abs(weightSum - 1.0) > 0.01
        warning('augmentation_synthesis:weightSum', ...
            'BACKGROUND_WEIGHTS sum to %.3f (expected 1.0), normalizing', weightSum);
        BACKGROUND_WEIGHTS = BACKGROUND_WEIGHTS / weightSum;
    end

    % Weighted sampling
    cumWeights = cumsum(BACKGROUND_WEIGHTS);
    r = rand() * cumWeights(end);
    surfaceType = find(r <= cumWeights, 1, 'first');
    if isempty(surfaceType)
        surfaceType = 1;
    end
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
        case 5  % White surface with ambient gradients
            baseGray = 245 + randi(6);  % [245, 250]
            baseRGB = [baseGray, baseGray, baseGray] + randi([-textureCfg.whiteRGBVariation, textureCfg.whiteRGBVariation], [1, 3]);
        otherwise  % Skin texture
            h = 0.03 + rand() * 0.07;
            s = 0.25 + rand() * 0.35;
            v = 0.55 + rand() * 0.35;
            baseRGB = round(255 * hsv2rgb([h, s, v]));
    end

    % Conditional clamping: preserve white tones for type 5
    if surfaceType ~= 5
        baseRGB = max(100, min(230, baseRGB));
    else
        baseRGB = max(200, min(255, baseRGB));  % Preserve white tones
    end

    numChannels = 3;  % RGB backgrounds only
    bg = repmat(reshape(single(baseRGB), [1, 1, numChannels]), [height, width, 1]);
    for c = 1:numChannels
        bg(:,:,c) = bg(:,:,c) + texture;
    end

    if rand() < 0.60
        bg = addLightingGradient(bg, width, height);
    end

    % Apply ambient radial gradients for white background (type 5)
    if surfaceType == 5
        numGradients = randi(textureCfg.ambientGradientCount);
        diagonal = sqrt(width^2 + height^2);

        [X, Y] = meshgrid(1:width, 1:height);
        X = single(X);
        Y = single(Y);

        for i = 1:numGradients
            % Random center
            cx = rand() * width;
            cy = rand() * height;

            % Random radius (30-60% of diagonal)
            radiusFraction = textureCfg.ambientGradientRadiusPercent(1) + ...
                             rand() * diff(textureCfg.ambientGradientRadiusPercent);
            radius = radiusFraction * diagonal;

            % Distance from center
            dist = sqrt((X - cx).^2 + (Y - cy).^2);

            % Gaussian falloff
            sigma = max(1, radius / 3);  % Minimum 1 pixel sigma
            gradient = exp(-dist.^2 / (2 * sigma^2));

            % Random strength (2-8% darkening for white backgrounds to preserve tone)
            strength = textureCfg.ambientGradientStrength(1) + ...
                       rand() * diff(textureCfg.ambientGradientStrength);
            darkening = 1 - strength * gradient;

            % Apply to all channels
            for c = 1:3
                bg(:,:,c) = bg(:,:,c) .* darkening;
            end
        end
    end

    bgType = surfaceType;
end

%% =========================================================================
%% BACKGROUND SYNTHESIS - TEXTURE POOL MANAGEMENT
%% =========================================================================

function texture = borrowBackgroundTexture(surfaceType, width, height, textureCfg)
    %% Borrow a texture from the pool with lazy initialization and jitter
    persistent poolState
    persistent dimensionChangeWarned

    if isempty(dimensionChangeWarned)
        dimensionChangeWarned = false;
    end

    if isempty(poolState) || texturePoolConfigChanged(poolState, width, height, textureCfg)
        if ~isempty(poolState) && ~dimensionChangeWarned
            oldWidth = poolState.width;
            oldHeight = poolState.height;
            widthDiff = abs(width - oldWidth) / max(oldWidth, 1);
            heightDiff = abs(height - oldHeight) / max(oldHeight, 1);
            if widthDiff > 0.01 || heightDiff > 0.01
                warning('augmentation_synthesis:poolDimensionChange', ...
                    'Variable image dimensions detected. Texture pool will regenerate per image (expected with variable sizing).');
                dimensionChangeWarned = true;
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
    surfaces = 5;
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
    poolState.surface = repmat(surfaceTemplate, 1, 5);
    for st = 1:5
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
        case 5  % White surface - minimal noise texture
            texture = single(randn(height, width)) .* single(textureCfg.whiteNoiseStrength);
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
    %       .sizeRangePercent - [min, max] size as fraction of diagonal
    %       .minSizePixels - Minimum artifact size
    %       .unitMaskSize - Resolution for ellipse/line unit masks
    %       .overhangMargin - Fraction of size for boundary overhang
    %       .lineWidthRatio - Line width as fraction of length
    %       .lineRotationPadding - Extra padding for line rotation
    %       .ellipseRadiusARange - [min, max] ellipse semi-major axis fraction
    %       .ellipseRadiusBRange - [min, max] ellipse semi-minor axis fraction
    %       .lineIntensityRange - [min, max] line intensity offset
    %       .blobDarkIntensityRange - [min, max] dark blob intensity
    %       .blobLightIntensityRange - [min, max] light blob intensity
    %       .countRange - [min, max] number of artifacts per image
    %     Note: rectangleSizeRange, quadSizeRange, triangleSizeRange are no longer used
    %
    % Output:
    %   bg - Background image with artifacts added
    %
    % Artifacts: ellipses (dust/spots) and lines (scratches) only
    % Count: configurable via artifactCfg.countRange (default: [5, 40])
    % Size: 1-100% of image diagonal (allows artifacts larger than frame)
    % Placement: unconstrained (artifacts can extend beyond boundaries for uniform spatial distribution)

    % Quick guard for tiny backgrounds
    width = max(1, round(width));
    height = max(1, round(height));
    if width < 8 || height < 8
        return;
    end

    % Number of artifacts: use configurable countRange from config
    if isfield(artifactCfg, 'countRange') && ~isempty(artifactCfg.countRange)
        countRange = artifactCfg.countRange;
    else
        countRange = [5, 40];  % Default fallback
    end
    numArtifacts = randi([countRange(1), countRange(2)]);

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
        % Simplified: only ellipses and lines (70% ellipses, 30% lines)
        if rand() < 0.70
            artifactType = 'ellipse';  % 70% ellipses (dust, spots)
        else
            artifactType = 'line';     % 30% lines (scratches)
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

    % Apply multiplier to scale distractor count (default multiplier=1.0 preserves original range)
    multiplier = 1.0;
    if isfield(distractorCfg, 'multiplier') && isfinite(distractorCfg.multiplier) && distractorCfg.multiplier > 0
        multiplier = distractorCfg.multiplier;
    end
    scaledMin = max(1, round(distractorCfg.minCount * multiplier));
    scaledMax = max(scaledMin, round(distractorCfg.maxCount * multiplier));
    targetCount = randi([scaledMin, scaledMax]);
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

function baseColor = sampleDistractorColor(~, numChannels)
    % Generate distractor colors: 70% paper-like, 30% colored objects
    %
    % Paper-like (70%): off-white, cream, light gray, warm white, ivory
    % Colored objects (30%): pens, tape, markers, misc items
    %
    % Note: First argument (textureCfg) unused but kept for API compatibility

    if nargin < 2 || isempty(numChannels)
        numChannels = 3;
    end

    % Probability of colored distractor (non-paper)
    COLORED_DISTRACTOR_PROB = 0.30;

    if rand() < COLORED_DISTRACTOR_PROB
        % Colored objects: pens, tape, markers, misc items
        coloredObjectColors = [
            % Pens
            40, 80, 180;    % Blue pen
            200, 40, 40;    % Red pen
            40, 150, 60;    % Green pen
            30, 30, 35;     % Black pen
            % Tape/stickers
            255, 230, 80;   % Yellow tape
            255, 140, 50;   % Orange tape
            255, 150, 180;  % Pink sticker
            130, 200, 255;  % Light blue tape
            150, 230, 150;  % Light green tape
            % Misc items
            140, 90, 50;    % Brown (wood)
            130, 130, 140;  % Gray (metal)
            180, 80, 160    % Purple marker
        ];

        idx = randi(size(coloredObjectColors, 1));
        base = coloredObjectColors(idx, :);

        % Add variation (±15 for colored objects)
        variation = randi([-15, 15], 1, 3);
        baseRGB = max(0, min(255, base + variation));
    else
        % Realistic paper-like base colors (70%)
        baseColors = [
            245, 245, 240;  % Off-white
            240, 235, 220;  % Cream
            235, 235, 235;  % Light gray
            250, 248, 245;  % Warm white
            238, 238, 230   % Ivory
        ];

        idx = randi(size(baseColors, 1));
        base = baseColors(idx, :);

        % Add small variation (±10)
        variation = randi([-10, 10], 1, 3);
        baseRGB = max(0, min(255, base + variation));
    end

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

    % Prepare masks for damage operations
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

    % Structural Cuts (corner clips only, applied to outer zone)
    % Note: maskEditable is already the outer zone (maskQuad & ~maskProtected)

    % Apply corner clip to outer zone only
    maskBeforeOp = maskEditable;
    maskEditable = apply_corner_clip(maskEditable, damageCorners, damageCfg);

    % Validate area removal
    currentArea = sum(maskEditable(:)) + sum(maskProtected(:));
    if ~isempty(minAllowedAreaPixels) && currentArea < minAllowedAreaPixels
        maskEditable = maskBeforeOp;
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

    % Phase 3: Edge Wear & Thickness Cues (DISABLED in Phase 3 simplification)
    % Removed: edge wave noise, fraying, thickness shadows
    % These are unrealistic or add excessive complexity

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

%% =========================================================================
%% SHADOW GENERATION - DROP SHADOWS FOR WHITE BACKGROUND
%% =========================================================================

function shadowMask = generateDropShadow(quadVertices, lightAngle, imgSize, params)
    % Generate soft drop shadow for a quadrilateral
    %
    % Inputs:
    %   quadVertices - 4x2 array of quad corners [x, y]
    %   lightAngle - light direction angle in degrees (0-360)
    %   imgSize - [height, width] of background
    %   params - struct with:
    %            .dropShadowOffsetRange - [min, max] offset in pixels
    %            .dropShadowBlurRange - [min, max] blur sigma in pixels
    %            .dropShadowDarknessRange - [min, max] multiplier (0.8-0.9)
    % Output:
    %   shadowMask - [H x W] single in [0,1], where 1=no shadow, ~0.8-0.9=shadow

    imgHeight = imgSize(1);
    imgWidth = imgSize(2);

    % Sample shadow parameters
    offset = params.dropShadowOffsetRange(1) + ...
             rand() * diff(params.dropShadowOffsetRange);
    blurSigma = params.dropShadowBlurRange(1) + ...
                rand() * diff(params.dropShadowBlurRange);
    darkness = params.dropShadowDarknessRange(1) + ...
               rand() * diff(params.dropShadowDarknessRange);

    % Convert light angle to offset direction (angle 0 = right, 90 = down)
    angleRad = deg2rad(lightAngle);
    offsetX = offset * cos(angleRad);
    offsetY = offset * sin(angleRad);

    % Offset quad vertices away from light (shadows fall opposite to light direction)
    shadowVertices = quadVertices - [offsetX, offsetY];

    % Clamp to image bounds (allow slight overflow for blur)
    shadowVertices(:,1) = max(-blurSigma*3, min(imgWidth + blurSigma*3, shadowVertices(:,1)));
    shadowVertices(:,2) = max(-blurSigma*3, min(imgHeight + blurSigma*3, shadowVertices(:,2)));

    % Create polygon mask for shadow
    try
        shadowBinaryMask = poly2mask(shadowVertices(:,1), shadowVertices(:,2), ...
                                     imgHeight, imgWidth);
    catch
        % Degenerate polygon (vertices outside bounds or invalid)
        shadowMask = ones(imgHeight, imgWidth, 'single');
        return;
    end

    if ~any(shadowBinaryMask(:))
        shadowMask = ones(imgHeight, imgWidth, 'single');
        return;
    end

    % Apply Gaussian blur for soft edges
    shadowSoft = single(shadowBinaryMask);
    shadowSoft = imgaussfilt(shadowSoft, blurSigma);

    % Convert to multiplicative mask (1 = no shadow, darkness = shadow)
    shadowMask = single(1 - (1 - darkness) * shadowSoft);
end

function [applyShadow, darkness] = sampleTieredShadow()
    % Sample shadow tier using probability distribution
    %
    % Tiers:
    %   None (15%): No shadow
    %   Subtle (35%): [0.94, 0.97] darkness
    %   Light (25%): [0.90, 0.94] darkness
    %   Medium (15%): [0.85, 0.90] darkness
    %   Strong (10%): [0.80, 0.88] darkness
    %
    % Outputs:
    %   applyShadow - boolean, true if shadow should be applied
    %   darkness - sampled darkness value (multiplier)

    % Shadow tier configuration
    SHADOW_TIERS = struct();
    SHADOW_TIERS.probs = [0.15, 0.35, 0.25, 0.15, 0.10];
    SHADOW_TIERS.darkness = {[], [0.94, 0.97], [0.90, 0.94], [0.85, 0.90], [0.80, 0.88]};

    % Sample tier
    r = rand();
    cumProbs = cumsum(SHADOW_TIERS.probs);
    tier = find(r <= cumProbs, 1, 'first');

    if isempty(tier)
        tier = 1;
    end

    % Tier 1 = no shadow
    if tier == 1
        applyShadow = false;
        darkness = 1.0;
        return;
    end

    % Sample darkness from tier range
    applyShadow = true;
    darknessRange = SHADOW_TIERS.darkness{tier};
    darkness = darknessRange(1) + rand() * diff(darknessRange);
end

%% =========================================================================
%% SCENE-SPACE CONVERSIONS
%% =========================================================================

function sceneSpaceWidth = getSceneSpaceWidth(compositeWidth, quadTransform, occlusionMidpoint)
    % Compute scene-space width from composite-space width using local Jacobian
    % quadTransform: homography mapping SCENE -> COMPOSITE
    % occlusionMidpoint: [x, y] midpoint in composite space

    SAFETY_MARGIN = 1.10;
    localScale = getLocalScale(quadTransform, occlusionMidpoint, []);
    sceneSpaceWidth = compositeWidth * localScale * SAFETY_MARGIN;
end

function localScale = getLocalScale(quadTransform, point, ~)
    % Compute local scale factor at a point (composite->scene conversion)
    % Returns max singular value of Jacobian of inverse transform
    % Third parameter ignored (config) for compatibility

    if size(quadTransform, 1) == 3
        H = quadTransform;
        x = point(1);
        y = point(2);
        Hinv = inv(H);
        w = Hinv(3,1)*x + Hinv(3,2)*y + Hinv(3,3);

        % Guard against degenerate transforms
        if abs(w) < 1e-10
            localScale = 1;
            return;
        end

        J = zeros(2,2);
        J(1,1) = (Hinv(1,1)*w - Hinv(3,1)*(Hinv(1,1)*x + Hinv(1,2)*y + Hinv(1,3))) / w^2;
        J(1,2) = (Hinv(1,2)*w - Hinv(3,2)*(Hinv(1,1)*x + Hinv(1,2)*y + Hinv(1,3))) / w^2;
        J(2,1) = (Hinv(2,1)*w - Hinv(3,1)*(Hinv(2,1)*x + Hinv(2,2)*y + Hinv(2,3))) / w^2;
        J(2,2) = (Hinv(2,2)*w - Hinv(3,2)*(Hinv(2,1)*x + Hinv(2,2)*y + Hinv(2,3))) / w^2;

        [~, S, ~] = svd(J);
        localScale = max(diag(S));
    else
        A_inv = inv(quadTransform(1:2, 1:2));
        [~, S, ~] = svd(A_inv);
        localScale = max(diag(S));
    end

    % Ensure valid scale
    if ~isfinite(localScale) || localScale <= 0
        localScale = 1;
    end
end

function sceneDiagonal = computeCoreDiagonal(coreQuadVertices, quadTransform, ~)
    % Compute core diagonal in scene space using polygon vertices
    % coreQuadVertices: Nx2 vertices in composite space
    % Third parameter ignored (config) for compatibility

    if isempty(coreQuadVertices) || size(coreQuadVertices, 1) < 2
        sceneDiagonal = 100;  % Fallback default
        return;
    end

    % Find max distance between any two vertices
    numVerts = size(coreQuadVertices, 1);
    maxDist = 0;
    for i = 1:numVerts
        for j = (i+1):numVerts
            dist = sqrt(sum((coreQuadVertices(i,:) - coreQuadVertices(j,:)).^2));
            maxDist = max(maxDist, dist);
        end
    end
    compositeDiagonal = maxDist;

    % Convert to scene space
    centroid = mean(coreQuadVertices, 1);
    localScale = getLocalScale(quadTransform, centroid, []);
    sceneDiagonal = compositeDiagonal * localScale;

    % Ensure valid result
    if ~isfinite(sceneDiagonal) || sceneDiagonal <= 0
        sceneDiagonal = 100;  % Fallback default
    end
end

%% =========================================================================
%% PAPER STAINS - TWO-TIER STAIN SYSTEM
%% =========================================================================

function compositeImg = apply_paper_stains(compositeImg, quadMask, coreMask, config)
    % Apply paper stains as post-composite augmentation (per-quad)
    % Two-tier: outer zone full opacity, core low opacity
    %
    % Inputs:
    %   compositeImg - RGB image (uint8)
    %   quadMask - Binary mask of full quad region
    %   coreMask - Binary mask of protected core region
    %   config - Configuration struct with stain parameters:
    %            .outerStainProbability - Probability of outer zone stain
    %            .maxOuterStainCoverage - Max coverage fraction for outer stains
    %            .outerStainOpacityRange - [min, max] opacity for outer stains
    %            .coreStainProbability - Probability of core zone stain
    %            .coreStainMaxOpacity - Max opacity for core stains
    %            .coreStainMaxCoverage - Max coverage for core stains
    %            .stainSemiMajorRange - [min, max] semi-major axis in pixels
    %            .stainSemiMinorRange - [min, max] semi-minor axis in pixels
    %
    % Output:
    %   compositeImg - Image with stains applied

    outerMask = quadMask & ~coreMask;
    [imgH, imgW, ~] = size(compositeImg);

    % Guard against degenerate quads with empty masks
    outerPixelCount = sum(outerMask(:));
    corePixelCount = sum(coreMask(:));

    % OUTER ZONE STAINS
    if outerPixelCount > 0 && rand() < config.outerStainProbability
        stainMask = generateIrregularStain(imgH, imgW, outerMask, config);

        % Enforce coverage limit
        outerCoverage = sum(stainMask(:) & outerMask(:)) / sum(outerMask(:));
        if outerCoverage > config.maxOuterStainCoverage
            stainMask = shrinkMaskToCoverage(stainMask, outerMask, config.maxOuterStainCoverage);
        end

        stainColor = sampleRealisticStainColor();
        opacityRange = config.outerStainOpacityRange;
        stainOpacity = opacityRange(1) + rand() * diff(opacityRange);
        compositeImg = blendStain(compositeImg, stainMask, stainColor, stainOpacity);
    end

    % CORE STAINS (lower opacity)
    if corePixelCount > 0 && rand() < config.coreStainProbability
        coreStainMask = generateIrregularStain(imgH, imgW, coreMask, config);

        % Enforce core coverage limit
        coreCoverage = sum(coreStainMask(:) & coreMask(:)) / sum(coreMask(:));
        if coreCoverage > config.coreStainMaxCoverage
            coreStainMask = shrinkMaskToCoverage(coreStainMask, coreMask, config.coreStainMaxCoverage);
        end

        stainColor = sampleRealisticStainColor();
        stainOpacity = rand() * config.coreStainMaxOpacity;
        compositeImg = blendStain(compositeImg, coreStainMask, stainColor, stainOpacity);
    end
end

function stainMask = generateIrregularStain(imgH, imgW, zoneMask, config)
    % Generate irregular stain mask within zone
    %
    % Inputs:
    %   imgH, imgW - Image dimensions
    %   zoneMask - Binary mask of valid zone for stain placement
    %   config - Configuration struct with stain size parameters
    %
    % Output:
    %   stainMask - Binary mask of stain region

    stainMask = false(imgH, imgW);

    % Find valid region bounds
    [rows, cols] = find(zoneMask);
    if isempty(rows)
        return;
    end

    % Random stain center within zone
    idx = randi(numel(rows));
    cx = cols(idx);
    cy = rows(idx);

    % Random elliptical shape from config ranges
    majorRange = config.stainSemiMajorRange;
    minorRange = config.stainSemiMinorRange;
    a = majorRange(1) + rand() * diff(majorRange);  % semi-major axis
    b = minorRange(1) + rand() * diff(minorRange);  % semi-minor axis
    theta = rand() * pi;

    [xx, yy] = meshgrid(1:imgW, 1:imgH);
    xxRot = (xx - cx) * cos(theta) + (yy - cy) * sin(theta);
    yyRot = -(xx - cx) * sin(theta) + (yy - cy) * cos(theta);

    ellipseMask = (xxRot.^2 / a^2 + yyRot.^2 / b^2) <= 1;

    % Add irregular edges using noise
    noise = imgaussfilt(randn(imgH, imgW), 5);
    irregularMask = ellipseMask & (noise > -0.3);

    stainMask = irregularMask & zoneMask;
end

function stainMask = shrinkMaskToCoverage(stainMask, zoneMask, targetCoverage)
    % Shrink stain mask to meet coverage target by erosion
    %
    % Inputs:
    %   stainMask - Binary mask to shrink
    %   zoneMask - Binary mask of valid zone
    %   targetCoverage - Target coverage fraction (0-1)
    %
    % Output:
    %   stainMask - Shrunk mask meeting coverage target

    currentCoverage = sum(stainMask(:) & zoneMask(:)) / sum(zoneMask(:));

    se = strel('disk', 2);
    maxIterations = 50;
    iter = 0;
    while currentCoverage > targetCoverage && any(stainMask(:)) && iter < maxIterations
        stainMask = imerode(stainMask, se);
        stainMask = stainMask & zoneMask;
        currentCoverage = sum(stainMask(:) & zoneMask(:)) / sum(zoneMask(:));
        iter = iter + 1;
    end
end

function stainColor = sampleRealisticStainColor()
    % Sample realistic stain colors (brown, gray, yellowish)
    %
    % Output:
    %   stainColor - uint8 RGB color [R, G, B]

    colorType = randi(3);
    switch colorType
        case 1  % Brown/coffee stain
            stainColor = uint8([90 + randi(40), 50 + randi(30), 30 + randi(20)]);
        case 2  % Gray/dust
            gray = 80 + randi(60);
            stainColor = uint8([gray, gray, gray] + randi([-10, 10], [1, 3]));
        case 3  % Yellowish/water stain
            stainColor = uint8([200 + randi(40), 180 + randi(40), 100 + randi(40)]);
    end
end

function img = blendStain(img, stainMask, stainColor, opacity)
    % Blend stain color into image with given opacity
    %
    % Inputs:
    %   img - RGB image (uint8)
    %   stainMask - Binary mask of stain region
    %   stainColor - uint8 RGB color [R, G, B]
    %   opacity - Blend opacity (0-1)
    %
    % Output:
    %   img - Image with stain blended

    if ~any(stainMask(:))
        return;
    end

    for c = 1:3
        channel = double(img(:,:,c));
        channel(stainMask) = channel(stainMask) * (1 - opacity) + double(stainColor(c)) * opacity;
        img(:,:,c) = uint8(channel);
    end
end

%% =========================================================================
%% SPECULAR HIGHLIGHT AUGMENTATION
%% =========================================================================

function img = apply_specular_highlight(img, quadVertices, textureCfg)
    % Apply localized specular highlight to simulate phone camera reflections
    %
    % Simulates bright spots from overhead lighting reflecting off glossy or
    % laminated paper surfaces. Common in real phone captures.
    %
    % Inputs:
    %   img - RGB image (uint8)
    %   quadVertices - 4x2 matrix of quad corner positions
    %   textureCfg - Texture configuration with specular params:
    %                .specularHighlightIntensityRange - [min, max] multiplier
    %                .specularHighlightRadiusRange - [min, max] fraction of quad diagonal
    %                .specularHighlightBlurRange - [min, max] fraction of radius
    %
    % Output:
    %   img - Image with specular highlight applied

    [imgH, imgW, ~] = size(img);

    % Compute quad center and diagonal (use max of both diagonals for robustness)
    centroid = mean(quadVertices, 1);
    diag13 = sqrt(sum((quadVertices(1,:) - quadVertices(3,:)).^2));
    diag24 = sqrt(sum((quadVertices(2,:) - quadVertices(4,:)).^2));
    quadDiag = max(diag13, diag24);

    % Sample highlight parameters
    intensityRange = textureCfg.specularHighlightIntensityRange;
    radiusRange = textureCfg.specularHighlightRadiusRange;
    blurRange = textureCfg.specularHighlightBlurRange;

    intensity = intensityRange(1) + rand() * diff(intensityRange);
    radiusFrac = radiusRange(1) + rand() * diff(radiusRange);
    blurFrac = blurRange(1) + rand() * diff(blurRange);

    radius = radiusFrac * quadDiag;
    blurSigma = blurFrac * radius;

    % Random offset from centroid (within 30% of quad size)
    offsetRange = 0.30 * quadDiag;
    cx = centroid(1) + (rand() - 0.5) * 2 * offsetRange;
    cy = centroid(2) + (rand() - 0.5) * 2 * offsetRange;

    % Clamp center to image bounds
    cx = max(1, min(imgW, cx));
    cy = max(1, min(imgH, cy));

    % Create highlight mask (radial gradient)
    [X, Y] = meshgrid(1:imgW, 1:imgH);
    distFromCenter = sqrt((X - cx).^2 + (Y - cy).^2);

    % Smooth falloff using Gaussian-like profile
    highlightMask = exp(-0.5 * (distFromCenter / radius).^2);

    % Apply Gaussian blur for soft edges
    if blurSigma > 0.5
        highlightMask = imgaussfilt(highlightMask, blurSigma);
    end

    % Apply highlight (multiplicative brightening)
    imgDouble = double(img);
    for c = 1:3
        channel = imgDouble(:,:,c);
        % Blend: original + (brightened - original) * mask
        brightened = channel * intensity;
        channel = channel + (brightened - channel) .* highlightMask;
        imgDouble(:,:,c) = channel;
    end

    % Clamp and convert
    img = uint8(max(0, min(255, imgDouble)));
end

