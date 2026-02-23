function occlusionUtils = occlusion_utils()
    %% OCCLUSION_UTILS Returns a struct of function handles for occlusion generation
    %
    % This utility module consolidates all occlusion synthesis for the augmentation
    % pipeline: line occlusions (thin/thick), blob occlusions (dirt, smudges), and
    % finger occlusions (hand in frame).
    %
    % Usage:
    %   occlusionUtils = occlusion_utils();
    %
    %   % Add occlusions to quad
    %   [img, augRecord] = occlusionUtils.addQuadOcclusions(img, quadMask, coreMask, ...
    %       coreQuadVertices, outerMask, quadTransform, config, augRecord);
    %
    %   % Low-level utilities
    %   [width, isThin] = occlusionUtils.sampleWidth(quadTransform, midpoint, config);
    %   mask = occlusionUtils.drawLine(img, startPt, endPt, width);
    %   img = occlusionUtils.applyColor(img, mask, color);
    %   color = occlusionUtils.generateColor();
    %   [mask, color] = occlusionUtils.generateBlob(imgH, imgW, minX, maxX, minY, maxY, quadDiag);
    %   [mask, color] = occlusionUtils.generateFinger(imgH, imgW, minX, maxX, minY, maxY, quadDiag);
    %   adjustedColor = occlusionUtils.enforceContrastLimit(img, corePortion, color, maxContrast);
    %   [clipStart, clipEnd] = occlusionUtils.clipToMaxLength(lineStart, lineEnd, maxLength);
    %
    % See also: augment_dataset, augmentation_synthesis

    %% Public API
    occlusionUtils.addQuadOcclusions = @add_quad_occlusions;
    occlusionUtils.sampleWidth = @sampleOcclusionWidth;
    occlusionUtils.drawLine = @drawLine;
    occlusionUtils.applyColor = @applyOcclusionColor;
    occlusionUtils.generateColor = @generateOcclusionColor;
    occlusionUtils.generateBlob = @generateBlobOcclusion;
    occlusionUtils.generateFinger = @generateFingerOcclusion;
    occlusionUtils.enforceContrastLimit = @enforceContrastLimit;
    occlusionUtils.clipToMaxLength = @clipToMaxLength;
end

%% -------------------------------------------------------------------------
%% Main Occlusion Orchestration
%% -------------------------------------------------------------------------

function [img, augRecord] = add_quad_occlusions(img, quadMask, coreMask, coreQuadVertices, outerMask, quadTransform, config, augRecord)
    % Draw core-aware occlusions (thin/thick) with bounded core entry
    % Thick occlusions (>3px scene-space) are outer-only
    % Thin occlusions (<=3px scene-space) can enter core with bounds

    % Initialize augRecord fields if needed
    if ~isfield(augRecord, 'occlusionApplied')
        augRecord.occlusionApplied = false;
    end
    if ~isfield(augRecord, 'thickOcclusionMask')
        augRecord.thickOcclusionMask = false(size(img, 1), size(img, 2));
    end
    if ~isfield(augRecord, 'thinOcclusionMask')
        augRecord.thinOcclusionMask = false(size(img, 1), size(img, 2));
    end
    if ~isfield(augRecord, 'thinOcclusionCount')
        augRecord.thinOcclusionCount = 0;
    end

    % Number of occlusion attempts
    numOcclusions = randi([1, 4]);

    thickMaskCombined = false(size(img, 1), size(img, 2));
    thinMaskCombined = false(size(img, 1), size(img, 2));
    thinCorePixels = 0;
    thinCount = 0;

    % Get quad bounding box
    if isempty(quadMask) || ~any(quadMask(:))
        return;
    end

    [rows, cols] = find(quadMask);
    if isempty(rows)
        return;
    end

    [imgH, imgW, ~] = size(img);
    minX = max(1, min(cols));
    maxX = min(imgW, max(cols));
    minY = max(1, min(rows));
    maxY = min(imgH, max(rows));

    % Compute core diagonal for length limits (using function handle from config)
    coreDiagonal = config.computeCoreDiagonal(coreQuadVertices, quadTransform, []);

    % Get occlusion type weights (default: 60% line, 25% blob, 15% finger)
    typeWeights = [0.60, 0.25, 0.15];
    if isfield(config, 'occlusionTypeWeights') && numel(config.occlusionTypeWeights) == 3 ...
            && all(config.occlusionTypeWeights >= 0) && sum(config.occlusionTypeWeights) > 0
        typeWeights = config.occlusionTypeWeights;
    end
    cumWeights = cumsum(typeWeights) / sum(typeWeights);

    for i = 1:numOcclusions
        % Sample occlusion type: 1=line, 2=blob, 3=finger
        r = rand();
        if r < cumWeights(1)
            occlusionType = 1;  % Line
        elseif r < cumWeights(2)
            occlusionType = 2;  % Blob
        else
            occlusionType = 3;  % Finger
        end

        cx = (minX + maxX) / 2;
        cy = (minY + maxY) / 2;
        quadDiag = sqrt((maxX - minX)^2 + (maxY - minY)^2);

        if occlusionType == 1
            % LINE OCCLUSION (original implementation)
            angle = rand() * 2 * pi;
            lineLength = 0.5 * quadDiag + rand() * 0.5 * quadDiag;

            lineStart = [cx - lineLength/2 * cos(angle), cy - lineLength/2 * sin(angle)];
            lineEnd = [cx + lineLength/2 * cos(angle), cy + lineLength/2 * sin(angle)];
            occlusionMidpoint = (lineStart + lineEnd) / 2;

            [compositeWidth, isThin] = sampleOcclusionWidth(quadTransform, occlusionMidpoint, config);
            occlusionColor = generateOcclusionColor();

            if ~isThin
                occMask = drawLine(img, lineStart, lineEnd, compositeWidth);
                occMask = occMask & outerMask;
                thickMaskCombined = thickMaskCombined | occMask;
                img = applyOcclusionColor(img, occMask, occlusionColor);
            else
                fullOccMask = drawLine(img, lineStart, lineEnd, compositeWidth);
                fullCorePortion = fullOccMask & coreMask;

                if ~any(fullCorePortion(:))
                    clippedFullMask = fullOccMask & quadMask;
                    thinMaskCombined = thinMaskCombined | clippedFullMask;
                    img = applyOcclusionColor(img, clippedFullMask, occlusionColor);
                else
                    maxLengthScene = coreDiagonal * config.thinOcclusionMaxLength;
                    localScale = config.getLocalScale(quadTransform, occlusionMidpoint, []);
                    maxLengthComposite = maxLengthScene / (localScale * 1.10);

                    [clippedStart, clippedEnd] = clipToMaxLength(lineStart, lineEnd, maxLengthComposite);
                    occMask = drawLine(img, clippedStart, clippedEnd, compositeWidth);
                    clippedCorePortion = occMask & coreMask;

                    if ~any(clippedCorePortion(:))
                        clippedToQuad = occMask & quadMask;
                        thinMaskCombined = thinMaskCombined | clippedToQuad;
                        img = applyOcclusionColor(img, clippedToQuad, occlusionColor);
                    else
                        if thinCount >= config.thinOcclusionMaxCount
                            continue;
                        end
                        newCorePixels = sum(clippedCorePortion(:));
                        totalCorePixels = sum(coreMask(:));
                        if (thinCorePixels + newCorePixels) / totalCorePixels > config.thinOcclusionMaxCoverage
                            continue;
                        end
                        occlusionColor = enforceContrastLimit(img, clippedCorePortion, occlusionColor, config.thinOcclusionMaxContrast);
                        clippedToQuad = occMask & quadMask;
                        thinMaskCombined = thinMaskCombined | clippedToQuad;
                        thinCorePixels = thinCorePixels + newCorePixels;
                        thinCount = thinCount + 1;
                        img = applyOcclusionColor(img, clippedToQuad, occlusionColor);
                    end
                end
            end

        elseif occlusionType == 2
            % BLOB OCCLUSION (dirt, smudge, shadow)
            [occMask, occlusionColor] = generateBlobOcclusion(imgH, imgW, minX, maxX, minY, maxY, quadDiag);
            occMask = occMask & outerMask;  % Outer zone only
            if any(occMask(:))
                thickMaskCombined = thickMaskCombined | occMask;
                img = applyOcclusionColor(img, occMask, occlusionColor);
            end

        else
            % FINGER OCCLUSION (hand in frame)
            [occMask, occlusionColor] = generateFingerOcclusion(imgH, imgW, minX, maxX, minY, maxY, quadDiag);
            occMask = occMask & outerMask;  % Outer zone only
            if any(occMask(:))
                thickMaskCombined = thickMaskCombined | occMask;
                img = applyOcclusionColor(img, occMask, occlusionColor);
            end
        end
    end

    augRecord.thickOcclusionMask = thickMaskCombined;
    augRecord.thinOcclusionMask = thinMaskCombined;
    augRecord.thinOcclusionCount = thinCount;
    augRecord.occlusionApplied = any(thickMaskCombined(:)) || any(thinMaskCombined(:));
end

%% -------------------------------------------------------------------------
%% Width Sampling and Line Drawing
%% -------------------------------------------------------------------------

function [compositeWidth, isThin] = sampleOcclusionWidth(quadTransform, occlusionMidpoint, config)
    % Sample occlusion width with probability-based thin/thick routing
    preferThin = rand() < config.thinOcclusionProbability;

    if preferThin
        minW = config.thinOcclusionMinWidth;
        maxW = config.thinOcclusionMaxWidth;
    else
        minW = config.thinOcclusionMaxWidth + config.thickOcclusionMinGap;
        maxW = config.thickOcclusionMaxWidth;
    end

    sceneWidth = minW + rand() * (maxW - minW);
    localScale = config.getLocalScale(quadTransform, occlusionMidpoint, []);
    compositeWidth = sceneWidth / localScale;
    compositeWidth = max(1, min(30, compositeWidth));

    actualSceneWidth = compositeWidth * localScale;
    isThin = actualSceneWidth <= config.thinOcclusionMaxWidth;
end

function mask = drawLine(img, lineStart, lineEnd, width)
    % Draw a line mask with specified width
    [imgH, imgW, ~] = size(img);
    mask = false(imgH, imgW);

    if width < 1
        return;
    end

    % Get bounding box
    minX = max(1, floor(min(lineStart(1), lineEnd(1)) - width));
    maxX = min(imgW, ceil(max(lineStart(1), lineEnd(1)) + width));
    minY = max(1, floor(min(lineStart(2), lineEnd(2)) - width));
    maxY = min(imgH, ceil(max(lineStart(2), lineEnd(2)) + width));

    if maxX < minX || maxY < minY
        return;
    end

    [X, Y] = meshgrid(minX:maxX, minY:maxY);

    % Distance from point to line segment
    dx = lineEnd(1) - lineStart(1);
    dy = lineEnd(2) - lineStart(2);
    lengthSq = dx^2 + dy^2;

    if lengthSq < 1e-6
        % Degenerate line (point)
        dist = sqrt((X - lineStart(1)).^2 + (Y - lineStart(2)).^2);
    else
        % Project each point onto line
        t = ((X - lineStart(1)) * dx + (Y - lineStart(2)) * dy) / lengthSq;
        t = max(0, min(1, t));
        projX = lineStart(1) + t * dx;
        projY = lineStart(2) + t * dy;
        dist = sqrt((X - projX).^2 + (Y - projY).^2);
    end

    lineMask = dist <= width / 2;
    mask(minY:maxY, minX:maxX) = lineMask;
end

%% -------------------------------------------------------------------------
%% Color Application and Generation
%% -------------------------------------------------------------------------

function img = applyOcclusionColor(img, mask, color)
    % Apply occlusion color to masked pixels
    if ~any(mask(:))
        return;
    end

    for c = 1:3
        plane = img(:,:,c);
        plane(mask) = color(c);
        img(:,:,c) = plane;
    end
end

function occlusionColor = generateOcclusionColor()
    % Generate occlusion color with full range (for outer-only occlusions)
    % Core-entering occlusions have contrast enforced separately

    % Color range constants for occlusion generation
    DARK_GRAY_RANGE = [0, 60];      % Base gray for dark occlusions
    BRIGHT_GRAY_RANGE = [195, 255]; % Base gray for bright occlusions
    COLOR_OFFSET_RANGE = [-15, 15]; % Per-channel color variation

    if rand() < 0.5
        % Dark occlusion
        baseGray = randi(DARK_GRAY_RANGE);
        colorOffset = randi(COLOR_OFFSET_RANGE, [1, 3]);
        occlusionColor = uint8(max(0, min(255, baseGray + colorOffset)));
    else
        % Bright occlusion
        baseGray = randi(BRIGHT_GRAY_RANGE);
        colorOffset = randi(COLOR_OFFSET_RANGE, [1, 3]);
        occlusionColor = uint8(max(0, min(255, baseGray + colorOffset)));
    end
end

%% -------------------------------------------------------------------------
%% Blob and Finger Occlusion Generation
%% -------------------------------------------------------------------------

function [mask, color] = generateBlobOcclusion(imgH, imgW, minX, maxX, minY, maxY, quadDiag)
    % Generate blob-shaped occlusion (dirt, smudge, shadow)
    %
    % Creates an irregular elliptical shape with noise-modulated edges,
    % simulating dirt spots, smudges, or cast shadows.
    %
    % Outputs:
    %   mask - Binary occlusion mask [imgH x imgW]
    %   color - uint8 RGB color for the blob

    mask = false(imgH, imgW);

    % Random blob center within outer zone of quad
    blobCx = minX + rand() * (maxX - minX);
    blobCy = minY + rand() * (maxY - minY);

    % Blob size: 5-15% of quad diagonal
    sizeFrac = 0.05 + rand() * 0.10;
    blobRadius = sizeFrac * quadDiag;

    % Create elliptical base (random aspect ratio)
    aspectRatio = 0.5 + rand() * 1.0;  % [0.5, 1.5]
    semiMajor = blobRadius;
    semiMinor = blobRadius * aspectRatio;
    if semiMinor > semiMajor
        [semiMajor, semiMinor] = deal(semiMinor, semiMajor);
    end

    % Random rotation
    theta = rand() * pi;

    % Compute bounding region
    blobMinX = max(1, floor(blobCx - semiMajor - 10));
    blobMaxX = min(imgW, ceil(blobCx + semiMajor + 10));
    blobMinY = max(1, floor(blobCy - semiMajor - 10));
    blobMaxY = min(imgH, ceil(blobCy + semiMajor + 10));

    if blobMaxX <= blobMinX || blobMaxY <= blobMinY
        color = uint8([60, 50, 40]);
        return;
    end

    [X, Y] = meshgrid(blobMinX:blobMaxX, blobMinY:blobMaxY);

    % Rotate coordinates
    Xrot = (X - blobCx) * cos(theta) + (Y - blobCy) * sin(theta);
    Yrot = -(X - blobCx) * sin(theta) + (Y - blobCy) * cos(theta);

    % Ellipse equation
    ellipseDist = (Xrot.^2 / semiMajor^2) + (Yrot.^2 / semiMinor^2);

    % Add noise to edges for irregular shape
    noiseScale = 0.3;
    noise = randn(size(X)) * noiseScale;
    noise = imgaussfilt(noise, 2);

    blobMask = (ellipseDist + noise) <= 1.0;
    mask(blobMinY:blobMaxY, blobMinX:blobMaxX) = blobMask;

    % Generate shadow/dirt color (dark browns, grays)
    shadowColors = [
        60, 50, 40;    % Dark brown (dirt)
        80, 70, 60;    % Medium brown
        50, 50, 55;    % Dark gray (shadow)
        70, 65, 60;    % Warm gray
        45, 40, 35     % Very dark brown
    ];
    idx = randi(size(shadowColors, 1));
    baseColor = shadowColors(idx, :);
    variation = randi([-10, 10], 1, 3);
    color = uint8(max(0, min(255, baseColor + variation)));
end

function [mask, color] = generateFingerOcclusion(imgH, imgW, minX, maxX, minY, maxY, quadDiag)
    % Generate finger-shaped occlusion (hand in frame)
    %
    % Creates a rounded rectangle shape extending from an edge,
    % simulating a finger or hand partially occluding the image.
    %
    % Outputs:
    %   mask - Binary occlusion mask [imgH x imgW]
    %   color - uint8 RGB color (skin tone)

    mask = false(imgH, imgW);

    % Finger dimensions: width 8-15% of quad diagonal, length 15-35%
    fingerWidth = (0.08 + rand() * 0.07) * quadDiag;
    fingerLength = (0.15 + rand() * 0.20) * quadDiag;

    % Choose which edge the finger comes from (0=top, 1=right, 2=bottom, 3=left)
    edge = randi([0, 3]);

    % Position along edge (20-80% to avoid corners)
    edgePos = 0.2 + rand() * 0.6;

    % Compute finger start position and direction
    switch edge
        case 0  % Top edge
            startX = minX + edgePos * (maxX - minX);
            startY = minY;
            dx = 0;
            dy = 1;  % Points down
        case 1  % Right edge
            startX = maxX;
            startY = minY + edgePos * (maxY - minY);
            dx = -1;  % Points left
            dy = 0;
        case 2  % Bottom edge
            startX = minX + edgePos * (maxX - minX);
            startY = maxY;
            dx = 0;
            dy = -1;  % Points up
        case 3  % Left edge
            startX = minX;
            startY = minY + edgePos * (maxY - minY);
            dx = 1;  % Points right
            dy = 0;
    end

    % Create finger shape (rounded rectangle)
    % Build mask using distance from centerline
    fingerMinX = max(1, floor(min(startX, startX + dx * fingerLength) - fingerWidth));
    fingerMaxX = min(imgW, ceil(max(startX, startX + dx * fingerLength) + fingerWidth));
    fingerMinY = max(1, floor(min(startY, startY + dy * fingerLength) - fingerWidth));
    fingerMaxY = min(imgH, ceil(max(startY, startY + dy * fingerLength) + fingerWidth));

    if fingerMaxX <= fingerMinX || fingerMaxY <= fingerMinY
        color = uint8([200, 160, 130]);
        return;
    end

    [X, Y] = meshgrid(fingerMinX:fingerMaxX, fingerMinY:fingerMaxY);

    % Distance along finger axis (0 to fingerLength)
    if dx ~= 0
        alongAxis = (X - startX) * dx;
        perpDist = abs(Y - startY);
    else
        alongAxis = (Y - startY) * dy;
        perpDist = abs(X - startX);
    end

    % Finger shape: rectangle with rounded tip
    inBody = (alongAxis >= 0) & (alongAxis <= fingerLength * 0.8) & (perpDist <= fingerWidth / 2);

    % Rounded tip
    tipCenterX = startX + dx * fingerLength * 0.8;
    tipCenterY = startY + dy * fingerLength * 0.8;
    tipRadius = fingerWidth / 2;
    distFromTip = sqrt((X - tipCenterX).^2 + (Y - tipCenterY).^2);
    inTip = (distFromTip <= tipRadius) & (alongAxis >= fingerLength * 0.8 - tipRadius);

    fingerMask = inBody | inTip;
    mask(fingerMinY:fingerMaxY, fingerMinX:fingerMaxX) = fingerMask;

    % Generate skin tone color (5 variants from light to dark)
    skinTones = [
        255, 220, 195;   % Light skin
        235, 190, 160;   % Medium-light skin
        200, 160, 130;   % Medium skin
        165, 125, 95;    % Medium-dark skin
        100, 70, 50      % Dark skin
    ];
    idx = randi(size(skinTones, 1));
    baseColor = skinTones(idx, :);
    variation = randi([-8, 8], 1, 3);
    color = uint8(max(0, min(255, baseColor + variation)));
end

%% -------------------------------------------------------------------------
%% Contrast and Length Constraints
%% -------------------------------------------------------------------------

function adjustedColor = enforceContrastLimit(img, corePortion, occlusionColor, maxContrast)
    % Enforce contrast limit for core-entering thin occlusions
    % Adjusts color to stay within maxContrast luminance delta from core pixels

    adjustedColor = occlusionColor;

    if ~any(corePortion(:))
        return;
    end

    % Extract core pixels
    corePixels = zeros(sum(corePortion(:)), 3);
    for c = 1:3
        plane = double(img(:,:,c));
        corePixels(:, c) = plane(corePortion);
    end

    % Compute mean luminance of core pixels
    coreLuminance = mean(0.299 * corePixels(:,1) + 0.587 * corePixels(:,2) + 0.114 * corePixels(:,3));

    % Compute occlusion luminance
    occLuminance = 0.299 * double(occlusionColor(1)) + 0.587 * double(occlusionColor(2)) + 0.114 * double(occlusionColor(3));

    % Check if contrast exceeds limit
    contrastDelta = occLuminance - coreLuminance;

    if abs(contrastDelta) > maxContrast
        % Adjust occlusion color to be within contrast limit
        targetLuminance = coreLuminance + sign(contrastDelta) * maxContrast;
        targetLuminance = max(0, min(255, targetLuminance));

        % Scale color to achieve target luminance while preserving hue
        currentLuminance = occLuminance;
        if currentLuminance > 1  % Avoid near-zero division
            scaleFactor = min(10, targetLuminance / currentLuminance);  % Cap scale factor
            adjustedColor = uint8(max(0, min(255, double(occlusionColor) * scaleFactor)));
        else
            % Near-black - shift to target gray
            adjustedColor = uint8([targetLuminance, targetLuminance, targetLuminance]);
        end
    end
end

function [clippedStart, clippedEnd] = clipToMaxLength(lineStart, lineEnd, maxLength)
    % Clip line to maximum length, keeping center fixed
    dx = lineEnd(1) - lineStart(1);
    dy = lineEnd(2) - lineStart(2);
    currentLength = sqrt(dx^2 + dy^2);

    if currentLength <= maxLength
        clippedStart = lineStart;
        clippedEnd = lineEnd;
        return;
    end

    scale = maxLength / currentLength;
    center = (lineStart + lineEnd) / 2;

    clippedStart = center + (lineStart - center) * scale;
    clippedEnd = center + (lineEnd - center) * scale;
end
