function utils = mask_utils()
    %% MASK_UTILS Returns a struct of function handles for mask operations
    %
    % This utility module provides functions for creating quadrilateral and ellipse
    % masks, with caching support for batch processing performance.
    %
    % Usage:
    %   masks = mask_utils();
    %   [cropped, mask] = masks.cropWithQuadMask(img, vertices);
    %   ellipseMask = masks.createEllipseMask([h, w], cx, cy, a, b, theta);
    %
    % Caching:
    %   For batch processing, use the cached ellipse mask function:
    %     cache = masks.createMaskCache();
    %     for each patch:
    %         mask = masks.createEllipseMaskCached([h, w], cx, cy, a, b, theta, cache);
    %
    %   Typical hit rate >90% for datasets with repeated patch geometries.
    %   Masks larger than 1e6 pixels are not cached.
    %
    % Ellipse Rotation Convention:
    %   Positive angle = clockwise rotation from horizontal major axis
    %   (matches MATLAB image coordinate system where Y increases downward)
    %
    % See also: coordinate_io, image_io, geometry_transform

    %% Public API
    % Quad masks
    utils.createQuadMask = @createQuadMask;
    utils.cropWithQuadMask = @cropWithQuadMask;

    % Ellipse masks
    utils.createEllipseMask = @createEllipseMask;
    utils.createEllipseMaskCached = @createEllipseMaskCached;
    utils.cropWithEllipseMask = @cropWithEllipseMask;

    % Caching
    utils.createMaskCache = @createMaskCache;

    % Utilities
    utils.computeEllipseBoundingBox = @computeEllipseBoundingBox;
    utils.applyMaskToImage = @applyMaskToImage;

    % Statistics
    utils.calculateChannelStats = @calculateChannelStats;
    utils.createValidPixelMask = @createValidPixelMask;
end

%% =========================================================================
%% QUAD MASKS
%% =========================================================================

function mask = createQuadMask(imageSize, vertices)
    % Create binary mask for quadrilateral region
    %
    % INPUTS:
    %   imageSize - [height, width] of mask
    %   vertices  - [N x 2] quad vertices [x, y]
    %
    % OUTPUTS:
    %   mask - Logical array (true inside quad)

    if numel(imageSize) >= 2
        h = imageSize(1);
        w = imageSize(2);
    else
        error('mask_utils:invalid_size', 'imageSize must have at least 2 elements [height, width]');
    end

    if isempty(vertices) || size(vertices, 1) < 3
        mask = false(h, w);
        return;
    end

    mask = poly2mask(vertices(:, 1), vertices(:, 2), h, w);
end

function [cropped, mask] = cropWithQuadMask(img, vertices)
    % Crop image to quadrilateral region with mask applied
    %
    % INPUTS:
    %   img      - Input image (H x W x C)
    %   vertices - [N x 2] quad vertices [x, y]
    %
    % OUTPUTS:
    %   cropped - Cropped and masked image (bounding box size)
    %   mask    - Binary mask for cropped region
    %
    % Pixels outside the quad are set to 0 (black).

    [h, w, c] = size(img);

    if isempty(vertices) || size(vertices, 1) < 3
        cropped = img;
        mask = true(h, w);
        return;
    end

    % Compute bounding box (optimized: create mask only for bbox)
    minX = max(1, floor(min(vertices(:, 1))));
    maxX = min(w, ceil(max(vertices(:, 1))));
    minY = max(1, floor(min(vertices(:, 2))));
    maxY = min(h, ceil(max(vertices(:, 2))));

    bboxW = maxX - minX + 1;
    bboxH = maxY - minY + 1;

    % Adjust vertices to bbox-relative coordinates
    verticesRelative = vertices;
    verticesRelative(:, 1) = verticesRelative(:, 1) - minX + 1;
    verticesRelative(:, 2) = verticesRelative(:, 2) - minY + 1;

    % Create mask for bbox region only (memory optimization)
    mask = poly2mask(verticesRelative(:, 1), verticesRelative(:, 2), bboxH, bboxW);

    % Extract and mask the region
    if c > 1
        cropped = img(minY:maxY, minX:maxX, :);
        mask3D = repmat(mask, [1, 1, c]);
        cropped(~mask3D) = 0;
    else
        cropped = img(minY:maxY, minX:maxX);
        cropped(~mask) = 0;
    end
end

%% =========================================================================
%% ELLIPSE MASKS
%% =========================================================================

function mask = createEllipseMask(imageSize, cx, cy, semiMajor, semiMinor, rotationAngle)
    % Create binary mask for ellipse region
    %
    % INPUTS:
    %   imageSize     - [height, width] of mask
    %   cx, cy        - Ellipse center coordinates
    %   semiMajor     - Semi-major axis length (must be >= semiMinor)
    %   semiMinor     - Semi-minor axis length
    %   rotationAngle - Rotation in degrees (clockwise from horizontal)
    %
    % OUTPUTS:
    %   mask - Logical array (true inside ellipse)
    %
    % Rotation Convention:
    %   Positive angle rotates the major axis clockwise from horizontal.
    %   This matches MATLAB's image coordinate system (Y increases downward).

    if numel(imageSize) >= 2
        h = imageSize(1);
        w = imageSize(2);
    else
        error('mask_utils:invalid_size', 'imageSize must have at least 2 elements [height, width]');
    end

    if semiMajor <= 0 || semiMinor <= 0
        mask = false(h, w);
        return;
    end

    % Enforce semiMajor >= semiMinor convention (swap and rotate 90° if needed)
    if semiMajor < semiMinor
        [semiMajor, semiMinor] = deal(semiMinor, semiMajor);
        rotationAngle = rotationAngle + 90;
    end

    % Create coordinate grids
    [X, Y] = meshgrid(1:w, 1:h);

    % Convert rotation to radians (ellipse rotation is specified clockwise;
    % we rotate points counterclockwise to undo it)
    theta = deg2rad(rotationAngle);

    % Translate to ellipse center
    dx = X - cx;
    dy = Y - cy;

    % Rotate coordinates into the ellipse's principal frame (CCW by theta)
    x_rot =  dx * cos(theta) - dy * sin(theta);
    y_rot =  dx * sin(theta) + dy * cos(theta);

    % Apply standard ellipse equation: (x/a)^2 + (y/b)^2 <= 1
    mask = (x_rot ./ semiMajor).^2 + (y_rot ./ semiMinor).^2 <= 1;
end

function mask = createEllipseMaskCached(imageSize, cx, cy, semiMajor, semiMinor, rotationAngle, cache)
    % Create ellipse mask with caching for performance
    %
    % INPUTS:
    %   imageSize     - [height, width] of mask
    %   cx, cy        - Ellipse center
    %   semiMajor     - Semi-major axis
    %   semiMinor     - Semi-minor axis
    %   rotationAngle - Rotation in degrees
    %   cache         - containers.Map from createMaskCache()
    %
    % OUTPUTS:
    %   mask - Logical array (true inside ellipse)
    %
    % Cache key format: 'h_w_cx_cy_a_b_theta' (4 decimal places)
    % Masks larger than 1e6 pixels are not cached to save memory.

    h = imageSize(1);
    w = imageSize(2);

    % Normalize parameters for consistent cache keys (semiMajor >= semiMinor)
    if semiMajor < semiMinor
        [semiMajor, semiMinor] = deal(semiMinor, semiMajor);
        rotationAngle = rotationAngle + 90;
    end

    % Generate cache key with normalized parameters
    cacheKey = sprintf('%d_%d_%.4f_%.4f_%.4f_%.4f_%.4f', ...
        h, w, cx, cy, semiMajor, semiMinor, rotationAngle);

    % Check cache
    if isKey(cache, cacheKey)
        mask = cache(cacheKey);
        return;
    end

    % Generate new mask
    mask = createEllipseMask(imageSize, cx, cy, semiMajor, semiMinor, rotationAngle);

    % Cache if not too large (1e6 pixels = 1000x1000)
    % containers.Map is a handle class - mutations persist to caller
    if h * w < 1e6
        cache(cacheKey) = mask;  %#ok<NASGU> - cache side effect
    end
end

function [cropped, mask] = cropWithEllipseMask(img, cx, cy, semiMajor, semiMinor, rotationAngle, cache)
    % Crop image to ellipse region with mask applied
    %
    % INPUTS:
    %   img           - Input image (H x W x C)
    %   cx, cy        - Ellipse center
    %   semiMajor     - Semi-major axis
    %   semiMinor     - Semi-minor axis
    %   rotationAngle - Rotation in degrees
    %   cache         - (Optional) Mask cache from createMaskCache()
    %
    % OUTPUTS:
    %   cropped - Cropped and masked image (bounding box size)
    %   mask    - Binary mask for cropped region
    %
    % Pixels outside the ellipse are set to 0 (black).

    [imgH, imgW, ~] = size(img);

    if semiMajor <= 0 || semiMinor <= 0
        cropped = [];
        mask = [];
        return;
    end

    % Compute axis-aligned bounding box
    [x1, y1, x2, y2] = computeEllipseBoundingBox(cx, cy, semiMajor, semiMinor, rotationAngle, imgW, imgH);

    % Extract region
    patchRegion = img(y1:y2, x1:x2, :);
    [patchH, patchW, ~] = size(patchRegion);

    % Compute center in patch coordinates
    centerX_patch = cx - x1 + 1;
    centerY_patch = cy - y1 + 1;

    % Create ellipse mask
    if nargin >= 7 && ~isempty(cache)
        mask = createEllipseMaskCached([patchH, patchW], centerX_patch, centerY_patch, ...
            semiMajor, semiMinor, rotationAngle, cache);
    else
        mask = createEllipseMask([patchH, patchW], centerX_patch, centerY_patch, ...
            semiMajor, semiMinor, rotationAngle);
    end

    % Apply mask
    cropped = applyMaskToImage(patchRegion, mask);
end

%% =========================================================================
%% CACHING
%% =========================================================================

function cache = createMaskCache()
    % Create cache for ellipse mask storage
    %
    % OUTPUTS:
    %   cache - containers.Map for mask caching
    %
    % Usage:
    %   cache = masks.createMaskCache();
    %   % Process batch of patches
    %   for i = 1:numPatches
    %       mask = masks.createEllipseMaskCached(size, cx, cy, a, b, theta, cache);
    %       % ... use mask
    %   end
    %
    % Cache persists for entire processing run. Do not reuse across
    % different runs as it does not invalidate on parameter changes.

    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
end

%% =========================================================================
%% UTILITIES
%% =========================================================================

function [x1, y1, x2, y2] = computeEllipseBoundingBox(cx, cy, semiMajor, semiMinor, rotationAngle, imgW, imgH)
    % Compute axis-aligned bounding box for rotated ellipse
    %
    % INPUTS:
    %   cx, cy        - Ellipse center
    %   semiMajor     - Semi-major axis
    %   semiMinor     - Semi-minor axis
    %   rotationAngle - Rotation in degrees
    %   imgW, imgH    - Image dimensions for clamping
    %
    % OUTPUTS:
    %   x1, y1, x2, y2 - Bounding box corners (clamped to image bounds)

    theta = deg2rad(rotationAngle);

    % Compute extent in each direction
    % Using parametric ellipse: x = a*cos(t)*cos(theta) - b*sin(t)*sin(theta)
    %                          y = a*cos(t)*sin(theta) + b*sin(t)*cos(theta)
    % Maximum extent occurs at: tan(t) = -b*tan(theta)/a for x
    %                          tan(t) = b*cot(theta)/a for y
    % Simplified formula using max extent:
    ux = sqrt((semiMajor * cos(theta))^2 + (semiMinor * sin(theta))^2);
    uy = sqrt((semiMajor * sin(theta))^2 + (semiMinor * cos(theta))^2);

    x1 = max(1, floor(cx - ux));
    y1 = max(1, floor(cy - uy));
    x2 = min(imgW, ceil(cx + ux));
    y2 = min(imgH, ceil(cy + uy));

    % Ensure valid bounds
    if x2 < x1, x2 = x1; end
    if y2 < y1, y2 = y1; end
end

function maskedImg = applyMaskToImage(img, mask)
    % Apply binary mask to image (set outside pixels to 0)
    %
    % INPUTS:
    %   img  - Input image (H x W x C)
    %   mask - Binary mask (H x W)
    %
    % OUTPUTS:
    %   maskedImg - Masked image

    maskedImg = img;
    numChannels = size(img, 3);

    if numChannels > 1
        inverseMask3D = repmat(~mask, [1, 1, numChannels]);
        maskedImg(inverseMask3D) = 0;
    else
        maskedImg(~mask) = 0;
    end
end

%% =========================================================================
%% STATISTICS
%% =========================================================================

function stats = calculateChannelStats(channelData, mask, statTypes)
    % Calculate statistics on masked channel data
    %
    % INPUTS:
    %   channelData - 2D channel data (H x W)
    %   mask        - Binary mask (H x W)
    %   statTypes   - Cell array of statistic names to compute
    %                 Supported: 'mean', 'std', 'skewness', 'kurtosis', 'min', 'max'
    %
    % OUTPUTS:
    %   stats - Struct with requested statistics as fields
    %
    % Example:
    %   stats = masks.calculateChannelStats(Rchannel, mask, {'mean', 'std'});
    %   fprintf('Mean: %.2f, Std: %.2f\n', stats.mean, stats.std);

    stats = struct();

    if ~any(mask(:))
        % Return zeros for empty mask
        for i = 1:length(statTypes)
            stats.(statTypes{i}) = 0;
        end
        return;
    end

    pixels = channelData(mask);

    % Vectorized computation - calculate all requested statistics at once
    if any(strcmp(statTypes, 'mean'))
        stats.mean = mean(pixels);
    end
    if any(strcmp(statTypes, 'std'))
        stats.std = std(pixels);
    end
    if any(strcmp(statTypes, 'skewness'))
        stats.skew = skewness(pixels);
    end
    if any(strcmp(statTypes, 'kurtosis'))
        stats.kurt = kurtosis(pixels);
    end
    if any(strcmp(statTypes, 'min'))
        stats.min = min(pixels);
    end
    if any(strcmp(statTypes, 'max'))
        stats.max = max(pixels);
    end
end

function mask = createValidPixelMask(image)
    % Create mask identifying valid (non-zero) pixels
    %
    % INPUTS:
    %   image - Input image (H x W x C), uint8 or double
    %
    % OUTPUTS:
    %   mask - Logical mask where true indicates non-zero pixel
    %
    % A pixel is considered valid if any channel has a value > 0.

    validateattributes(image, {'uint8', 'double'}, {'3d'}, 'createValidPixelMask', 'image');
    mask = any(im2double(image) > 0, 3);
end
