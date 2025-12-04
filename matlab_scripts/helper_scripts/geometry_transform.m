function geomTform = geometry_transform()
    %% GEOMETRY_TRANSFORM Returns a struct of function handles for geometry and homography operations
    %
    % This utility module consolidates pure geometry math (quad/ellipse transformations)
    % and projective/homography operations for the microPAD processing pipeline.
    %
    % Usage:
    %   geomTform = geometry_transform();
    %
    %   % Geometry operations
    %   quads = geomTform.geom.calculateDefaultQuads(width, height, cfg);
    %   [rotated, newSize] = geomTform.geom.rotateQuadsDiscrete(quads, imageSize, 90);
    %
    %   % Homography operations
    %   tform = geomTform.homog.computeHomography(imageSize, viewParams, cameraCfg);
    %   ellipseOut = geomTform.homog.transformEllipse(ellipseIn, tform);
    %
    % Coordinate Conventions:
    %   - Image coordinates: (1,1) is top-left, Y increases downward
    %   - Rotation: Positive = clockwise in image coordinates
    %   - Quad vertices: 4x2 matrix, clockwise from top-left
    %   - Ellipse format: [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
    %
    % See also: coordinate_io, mask_utils

    %% Public API - Geometry: Default geometry generation
    geomTform.geom.calculateDefaultQuads = @calculateDefaultQuads;
    geomTform.geom.scaleAndCenterQuads = @scaleAndCenterQuads;

    %% Public API - Geometry: Quad transformations
    geomTform.geom.rotateQuadsDiscrete = @rotateQuadsDiscrete;
    geomTform.geom.rotatePointsDiscrete = @rotatePointsDiscrete;
    geomTform.geom.clampQuadToImage = @clampQuadToImage;
    geomTform.geom.scaleQuadsForImageSize = @scaleQuadsForImageSize;

    %% Public API - Geometry: Coordinate frame transformations
    geomTform.geom.inverseRotatePoints = @inverseRotatePoints;
    geomTform.geom.forwardRotatePoints = @forwardRotatePoints;

    %% Public API - Geometry: Display/base coordinate conversion
    geomTform.geom.computeDisplayImageSize = @computeDisplayImageSize;
    geomTform.geom.convertBaseQuadsToDisplay = @convertBaseQuadsToDisplay;
    geomTform.geom.convertDisplayQuadsToBase = @convertDisplayQuadsToBase;

    %% Public API - Geometry: Ellipse transformations
    geomTform.geom.scaleEllipsesForQuadChange = @scaleEllipsesForQuadChange;
    geomTform.geom.convertBaseEllipsesToDisplay = @convertBaseEllipsesToDisplay;
    geomTform.geom.enforceEllipseAxisLimits = @enforceEllipseAxisLimits;
    geomTform.geom.computeEllipseAxisBounds = @computeEllipseAxisBounds;

    %% Public API - Geometry: Bounds calculation
    geomTform.geom.computeQuadBounds = @computeQuadBounds;
    geomTform.geom.computeEllipseBounds = @computeEllipseBounds;

    %% Public API - Homography: Core computation
    geomTform.homog.computeHomography = @compute_homography;
    geomTform.homog.computeHomographyFromPoints = @compute_homography_from_points;
    geomTform.homog.projectCorners = @project_corners;
    geomTform.homog.fitPointsToFrame = @fit_points_to_frame;

    %% Public API - Homography: Quad transformation
    geomTform.homog.transformQuad = @transform_quad;
    geomTform.homog.transformQuadContent = @transform_quad_content;

    %% Public API - Homography: Ellipse transformation
    geomTform.homog.transformEllipse = @transform_ellipse;
    geomTform.homog.transformRegionEllipses = @transform_region_ellipses;
    geomTform.homog.ellipseToConic = @ellipse_to_conic;
    geomTform.homog.conicToEllipse = @conic_to_ellipse;
    geomTform.homog.mapEllipseCropToImage = @map_ellipse_crop_to_image;
    geomTform.homog.invalidEllipse = @invalid_ellipse;

    %% Public API - Homography: Rotation and affine
    geomTform.homog.centeredRotationTform = @centered_rotation_tform;

    %% Public API - Homography: Viewpoint sampling
    geomTform.homog.sampleViewpoint = @sample_viewpoint;
    geomTform.homog.normalizeToAngle = @normalize_to_angle;
    geomTform.homog.randRange = @rand_range;

    %% Public API - Shared utilities
    geomTform.isMultipleOfNinety = @isMultipleOfNinety;
    geomTform.normalizeAngle = @normalizeAngle;

    %% Constants
    geomTform.DEFAULT_ANGLE_TOLERANCE = 0.01;  % degrees
end

%% =========================================================================
%% GEOMETRY: DEFAULT GEOMETRY GENERATION
%% =========================================================================

function quads = calculateDefaultQuads(imageWidth, imageHeight, cfg)
    % Generate default quadrilateral positions using geometry parameters
    %
    % INPUTS:
    %   imageWidth  - Image width in pixels
    %   imageHeight - Image height in pixels
    %   cfg         - Configuration struct with fields:
    %                 .numSquares - Number of quads
    %                 .geometry.aspectRatio - Width/height ratio per quad
    %                 .geometry.gapPercentWidth - Gap as fraction of quad width
    %                 .coverage - Fraction of image width to cover
    %
    % OUTPUTS:
    %   quads - [N x 4 x 2] array of quad vertices

    n = cfg.numSquares;

    % Build world coordinates
    aspect = cfg.geometry.aspectRatio;
    aspect = max(aspect, eps);
    totalGridWidth = 1.0;

    % Compute gap size and individual rectangle width
    gp = cfg.geometry.gapPercentWidth;
    denom = n + max(n-1, 0) * gp;
    if denom <= 0
        denom = max(n, 1);
    end
    w = totalGridWidth / denom;
    gapSizeWorld = gp * w;

    % Calculate height based on individual rectangle width
    rectHeightWorld = w / aspect;

    % Build world corners
    worldCorners = zeros(n, 4, 2);
    xi = -totalGridWidth / 2;
    yi = -rectHeightWorld / 2;

    for i = 1:n
        worldCorners(i, :, :) = [
            xi,       yi;
            xi + w,   yi;
            xi + w,   yi + rectHeightWorld;
            xi,       yi + rectHeightWorld
        ];
        xi = xi + w + gapSizeWorld;
    end

    % Scale and center to image
    quads = scaleAndCenterQuads(worldCorners, imageWidth, imageHeight, cfg);
end

function quads = scaleAndCenterQuads(worldCorners, imageWidth, imageHeight, cfg)
    % Scale world coordinates to fit image with coverage factor
    %
    % INPUTS:
    %   worldCorners - [N x 4 x 2] array in normalized world coordinates
    %   imageWidth   - Target image width
    %   imageHeight  - Target image height
    %   cfg          - Configuration with .coverage field
    %
    % OUTPUTS:
    %   quads - [N x 4 x 2] array in pixel coordinates

    n = size(worldCorners, 1);
    quads = zeros(n, 4, 2);

    % Find bounding box of all world corners
    allX = worldCorners(:, :, 1);
    minX = min(allX(:));
    maxX = max(allX(:));
    worldW = maxX - minX;

    % Validate non-degenerate input
    if worldW <= 0
        error('geometry_transform:degenerateQuad', ...
            'World corners have zero width - cannot scale degenerate quad');
    end

    % Scale to fit image width with coverage factor
    targetWidth = imageWidth * cfg.coverage;
    scale = targetWidth / worldW;

    % Center in image
    centerX = imageWidth / 2;
    centerY = imageHeight / 2;

    for i = 1:n
        corners = squeeze(worldCorners(i, :, :));
        scaled = corners * scale;
        scaled(:, 1) = scaled(:, 1) + centerX;
        scaled(:, 2) = scaled(:, 2) + centerY;
        quads(i, :, :) = scaled;
    end
end

%% =========================================================================
%% GEOMETRY: QUAD TRANSFORMATIONS
%% =========================================================================

function [rotatedQuads, newSize] = rotateQuadsDiscrete(quads, imageSize, rotation)
    % Rotate quads by multiples of 90 degrees using rot90 conventions
    %
    % INPUTS:
    %   quads     - [N x 4 x 2] array of quad vertices
    %   imageSize - [height, width] of image
    %   rotation  - Rotation angle in degrees (will be rounded to nearest 90°)
    %
    % OUTPUTS:
    %   rotatedQuads - [N x 4 x 2] rotated vertices
    %   newSize      - [height, width] of rotated image

    imageSize = imageSize(1:2);
    [numQuads, numVertices, ~] = size(quads);
    rotatedQuads = quads;
    newSize = imageSize;

    if isempty(quads)
        return;
    end

    k = mod(round(rotation / 90), 4);
    if k == 0
        return;
    end

    H = imageSize(1);
    W = imageSize(2);
    rotatedQuads = zeros(size(quads));

    switch k
        case 1  % 90 degrees clockwise
            newSize = [W, H];
            for i = 1:numQuads
                quad = squeeze(quads(i, :, :));
                transformed = zeros(numVertices, 2);
                transformed(:, 1) = H - quad(:, 2) + 1;
                transformed(:, 2) = quad(:, 1);
                rotatedQuads(i, :, :) = clampQuadToImage(transformed, newSize);
            end
        case 2  % 180 degrees
            newSize = [H, W];
            for i = 1:numQuads
                quad = squeeze(quads(i, :, :));
                transformed = zeros(numVertices, 2);
                transformed(:, 1) = W - quad(:, 1) + 1;
                transformed(:, 2) = H - quad(:, 2) + 1;
                rotatedQuads(i, :, :) = clampQuadToImage(transformed, newSize);
            end
        case 3  % 270 degrees clockwise (90 counter-clockwise)
            newSize = [W, H];
            for i = 1:numQuads
                quad = squeeze(quads(i, :, :));
                transformed = zeros(numVertices, 2);
                transformed(:, 1) = quad(:, 2);
                transformed(:, 2) = W - quad(:, 1) + 1;
                rotatedQuads(i, :, :) = clampQuadToImage(transformed, newSize);
            end
    end
end

function [rotatedPoints, newSize] = rotatePointsDiscrete(points, imageSize, rotation)
    % Rotate points by multiples of 90 degrees
    %
    % INPUTS:
    %   points    - [N x 2] array of points [x, y]
    %   imageSize - [height, width] of image
    %   rotation  - Rotation angle in degrees
    %
    % OUTPUTS:
    %   rotatedPoints - [N x 2] rotated points
    %   newSize       - [height, width] of rotated image

    rotatedPoints = points;
    newSize = imageSize(1:2);

    if isempty(points)
        return;
    end

    k = mod(round(rotation / 90), 4);
    if k == 0
        return;
    end

    H = imageSize(1);
    W = imageSize(2);
    rotatedPoints = zeros(size(points));

    switch k
        case 1 % 90 degrees clockwise
            newSize = [W, H];
            rotatedPoints(:, 1) = H - points(:, 2) + 1;
            rotatedPoints(:, 2) = points(:, 1);
        case 2 % 180 degrees
            newSize = [H, W];
            rotatedPoints(:, 1) = W - points(:, 1) + 1;
            rotatedPoints(:, 2) = H - points(:, 2) + 1;
        case 3 % 270 degrees clockwise (90 ccw)
            newSize = [W, H];
            rotatedPoints(:, 1) = points(:, 2);
            rotatedPoints(:, 2) = W - points(:, 1) + 1;
    end
end

function clamped = clampQuadToImage(quad, imageSize)
    % Clamp quad coordinates to lie within image extents
    %
    % INPUTS:
    %   quad      - [N x 2] quad vertices [x, y]
    %   imageSize - [height, width] of image
    %
    % OUTPUTS:
    %   clamped - Clamped quad vertices

    if isempty(quad)
        clamped = quad;
        return;
    end

    width = imageSize(2);
    height = imageSize(1);
    clamped = quad;
    clamped(:, 1) = max(1, min(width, clamped(:, 1)));
    clamped(:, 2) = max(1, min(height, clamped(:, 2)));
end

function scaledQuads = scaleQuadsForImageSize(quads, oldSize, newSize, expectedCount)
    % Scale quad coordinates when image dimensions change
    %
    % INPUTS:
    %   quads         - [N x 4 x 2] array of quad vertices
    %   oldSize       - [height, width] of previous image
    %   newSize       - [height, width] of current image
    %   expectedCount - (Optional) expected number of quads
    %
    % OUTPUTS:
    %   scaledQuads - Scaled quad array, or [] if count mismatch

    if isempty(oldSize) || any(oldSize <= 0) || isempty(newSize) || any(newSize <= 0)
        error('geometry_transform:invalid_dimensions', ...
            'Cannot scale quads: invalid dimensions [%d %d] -> [%d %d]', ...
            oldSize(1), oldSize(2), newSize(1), newSize(2));
    end

    % Validate quad count if expectedCount is provided
    numQuads = size(quads, 1);
    if nargin >= 4 && ~isempty(expectedCount) && numQuads ~= expectedCount
        scaledQuads = [];
        return;
    end

    oldHeight = oldSize(1);
    oldWidth = oldSize(2);
    newHeight = newSize(1);
    newWidth = newSize(2);

    if oldHeight == newHeight && oldWidth == newWidth
        scaledQuads = quads;
        return;
    end

    scaleX = newWidth / oldWidth;
    scaleY = newHeight / oldHeight;

    scaledQuads = zeros(size(quads));

    for i = 1:numQuads
        quad = squeeze(quads(i, :, :));
        quad(:, 1) = quad(:, 1) * scaleX;
        quad(:, 2) = quad(:, 2) * scaleY;
        quad(:, 1) = max(1, min(quad(:, 1), newWidth));
        quad(:, 2) = max(1, min(quad(:, 2), newHeight));
        scaledQuads(i, :, :) = quad;
    end
end

%% =========================================================================
%% GEOMETRY: COORDINATE FRAME TRANSFORMATIONS
%% =========================================================================

function transformedPoints = inverseRotatePoints(points, rotatedSize, originalSize, rotation, angleTolerance)
    % Transform points from rotated image frame back to original frame
    %
    % ROTATION CONVENTION: Positive rotation = clockwise in image coordinates
    % TRANSFORMATION: rotated -> original (inverse/reverse rotation)
    %
    % INPUTS:
    %   points         - [N x 2] points in rotated frame
    %   rotatedSize    - [height, width] of rotated image
    %   originalSize   - [height, width] of original image
    %   rotation       - Applied rotation angle (degrees)
    %   angleTolerance - Tolerance for 90° detection (default: 0.01)
    %
    % OUTPUTS:
    %   transformedPoints - [N x 2] points in original frame

    if nargin < 5
        angleTolerance = 0.01;
    end

    if rotation == 0 || isempty(points)
        transformedPoints = points;
        return;
    end

    % Handle exact 90-degree rotations
    if abs(mod(rotation, 90)) < angleTolerance
        numRotations = mod(round(rotation / 90), 4);

        switch numRotations
            case 1  % -90 degrees (rot90(..., -1))
                x_orig = points(:, 2);
                y_orig = rotatedSize(2) - points(:, 1) + 1;
            case 2  % 180 degrees
                x_orig = rotatedSize(2) - points(:, 1) + 1;
                y_orig = rotatedSize(1) - points(:, 2) + 1;
            case 3  % 90 degrees (rot90(..., 1))
                x_orig = rotatedSize(1) - points(:, 2) + 1;
                y_orig = points(:, 1);
            otherwise  % 0 degrees
                x_orig = points(:, 1);
                y_orig = points(:, 2);
        end

        transformedPoints = [x_orig, y_orig];
    else
        % For non-90-degree rotations, use geometric transform (inverse rotation)
        theta = -deg2rad(rotation);  % Inverse rotation
        cosTheta = cos(theta);
        sinTheta = sin(theta);

        % Centers of rotated and original images
        centerRotated = [rotatedSize(2)/2, rotatedSize(1)/2];
        centerOriginal = [originalSize(2)/2, originalSize(1)/2];

        % Translate to origin, rotate, translate back
        pointsCentered = points - centerRotated;
        x_orig = pointsCentered(:, 1) * cosTheta + pointsCentered(:, 2) * sinTheta;
        y_orig = -pointsCentered(:, 1) * sinTheta + pointsCentered(:, 2) * cosTheta;

        transformedPoints = [x_orig + centerOriginal(1), y_orig + centerOriginal(2)];
    end
end

function transformedPoints = forwardRotatePoints(points, originalSize, rotatedSize, rotation, angleTolerance)
    % Transform points from original image frame to rotated frame
    %
    % ROTATION CONVENTION: Positive rotation = clockwise in image coordinates
    % TRANSFORMATION: original -> rotated (forward rotation)
    %
    % INPUTS:
    %   points         - [N x 2] points in original frame
    %   originalSize   - [height, width] of original image
    %   rotatedSize    - [height, width] of rotated image
    %   rotation       - Rotation angle (degrees)
    %   angleTolerance - Tolerance for 90° detection (default: 0.01)
    %
    % OUTPUTS:
    %   transformedPoints - [N x 2] points in rotated frame

    if nargin < 5
        angleTolerance = 0.01;
    end

    if rotation == 0 || isempty(points)
        transformedPoints = points;
        return;
    end

    % Handle exact 90-degree rotations
    if abs(mod(rotation, 90)) < angleTolerance
        numRotations = mod(round(rotation / 90), 4);
        H = originalSize(1);
        W = originalSize(2);

        switch numRotations
            case 1  % 90 degrees clockwise
                x_rot = H - points(:, 2) + 1;
                y_rot = points(:, 1);
            case 2  % 180 degrees
                x_rot = W - points(:, 1) + 1;
                y_rot = H - points(:, 2) + 1;
            case 3  % 270 degrees clockwise (= 90 CCW)
                x_rot = points(:, 2);
                y_rot = W - points(:, 1) + 1;
            otherwise  % 0 degrees
                x_rot = points(:, 1);
                y_rot = points(:, 2);
        end

        transformedPoints = [x_rot, y_rot];
    else
        % For non-90-degree rotations, use geometric transform
        theta = deg2rad(rotation);
        cosTheta = cos(theta);
        sinTheta = sin(theta);

        % Centers of original and rotated images
        centerOriginal = [originalSize(2)/2, originalSize(1)/2];
        centerRotated = [rotatedSize(2)/2, rotatedSize(1)/2];

        % Translate to origin, rotate, translate back
        pointsCentered = points - centerOriginal;
        x_rot = pointsCentered(:, 1) * cosTheta + pointsCentered(:, 2) * sinTheta;
        y_rot = -pointsCentered(:, 1) * sinTheta + pointsCentered(:, 2) * cosTheta;

        transformedPoints = [x_rot + centerRotated(1), y_rot + centerRotated(2)];
    end
end

%% =========================================================================
%% GEOMETRY: DISPLAY/BASE COORDINATE CONVERSION
%% =========================================================================

function displaySize = computeDisplayImageSize(baseSize, rotation, angleTolerance)
    % Calculate image dimensions after rotation
    %
    % INPUTS:
    %   baseSize       - [height, width] of original image
    %   rotation       - Rotation angle (degrees)
    %   angleTolerance - (Optional) tolerance for 90° detection (default: 0.01)
    %
    % OUTPUTS:
    %   displaySize - [height, width] after rotation

    if nargin < 3
        angleTolerance = 0.01;
    end

    displaySize = baseSize;
    if rotation == 0
        return;
    end

    if abs(mod(rotation, 90)) < angleTolerance
        k = mod(round(rotation / 90), 2);
        if k == 1
            displaySize = fliplr(baseSize);
        end
    end
end

function displayQuads = convertBaseQuadsToDisplay(baseQuads, baseImageSize, displayImageSize, rotation, angleTolerance)
    % Convert quads from base (unrotated) coordinates to display (rotated) coordinates
    %
    % INPUTS:
    %   baseQuads        - [N x 4 x 2] in original image frame
    %   baseImageSize    - [height, width] of original image
    %   displayImageSize - [height, width] of display image
    %   rotation         - Applied rotation (degrees)
    %   angleTolerance   - (Optional) tolerance for 90° detection
    %
    % OUTPUTS:
    %   displayQuads - [N x 4 x 2] in display frame

    if nargin < 5
        angleTolerance = 0.01;
    end

    displayQuads = baseQuads;
    if isempty(baseQuads)
        return;
    end

    originalSize = baseImageSize(1:2);

    if rotation ~= 0 && isMultipleOfNinety(rotation, angleTolerance)
        [rotatedQuads, rotatedSize] = rotateQuadsDiscrete(baseQuads, originalSize, rotation);
    else
        rotatedQuads = baseQuads;
        rotatedSize = originalSize;

        if rotation ~= 0
            numQuads = size(baseQuads, 1);
            rotatedQuads = zeros(size(baseQuads));
            for i = 1:numQuads
                quadBase = squeeze(baseQuads(i, :, :));
                quadBase = clampQuadToImage(quadBase, originalSize);
                quadRot = forwardRotatePoints(quadBase, originalSize, rotatedSize, rotation, angleTolerance);
                rotatedQuads(i, :, :) = clampQuadToImage(quadRot, rotatedSize);
            end
        end
    end

    targetSize = displayImageSize(1:2);
    if any(rotatedSize ~= targetSize)
        displayQuads = scaleQuadsForImageSize(rotatedQuads, rotatedSize, targetSize, size(baseQuads, 1));
    else
        displayQuads = rotatedQuads;
    end
end

function baseQuads = convertDisplayQuadsToBase(displayQuads, displayImageSize, baseImageSize, rotation, angleTolerance)
    % Convert quads from display (rotated) coordinates to base (unrotated) coordinates
    %
    % INPUTS:
    %   displayQuads     - [N x 4 x 2] in display frame
    %   displayImageSize - [height, width] of display image
    %   baseImageSize    - [height, width] of original image
    %   rotation         - Applied rotation (degrees)
    %   angleTolerance   - (Optional) tolerance for 90° detection
    %
    % OUTPUTS:
    %   baseQuads - [N x 4 x 2] in original frame

    if nargin < 5
        angleTolerance = 0.01;
    end

    baseQuads = displayQuads;
    if isempty(displayQuads)
        return;
    end

    displaySize = displayImageSize(1:2);
    originalSize = baseImageSize(1:2);
    numQuads = size(displayQuads, 1);

    % Compute what size would be after rotation (for inverse transform)
    rotatedSize = computeDisplayImageSize(originalSize, rotation, angleTolerance);

    % First scale from display to rotated size if needed
    if any(displaySize ~= rotatedSize)
        scaledQuads = scaleQuadsForImageSize(displayQuads, displaySize, rotatedSize, numQuads);
    else
        scaledQuads = displayQuads;
    end

    % Then inverse rotate to get base coordinates
    if rotation ~= 0
        baseQuads = zeros(size(scaledQuads));
        for i = 1:numQuads
            quadDisplay = squeeze(scaledQuads(i, :, :));
            quadBase = inverseRotatePoints(quadDisplay, rotatedSize, originalSize, rotation, angleTolerance);
            baseQuads(i, :, :) = clampQuadToImage(quadBase, originalSize);
        end
    else
        baseQuads = scaledQuads;
    end
end

%% =========================================================================
%% GEOMETRY: ELLIPSE TRANSFORMATIONS
%% =========================================================================

function scaledEllipses = scaleEllipsesForQuadChange(oldCorners, newCorners, oldEllipses, imageSize, cfg)
    % Scale ellipse positions when quad geometry changes
    %
    % INPUTS:
    %   oldCorners  - [4 x 2] old quad vertices
    %   newCorners  - [4 x 2] new quad vertices
    %   oldEllipses - [N x 7] matrix [concIdx, repIdx, x, y, semiMajor, semiMinor, rotation]
    %   imageSize   - [height, width] for bounds calculation
    %   cfg         - Configuration struct
    %
    % OUTPUTS:
    %   scaledEllipses - [N x 7] scaled ellipse parameters

    if isempty(oldEllipses)
        scaledEllipses = oldEllipses;
        return;
    end

    % Calculate old quad centroid and dimensions
    oldCentroid = mean(oldCorners, 1);
    oldMinX = min(oldCorners(:, 1));
    oldMaxX = max(oldCorners(:, 1));
    oldMinY = min(oldCorners(:, 2));
    oldMaxY = max(oldCorners(:, 2));
    oldWidth = oldMaxX - oldMinX;
    oldHeight = oldMaxY - oldMinY;

    % Calculate new quad centroid and dimensions
    newCentroid = mean(newCorners, 1);
    newMinX = min(newCorners(:, 1));
    newMaxX = max(newCorners(:, 1));
    newMinY = min(newCorners(:, 2));
    newMaxY = max(newCorners(:, 2));
    newWidth = newMaxX - newMinX;
    newHeight = newMaxY - newMinY;

    % Validate quad dimensions
    if oldWidth <= 0 || oldHeight <= 0 || newWidth <= 0 || newHeight <= 0
        warning('geometry_transform:degenerate_quad', ...
                'Quad has zero or negative dimensions - returning empty ellipses');
        scaledEllipses = [];
        return;
    end

    % Compute uniform scale factor using geometric mean
    scaleX = newWidth / oldWidth;
    scaleY = newHeight / oldHeight;
    axisScale = sqrt(scaleX * scaleY);

    % Scale ellipse parameters
    numEllipses = size(oldEllipses, 1);
    scaledEllipses = zeros(size(oldEllipses));

    bounds = computeEllipseAxisBounds(newCorners, imageSize, cfg);

    for i = 1:numEllipses
        concIdx = oldEllipses(i, 1);
        repIdx = oldEllipses(i, 2);
        oldX = oldEllipses(i, 3);
        oldY = oldEllipses(i, 4);
        oldSemiMajor = oldEllipses(i, 5);
        oldSemiMinor = oldEllipses(i, 6);
        oldRotation = oldEllipses(i, 7);

        % Transform centers relative to quad centroid
        newX = (oldX - oldCentroid(1)) * axisScale + newCentroid(1);
        newY = (oldY - oldCentroid(2)) * axisScale + newCentroid(2);

        % Scale axes uniformly
        newSemiMajor = oldSemiMajor * axisScale;
        newSemiMinor = oldSemiMinor * axisScale;

        % Enforce constraint: semiMajor >= semiMinor
        if newSemiMinor > newSemiMajor
            temp = newSemiMajor;
            newSemiMajor = newSemiMinor;
            newSemiMinor = temp;
            newRotation = mod(oldRotation + 90, 360);
        else
            newRotation = oldRotation;
        end

        % Enforce configured limits
        [newSemiMajor, newSemiMinor, newRotation] = enforceEllipseAxisLimits(newSemiMajor, newSemiMinor, newRotation, bounds);

        scaledEllipses(i, :) = [concIdx, repIdx, newX, newY, newSemiMajor, newSemiMinor, newRotation];
    end
end

function displayEllipses = convertBaseEllipsesToDisplay(ellipseData, baseImageSize, displayImageSize, rotation, angleTolerance)
    % Convert ellipses from base to display coordinates
    %
    % INPUTS:
    %   ellipseData      - [N x 7] matrix in base frame
    %   baseImageSize    - [height, width] of original image
    %   displayImageSize - [height, width] of display image
    %   rotation         - Applied rotation (degrees)
    %   angleTolerance   - (Optional) tolerance for 90° detection
    %
    % OUTPUTS:
    %   displayEllipses - [N x 7] matrix in display frame

    if nargin < 5
        angleTolerance = 0.01;
    end

    displayEllipses = ellipseData;
    if isempty(ellipseData)
        return;
    end

    if rotation ~= 0 && isMultipleOfNinety(rotation, angleTolerance)
        [rotCenters, ~] = rotatePointsDiscrete(ellipseData(:, 3:4), baseImageSize, rotation);
    else
        rotCenters = ellipseData(:, 3:4);
    end

    rotatedSize = computeDisplayImageSize(baseImageSize, rotation, angleTolerance);
    targetSize = displayImageSize(1:2);

    if any(rotatedSize ~= targetSize)
        scaleX = targetSize(2) / rotatedSize(2);
        scaleY = targetSize(1) / rotatedSize(1);
        rotCenters(:, 1) = rotCenters(:, 1) * scaleX;
        rotCenters(:, 2) = rotCenters(:, 2) * scaleY;
    end

    displayEllipses(:, 3:4) = rotCenters;
    displayEllipses(:, 7) = mod(ellipseData(:, 7) + rotation + 180, 360) - 180;
end

function [semiMajor, semiMinor, rotationAngle] = enforceEllipseAxisLimits(semiMajor, semiMinor, rotationAngle, bounds)
    % Clamp ellipse axes to configured bounds and normalize rotation
    %
    % INPUTS:
    %   semiMajor     - Semi-major axis length
    %   semiMinor     - Semi-minor axis length
    %   rotationAngle - Rotation in degrees
    %   bounds        - (Optional) struct with .minAxis and .maxAxis
    %
    % OUTPUTS:
    %   semiMajor, semiMinor, rotationAngle - Enforced values
    %
    % Ensures semiMajor >= semiMinor (swaps and rotates 90° if needed)

    if semiMinor > semiMajor
        tmp = semiMajor;
        semiMajor = semiMinor;
        semiMinor = tmp;
        rotationAngle = rotationAngle + 90;
    end

    if nargin < 4 || isempty(bounds)
        rotationAngle = mod(rotationAngle + 180, 360) - 180;
        return;
    end

    minAxis = bounds.minAxis;
    maxAxis = bounds.maxAxis;

    semiMajor = min(max(semiMajor, minAxis), maxAxis);
    semiMinor = min(max(semiMinor, minAxis), semiMajor);
    rotationAngle = mod(rotationAngle + 180, 360) - 180;
end

function bounds = computeEllipseAxisBounds(~, imageSize, cfg)
    % Compute min/max semi-axis lengths for ellipse editing
    %
    % INPUTS:
    %   (unused)  - Quad vertices (unused, for API compatibility)
    %   imageSize - [height, width] of image
    %   cfg       - Configuration with .ellipse.minAxisPercent
    %
    % OUTPUTS:
    %   bounds - struct with .minAxis and .maxAxis

    imgHeight = imageSize(1);
    imgWidth = imageSize(2);
    baseExtent = max(imgWidth, imgHeight);
    minAxis = max(1, baseExtent * cfg.ellipse.minAxisPercent);
    maxAxis = baseExtent;
    bounds = struct('minAxis', minAxis, 'maxAxis', maxAxis);
end

%% =========================================================================
%% GEOMETRY: BOUNDS CALCULATION
%% =========================================================================

function bounds = computeQuadBounds(quads)
    % Compute bounding box of all quad vertices
    %
    % INPUTS:
    %   quads - [N x 4 x 2] array or cell array of quad vertices
    %
    % OUTPUTS:
    %   bounds - struct with .minX, .maxX, .minY, .maxY

    if iscell(quads)
        % Handle cell array of drawpolygon objects or vertex matrices
        allX = [];
        allY = [];
        for i = 1:numel(quads)
            if isobject(quads{i}) && isprop(quads{i}, 'Position')
                pos = quads{i}.Position;
                allX = [allX; pos(:, 1)]; %#ok<AGROW>
                allY = [allY; pos(:, 2)]; %#ok<AGROW>
            elseif isnumeric(quads{i})
                allX = [allX; quads{i}(:, 1)]; %#ok<AGROW>
                allY = [allY; quads{i}(:, 2)]; %#ok<AGROW>
            end
        end
    else
        % Handle [N x 4 x 2] array
        allX = quads(:, :, 1);
        allY = quads(:, :, 2);
        allX = allX(:);
        allY = allY(:);
    end

    if isempty(allX)
        bounds = struct('minX', 0, 'maxX', 0, 'minY', 0, 'maxY', 0);
        return;
    end

    bounds = struct('minX', min(allX), 'maxX', max(allX), ...
                    'minY', min(allY), 'maxY', max(allY));
end

function bounds = computeEllipseBounds(x, y, semiMajor, semiMinor, rotationAngle)
    % Compute axis-aligned bounding box of a rotated ellipse
    %
    % INPUTS:
    %   x, y              - Ellipse center
    %   semiMajor         - Semi-major axis length
    %   semiMinor         - Semi-minor axis length
    %   rotationAngle     - Rotation in degrees
    %
    % OUTPUTS:
    %   bounds - struct with .minX, .maxX, .minY, .maxY

    theta = deg2rad(rotationAngle);

    % Compute axis-aligned extents
    ux = sqrt((semiMajor * cos(theta))^2 + (semiMinor * sin(theta))^2);
    uy = sqrt((semiMajor * sin(theta))^2 + (semiMinor * cos(theta))^2);

    bounds = struct('minX', x - ux, 'maxX', x + ux, ...
                    'minY', y - uy, 'maxY', y + uy);
end

%% =========================================================================
%% HOMOGRAPHY: CORE COMPUTATION
%% =========================================================================

function tform = compute_homography(imageSize, viewParams, cameraCfg)
    % Compute projective transformation from camera viewpoint parameters
    %
    % Inputs:
    %   imageSize: [height, width] of the image
    %   viewParams: struct with fields vx (yaw param), vy (pitch param), vz (depth)
    %   cameraCfg: struct with camera configuration (xRange, yRange, zRange, maxAngleDeg, etc.)
    %
    % Output:
    %   tform: projective2d transformation object

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

function tform = compute_homography_from_points(srcPoints, dstPoints)
    % Compute homography from source to destination point correspondences
    %
    % Inputs:
    %   srcPoints: Nx2 matrix of source points (e.g., unit square corners)
    %   dstPoints: Nx2 matrix of destination points (e.g., quad vertices)
    %
    % Output:
    %   tform: projective2d transformation object

    tform = fitgeotrans(srcPoints, dstPoints, 'projective');
end

function projected = project_corners(imgWidth, imgHeight, yawDeg, pitchDeg, viewZ)
    % Perform 3D perspective projection of image corners
    %
    % Applies yaw (Y-axis) and pitch (X-axis) rotations, then projects
    % onto a virtual camera plane using pinhole model.

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
    % Align projected points to fit within the output image frame
    %
    % Scales and centers the projected points to ensure the transformed
    % content stays visible within frame boundaries.

    MIN_DIMENSION = 1.0;  % Minimum valid dimension in pixels

    minX = min(projected(:,1));
    maxX = max(projected(:,1));
    minY = min(projected(:,2));
    maxY = max(projected(:,2));
    width = maxX - minX;
    height = maxY - minY;
    if width < MIN_DIMENSION || height < MIN_DIMENSION
        aligned = projected;
        return;
    end

    scale = coverage * min((imgWidth - 1) / width, (imgHeight - 1) / height);
    center = [(maxX + minX) / 2, (maxY + minY) / 2];
    targetCenter = [(imgWidth + 1) / 2, (imgHeight + 1) / 2];

    aligned = (projected - center) * scale + targetCenter;
end

%% =========================================================================
%% HOMOGRAPHY: QUAD TRANSFORMATION
%% =========================================================================

function quadOut = transform_quad(vertices, tform)
    % Apply geometric transformation to quad vertices
    %
    % Inputs:
    %   vertices: Nx2 matrix of quad vertex coordinates
    %   tform: geometric transformation object (projective2d or affine2d)
    %
    % Output:
    %   quadOut: Nx2 matrix of transformed vertex coordinates

    [x, y] = transformPointsForward(tform, vertices(:,1), vertices(:,2));
    quadOut = [x, y];
end

function augImg = transform_quad_content(content, origVerts, augVerts, bbox)
    % Warp quad image content to match target geometry
    %
    % Inputs:
    %   content: image patch containing the quad
    %   origVerts: 4x2 original quad vertices
    %   augVerts: 4x2 target quad vertices
    %   bbox: [x, y, width, height] bounding box of content in original image
    %
    % Output:
    %   augImg: warped image content

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

%% =========================================================================
%% HOMOGRAPHY: ELLIPSE TRANSFORMATION
%% =========================================================================

function ellipseOut = transform_ellipse(ellipseIn, tform)
    % Transform an ellipse through a homography/projective transformation
    %
    % Uses conic matrix representation: converts ellipse to 3x3 conic,
    % applies homography conjugate transformation (H^-T * C * H^-1),
    % then converts back to ellipse parameters.
    %
    % Inputs:
    %   ellipseIn: struct with fields center, semiMajor, semiMinor, rotation, valid
    %   tform: geometric transformation object
    %
    % Output:
    %   ellipseOut: transformed ellipse struct (or invalid_ellipse if degenerate)

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

    % Use backslash for numerical stability instead of inv()
    Hinv = H \ eye(3);
    transformedConic = Hinv' * conic * Hinv;
    ellipseOut = conic_to_ellipse(transformedConic);
end

function ellipseCropList = transform_region_ellipses(ellipseList, paperBase, concentration, ...
                                                     origVertices, contentBbox, augVertices, ...
                                                     minXCrop, minYCrop, tformPersp, tformRot, ...
                                                     tformIndepRot, totalAppliedRotation)
    % Transform ellipse annotations through multiple sequential transformations
    %
    % Orchestrates perspective + shared rotation + independent rotation transforms,
    % validates constraints, and maps to augmented crop coordinates.
    %
    % Inputs:
    %   ellipseList: array of ellipse structs with replicate, center, semiMajor, semiMinor, rotation
    %   paperBase: base name for logging
    %   concentration: concentration index for logging
    %   origVertices: 4x2 original quad vertices
    %   contentBbox: [x, y, w, h] bounding box of content
    %   augVertices: 4x2 augmented quad vertices
    %   minXCrop, minYCrop: crop offset for coordinate conversion
    %   tformPersp: perspective transformation
    %   tformRot: shared rotation transformation
    %   tformIndepRot: independent rotation transformation
    %   totalAppliedRotation: total rotation angle for reference frame adjustment
    %
    % Output:
    %   ellipseCropList: array of transformed ellipse structs in crop coordinates

    if isempty(ellipseList)
        ellipseCropList = struct('replicate', {}, 'center', {}, ...
                                 'semiMajor', {}, 'semiMinor', {}, 'rotation', {});
        return;
    end

    % Pre-allocate for all ellipses (trim to actual count after loop)
    maxEllipses = numel(ellipseList);
    ellipseCropList = repmat(struct('replicate', [], 'center', [], ...
                                    'semiMajor', [], 'semiMinor', [], 'rotation', []), ...
                             maxEllipses, 1);
    validCount = 0;

    for idx = 1:numel(ellipseList)
        ellipseIn = ellipseList(idx);

        % Map ellipse from crop space to original image coordinates
        ellipseInImageSpace = map_ellipse_crop_to_image(ellipseIn, contentBbox);

        % Validate ellipse lies inside the original quad before augmentations
        if ~inpolygon(ellipseInImageSpace.center(1), ellipseInImageSpace.center(2), ...
                      origVertices(:,1), origVertices(:,2))
            warning('geometry_transform:ellipseOutsideOriginal', ...
                    '  ! Ellipse %s con %d rep %d outside original quad. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Apply shared perspective + rotation + independent rotation transforms
        ellipseAug = transform_ellipse(ellipseInImageSpace, tformPersp);
        if ~ellipseAug.valid
            warning('geometry_transform:ellipseInvalid1', ...
                    '  ! Ellipse %s con %d rep %d invalid after perspective. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        ellipseAug = transform_ellipse(ellipseAug, tformRot);
        if ~ellipseAug.valid
            warning('geometry_transform:ellipseInvalid2', ...
                    '  ! Ellipse %s con %d rep %d invalid after shared rotation. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        ellipseAug = transform_ellipse(ellipseAug, tformIndepRot);
        if ~ellipseAug.valid
            warning('geometry_transform:ellipseInvalid3', ...
                    '  ! Ellipse %s con %d rep %d invalid after independent rotation. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Ensure ellipse stays inside the augmented quad footprint
        if ~inpolygon(ellipseAug.center(1), ellipseAug.center(2), ...
                      augVertices(:,1), augVertices(:,2))
            warning('geometry_transform:ellipseOutside', ...
                    '  ! Ellipse %s con %d rep %d outside transformed quad. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Convert to quad-crop coordinates
        ellipseCrop = ellipseAug;
        ellipseCrop.center = ellipseAug.center - [minXCrop, minYCrop];

        if ~isfinite(ellipseCrop.semiMajor) || ~isfinite(ellipseCrop.semiMinor) || ...
           ellipseCrop.semiMajor <= 0 || ellipseCrop.semiMinor <= 0
            warning('geometry_transform:invalidAxes', ...
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

        % Adjust rotation relative to the augmented patch reference frame
        ellipseCrop.rotation = normalizeAngle(ellipseCrop.rotation - totalAppliedRotation);

        validCount = validCount + 1;
        ellipseCropList(validCount) = struct( ...
            'replicate', ellipseIn.replicate, ...
            'center', ellipseCrop.center, ...
            'semiMajor', ellipseCrop.semiMajor, ...
            'semiMinor', ellipseCrop.semiMinor, ...
            'rotation', ellipseCrop.rotation);
    end

    % Trim to actual valid ellipse count
    ellipseCropList = ellipseCropList(1:validCount);
end

function conic = ellipse_to_conic(ellipse)
    % Convert ellipse parameters to 3x3 conic matrix representation
    %
    % The conic matrix C represents the ellipse equation x'*C*x = 0
    % in homogeneous coordinates.
    %
    % Input:
    %   ellipse: struct with center, semiMajor, semiMinor, rotation fields
    %
    % Output:
    %   conic: 3x3 symmetric matrix (or zeros if invalid)

    MIN_AXIS = 0.1;  % Minimum ellipse axis length in pixels

    xc = ellipse.center(1);
    yc = ellipse.center(2);
    a = ellipse.semiMajor;
    b = ellipse.semiMinor;
    theta = deg2rad(ellipse.rotation);

    % Validate axes are positive to prevent division by zero
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
    % Convert 3x3 conic matrix to ellipse parameters
    %
    % Extracts center, axes lengths, and rotation through algebraic manipulation.
    %
    % Input:
    %   C: 3x3 conic matrix
    %
    % Output:
    %   ellipse: struct with center, semiMajor, semiMinor, rotation, valid fields

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

    % Defensive check (should never trigger with validated inputs upstream)
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

function ellipseImageSpace = map_ellipse_crop_to_image(ellipseCrop, cropBbox)
    % Map ellipse from crop space to full image space
    %
    % Inputs:
    %   ellipseCrop: ellipse struct in crop-relative coordinates
    %   cropBbox: [x, y, width, height] bounding box of crop in image
    %
    % Output:
    %   ellipseImageSpace: ellipse struct in full image coordinates

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

function ellipse = invalid_ellipse()
    % Create a sentinel struct representing a degenerate/invalid ellipse
    ellipse = struct('center', [NaN, NaN], 'semiMajor', NaN, 'semiMinor', NaN, ...
                     'rotation', NaN, 'valid', false);
end

%% =========================================================================
%% HOMOGRAPHY: ROTATION AND AFFINE TRANSFORMATIONS
%% =========================================================================

function tform = centered_rotation_tform(imageSize, angleDeg)
    % Create affine transformation that rotates around image center
    %
    % Inputs:
    %   imageSize: [height, width] of the image
    %   angleDeg: rotation angle in degrees (positive = counter-clockwise)
    %
    % Output:
    %   tform: affine2d transformation object

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

%% =========================================================================
%% HOMOGRAPHY: VIEWPOINT SAMPLING AND UTILITIES
%% =========================================================================

function viewParams = sample_viewpoint(cameraCfg)
    % Randomly sample camera viewpoint parameters
    %
    % Input:
    %   cameraCfg: struct with xRange, yRange, zRange fields
    %
    % Output:
    %   viewParams: struct with vx, vy, vz fields

    vx = cameraCfg.xRange(1) + rand() * diff(cameraCfg.xRange);
    vy = cameraCfg.yRange(1) + rand() * diff(cameraCfg.yRange);
    vz = cameraCfg.zRange(1) + rand() * diff(cameraCfg.zRange);

    viewParams = struct('vx', vx, 'vy', vy, 'vz', vz);
end

function angleDeg = normalize_to_angle(value, range, maxAngle)
    % Convert a sampled parameter to a rotation angle in degrees
    %
    % Maps value from the given range to [-maxAngle, maxAngle].
    %
    % Inputs:
    %   value: sampled parameter value
    %   range: [min, max] bounds of the parameter
    %   maxAngle: maximum output angle in degrees
    %
    % Output:
    %   angleDeg: rotation angle in degrees

    if numel(range) ~= 2
        error('geometry_transform:invalidRange', ...
            'Range parameter must have exactly 2 elements. Got %d.', numel(range));
    end

    mid = mean(range);
    span = range(2) - range(1);
    if span <= 0
        angleDeg = 0;
        return;
    end
    normalized = 2 * (value - mid) / span;
    angleDeg = normalized * maxAngle;
end

function val = rand_range(range)
    % Sample uniformly from a [min, max] range
    val = range(1) + (range(2) - range(1)) * rand();
end

%% =========================================================================
%% SHARED UTILITIES
%% =========================================================================

function tf = isMultipleOfNinety(angle, tolerance)
    % Check if angle is effectively a multiple of 90 degrees
    %
    % INPUTS:
    %   angle     - Angle in degrees
    %   tolerance - Tolerance in degrees (default: 0.01)
    %
    % OUTPUTS:
    %   tf - Boolean

    if nargin < 2
        tolerance = 0.01;
    end

    if isnan(angle) || isinf(angle)
        tf = false;
        return;
    end

    tf = abs(angle / 90 - round(angle / 90)) <= tolerance;
end

function normalized = normalizeAngle(angle)
    % Normalize angle to [-180, 180] range
    %
    % INPUTS:
    %   angle - Angle in degrees
    %
    % OUTPUTS:
    %   normalized - Angle in [-180, 180]

    normalized = mod(angle + 180, 360) - 180;
end
