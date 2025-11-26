function utils = homography_utils()
    % HOMOGRAPHY_UTILS Returns a struct of function handles for homography operations
    %
    % This utility module provides functions for:
    %   - Computing homography transformations from viewpoint parameters
    %   - Computing homography from point correspondences
    %   - Transforming polygons through projective transformations
    %   - Transforming ellipses through homography via conic matrix math
    %   - Rotation and affine transformations
    %
    % Usage:
    %   h = homography_utils();
    %   tform = h.compute_homography(imageSize, viewParams, cameraCfg);
    %   ellipseOut = h.transform_ellipse(ellipseIn, tform);
    %
    % Extracted from augment_dataset.m for reuse across pipeline scripts.

    % Core homography computation
    utils.compute_homography = @compute_homography;
    utils.compute_homography_from_points = @compute_homography_from_points;
    utils.project_corners = @project_corners;
    utils.fit_points_to_frame = @fit_points_to_frame;

    % Polygon transformation
    utils.transform_polygon = @transform_polygon;
    utils.transform_polygon_content = @transform_polygon_content;

    % Ellipse transformation
    utils.transform_ellipse = @transform_ellipse;
    utils.transform_region_ellipses = @transform_region_ellipses;
    utils.ellipse_to_conic = @ellipse_to_conic;
    utils.conic_to_ellipse = @conic_to_ellipse;
    utils.map_ellipse_crop_to_image = @map_ellipse_crop_to_image;
    utils.invalid_ellipse = @invalid_ellipse;

    % Rotation and affine
    utils.centered_rotation_tform = @centered_rotation_tform;
    utils.normalizeAngle = @normalizeAngle;

    % Viewpoint sampling and utilities
    utils.sample_viewpoint = @sample_viewpoint;
    utils.normalize_to_angle = @normalize_to_angle;
    utils.rand_range = @rand_range;
end

%% =========================================================================
%% CORE HOMOGRAPHY COMPUTATION
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
    %   dstPoints: Nx2 matrix of destination points (e.g., polygon vertices)
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
%% POLYGON TRANSFORMATION
%% =========================================================================

function polygonOut = transform_polygon(vertices, tform)
    % Apply geometric transformation to polygon vertices
    %
    % Inputs:
    %   vertices: Nx2 matrix of polygon vertex coordinates
    %   tform: geometric transformation object (projective2d or affine2d)
    %
    % Output:
    %   polygonOut: Nx2 matrix of transformed vertex coordinates

    [x, y] = transformPointsForward(tform, vertices(:,1), vertices(:,2));
    polygonOut = [x, y];
end

function augImg = transform_polygon_content(content, origVerts, augVerts, bbox)
    % Warp polygon image content to match target geometry
    %
    % Inputs:
    %   content: image patch containing the polygon
    %   origVerts: 4x2 original polygon vertices
    %   augVerts: 4x2 target polygon vertices
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
%% ELLIPSE TRANSFORMATION
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

    Hinv = inv(H);
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
    %   origVertices: 4x2 original polygon vertices
    %   contentBbox: [x, y, w, h] bounding box of content
    %   augVertices: 4x2 augmented polygon vertices
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

        % Validate ellipse lies inside the original polygon before augmentations
        if ~inpolygon(ellipseInImageSpace.center(1), ellipseInImageSpace.center(2), ...
                      origVertices(:,1), origVertices(:,2))
            warning('homography_utils:ellipseOutsideOriginal', ...
                    '  ! Ellipse %s con %d rep %d outside original polygon. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Apply shared perspective + rotation + independent rotation transforms
        ellipseAug = transform_ellipse(ellipseInImageSpace, tformPersp);
        if ~ellipseAug.valid
            warning('homography_utils:ellipseInvalid1', ...
                    '  ! Ellipse %s con %d rep %d invalid after perspective. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        ellipseAug = transform_ellipse(ellipseAug, tformRot);
        if ~ellipseAug.valid
            warning('homography_utils:ellipseInvalid2', ...
                    '  ! Ellipse %s con %d rep %d invalid after shared rotation. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        ellipseAug = transform_ellipse(ellipseAug, tformIndepRot);
        if ~ellipseAug.valid
            warning('homography_utils:ellipseInvalid3', ...
                    '  ! Ellipse %s con %d rep %d invalid after independent rotation. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Ensure ellipse stays inside the augmented polygon footprint
        if ~inpolygon(ellipseAug.center(1), ellipseAug.center(2), ...
                      augVertices(:,1), augVertices(:,2))
            warning('homography_utils:ellipseOutside', ...
                    '  ! Ellipse %s con %d rep %d outside transformed polygon. Skipping.', ...
                    paperBase, concentration, ellipseIn.replicate);
            continue;
        end

        % Convert to polygon-crop coordinates
        ellipseCrop = ellipseAug;
        ellipseCrop.center = ellipseAug.center - [minXCrop, minYCrop];

        if ~isfinite(ellipseCrop.semiMajor) || ~isfinite(ellipseCrop.semiMinor) || ...
           ellipseCrop.semiMajor <= 0 || ellipseCrop.semiMinor <= 0
            warning('homography_utils:invalidAxes', ...
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
%% ROTATION AND AFFINE TRANSFORMATIONS
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

function angle = normalizeAngle(angle)
    % Normalize angle to range [-180, 180] degrees
    angle = mod(angle + 180, 360) - 180;
end

%% =========================================================================
%% VIEWPOINT SAMPLING AND UTILITIES
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
        error('homography_utils:invalidRange', ...
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
