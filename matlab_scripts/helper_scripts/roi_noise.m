function roiNoise = roi_noise()
    %% ROI_NOISE Returns a struct of function handles for realistic sensor noise augmentation
    %
    % This utility module provides noise augmentation functions that simulate
    % realistic capture device artifacts on ROI regions (test zones) while
    % preserving background textures. Supports camera sensor noise, screen
    % capture artifacts, old photo degradation, and JPEG compression.
    %
    % Usage:
    %   roiNoise = roi_noise();
    %
    %   % Apply noise to ROIs with soft edge blending
    %   noisyImg = roiNoise.applyToROIs(img, roiMask, 'camera', rngSeed, featherWidth);
    %
    %   % Create ROI mask from quad vertices
    %   mask = roiNoise.createMaskFromQuads(quads, imgHeight, imgWidth);
    %
    %   % Select noise profile with weighted random selection
    %   profile = roiNoise.selectProfile(weights);
    %
    % Noise Types:
    %   - 'camera': Camera sensor noise (Poisson + Gaussian)
    %   - 'screen': Screen capture artifacts (color banding + gamma mismatch)
    %   - 'old_photo': Old photo degradation (blur + color shift + grain)
    %   - 'jpeg': JPEG compression artifacts
    %
    % See also: augment_dataset, augmentation_synthesis

    %% Public API - ROI mask creation
    roiNoise.createMaskFromQuads = @createMaskFromQuads;

    %% Public API - Noise application
    roiNoise.applyToROIs = @applyNoiseToROIs;
    roiNoise.selectProfile = @selectNoiseProfile;

    %% Public API - Individual noise generators
    roiNoise.cameraSensorNoise = @applyCameraSensorNoise;
    roiNoise.screenArtifacts = @applyScreenArtifacts;
    roiNoise.oldPhotoArtifacts = @applyOldPhotoArtifacts;
    roiNoise.jpegArtifacts = @applyJpegArtifacts;
end

%% =========================================================================
%% ROI MASK CREATION
%% =========================================================================

function roiMask = createMaskFromQuads(quadVertices, imgHeight, imgWidth)
    % Create binary ROI mask from quadrilateral vertices
    %
    % Inputs:
    %   quadVertices - Cell array of Nx2 matrices, each containing quad vertices [x, y]
    %   imgHeight - Image height in pixels
    %   imgWidth - Image width in pixels
    %
    % Output:
    %   roiMask - logical [imgHeight x imgWidth] mask, true for ROI pixels

    roiMask = false(imgHeight, imgWidth);

    if isempty(quadVertices)
        return;
    end

    for i = 1:numel(quadVertices)
        quad = quadVertices{i};

        % Validate quad structure
        if isempty(quad) || size(quad, 2) ~= 2 || size(quad, 1) < 3
            continue;
        end

        % Warn if not exactly 4 vertices (unexpected polygon)
        if size(quad, 1) ~= 4
            warning('roi_noise:unexpectedVertexCount', ...
                'Quad %d has %d vertices (expected 4). Processing as polygon.', i, size(quad, 1));
        end

        % Compute quad area to skip degenerate quads
        area = polyarea(quad(:,1), quad(:,2));
        if area < 100  % Skip quads smaller than 100 pixels
            continue;
        end

        % Create polygon mask for this quad
        try
            quadMask = poly2mask(quad(:,1), quad(:,2), imgHeight, imgWidth);
            roiMask = roiMask | quadMask;
        catch
            % Skip invalid polygons (vertices outside bounds, etc.)
            continue;
        end
    end
end

%% =========================================================================
%% NOISE PROFILE SELECTION
%% =========================================================================

function profileName = selectNoiseProfile(profileWeights)
    % Select noise profile using weighted random selection
    %
    % Input:
    %   profileWeights - struct with fields:
    %                    .camera (default 0.35)
    %                    .screen (default 0.25)
    %                    .old_photo (default 0.20)
    %                    .jpeg (default 0.20)
    %
    % Output:
    %   profileName - 'camera', 'screen', 'old_photo', or 'jpeg'
    %
    % Note: This function relies on the global RNG state. Ensure RNG is seeded
    %       before calling if reproducibility is required.

    % Default weights
    weights = [0.35, 0.25, 0.20, 0.20];
    profileNames = {'camera', 'screen', 'old_photo', 'jpeg'};

    % Override with provided weights if available
    if nargin > 0 && isstruct(profileWeights)
        if isfield(profileWeights, 'camera')
            weights(1) = profileWeights.camera;
        end
        if isfield(profileWeights, 'screen')
            weights(2) = profileWeights.screen;
        end
        if isfield(profileWeights, 'old_photo')
            weights(3) = profileWeights.old_photo;
        end
        if isfield(profileWeights, 'jpeg')
            weights(4) = profileWeights.jpeg;
        end
    end

    % Normalize weights
    totalWeight = sum(weights);
    if totalWeight <= 0
        weights = [0.35, 0.25, 0.20, 0.20];
        totalWeight = 1.0;
    end
    weights = weights / totalWeight;

    % Weighted random selection
    cumWeights = cumsum(weights);
    r = rand();
    idx = find(r <= cumWeights, 1, 'first');

    if isempty(idx)
        idx = 1;  % Fallback to camera
    end

    profileName = profileNames{idx};
end

%% =========================================================================
%% NOISE APPLICATION
%% =========================================================================

function noisyImg = applyNoiseToROIs(img, roiMask, profileName, rngSeed, featherWidth)
    % Apply noise to ROI regions with soft edge blending
    %
    % Uses a feathered mask to blend noise smoothly at ROI boundaries,
    % avoiding the hard edges that can make augmented regions distinguishable.
    %
    % Inputs:
    %   img - uint8 RGB image [H x W x 3]
    %   roiMask - logical [H x W] mask, true for ROI pixels
    %   profileName - 'camera', 'screen', 'old_photo', or 'jpeg'
    %   rngSeed - optional RNG seed for reproducibility
    %   featherWidth - optional edge feather width in pixels (default: 3)
    %
    % Output:
    %   noisyImg - uint8 RGB image with noise applied to ROIs

    % Default feather width for soft blending
    if nargin < 5 || isempty(featherWidth)
        featherWidth = 3;
    end

    % Set RNG seed if provided, with state preservation
    if nargin >= 4 && ~isempty(rngSeed)
        oldRng = rng;  % Save current state
        rng(rngSeed);
        cleanupRng = onCleanup(@() rng(oldRng));  % Restore on function exit (destructor-based)
    end

    % Check if any ROI pixels exist
    if ~any(roiMask(:))
        noisyImg = img;
        return;
    end

    % Generate noisy version of entire image
    switch profileName
        case 'camera'
            noisyFull = applyCameraSensorNoise(img);
        case 'screen'
            noisyFull = applyScreenArtifacts(img);
        case 'old_photo'
            noisyFull = applyOldPhotoArtifacts(img);
        case 'jpeg'
            noisyFull = applyJpegArtifacts(img);
        otherwise
            % Unknown profile, return original
            noisyImg = img;
            return;
    end

    % Create soft blend mask with feathered edges
    % This prevents hard boundaries between noisy ROIs and clean background
    if featherWidth > 0
        % Adaptive featherWidth clamping: prevent erosion from eliminating small ROIs
        % Clamp to 1/4 of minimum ROI dimension to ensure core region survives
        [rows, cols] = find(roiMask);
        if isempty(rows)
            % Edge case: roiMask became empty (shouldn't happen after earlier check)
            softMask = double(roiMask);
            noisyImg = img;
            return;
        end
        roiHeight = max(rows) - min(rows) + 1;
        roiWidth = max(cols) - min(cols) + 1;
        minDim = min(roiHeight, roiWidth);
        effectiveFeatherWidth = min(featherWidth, floor(minDim / 4));

        if effectiveFeatherWidth <= 0
            % ROI too small for feathering, use hard mask
            softMask = double(roiMask);
        else
            % Erode mask inward then blur for smooth falloff
            se = strel('disk', effectiveFeatherWidth, 0);
            erodedMask = imerode(roiMask, se);

            % Guard: if erosion still eliminates all ROIs, fall back to hard mask
            if ~any(erodedMask(:))
                softMask = double(roiMask);
            else
                sigma = max(0.5, effectiveFeatherWidth / 2);
                softMask = imgaussfilt(double(erodedMask), sigma);
                % Ensure core region stays fully opaque
                softMask = max(softMask, double(erodedMask));
                % Constrain to original mask boundary (don't extend noise outside)
                softMask = softMask .* double(roiMask);
                softMask = max(0, min(1, softMask));
            end
        end
    else
        softMask = double(roiMask);
    end

    % Blend using soft mask: result = noisy * alpha + original * (1 - alpha)
    imgDouble = double(img);
    noisyDouble = double(noisyFull);

    result = zeros(size(imgDouble));
    for c = 1:size(imgDouble, 3)
        result(:,:,c) = noisyDouble(:,:,c) .* softMask + imgDouble(:,:,c) .* (1 - softMask);
    end

    % Convert back to uint8
    noisyImg = uint8(result);
end

%% =========================================================================
%% CAMERA SENSOR NOISE
%% =========================================================================

function noisyImg = applyCameraSensorNoise(img)
    % Apply realistic camera sensor noise (signal-dependent model)
    %
    % Uses Poisson (shot noise) + Gaussian (read noise) model
    %
    % Input:
    %   img - uint8 RGB [H x W x 3]
    %
    % Output:
    %   noisyImg - uint8 RGB with sensor noise

    % Noise parameters (sampled randomly)
    poissonScale = 5 + rand() * 10;   % [5, 15]
    gaussianSigma = 3 + rand() * 4;   % [3, 7]

    % Convert to double [0, 255]
    imgDouble = double(img);

    % Apply signal-dependent noise model per channel
    noisyDouble = zeros(size(imgDouble));

    for c = 1:3
        channel = imgDouble(:,:,c);

        % 1. Poisson (shot noise) - signal-dependent
        scaled = channel / poissonScale;
        noisyChannel = poissrnd(scaled) * poissonScale;

        % 2. Gaussian (read noise) - signal-independent
        gaussNoise = gaussianSigma * randn(size(channel));
        noisyChannel = noisyChannel + gaussNoise;

        noisyDouble(:,:,c) = noisyChannel;
    end

    % Clamp and convert
    noisyImg = uint8(max(0, min(255, noisyDouble)));
end

%% =========================================================================
%% SCREEN CAPTURE ARTIFACTS
%% =========================================================================

function noisyImg = applyScreenArtifacts(img)
    % Apply screen capture artifacts (color banding + gamma mismatch)
    %
    % Input:
    %   img - uint8 RGB [H x W x 3]
    %
    % Output:
    %   noisyImg - uint8 RGB with screen artifacts

    % Parameters
    bitDepth = randi([6, 7]);  % Bit depth reduction
    gammaShift = 0.05 + rand() * 0.07;  % [0.05, 0.12]

    % Convert to double [0, 1]
    imgDouble = double(img) / 255.0;

    % 1. Color banding (bit depth reduction)
    levels = 2^bitDepth;
    quantized = round(imgDouble * (levels - 1)) / (levels - 1);

    % 2. Gamma mismatch
    gamma = 2.2 + (rand() * 2 - 1) * gammaShift;
    noisyDouble = quantized .^ (1.0 / gamma);

    % Clamp and convert
    noisyDouble = max(0, min(1, noisyDouble));
    noisyImg = uint8(noisyDouble * 255);
end

%% =========================================================================
%% OLD PHOTO ARTIFACTS
%% =========================================================================

function noisyImg = applyOldPhotoArtifacts(img)
    % Apply old photo degradation (blur + color shift + grain)
    %
    % Input:
    %   img - uint8 RGB [H x W x 3]
    %
    % Output:
    %   noisyImg - uint8 RGB with old photo artifacts

    % Parameters
    blurSigma = 0.3 + rand() * 0.5;  % [0.3, 0.8]
    colorShiftB = 5 + rand() * 5;     % [5, 10]
    colorShiftG = 8 + rand() * 4;     % [8, 12]
    colorShiftR = 10 + rand() * 5;    % [10, 15]
    grainSigma = 3.0;

    % Convert to double
    imgDouble = double(img);

    % 1. Slight blur
    blurred = imgaussfilt(imgDouble, blurSigma);

    % 2. Color shift (sepia/warm tone: increase red most, green moderate, blue least)
    shifted = blurred;
    shifted(:,:,1) = shifted(:,:,1) + colorShiftR;  % R (red channel, largest shift)
    shifted(:,:,2) = shifted(:,:,2) + colorShiftG;  % G (green channel, moderate shift)
    shifted(:,:,3) = shifted(:,:,3) + colorShiftB;  % B (blue channel, smallest shift)

    % 3. Film grain
    grain = grainSigma * randn(size(imgDouble));
    noisyDouble = shifted + grain;

    % Clamp and convert
    noisyImg = uint8(max(0, min(255, noisyDouble)));
end

%% =========================================================================
%% JPEG COMPRESSION ARTIFACTS
%% =========================================================================

function noisyImg = applyJpegArtifacts(img)
    % Apply JPEG compression artifacts via encode-decode cycle
    %
    % Input:
    %   img - uint8 RGB [H x W x 3]
    %
    % Output:
    %   noisyImg - uint8 RGB with JPEG artifacts

    % Quality parameter
    quality = 60 + randi(26);  % [60, 85]

    % Create temporary file for compression
    tempFile = [tempname() '.jpg'];

    try
        % Encode
        imwrite(img, tempFile, 'Quality', quality);

        % Decode
        noisyImg = imread(tempFile);

        % Clean up
        delete(tempFile);
    catch
        % If compression fails, return original
        noisyImg = img;
        if exist(tempFile, 'file')
            delete(tempFile);
        end
    end
end
