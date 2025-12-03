function colorAnalysis = color_analysis()
    %% COLOR_ANALYSIS Returns a struct of function handles for color science utilities
    %
    % This utility module provides functions for color temperature estimation,
    % chromatic adaptation, and other color science computations used in
    % colorimetric analysis.
    %
    % Usage:
    %   color = color_analysis();
    %   cct = color.estimateColorTemperature(paperRGB);
    %   adaptation = color.calculateChromaticAdaptation(paperRGB);
    %
    % Color Temperature Constants:
    %   McCamy's approximation coefficients and standard illuminant values
    %   are provided as defaults. Custom parameters can be passed to functions.
    %
    % See also: mask_utils, image_io

    %% Public API
    colorAnalysis.estimateColorTemperature = @estimateColorTemperature;
    colorAnalysis.calculateChromaticAdaptation = @calculateChromaticAdaptation;

    % Constants
    colorAnalysis.getDefaultColorTempParams = @getDefaultColorTempParams;
end

%% =========================================================================
%% CONSTANTS
%% =========================================================================

function params = getDefaultColorTempParams()
    % Get default parameters for color temperature estimation
    %
    % OUTPUTS:
    %   params - Struct with McCamy's approximation coefficients and thresholds
    %
    % These are standard values for McCamy's CCT approximation formula.

    % McCamy's approximation reference point (daylight locus)
    params.mccamyXRef = 0.3320;
    params.mccamyYRef = 0.1858;

    % McCamy's polynomial coefficients
    params.mccamyCoeff0 = 5520.33;
    params.mccamyCoeff1 = 449.0;
    params.mccamyCoeff2 = 99.11;
    params.mccamyCoeff3 = -449.0;

    % Valid CCT range
    params.cctMinKelvin = 2500;
    params.cctMaxKelvin = 10000;

    % R/B ratio thresholds for fallback estimation
    params.rbRatioTungstenThreshold = 1.2;
    params.rbRatioMixedThreshold = 0.9;

    % Fixed CCT values for fallback categories
    params.cctTungstenK = 3200;
    params.cctMixedK = 4500;
    params.cctDaylightK = 6500;
    params.fallbackK = 5500;
end

%% =========================================================================
%% COLOR TEMPERATURE
%% =========================================================================

function colorTemp = estimateColorTemperature(paperRGB, params)
    % Estimate color temperature from paper white point using McCamy's CCT approximation
    %
    % INPUTS:
    %   paperRGB - RGB values of paper white point [R, G, B] (0-255 scale)
    %   params   - (Optional) Parameter struct from getDefaultColorTempParams()
    %
    % OUTPUTS:
    %   colorTemp - Estimated color temperature in Kelvin
    %
    % McCamy's approximation provides accurate CCT estimates for illuminants
    % near the Planckian locus. For unusual lighting conditions, a fallback
    % R/B ratio method is used.
    %
    % References:
    %   McCamy, C. S. (1992). "Correlated color temperature as an explicit
    %   function of chromaticity coordinates"

    if nargin < 2
        params = getDefaultColorTempParams();
    end

    fallbackK = params.fallbackK;

    try
        % Normalize to avoid absolute brightness effects
        rgb_sum = sum(paperRGB);
        if rgb_sum > 0
            % Chromaticity coordinates
            r_chrom = paperRGB(1) / rgb_sum;
            g_chrom = paperRGB(2) / rgb_sum;

            % McCamy's approximation for CCT from chromaticity
            x = r_chrom;
            y = g_chrom;

            if y > 0
                n = (x - params.mccamyXRef) / (params.mccamyYRef - y);
                cct = params.mccamyCoeff3 * n^3 + ...
                      params.mccamyCoeff2 * n^2 + ...
                      params.mccamyCoeff1 * n + ...
                      params.mccamyCoeff0;

                % Clamp to reasonable range
                colorTemp = min(max(cct, params.cctMinKelvin), params.cctMaxKelvin);
            else
                % Fallback to simple R/B ratio method
                if paperRGB(3) > 0
                    rb_ratio = paperRGB(1) / paperRGB(3);
                    if rb_ratio > params.rbRatioTungstenThreshold
                        colorTemp = params.cctTungstenK; % Tungsten
                    elseif rb_ratio > params.rbRatioMixedThreshold
                        colorTemp = params.cctMixedK; % Mixed
                    else
                        colorTemp = params.cctDaylightK; % Daylight
                    end
                else
                    colorTemp = fallbackK;
                end
            end
        else
            colorTemp = fallbackK;
        end

    catch
        colorTemp = fallbackK;
    end
end

%% =========================================================================
%% CHROMATIC ADAPTATION
%% =========================================================================

function chromaticAdaptation = calculateChromaticAdaptation(paperRGB)
    % Calculate chromatic adaptation factors using von Kries-style adaptation
    %
    % INPUTS:
    %   paperRGB - RGB values of paper white point [R, G, B] (0-255 scale)
    %
    % OUTPUTS:
    %   chromaticAdaptation - [R, G, B] scaling factors for color correction
    %
    % The von Kries adaptation model normalizes colors relative to a reference
    % white point. This is useful for correcting color casts due to different
    % illumination conditions.
    %
    % Factors are clamped to [0.5, 2.0] to prevent extreme corrections.

    try
        % Normalize paper to its maximum component (preserve color ratios)
        maxComponent = max(paperRGB);
        if maxComponent > 10 % Avoid division by very small numbers
            % von Kries-style adaptation: normalize by each component
            % This preserves the relative color ratios while standardizing brightness
            chromaticAdaptation = maxComponent ./ max(paperRGB, 1);

            % Clamp to physically reasonable range
            chromaticAdaptation = min(max(chromaticAdaptation, 0.5), 2.0);
        else
            chromaticAdaptation = [1.0, 1.0, 1.0];
        end

    catch
        chromaticAdaptation = [1.0, 1.0, 1.0];
    end
end
