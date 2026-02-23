function featPipe = feature_pipeline()
    %% FEATURE_PIPELINE Returns a struct of function handles for feature extraction pipeline
    %
    % This utility module consolidates feature group definitions, preset management,
    % feature export, train/test splitting, and column pruning for extract_features.
    %
    % Usage:
    %   featPipe = feature_pipeline();
    %
    %   % Registry operations
    %   reg = featPipe.registry.get();
    %   featureCfg = featPipe.registry.getConfiguration('robust');
    %   customCfg = featPipe.registry.completeSelection(partialStruct);
    %
    %   % Output operations
    %   featPipe.output.generateExcel(cfg, externalDeps);
    %   [trainTable, testTable] = featPipe.output.splitTable(featureTable, testFraction, groupColumn, seed);
    %
    % See also: extract_features

    %% Public API - Registry (feature definitions and presets)
    featPipe.registry.get = @getFeatureRegistry;
    featPipe.registry.getConfiguration = @getFeatureConfiguration;
    featPipe.registry.completeSelection = @completeFeatureSelection;
    featPipe.registry.create = @createFeatureRegistry;

    %% Public API - Output (export and splitting)
    featPipe.output.generateExcel = @generateConsolidatedExcelFile;
    featPipe.output.storePhoneData = @storePhoneFeatureData;
    featPipe.output.pruneColumns = @pruneFeatureColumns;
    featPipe.output.roundNumeric = @roundNumericColumns;
    featPipe.output.splitTable = @splitFeatureTable;
    featPipe.output.createConfig = @createOutputConfiguration;
    featPipe.output.writeTable = @writeTableWithFormat;
    featPipe.output.isExcelExtension = @isExcelFileExtension;
    featPipe.output.calculateValidationCount = @calculateValidationCount;
    featPipe.output.resolveSplitGroupColumn = @resolveSplitGroupColumn;

    %% Public API - Debug
    featPipe.debug.visualizeMask = @debugVisualizeMaskIfEnabled;
end

%% =========================================================================
%% REGISTRY - CREATION
%% =========================================================================

function registry = createFeatureRegistry()
    %% CENTRALIZED FEATURE GROUP DEFINITIONS
    % Feature group arrays are defined here to avoid duplication throughout the script.
    %
    % TO ADD/MODIFY FEATURE GROUPS: Edit the registry.groups array below
    % TO CHANGE PRESETS: Update the 'tier' assignment; preset membership is derived automatically.
    % GROUPING: 'groupType' is one of {'background','patch','normalized'}

    registry = struct();

    % Feature groups with properties:
    % - name: Feature group identifier
    % - outputCols: Column names this feature generates in output
    % - groupType: {'background','patch','normalized'}
    % - isBasic: true for basic color features, false for extended features
    % - tier: {'mustHave','bestScore','experimental','customOnly'} controls preset inclusion
    registry.groups = {
        % Background features (from original image / paper stats)
        struct('name', 'Background', 'outputCols', {{'paper_R','paper_G','paper_B','paper_L','paper_a','paper_b','paper_tempK'}}, 'groupType', 'background', 'isBasic', false, 'tier', 'customOnly', 'presets', []); ...

        % Patch features (computed on elliptical patch mask)
        struct('name', 'RGB', 'outputCols', {{'R', 'G', 'B'}}, 'groupType', 'patch', 'isBasic', true, 'tier', 'experimental', 'presets', []); ...
        struct('name', 'HSV', 'outputCols', {{'H', 'S', 'V'}}, 'groupType', 'patch', 'isBasic', true, 'tier', 'mustHave', 'presets', []); ...
        struct('name', 'Lab', 'outputCols', {{'L', 'a', 'b'}}, 'groupType', 'patch', 'isBasic', true, 'tier', 'mustHave', 'presets', []); ...
        struct('name', 'Skewness', 'outputCols', {{'R_skew', 'G_skew', 'B_skew', 'H_skew', 'S_skew', 'V_skew', 'L_skew', 'a_skew', 'b_skew'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'experimental', 'presets', []); ...
        struct('name', 'Kurtosis', 'outputCols', {{'R_kurto', 'G_kurto', 'B_kurto', 'H_kurto', 'S_kurto', 'V_kurto', 'L_kurto', 'a_kurto', 'b_kurto'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'experimental', 'presets', []); ...
        struct('name', 'GLCM', 'outputCols', {{'stripe_correlation', 'stripe_contrast', 'stripe_energy', 'stripe_homogeneity'}}, 'groupType', 'patch', 'isBasic', true, 'tier', 'bestScore', 'presets', []); ...
        struct('name', 'Entropy', 'outputCols', {{'entropyValue'}}, 'groupType', 'patch', 'isBasic', true, 'tier', 'bestScore', 'presets', []); ...
        struct('name', 'ColorRatios', 'outputCols', {{'RG_ratio', 'RB_ratio', 'GB_ratio'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'bestScore', 'presets', []); ...
        struct('name', 'Chromaticity', 'outputCols', {{'r_chromaticity', 'g_chromaticity', 'chroma_magnitude', 'dominant_chroma', 'chromaticity_std'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'mustHave', 'presets', []); ...
        struct('name', 'IlluminantInvariant', 'outputCols', {{'ab_magnitude', 'ab_angle', 'hue_circular_mean', 'saturation_mean'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'bestScore', 'presets', []); ...
        struct('name', 'ColorUniformity', 'outputCols', {{'RGB_CV_R', 'RGB_CV_G', 'RGB_CV_B', 'Saturation_uniformity', 'Value_uniformity', 'L_uniformity', 'chroma_uniformity'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'mustHave', 'presets', []); ...
        struct('name', 'RobustColorStats', 'outputCols', {{'L_median', 'L_iqr', 'a_median', 'a_iqr', 'b_median', 'b_iqr', 'S_median', 'S_iqr', 'V_median', 'V_iqr'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'bestScore', 'presets', []); ...
        struct('name', 'ColorGradients', 'outputCols', {{'L_gradient_mean', 'L_gradient_std', 'L_gradient_max', 'a_gradient_mean', 'a_gradient_std', 'a_gradient_max', 'b_gradient_mean', 'b_gradient_std', 'b_gradient_max', 'edge_density'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'bestScore', 'presets', []); ...
        struct('name', 'SpatialDistribution', 'outputCols', {{'L_spatial_std', 'a_spatial_std', 'b_spatial_std', 'radial_L_gradient', 'spatial_uniformity'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'bestScore', 'presets', []); ...
        struct('name', 'RadialProfile', 'outputCols', {{'radial_L_inner', 'radial_L_outer', 'radial_L_ratio', 'radial_chroma_slope', 'radial_saturation_slope'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'bestScore', 'presets', []); ...
        struct('name', 'ConcentrationMetrics', 'outputCols', {{'saturation_range', 'chroma_intensity', 'chroma_max', 'Lab_L_range', 'Lab_a_range', 'Lab_b_range'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'mustHave', 'presets', []); ...

        struct('name', 'LogarithmicColorTransforms', 'outputCols', {{'log_RG', 'log_GB', 'log_RB', 'log_RGB_magnitude', 'log_RGB_angle'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'experimental', 'presets', []); ...
        struct('name', 'FrequencyEnergy', 'outputCols', {{'fft_low_energy', 'fft_high_energy', 'fft_band_contrast'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'bestScore', 'presets', []); ...

        struct('name', 'AdvancedColorAnalysis', 'outputCols', {{'absorption_estimate', 'hue_shift_from_paper', 'chroma_difference'}}, 'groupType', 'patch', 'isBasic', false, 'tier', 'experimental', 'presets', []); ...

        struct('name', 'PaperNormalization', 'outputCols', {{'R_paper_ratio', 'G_paper_ratio', 'B_paper_ratio'}}, 'groupType', 'normalized', 'isBasic', false, 'tier', 'mustHave', 'presets', []); ...
        struct('name', 'EnhancedNormalization', 'outputCols', {{'L_corrected_mean', 'L_corrected_median', 'a_corrected_mean', 'a_corrected_median', 'b_corrected_mean', 'b_corrected_median', 'delta_E_from_paper', 'delta_E_median'}}, 'groupType', 'normalized', 'isBasic', false, 'tier', 'mustHave', 'presets', []); ...
        struct('name', 'PaperNormalizationExtras', 'outputCols', {{'R_norm','G_norm','B_norm','R_reflectance','G_reflectance','B_reflectance','R_chromatic_adapted','G_chromatic_adapted','B_chromatic_adapted','L_norm','a_norm','b_norm'}}, 'groupType', 'normalized', 'isBasic', false, 'tier', 'customOnly', 'presets', [])
    };

    registry.tierLegend = createFeatureTierLegend();

    for i = 1:numel(registry.groups)
        registry.groups{i}.presets = tierToPresetVector(registry.groups{i}.tier);
    end

    registry.presetNames = {'minimal', 'robust', 'full'};
    registry.presetDescriptions = struct( ...
        'minimal', 'Must-have baseline features for reproducible signal capture.', ...
        'robust', 'Adds validated feature groups that consistently improve cross-validation scores.', ...
        'full', 'Superset with exploratory groups that may or may not improve models.' ...
    );

    registry = addDerivedFeatureData(registry);

    minimalCount = sum(registry.presetMatrix(:, 1));
    robustCount = sum(registry.presetMatrix(:, 2));
    fullCount = sum(registry.presetMatrix(:, 3));

    registry.presetLabels = {
        sprintf('Minimal - Must-have (%d)', minimalCount), ...
        sprintf('Robust - Best validation (%d)', robustCount), ...
        sprintf('Full - Exploratory (%d)', fullCount), ...
        'Custom'
    };
end

function legend = createFeatureTierLegend()
    %% Create tier legend with descriptions
    legend = struct( ...
        'mustHave', struct('tag', 'Must-have', 'description', 'Baseline coverage included in Minimal, Robust, and Full presets.'), ...
        'bestScore', struct('tag', 'Best-score', 'description', 'Validated groups that lift model accuracy; included in Robust and Full.'), ...
        'experimental', struct('tag', 'Exploratory', 'description', 'Higher variance metrics reserved for the Full preset.'), ...
        'customOnly', struct('tag', 'Custom-only', 'description', 'Available when using the Custom preset; excluded from canned presets.') ...
    );
end

function vec = tierToPresetVector(tier)
    %% Convert tier string to preset inclusion vector [minimal, robust, full]
    if isstring(tier)
        tier = char(tier);
    end
    switch tier
        case 'mustHave'
            vec = [1, 1, 1];
        case 'bestScore'
            vec = [0, 1, 1];
        case 'experimental'
            vec = [0, 0, 1];
        case 'customOnly'
            vec = [0, 0, 0];
        otherwise
            error('feature_pipeline:unknownTier', 'Unknown feature tier: %s', string(tier));
    end
end

%% =========================================================================
%% REGISTRY - ACCESS
%% =========================================================================

function registry = getFeatureRegistry()
    %% Get centralized feature registry with persistent cache
    persistent cachedRegistry

    if isempty(cachedRegistry)
        cachedRegistry = createFeatureRegistry();
    end

    registry = cachedRegistry;
end

%% =========================================================================
%% REGISTRY - DERIVED DATA
%% =========================================================================

function registry = addDerivedFeatureData(registry)
    %% Add computed feature data for easy access

    % Extract feature group names
    registry.featureNames = cellfun(@(x) x.name, registry.groups, 'UniformOutput', false);

    % Create preset matrix
    numFeatures = length(registry.groups);
    numPresets = length(registry.presetNames);
    registry.presetMatrix = zeros(numFeatures, numPresets);

    for i = 1:numFeatures
        registry.presetMatrix(i, :) = registry.groups{i}.presets;
    end

    registry.tiers = cell(numFeatures, 1);
    registry.displayNames = cell(numFeatures, 1);

    for i = 1:numFeatures
        group = registry.groups{i};
        if isfield(group, 'tier')
            tierKey = char(string(group.tier));
        else
            tierKey = 'customOnly';
        end
        registry.tiers{i} = tierKey;

        if isfield(registry, 'tierLegend') && isfield(registry.tierLegend, tierKey)
            legendEntry = registry.tierLegend.(tierKey);
            if isfield(legendEntry, 'tag')
                tagText = char(string(legendEntry.tag));
            else
                tagText = tierKey;
            end
            registry.displayNames{i} = sprintf('%s [%s]', group.name, tagText);
        else
            registry.displayNames{i} = group.name;
        end
    end

    % Categorize feature groups
    registry.basicFeatureGroups = registry.featureNames(cellfun(@(x) x.isBasic, registry.groups));
    registry.advancedFeatureGroups = registry.featureNames(~cellfun(@(x) x.isBasic, registry.groups));

    % Grouping by Background/Patch/Normalized
    registry.backgroundFeatureCols = {};
    registry.patchFeatureCols = {};
    registry.normalizedFeatureCols = {};

    for i = 1:numFeatures
        group = registry.groups{i};
        switch lower(group.groupType)
            case 'background'
                registry.backgroundFeatureCols = [registry.backgroundFeatureCols, group.outputCols{:}];
            case 'patch'
                registry.patchFeatureCols = [registry.patchFeatureCols, group.outputCols{:}];
            case 'normalized'
                registry.normalizedFeatureCols = [registry.normalizedFeatureCols, group.outputCols{:}];
        end
    end

    % All expected feature groups for validation
    registry.allExpectedFeatureGroups = registry.featureNames;
end

%% =========================================================================
%% REGISTRY - CONFIGURATION
%% =========================================================================

function featureConfig = getFeatureConfiguration(presetName)
    %% DATA-DRIVEN FEATURE CONFIGURATION - Uses centralized feature registry

    % Get centralized feature registry
    registry = getFeatureRegistry();

    % Handle custom preset (should not reach here with custom, but just in case)
    if strcmp(presetName, 'custom')
        error('feature_pipeline:customPresetUnsupported', 'Custom preset requires a features struct to be provided by the caller.');
    end

    % Map preset name to column index
    presetMap = containers.Map(registry.presetNames, {1, 2, 3});

    if ~isKey(presetMap, presetName)
        error('feature_pipeline:invalidPreset', 'Invalid preset name: %s', presetName);
    end

    colIndex = presetMap(presetName);

    % Create configuration structure from registry
    featureConfig = struct();
    for i = 1:length(registry.featureNames)
        featureName = registry.featureNames{i};
        featureConfig.(featureName) = logical(registry.presetMatrix(i, colIndex));
    end
end

function out = completeFeatureSelection(in)
    %% Ensure custom feature selection covers all feature groups
    %
    % Input:
    %   in - Partial feature selection struct with some features enabled
    %
    % Output:
    %   out - Complete feature selection struct with all groups (missing = false)

    registry = getFeatureRegistry();
    out = struct();
    for i = 1:length(registry.featureNames)
        featureName = registry.featureNames{i};
        if isfield(in, featureName)
            out.(featureName) = logical(in.(featureName));
        else
            out.(featureName) = false;
        end
    end
end

%% =========================================================================
%% OUTPUT - MAIN OPERATIONS
%% =========================================================================

function storePhoneFeatureData(phoneName, featureData, cfg)
    %% Store phone feature data to temporary MAT file for later consolidation
    %
    % Inputs:
    %   phoneName - Name of the phone directory
    %   featureData - Struct array of feature data
    %   cfg - Configuration struct with projectRoot field

    tempDir = fullfile(cfg.projectRoot, 'temp_feature_data');
    if ~exist(tempDir, 'dir')
        mkdir(tempDir);
    end

    filename = fullfile(tempDir, [phoneName '_features.mat']);
    save(filename, 'featureData');

    fprintf('  >> Stored: %s (%d records)\n', phoneName, length(featureData));
end

function generateConsolidatedExcelFile(cfg, externalDeps)
    %% Consolidate feature data from all phones into Excel/CSV files
    %
    % Inputs:
    %   cfg - Configuration struct with paths, output settings, and performance config
    %   externalDeps - Struct with external dependencies:
    %       .listSubfolders - Function handle for listing subfolders
    %       .getFeatureRegistry - Function handle for feature registry
    %       .calculateBatchSize - Function handle for batch size calculation
    %       .checkMemoryPressure - Function handle for memory pressure check

    fprintf('\n=== Consolidating Features from %s phones ===\n', ...
        num2str(length(externalDeps.listSubfolders(cfg.originalImagesPath))));

    tempDir = fullfile(cfg.projectRoot, 'temp_feature_data');
    if ~exist(tempDir, 'dir')
        fprintf('!! No feature data found. Skipping Excel generation.\n');
        return;
    end

    matFiles = dir(fullfile(tempDir, '*_features.mat'));
    numFiles = length(matFiles);
    fprintf('Processing %d feature files -> Excel...\n', numFiles);

    % Adaptive batch sizing based on dataset size and memory
    if cfg.performance.adaptiveBatchSize && numFiles > 0
        batchSize = externalDeps.calculateBatchSize(numFiles, cfg);
    else
        batchSize = cfg.performance.baseBatchSize;
    end

    % Handle small datasets without batching overhead
    if numFiles <= 3
        fprintf('Loading %d files: ', numFiles);
        allFeatureData = cell(numFiles, 1);
        for i = 1:numFiles
            loadedData = load(fullfile(tempDir, matFiles(i).name), 'featureData');
            allFeatureData{i} = loadedData.featureData;
            fprintf('%d ', i);
        end
        fprintf('OK\n');
    else
        % Large dataset: batch processing with dynamic capacity management
        fprintf('Batch loading (%d files, batch=%d): ', numFiles, batchSize);

        % Start with smaller pre-allocation, grow exponentially if needed
        initialCapacity = min(20, ceil(numFiles / batchSize));
        allFeatureData = cell(initialCapacity, 1);
        batchCount = 0;

        for batchStart = 1:batchSize:numFiles
            batchEnd = min(batchStart + batchSize - 1, numFiles);
            currentBatchSize = batchEnd - batchStart + 1;

            % Pre-allocate batch data
            batchData = cell(currentBatchSize, 1);

            % Load current batch
            for i = batchStart:batchEnd
                loadedData = load(fullfile(tempDir, matFiles(i).name), 'featureData');
                batchData{i - batchStart + 1} = loadedData.featureData;
                if mod(i, 5) == 0 || i == batchEnd
                    fprintf('.');
                end
            end

            % Consolidate current batch
            consolidatedBatch = vertcat(batchData{:});
            batchCount = batchCount + 1;

            % Grow capacity if needed (exponential growth)
            if batchCount > length(allFeatureData)
                newCapacity = min(ceil(numFiles / batchSize), length(allFeatureData) * 2);
                allFeatureData{newCapacity, 1} = [];  % Grow array
            end

            allFeatureData{batchCount} = consolidatedBatch;

            % Clear intermediate variables immediately
            clear batchData loadedData consolidatedBatch;

            % Memory monitoring and adaptive adjustment
            if externalDeps.checkMemoryPressure(cfg.performance.memoryThreshold) && batchStart + batchSize < numFiles
                newBatchSize = max(5, round(batchSize * 0.7));
                if newBatchSize ~= batchSize
                    fprintf('\n  Memory pressure - reducing batch size %d->%d\n', batchSize, newBatchSize);
                    batchSize = newBatchSize;
                end
            end
        end

        % Trim to actual size
        allFeatureData = allFeatureData(1:batchCount);
        fprintf(' OK\n');
    end

    if ~isempty(allFeatureData)
        allFeatureData = vertcat(allFeatureData{:});
    else
        allFeatureData = [];
    end

    if isempty(allFeatureData)
        fprintf('No feature data to process.\n');
        return;
    end

    featureTable = struct2table(allFeatureData);

    trainTable = [];
    testTable = [];
    requestedGroupColumn = '';
    if isfield(cfg.output, 'splitGroupColumn') && ~isempty(cfg.output.splitGroupColumn)
        requestedGroupColumn = char(cfg.output.splitGroupColumn);
    end

    if cfg.output.trainTestSplit
        groupColumnResolved = resolveSplitGroupColumn(featureTable, requestedGroupColumn);
        randomSeed = [];
        if isfield(cfg.output, 'randomSeed')
            randomSeed = cfg.output.randomSeed;
        end
        [trainTable, testTable] = splitFeatureTable(featureTable, cfg.output.testSize, groupColumnResolved, randomSeed);

        % Display split information
        if strcmpi(groupColumnResolved, 'PhoneType')
            % Phone-level splitting with dynamic validation count
            uniquePhones = unique(featureTable.(groupColumnResolved));
            numPhones = length(uniquePhones);
            numValPhones = calculateValidationCount(numPhones);

            fprintf('\n=== Phone-Level Train/Validation Split ===\n');
            fprintf('Using dynamic validation count formula: ceil(N/5)\n');
            fprintf('Total phones: %d | Validation phones: %d (%.1f%%)\n', ...
                numPhones, numValPhones, 100 * numValPhones / numPhones);

            if numValPhones == 0
                warning('feature_pipeline:insufficientPhones', ...
                    'Insufficient phones for validation split (need >= 3). Using all data for training.');
            else
                % Display train phone names
                trainPhones = unique(trainTable.(groupColumnResolved));
                if iscell(trainPhones)
                    trainPhoneStr = strjoin(trainPhones, ', ');
                else
                    trainPhoneStr = strjoin(string(trainPhones), ', ');
                end
                fprintf('Train phones (%d): %s\n', length(trainPhones), trainPhoneStr);

                % Display validation phone names
                valPhones = unique(testTable.(groupColumnResolved));
                if iscell(valPhones)
                    valPhoneStr = strjoin(valPhones, ', ');
                else
                    valPhoneStr = strjoin(string(valPhones), ', ');
                end
                fprintf('Validation phones (%d): %s\n', length(valPhones), valPhoneStr);
            end
            fprintf('==========================================\n\n');
        end
    else
        groupColumnResolved = requestedGroupColumn;
    end

    featureTable = pruneFeatureColumns(featureTable, cfg, groupColumnResolved);
    if cfg.output.trainTestSplit
        trainTable = pruneFeatureColumns(trainTable, cfg, groupColumnResolved);
        testTable = pruneFeatureColumns(testTable, cfg, groupColumnResolved);
    end

    if cfg.output.roundDecimals > 0
        if ~isempty(featureTable) && width(featureTable) > 0
            featureTable = roundNumericColumns(featureTable, cfg.output.roundDecimals);
        end
        if cfg.output.trainTestSplit
            if ~isempty(trainTable) && width(trainTable) > 0
                trainTable = roundNumericColumns(trainTable, cfg.output.roundDecimals);
            end
            if ~isempty(testTable) && width(testTable) > 0
                testTable = roundNumericColumns(testTable, cfg.output.roundDecimals);
            end
        end
    end

    if ~exist(cfg.outputPath, 'dir')
        mkdir(cfg.outputPath);
        fprintf('Created output directory: %s\n', cfg.outputPath);
    end

    outputFilename = sprintf('%s_%s_%s_features%s', cfg.featurePreset, cfg.chemicalName, cfg.tNValue, cfg.output.excelExtension);
    outputPath = fullfile(cfg.outputPath, outputFilename);

    isExcelFormat = isExcelFileExtension(cfg.output.excelExtension);

    try
        writeTableWithFormat(featureTable, outputPath, isExcelFormat);
        fprintf('\n>> Excel created: %s\n', outputFilename);
        fprintf('   Records: %d | Features: %d\n', height(featureTable), width(featureTable));

        % Show feature groups present using registry
        colNames = featureTable.Properties.VariableNames;
        registry = externalDeps.getFeatureRegistry();
        bgPresent = sum(ismember(registry.backgroundFeatureCols, colNames));
        patchPresent = sum(ismember(registry.patchFeatureCols, colNames));
        normPresent = sum(ismember(registry.normalizedFeatureCols, colNames));
        fprintf('   Background: %d/%d | Patch: %d/%d | Normalized: %d/%d\n', ...
            bgPresent, length(registry.backgroundFeatureCols), ...
            patchPresent, length(registry.patchFeatureCols), ...
            normPresent, length(registry.normalizedFeatureCols));

        if isfield(cfg.output, 'trainTestSplit') && cfg.output.trainTestSplit
            trainFilename = ['train_' outputFilename];
            testFilename = ['test_' outputFilename];
            trainPath = fullfile(cfg.outputPath, trainFilename);
            testPath = fullfile(cfg.outputPath, testFilename);
            try
                writeTableWithFormat(trainTable, trainPath, isExcelFormat);
                writeTableWithFormat(testTable, testPath, isExcelFormat);
                fprintf('   Train/Test split saved -> %s (%d rows), %s (%d rows)\n', ...
                    trainFilename, height(trainTable), testFilename, height(testTable));
            catch splitME
                warning('feature_pipeline:trainTestSplit', ...
                    'Failed to create train/test split: %s', splitME.message);
            end
        end

    catch ME
        warning('feature_pipeline:excelCreation', 'Failed to create feature file: %s', ME.message);
        if isExcelFormat
            fprintf('Attempting to save as CSV instead...\n');
            csvPath = strrep(outputPath, cfg.output.excelExtension, '.csv');
            try
                writeTableWithFormat(featureTable, csvPath, false);
                fprintf('>> CSV file created: %s\n', csvPath);
            catch
                warning('feature_pipeline:saveFailed', 'Failed to save data in any format.');
            end
        else
            warning('feature_pipeline:saveFailed', 'Failed to save data in requested format and no fallback available.');
        end
    end

    try
        rmdir(tempDir, 's');
        fprintf('Cleaned up temporary files.\n');
    catch
        warning('feature_pipeline:cleanupFailed', 'Could not clean up temporary directory: %s', tempDir);
    end
end

%% =========================================================================
%% OUTPUT - TABLE TRANSFORMATIONS
%% =========================================================================

function tableOut = pruneFeatureColumns(tableIn, cfg, groupColumn)
    %% Remove metadata and internal columns from feature table for export
    %
    % Inputs:
    %   tableIn - Input feature table
    %   cfg - Configuration struct with output settings
    %   groupColumn - Column name used for train/test grouping (preserved if needed)
    %
    % Output:
    %   tableOut - Pruned table with metadata columns removed

    tableOut = tableIn;
    if isempty(tableOut)
        return;
    end

    allColumns = tableOut.Properties.VariableNames;
    columnsToRemove = {};

    metadataColumns = {'Concentration', 'PatchID', 'Replicate'};
    columnsToRemove = [columnsToRemove, metadataColumns(ismember(metadataColumns, allColumns))];

    if ~cfg.output.includeLabelInExcel && any(strcmpi(allColumns, 'Label'))
        columnsToRemove{end+1} = 'Label';
    end

    % Robust preset excludes spatial_uniformity from exports (computed internally only)
    if isfield(cfg, 'featurePreset') && strcmpi(cfg.featurePreset, 'robust')
        robustDropCols = {'spatial_uniformity'};
        columnsToRemove = [columnsToRemove, robustDropCols(ismember(robustDropCols, allColumns))];
    end

    if ~isempty(groupColumn)
        keepMask = strcmpi(columnsToRemove, groupColumn);
        if any(keepMask)
            % Allow removal when user explicitly disables Label export
            if strcmpi(groupColumn, 'Label') && isfield(cfg.output, 'includeLabelInExcel') && ~cfg.output.includeLabelInExcel
                keepMask(strcmpi(columnsToRemove, groupColumn)) = false;
            end
            columnsToRemove(keepMask) = [];
        end
    end

    columnsToRemove = unique(columnsToRemove);
    if ~isempty(columnsToRemove)
        tableOut(:, columnsToRemove) = [];
    end
end

function roundedTable = roundNumericColumns(inputTable, decimals)
    %% Round all numeric table variables to specified decimal places
    %
    % Inputs:
    %   inputTable - Input table with numeric and non-numeric columns
    %   decimals - Number of decimal places to round to
    %
    % Output:
    %   roundedTable - Table with numeric columns rounded

    % Use varfun to vectorize rounding across all numeric variables
    roundedTable = varfun(@(x) round(x, decimals), inputTable, ...
        'InputVariables', @isnumeric);

    % Restore original variable names
    roundedTable.Properties.VariableNames = inputTable.Properties.VariableNames(varfun(@isnumeric, inputTable, 'OutputFormat', 'uniform'));

    % Add back non-numeric columns
    nonNumericVars = inputTable.Properties.VariableNames(~varfun(@isnumeric, inputTable, 'OutputFormat', 'uniform'));
    for i = 1:length(nonNumericVars)
        roundedTable.(nonNumericVars{i}) = inputTable.(nonNumericVars{i});
    end

    % Restore original column order
    roundedTable = roundedTable(:, inputTable.Properties.VariableNames);
end

function [trainTable, testTable] = splitFeatureTable(featureTable, testFraction, groupColumn, randomSeed)
    %% Randomly split features into train/test partitions while keeping groups intact
    %
    % Inputs:
    %   featureTable - Table to split
    %   testFraction - Fraction of groups for test set (0-1)
    %   groupColumn  - Column name for grouping (keeps all rows from same group together)
    %   randomSeed   - Optional random seed for reproducibility (default: not set)
    %
    % Outputs:
    %   trainTable - Training partition
    %   testTable  - Test partition

    validateattributes(featureTable, {'table'}, {'nonempty'}, 'splitFeatureTable', 'featureTable');
    validateattributes(testFraction, {'double','single'}, {'scalar','>',0,'<',1}, ...
        'splitFeatureTable', 'testFraction');

    if nargin < 3 || isempty(groupColumn)
        error('feature_pipeline:splitMissingGroup', ...
            'Provide a grouping column to keep related rows together when splitting.');
    end

    groupColumn = resolveSplitGroupColumn(featureTable, groupColumn);
    groupData = featureTable.(groupColumn);
    if ischar(groupData)
        groupData = cellstr(groupData);
    end

    groupVector = groupData(:);
    rowCount = height(featureTable);
    if numel(groupVector) ~= rowCount
        error('feature_pipeline:splitGroupLength', ...
            'Grouping column ''%s'' must contain one entry per table row.', groupColumn);
    end

    if iscell(groupVector)
        emptyMask = cellfun(@isempty, groupVector);
        if any(emptyMask)
            error('feature_pipeline:splitMissingValues', ...
                'Grouping column ''%s'' contains missing entries. Populate it before splitting.', groupColumn);
        end
    elseif isstring(groupVector)
        if any(ismissing(groupVector) | strlength(groupVector) == 0)
            error('feature_pipeline:splitMissingValues', ...
                'Grouping column ''%s'' contains missing entries. Populate it before splitting.', groupColumn);
        end
    elseif iscategorical(groupVector)
        if any(isundefined(groupVector))
            error('feature_pipeline:splitMissingValues', ...
                'Grouping column ''%s'' contains missing entries. Populate it before splitting.', groupColumn);
        end
    elseif isnumeric(groupVector)
        if any(isnan(groupVector))
            error('feature_pipeline:splitMissingValues', ...
                'Grouping column ''%s'' contains missing entries. Populate it before splitting.', groupColumn);
        end
    end

    try
        [groupIds, ~] = findgroups(groupVector);
    catch
        groupVector = string(groupVector);
        [groupIds, ~] = findgroups(groupVector);
    end

    if any(groupIds == 0)
        error('feature_pipeline:splitMissingValues', ...
            'Grouping column ''%s'' contains missing entries. Populate it before splitting.', groupColumn);
    end

    numGroups = max(groupIds);

    % Determine test group count based on grouping column type
    if strcmpi(groupColumn, 'PhoneType')
        % Phone-level splitting: use dynamic validation count formula
        testGroupCount = calculateValidationCount(numGroups);

        if testGroupCount == 0
            % Zero validation case: return entire table as train, empty table as test
            trainTable = featureTable;
            testTable = featureTable([], :);
            return;
        end
    else
        % Non-phone splitting: use testFraction parameter (original behavior)
        if numGroups < 2
            error('feature_pipeline:splitInsufficientGroups', ...
                'Train/test split requires at least two unique groups in column ''%s''.', groupColumn);
        end

        testFraction = double(testFraction);
        desiredGroups = max(1, round(testFraction * numGroups));
        testGroupCount = min(desiredGroups, numGroups - 1);
    end

    % Set random seed if provided for reproducibility
    if nargin >= 4 && ~isempty(randomSeed)
        rng(randomSeed, 'twister');
    end

    selectedGroups = randperm(numGroups, testGroupCount);
    testMask = ismember(groupIds, selectedGroups);
    trainMask = ~testMask;

    if ~any(testMask) || ~any(trainMask)
        error('feature_pipeline:splitEmptyPartition', ...
            'Generated train/test partitions are empty. Adjust testSize or grouping.');
    end

    trainTable = featureTable(trainMask, :);
    testTable = featureTable(testMask, :);
end

%% =========================================================================
%% OUTPUT - CONFIGURATION
%% =========================================================================

function outputConfig = createOutputConfiguration(chemicalName, outputDecimals, outputExtension, includeLabelInExcel, trainTestSplit, testSize, splitGroupColumn, randomSeed)
    %% Create output configuration struct for feature export
    %
    % Inputs:
    %   chemicalName - Name of the chemical being analyzed
    %   outputDecimals - Number of decimal places for rounding
    %   outputExtension - File extension for output ('.xlsx', '.csv', etc.)
    %   includeLabelInExcel - Whether to include Label column in export
    %   trainTestSplit - Whether to create train/test split
    %   testSize - Fraction of data for test set (0-1)
    %   splitGroupColumn - Column name for grouping during split
    %   randomSeed - Random seed for reproducibility
    %
    % Output:
    %   outputConfig - Struct with output configuration

    validateattributes(chemicalName, {'char', 'string'}, {'nonempty'}, 'createOutputConfiguration', 'chemicalName');
    validateattributes(includeLabelInExcel, {'logical','numeric'}, {'scalar'}, 'createOutputConfiguration', 'includeLabelInExcel');
    validateattributes(splitGroupColumn, {'char', 'string'}, {'nonempty'}, 'createOutputConfiguration', 'splitGroupColumn');

    includeLabelFlag = logical(includeLabelInExcel);
    trainSplitFlag = logical(trainTestSplit);
    testSize = double(testSize);
    if trainSplitFlag
        validateattributes(testSize, {'double'}, {'scalar','>',0,'<',1}, ...
            'createOutputConfiguration', 'testSize');
    end

    outputConfig = struct('excelExtension', outputExtension, 'roundDecimals', outputDecimals, ...
                         'saveEmptyConcentrations', false, ...
                         'includeLabelInExcel', includeLabelFlag, ...
                         'chemicalName', char(chemicalName), ...
                         'trainTestSplit', trainSplitFlag, ...
                         'testSize', testSize, ...
                         'splitGroupColumn', char(splitGroupColumn), ...
                         'randomSeed', randomSeed);
end

%% =========================================================================
%% OUTPUT - UTILITIES
%% =========================================================================

function writeTableWithFormat(tableData, filePath, useExcelFormat)
    %% Write table to file in Excel or CSV format
    %
    % Inputs:
    %   tableData - Table to write
    %   filePath - Output file path
    %   useExcelFormat - True for Excel format with sheet, false for CSV

    if useExcelFormat
        writetable(tableData, filePath, 'Sheet', 'FeatureData');
    else
        writetable(tableData, filePath);
    end
end

function tf = isExcelFileExtension(ext)
    %% Check if file extension is an Excel format
    %
    % Input:
    %   ext - File extension string (e.g., '.xlsx')
    %
    % Output:
    %   tf - True if extension is Excel format

    excelExtensions = {'.xls', '.xlsx', '.xlsm', '.xlsb'};
    tf = any(strcmpi(ext, excelExtensions));
end

function numVal = calculateValidationCount(numPhones)
    %% Calculate validation phone count using dynamic formula
    %
    % Inputs:
    %   numPhones - Total number of phone directories (positive integer)
    %
    % Outputs:
    %   numVal - Number of phones to use for validation (non-negative integer)
    %
    % Formula: ceil(numPhones / 5) when numPhones >= 3, otherwise 0

    validateattributes(numPhones, {'numeric'}, {'scalar','integer','nonnegative'}, ...
        'calculateValidationCount', 'numPhones');

    numPhones = double(numPhones);

    if numPhones >= 3
        numVal = ceil(numPhones / 5);
    else
        numVal = 0;
    end
end

function resolvedColumn = resolveSplitGroupColumn(featureTable, requestedColumn)
    %% Resolve the grouping column for train/test splitting with flexible matching
    %
    % Inputs:
    %   featureTable - Feature table to search for column
    %   requestedColumn - Requested column name
    %
    % Output:
    %   resolvedColumn - Resolved column name from table

    varNames = featureTable.Properties.VariableNames;
    requestedColumn = char(requestedColumn);

    matchMask = strcmp(varNames, requestedColumn);
    if ~any(matchMask)
        matchMask = strcmpi(varNames, requestedColumn);
    end

    if ~any(matchMask)
        normalizedName = matlab.lang.makeValidName(requestedColumn);
        if ~strcmp(normalizedName, requestedColumn)
            matchMask = strcmp(varNames, normalizedName);
            if ~any(matchMask)
                matchMask = strcmpi(varNames, normalizedName);
            end
        end
    end

    if ~any(matchMask)
        if isempty(varNames)
            availableColumns = '(none)';
        else
            availableColumns = strjoin(varNames, ', ');
        end
        error('feature_pipeline:splitMissingGroup', ...
            'Grouping column ''%s'' not found in feature table. Available columns: %s. Ensure metadata retention includes it.', ...
            requestedColumn, availableColumns);
    end

    resolvedColumn = varNames{find(matchMask, 1)};
end

%% =========================================================================
%% DEBUG
%% =========================================================================

function debugVisualizeMaskIfEnabled(originalImage, mask, imageName, cfg)
    %% Save an overlay of paper mask on original image for a random subset when enabled
    %
    % Inputs:
    %   originalImage - Original RGB image
    %   mask - Binary mask to visualize
    %   imageName - Name of the image for filename
    %   cfg - Configuration struct with debug settings

    try
        if ~isfield(cfg, 'debug') || ~isfield(cfg.debug, 'visualizeMasks') || ~cfg.debug.visualizeMasks
            return;
        end
        if ~isfield(cfg.debug, 'sampleProb') || rand() > cfg.debug.sampleProb
            return;
        end
        I = im2double(originalImage);
        M = logical(mask);
        if ~any(M(:))
            return;
        end
        M3 = repmat(M, [1, 1, 3]);
        green = cat(3, zeros(size(I,1), size(I,2)), ones(size(I,1), size(I,2)), zeros(size(I,1), size(I,2)));
        overlay = I .* 0.6 + green .* 0.4 .* M3;
        outImg = im2uint8(overlay);

        if isfield(cfg.debug, 'saveToDisk') && cfg.debug.saveToDisk
            outDir = fullfile(cfg.projectRoot, 'temp_masks');
            if ~exist(outDir, 'dir'), mkdir(outDir); end
            [~, base, ~] = fileparts(imageName);
            outPath = fullfile(outDir, [base '_mask.png']);
            imwrite(outImg, outPath);
        else
            figure('Visible', 'off'); imshow(outImg); drawnow; close(gcf);
        end
    catch
        % Swallow any debug visualization errors silently
    end
end
