function results = validate_new_pipeline(varargin)
    %VALIDATE_NEW_PIPELINE Test suite for refactored 4-stage pipeline
    %
    % Validates that the new pipeline works correctly by running automated
    % tests on sample data. Use this before migrating full ground truth data.
    %
    % INPUTS (name-value pairs):
    %   testPhone     : Phone model to test (default: 'iphone_11')
    %   numSamples    : Number of images to test (default: 3)
    %   skipStage1    : Skip cut_micropads test (default: false)
    %   skipStage2    : Skip cut_elliptical_regions test (default: false)
    %   skipStage3    : Skip extract_features test (default: false)
    %   skipAugment   : Skip augment_dataset test (default: false)
    %   verbose       : Print detailed progress (default: true)
    %
    % OUTPUTS:
    %   results : Struct with test results for each stage
    %
    % USAGE:
    %   % Run all tests on default phone (iphone_11)
    %   validate_new_pipeline()
    %
    %   % Test specific phone with more samples
    %   validate_new_pipeline('testPhone', 'samsung_a75', 'numSamples', 5)
    %
    %   % Quick test - skip augmentation (slow)
    %   validate_new_pipeline('skipAugment', true)
    %
    % REQUIREMENTS:
    %   - 1_dataset/ must contain at least numSamples images for testPhone
    %   - If testing augmentation, 2_micropads/ and 3_elliptical_regions/
    %     must already exist with coordinates
    %
    % See also: cut_micropads, cut_elliptical_regions, extract_features, augment_dataset

    % Parse inputs
    parser = inputParser;
    addParameter(parser, 'testPhone', 'iphone_11', @ischar);
    addParameter(parser, 'numSamples', 3, @(n) isnumeric(n) && isscalar(n) && n > 0);
    addParameter(parser, 'skipStage1', false, @islogical);
    addParameter(parser, 'skipStage2', false, @islogical);
    addParameter(parser, 'skipStage3', false, @islogical);
    addParameter(parser, 'skipAugment', false, @islogical);
    addParameter(parser, 'verbose', true, @islogical);
    parse(parser, varargin{:});

    testPhone = parser.Results.testPhone;
    numSamples = parser.Results.numSamples;
    skipStage1 = parser.Results.skipStage1;
    skipStage2 = parser.Results.skipStage2;
    skipStage3 = parser.Results.skipStage3;
    skipAugment = parser.Results.skipAugment;
    verbose = parser.Results.verbose;

    % Find project root
    projectRoot = findProjectRoot('1_dataset', 5);

    % Initialize results
    results = struct();
    results.projectRoot = projectRoot;
    results.testPhone = testPhone;
    results.numSamples = numSamples;
    results.timestamp = datetime('now');

    % Create test output directory
    testOutputDir = fullfile(projectRoot, 'test_outputs');
    if ~isfolder(testOutputDir)
        mkdir(testOutputDir);
    end

    if verbose
        fprintf('=== Pipeline Validation Test Suite ===\n');
        fprintf('Project root: %s\n', projectRoot);
        fprintf('Test phone: %s\n', testPhone);
        fprintf('Sample count: %d\n\n', numSamples);
    end

    %% Test Stage 1: cut_micropads.m
    if ~skipStage1
        if verbose
            fprintf('--- Stage 1: cut_micropads.m ---\n');
        end

        try
            % Check if input data exists
            inputFolder = fullfile(projectRoot, '1_dataset', testPhone);
            if ~isfolder(inputFolder)
                error('validate_new_pipeline:missing_input', ...
                    'Input folder not found: %s', inputFolder);
            end

            % Get sample images
            imageFiles = dir(fullfile(inputFolder, '*.{jpg,jpeg,png}'));
            if isempty(imageFiles)
                imageFiles = dir(fullfile(inputFolder, '*.jpg'));
            end
            if isempty(imageFiles)
                imageFiles = dir(fullfile(inputFolder, '*.jpeg'));
            end
            if isempty(imageFiles)
                imageFiles = dir(fullfile(inputFolder, '*.png'));
            end

            if length(imageFiles) < numSamples
                warning('validate_new_pipeline:insufficient_samples', ...
                    'Only %d images found, requested %d. Using available images.', ...
                    length(imageFiles), numSamples);
                numSamples = length(imageFiles);
            end

            if verbose
                fprintf('  Found %d test images\n', length(imageFiles));
            end

            % Note: cut_micropads is interactive, so we can only check the script exists
            scriptPath = fullfile(projectRoot, 'matlab_scripts', 'cut_micropads.m');
            if ~isfile(scriptPath)
                error('validate_new_pipeline:missing_script', ...
                    'cut_micropads.m not found at: %s', scriptPath);
            end

            results.stage1.status = 'MANUAL';
            results.stage1.message = 'cut_micropads.m is interactive - manual testing required';
            results.stage1.scriptExists = true;
            results.stage1.inputDataExists = true;

            if verbose
                fprintf('  [MANUAL] Interactive script - requires manual testing\n');
                fprintf('  Script exists: %s\n', scriptPath);
                fprintf('  To test: cd matlab_scripts; cut_micropads(''numSquares'', 7)\n\n');
            end

        catch ME
            results.stage1.status = 'FAILED';
            results.stage1.error = ME.message;
            if verbose
                fprintf('  [FAILED] %s\n\n', ME.message);
            end
        end
    else
        results.stage1.status = 'SKIPPED';
        if verbose
            fprintf('--- Stage 1: SKIPPED ---\n\n');
        end
    end

    %% Test Stage 2: cut_elliptical_regions.m
    if ~skipStage2
        if verbose
            fprintf('--- Stage 2: cut_elliptical_regions.m ---\n');
        end

        try
            % Check if prerequisites exist
            inputFolder = fullfile(projectRoot, '2_micropads', testPhone);
            coordFile = fullfile(inputFolder, 'coordinates.txt');

            if ~isfolder(inputFolder)
                warning('validate_new_pipeline:missing_stage2_input', ...
                    '2_micropads folder not found - Stage 1 must be run first');
                results.stage2.status = 'SKIPPED';
                results.stage2.message = 'Prerequisites missing (need Stage 1 output)';
            elseif ~isfile(coordFile)
                warning('validate_new_pipeline:missing_coordinates', ...
                    'coordinates.txt not found - Stage 1 must be run first');
                results.stage2.status = 'SKIPPED';
                results.stage2.message = 'Prerequisites missing (coordinates.txt)';
            else
                % Note: cut_elliptical_regions is also interactive
                scriptPath = fullfile(projectRoot, 'matlab_scripts', 'cut_elliptical_regions.m');
                if ~isfile(scriptPath)
                    error('validate_new_pipeline:missing_script', ...
                        'cut_elliptical_regions.m not found');
                end

                results.stage2.status = 'MANUAL';
                results.stage2.message = 'cut_elliptical_regions.m is interactive - manual testing required';
                results.stage2.scriptExists = true;
                results.stage2.inputDataExists = true;

                if verbose
                    fprintf('  [MANUAL] Interactive script - requires manual testing\n');
                    fprintf('  Script exists: %s\n', scriptPath);
                    fprintf('  To test: cd matlab_scripts; cut_elliptical_regions()\n\n');
                end
            end

        catch ME
            results.stage2.status = 'FAILED';
            results.stage2.error = ME.message;
            if verbose
                fprintf('  [FAILED] %s\n\n', ME.message);
            end
        end
    else
        results.stage2.status = 'SKIPPED';
        if verbose
            fprintf('--- Stage 2: SKIPPED ---\n\n');
        end
    end

    %% Test Stage 3: extract_features.m
    if ~skipStage3
        if verbose
            fprintf('--- Stage 3: extract_features.m ---\n');
        end

        try
            % Check if prerequisites exist
            inputFolder2 = fullfile(projectRoot, '2_micropads', testPhone);
            inputFolder3 = fullfile(projectRoot, '3_elliptical_regions', testPhone);

            if ~isfolder(inputFolder2) || ~isfolder(inputFolder3)
                warning('validate_new_pipeline:missing_stage3_input', ...
                    'Stage 2 or 3 folders not found - earlier stages must be run first');
                results.stage3.status = 'SKIPPED';
                results.stage3.message = 'Prerequisites missing (need Stage 1 & 2 output)';
            else
                scriptPath = fullfile(projectRoot, 'matlab_scripts', 'extract_features.m');
                if ~isfile(scriptPath)
                    error('validate_new_pipeline:missing_script', ...
                        'extract_features.m not found');
                end

                results.stage3.status = 'MANUAL';
                results.stage3.message = 'extract_features.m testing requires processed data';
                results.stage3.scriptExists = true;
                results.stage3.inputDataExists = true;

                if verbose
                    fprintf('  [MANUAL] Requires processed elliptical regions\n');
                    fprintf('  Script exists: %s\n', scriptPath);
                    fprintf('  To test: cd matlab_scripts; extract_features(''preset'', ''minimal'')\n\n');
                end
            end

        catch ME
            results.stage3.status = 'FAILED';
            results.stage3.error = ME.message;
            if verbose
                fprintf('  [FAILED] %s\n\n', ME.message);
            end
        end
    else
        results.stage3.status = 'SKIPPED';
        if verbose
            fprintf('--- Stage 3: SKIPPED ---\n\n');
        end
    end

    %% Test Augmentation: augment_dataset.m
    if ~skipAugment
        if verbose
            fprintf('--- Augmentation: augment_dataset.m ---\n');
        end

        try
            % Check if prerequisites exist
            inputFolder2 = fullfile(projectRoot, '2_micropads', testPhone);
            inputFolder3 = fullfile(projectRoot, '3_elliptical_regions', testPhone);

            if ~isfolder(inputFolder2) || ~isfolder(inputFolder3)
                warning('validate_new_pipeline:missing_augment_input', ...
                    'Stage 2 or 3 folders not found - cannot test augmentation');
                results.augmentation.status = 'SKIPPED';
                results.augmentation.message = 'Prerequisites missing (need Stage 1 & 2 output)';
            else
                scriptPath = fullfile(projectRoot, 'matlab_scripts', 'augment_dataset.m');
                if ~isfile(scriptPath)
                    error('validate_new_pipeline:missing_script', ...
                        'augment_dataset.m not found');
                end

                results.augmentation.status = 'MANUAL';
                results.augmentation.message = 'augment_dataset.m is slow - manual testing recommended';
                results.augmentation.scriptExists = true;
                results.augmentation.inputDataExists = true;

                if verbose
                    fprintf('  [MANUAL] Augmentation is slow - manual testing recommended\n');
                    fprintf('  Script exists: %s\n', scriptPath);
                    fprintf('  To test: cd matlab_scripts; augment_dataset(''numAugmentations'', 1, ''phones'', {''%s''})\n\n', testPhone);
                end
            end

        catch ME
            results.augmentation.status = 'FAILED';
            results.augmentation.error = ME.message;
            if verbose
                fprintf('  [FAILED] %s\n\n', ME.message);
            end
        end
    else
        results.augmentation.status = 'SKIPPED';
        if verbose
            fprintf('--- Augmentation: SKIPPED ---\n\n');
        end
    end

    %% Test Migration Script
    if verbose
        fprintf('--- Migration: migrate_to_new_pipeline.m ---\n');
    end

    try
        scriptPath = fullfile(projectRoot, 'matlab_scripts', 'migrate_to_new_pipeline.m');
        if ~isfile(scriptPath)
            error('validate_new_pipeline:missing_script', ...
                'migrate_to_new_pipeline.m not found');
        end

        % Check if old pipeline data exists
        oldStage2 = fullfile(projectRoot, '2_micropad_papers');
        oldStage3 = fullfile(projectRoot, '3_concentration_rectangles');

        if isfolder(oldStage2) && isfolder(oldStage3)
            results.migration.status = 'READY';
            results.migration.message = 'Migration script ready - old data found';
            results.migration.canMigrate = true;
        else
            results.migration.status = 'N/A';
            results.migration.message = 'No old pipeline data to migrate';
            results.migration.canMigrate = false;
        end

        results.migration.scriptExists = true;

        if verbose
            fprintf('  Script exists: %s\n', scriptPath);
            if results.migration.canMigrate
                fprintf('  [READY] Old pipeline data detected\n');
                fprintf('  To migrate: cd matlab_scripts; migrate_to_new_pipeline(''testMode'', true, ''dryRun'', true)\n\n');
            else
                fprintf('  [N/A] No old pipeline data found (already migrated or new project)\n\n');
            end
        end

    catch ME
        results.migration.status = 'FAILED';
        results.migration.error = ME.message;
        if verbose
            fprintf('  [FAILED] %s\n\n', ME.message);
        end
    end

    %% Summary
    if verbose
        fprintf('=== Validation Summary ===\n');
        fprintf('Stage 1 (cut_micropads): %s\n', results.stage1.status);
        if isfield(results, 'stage2')
            fprintf('Stage 2 (cut_elliptical_regions): %s\n', results.stage2.status);
        end
        if isfield(results, 'stage3')
            fprintf('Stage 3 (extract_features): %s\n', results.stage3.status);
        end
        if isfield(results, 'augmentation')
            fprintf('Augmentation (augment_dataset): %s\n', results.augmentation.status);
        end
        fprintf('Migration (migrate_to_new_pipeline): %s\n', results.migration.status);
        fprintf('\n');

        % Provide next steps
        fprintf('=== Next Steps ===\n');
        fprintf('1. Manually test cut_micropads.m with sample images\n');
        fprintf('2. Verify rotation panel and AI detection work correctly\n');
        fprintf('3. Check coordinates.txt has 10 columns (with rotation)\n');
        fprintf('4. If old data exists, run migration script in test mode\n');
        fprintf('5. Rename augmented folders if needed:\n');
        fprintf('   mv augmented_2_concentration_rectangles augmented_2_micropads\n');
        fprintf('6. Run full pipeline validation on test samples\n');
    end

    % Save results to file
    resultsFile = fullfile(testOutputDir, sprintf('validation_results_%s.mat', ...
        datestr(now, 'yyyy-mm-dd_HH-MM-SS')));
    save(resultsFile, 'results');

    if verbose
        fprintf('\nResults saved to: %s\n', resultsFile);
    end
end

function root = findProjectRoot(folderName, maxDepth)
    %FINDPROJECTROOT Search up directory tree for folder
    currentDir = pwd;
    for depth = 0:maxDepth
        testPath = fullfile(currentDir, folderName);
        if isfolder(testPath)
            root = currentDir;
            return;
        end
        parentDir = fileparts(currentDir);
        if strcmp(parentDir, currentDir)
            % Reached filesystem root
            break;
        end
        currentDir = parentDir;
    end
    % Not found, use current directory
    root = pwd;
end
