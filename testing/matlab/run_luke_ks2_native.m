function run_luke_ks2_native(output_dir, kilosort_dir)
% Run pinned upstream KS2 while preserving native waveform-state diagnostics.
%
% SpikeInterface prepares only the accepted int16 binary, chanMap.mat, ops.mat,
% and npy-matlab helpers. Every scientific sorting stage below is the upstream
% Kilosort2 v2.0.2 implementation.

set(groot, 'defaultFigureVisible', 'off');
addpath(genpath(kilosort_dir));
addpath(genpath(output_dir));

load(fullfile(output_dir, 'chanMap.mat'));
load(fullfile(output_dir, 'ops.mat'));

assert(ops.ntbuff == 64, 'Executed KS2 ntbuff is not the pinned value.');
assert(any(ops.NT == [65472 65600]), ...
    'Executed KS2 NT is not an audited installation-gate value.');
assert(getOr(ops, 'reorder', 0) == 1, 'Native batch reordering must be enabled.');
assert(getOr(ops, 'skip_kilosort_preprocessing', 0) == 0, ...
    'KS2 native preprocessing must not be skipped.');

diary(fullfile(output_dir, 'ks2_native_matlab.log'));
cleanup_diary = onCleanup(@() diary('off'));
fprintf('Luke KS2 native runner started %s\n', datestr(now, 30));
fprintf('MATLAB %s\n', version);
fprintf('KS2 source %s\n', kilosort_dir);
fprintf('NT=%d ntbuff=%d stride=%d\n', ops.NT, ops.ntbuff, ...
    ops.NT - ops.ntbuff);
gpu = gpuDevice;
fprintf('GPU=%s compute=%s available_memory=%g\n', gpu.Name, ...
    gpu.ComputeCapability, gpu.AvailableMemory);

rez = preprocessDataSub(ops);
rez = clusterSingleBatches(rez);
[~, audit_iorig, audit_xs] = sortBatches2(rez.ccb);
audit_xs = gather(audit_xs);
audit_iorig = gather(audit_iorig);
assert(isequal(audit_iorig(:), rez.iorig(:)), ...
    'Recomputed native batch order differs from clusterSingleBatches output.');
save(fullfile(output_dir, 'rez_pretracking.mat'), 'rez', 'audit_xs', ...
    'audit_iorig', '-v7.3');
saveas(gcf, fullfile(output_dir, 'native_batch_reordering.png'));

rez = learnAndSolve8b(rez);
save(fullfile(output_dir, 'rez_tracking.mat'), 'rez', 'audit_xs', ...
    'audit_iorig', '-v7.3');

rez = find_merges(rez, 1);
rez = splitAllClusters(rez, 1);
rez = splitAllClusters(rez, 0);
rez = set_cutoff(rez);
fprintf('Found %d good units\n', sum(rez.good > 0));

rezToPhy(rez, output_dir);
save(fullfile(output_dir, 'rez_final.mat'), 'rez', 'audit_xs', ...
    'audit_iorig', '-v7.3');
fprintf('Luke KS2 native runner completed %s\n', datestr(now, 30));
clear cleanup_diary;
diary('off');
end
