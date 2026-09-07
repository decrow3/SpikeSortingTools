# Luke full-session interruption and user-requested hold

The restarted run is cancelled. The systemd user service is inactive, MainPID=0,
and neither the sorter nor the comparison process remains. Logs and partial
outputs are preserved. `HOLD.json` prevents the sorting entry point from being
restarted until the hold is deliberately resolved with subsequent authorization.

## Original interruption: strongest evidence is launcher shutdown

The original `run.log` ends at 2026-09-06 18:05:00.215508519 America/Los_Angeles
(2026-09-07 01:05:00 UTC). At the same second the local Codex log database records:

- Shutdown of this thread's Codex instance, thread
  `01a0731b-1ac8-7920-be37-8971476a9d05`.
- A `thread/revert` request followed by restoration of that thread.
- Shutdown of another active Codex instance.

The sorter and comparison waiter were launched as tool-owned processes; both
were subsequently absent. The sorting log contains no Python traceback. No
OOM kill, GPU Xid, or segfault was found in the checked kernel-log interval
(17:30–18:30 local). This strongly supports termination during Codex session
shutdown, rather than a demonstrated Kilosort crash. The exact process signal
and exit status were not retained, so this is a strongly supported diagnosis,
not a recovered kill receipt. The logs do not establish what UI action caused
the revert. Do not attribute intent to the user.

Relevant launcher log rows are copied to
`testing/outputs/luke_full_session_rigid_v1/termination_evidence.json`.

## Why there were no checkpoints

The installed SpikeInterface 0.102.1 wrapper calls, in sequence:
`compute_preprocessing`, `compute_drift_correction`, `detect_spikes`,
`cluster_spikes`, then `save_sorting`. It does not persist stage checkpoints.
`save_extra_vars=True` affects the final save, not intermediate persistence.

Inside installed Kilosort 4.0.27, `detect_spikes` itself contains the initial
template extraction, initial clustering, and learned-template extraction. These
states remain in memory. The first run completed drift estimation (5,191 s),
initial extraction (5,075 s; 9,775,172 detections), and initial clustering
(198 s; 1,370 initial clusters). It stopped at batch 254/5,237 of learned-template
extraction. These are intermediate counts, not a completed sorting result.

The repository's `run_sorter_config` only caches a **completed sort**. Existing
partial directories are archived on retry, and sorting starts from the beginning.
It is restartable between completed stages, not resumable inside sorting.
The surviving partial directory contains only wrapper configuration, recording
metadata and probe geometry; there are no saved spike/features/cluster states
from which this run can resume.

Source locations inspected:

- Installed `spikeinterface/sorters/external/kilosort4.py`, lines 318–361.
- Installed `kilosort/run_kilosort.py`, `detect_spikes`, lines 614–699.
- `testing/ladder_sorter.py`, `run_sorter_config`, lines 204–272.

## Responsibility and prerequisites for another launch

The initial launch should have used a persistent job service, and checkpoint
capability should have been checked before committing hours of compute. The
subsequent restart should have waited for this investigation. Earlier status
reports inferred liveness from a stale log; future checks must inspect the
actual service/process as well.

The persistent service used for the now-cancelled restart addresses ownership
of the job, but persistence through chat shutdown has not yet been demonstrated
with a controlled test. It does not add Kilosort checkpoints.

Before proposing another full run:

1. Demonstrate independent job survival and retained exit status using a cheap
   dummy process, including an intentional launcher disconnection.
2. Decide explicitly whether to accept whole-sort restart risk or implement
   stage checkpoints. A checkpoint implementation must preserve ops, arrays,
   random states, recording identity and environment, and reconstruct the binary
   reader. To preserve the completed initial clustering, it needs an integration
   point **inside** `detect_spikes`, not merely around that function.
3. Validate interrupted/resumed output against uninterrupted output on a small
   fixture before spending full-session compute. Do not improvise partial saves
   in the installed production package.

No new sorting or checkpoint implementation is authorized by this incident
investigation. No scientific settings or thresholds were changed.
