# Queued full-session rigid correction and Kilosort on huklaban1

**Armed 2026-09-09 at approximately 05:11 UTC.** Actual service inspection
after launcher exit confirmed active/running, MainPID=2202578, with the controller
at `queued_waiting_for_medicine`. The producer was independently observed
active/running, MainPID=2195155. This is launch-time evidence, not a completed
sort; inspect the live service and receipts below for current status.

User request: queue rigid motion correction and Kilosort after the active shared
MEDiCINe estimate finishes; all work must be independent of the chat session.
This is explicit authorization for the new development run. It supersedes the
earlier plan to stop and obtain another launch decision after field publication.
It does not promote the estimate to scientific ground truth or production use.

## Execution chain

`luke-full-session-medicine-20260909-v1.service` completes extraction, one
full-session fit and publication. The new independent
`luke-medicine-rigid-sort-20260909-v1.service` waits for actual producer exit,
successful managed-process and systemd exit receipts, and verified artifact
hashes. It then executes:

1. Verify source recording checksum, field integrity/time coverage, locked
   runtime, baseline settings, geometry and local disk capacity.
2. Consume the existing seed-referenced equal-depth rigid projection. Do not
   estimate motion again or enable native KS4 motion estimation.
3. Apply external rigid correction once to accepted RESCUE voltage, then
   materialize locally with a content receipt.
4. Run Kilosort 4.0.27 with frozen RESCUE 12/9 settings and effective internal
   correction off. Check saved `nblocks=0` and `dshift=None`.
5. Run existing curation, QC and MATLAB export and retain terminal receipts.

Estimator failure/cancellation, missing success receipts, changed hashes,
invalid coordinates, insufficient supported channels, disk shortage or invalid
voltage stops the queue with evidence retained. There is no automatic retry.
A field labelled `requires_review` remains labelled that way; this is a
user-authorized descriptive comparison, not an automatic scientific pass.

## Frozen application policy

Use SpikeInterface 0.102.1 `InterpolateMotionRecording`, kriging sigma=20 µm,
exponent p=1 (explicitly pinning the prior Option A default), float32 interpolation,
gain=1, and 0.25 s field bins. Round to nearest int16 after checking all returned
samples for nonfinite values and overflow; do not clip or wrap silently.

The saved accepted recording has a nonzero acquisition-time origin. Add its
measured `sample_index_to_time(0)` to the recording-relative field time bins
before passing them to SpikeInterface. This prevents a 3,057.677 s clock error.
All 314,204,894 time samples are retained. Nearest terminal-field values may
cover no more than one second at either recording edge, with durations recorded;
irregular internal field time grids are refused. No new temporal smoothing,
sign fitting, gain fitting or estimator tuning is introduced.

Apply on the full 384-contact source geometry, then keep a fixed interior
channel set supported across the entire rigid AND nonrigid fields, with a
60 µm kernel-support margin and no off-probe extrapolation. This may remove
edge channels; the exact set is resolved from the completed field before any
voltage is materialized. Fewer than 64 supported contacts refuses the run.
The reused baseline was sorted on the full probe, so any channel restriction
is recorded as a comparison limitation. The common domain is published for
huklaban5 to use, rather than allowing each arm to choose a different crop.

## Persistence, ownership and cancellation

The queue controller and every child belong to an independent systemd user
service with `KillMode=control-group`, user lingering enabled, persistent
stdout/stderr, managed exit receipt and `ExecStopPost` service-result receipt.
The source tree and production lockfile are copied into a hashed job bundle;
later repository edits or pulls cannot change queued execution.

Job directory:
`/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1_job/`.
Output directory:
`/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1/`.

To inspect actual state:

```bash
systemctl --user show luke-medicine-rigid-sort-20260909-v1.service \
  -p ActiveState -p SubState -p MainPID -p MemoryCurrent -p Result
```

`status.json` records the pipeline stage; `dependency_state.json` records each
actual producer-service check. These are useful progress artifacts, not alone
proof of liveness. Final success requires `summary.json`, `receipt.json` and
`service_result.json` with successful exits.

To cancel the queue or the active rigid run and all its child processes:

```bash
systemctl --user stop luke-medicine-rigid-sort-20260909-v1.service
```

This does not cancel the separate estimation service. A `HOLD.json` in either
new job/output directory prevents advancement while waiting or between stages;
use systemctl stop for immediate cancellation during a stage. Historical
`configs/luke_full_session_rigid.HOLD.json` and the cancelled native-rigid run
remain intact and are not reused.

Kilosort has no validated within-sort checkpoint. An interrupted sort must
restart from its beginning after investigation. Completed sealed estimation
and recording materialization are stage reuse, not within-sort resume.

## Checks and launch

- Seven new checks pass: dependency success/failure/cancellation, time support,
  common channel domain, signed/fractional/large translation with a nonzero
  clock origin, and safe int16 materialization.
- Nine existing external-warp and managed-job tests pass.
- A 3,000-sample real-voltage smoke using the existing five-minute field passed
  time mapping, materialization/reload equality and accepted-recording hashing.
  It used 336 supported contacts; that is not the final full-session channel set.
- The exact queue launcher/controller is exercised with a cheap independent
  dummy and checked after launcher disconnection before the real queue is armed.
- Locked runtime, CUDA, installed KS4 compatibility repair and exact baseline
  setting equality passed preflight.

```bash
environments/rescue-production/.venv/bin/python -m testing.launch_luke_medicine_rigid_queue \
  --unit luke-medicine-rigid-sort-20260909-v1 \
  --job-dir /media/huklab/Data/luke_medicine_rigid_sort_20260909_v1_job \
  --output /media/huklab/Data/luke_medicine_rigid_sort_20260909_v1 \
  --proof /home/huklab/Documents/RyanSorting/SpikeSortingTools/testing/outputs/luke_medicine_rigid_queue_dummy_v1_job
```

Consult the job receipts for execution status; this document is not a claim
that estimation, correction or sorting has completed. No new baseline sort
or nonrigid sort is launched by this controller.
