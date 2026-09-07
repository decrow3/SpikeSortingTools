# Group 2 execution on huklaban1

User authorization: prepare/test Group 2, then proceed on a suitable local disk.
This is separate from the cancelled full-probe rigid experiment; its tracked
hold remains intact.

Arms, in serial order: `rescue_10_9_motion_off`, `rescue_9_9_motion_off`,
`rescue_9_8_motion_off`. All use motion off and shared curation/QC/export.
No `legacy_style`, motion-arm sort, or additional threshold is launched here.
The reference `rescue_12_9_motion_off` must come from Group 1 on the same
contracted time/depth selection before comparisons can be completed.

## Shared brief and cross-host comparison

The governing design is [Implementation Brief: Standard-Pipeline Spike-Sorting
Development Ladder](Implementation%20Brief_%20Standard-Pipeline%20Spike-Sorting%20Development%20Ladder%20%281%29.md).
The user confirmed that a separate agent on huklaban5 owns Group 1: the 12/9
motion-off reference, native rigid, and native nonrigid arms. This host owns
Group 2 only; it does not rerun the shared reference by default.

Before combining results, verify both hosts' receipts against the shared
contract: source content, sample interval and clock origin, physical channel
selection/order and geometry, sorter/runtime versions, effective settings,
and curation/QC definitions. Preserve original manifests and content digests
when transferring artifacts. Host-specific paths must be reconciled explicitly;
neither identical filenames nor different path-dependent hashes establish
whether the underlying recordings are identical. The remote setup has not
been independently verified here.

The combined long-strip comparison is the first scientific selection stage.
Use the shared reference and common-time 1,000-spike amplitude fits, report
baseline-cohort coverage and identity/contamination guardrails, and advance at
most two supported finalists to full-probe testing. Completion of these sorts
alone does not establish a better pipeline.

The new explicit authorization covers these development threshold tests;
earlier frozen Stage 2 results and production thresholds are unchanged.

## Data and execution

- Local output: `/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/group2`.
  This is the mounted local SATA data disk, with about 990 GB free at preflight.
  The job requires at least 200 GiB free and refuses an absent mount.
- Parent contract: `testing/configs/luke0804_hindsight_ladder_v1.json`.
- Host execution plan: `testing/configs/luke0804_group2_huklaban1_v1.json`.
- Smoke: first 120 seconds, all three Group 2 arms, engineering only.
- Long run: all 10,473.553728 seconds, processing depths 1400–2380 µm,
  scoring interior 1600–2180 µm, as specified in the parent contract.
  Full duration does not mean full probe width in this ladder.
- Orchestration: `testing/luke_group2_job.py`, launched under the independent
  systemd user service `luke-group2-threshold-v1`.
- Preparation, smoke sorting/QC/export, long-strip preparation, long sorting/QC/
  export, and finalization execute sequentially. Any stage failure stops the
  sequence. Each command has a separate receipt and the service retains output
  logs and final process status. There are no automatic retries.

## Checks completed before launch

The existing focused suite passed 76 tests. The targeted Group 2, runner, smoke
and managed-job suite passed 25 tests, including rejection of the wrong group or
contract and proof that a failed smoke prevents the long run from launching.

Two harmless systemd-managed jobs were launched, then their disposable Python
launcher was deliberately killed with SIGKILL (launcher exit 137). Both jobs
subsequently printed `survived launcher death`; the success job retained exit 0
and the intentional failure retained exit 7 in both JSON receipts and systemd.
Evidence is in `testing/outputs/group2_preparation/`. This directly tests
launcher death, without reverting or shutting down the user's actual chat.

## Remaining limits

The real-data smoke runtime gate passed for all three Group 2 arms.
Kilosort still has no within-sort checkpoints. A completed arm can be reused,
but interruption inside sorting loses that arm's in-memory work. Unexpected
termination must be investigated before a new launch; this controller refuses
to reuse an existing launch receipt automatically.

No scientific ranking is drawn from the smoke. Full comparisons await the
shared Group 1 reference, with matched-unit/common-time QC and the baseline
coverage denominator retained.

## Execution milestones

- The independent service started at 2026-09-06 20:30:50 PDT as
  `luke-group2-threshold-v1.service` (main PID 1176876).
- Smoke preparation completed with return code 0 at 21:06:30 PDT. The three
  smoke arms, including shared curation/QC/export, completed with return code 0
  at 21:12:03 PDT. Their post-curation summaries were: 10/9, 74,998 spikes in
  67 units; 9/9, 82,588 spikes in 66 units; and 9/8, 112,812 spikes in 66 units.
  The group receipt is complete and these results remain engineering-only.
- Long-strip preparation began at 21:12:03 PDT. At 21:24 PDT the required
  source-integrity hash was 88.1/241.3 GB through the accepted recording and
  still advancing. The service was active, the host had 176 GiB available RAM,
  and the selected data disk had about 988 GiB free. No relaunch was performed.
- The only warning-like smoke log entry was a non-fatal Matplotlib inability to
  inspect the Noto Color Emoji font. No traceback, runtime failure, CUDA-memory
  error, OOM, or killed-process report was present.
