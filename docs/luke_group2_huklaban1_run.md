# Group 2 execution on huklaban1

User authorization: prepare/test Group 2, then proceed on a suitable local disk.
This is separate from the cancelled full-probe rigid experiment; its tracked
hold remains intact.

Arms, in serial order: `rescue_10_9_motion_off`, `rescue_9_9_motion_off`,
`rescue_9_8_motion_off`. All use motion off and shared curation/QC/export.
No `legacy_style`, motion-arm sort, or additional threshold is launched here.
The reference `rescue_12_9_motion_off` must come from Group 1 on the same
contracted time/depth selection before comparisons can be completed.

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

Real-data smoke completion is a runtime gate, not yet an established result.
Kilosort still has no within-sort checkpoints. A completed arm can be reused,
but interruption inside sorting loses that arm's in-memory work. Unexpected
termination must be investigated before a new launch; this controller refuses
to reuse an existing launch receipt automatically.

No scientific ranking is drawn from the smoke. Full comparisons await the
shared Group 1 reference, with matched-unit/common-time QC and the baseline
coverage denominator retained.
