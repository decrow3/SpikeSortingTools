# huklaban5 handoff: improved nonrigid motion, full Luke0804 imec0

Read [the governing two-machine plan](luke_full_session_improved_motion_two_machine_plan.md)
and local AGENTS.md first. The user requests two whole-duration corrected sorts:
rigid on huklaban1, nonrigid on huklaban5. Existing full-session no-correction
and legacy sorts are reused. This handoff is preparation for the new experiment;
it is not a restart of the cancelled native-rigid v1 job.

**User clarification:** all full-session motion estimation runs on huklaban1
only. Do not extract another peak population or fit MEDiCINe on huklaban5.
Consume the shared hash-verified field package after it is published and reviewed.
The producing job is documented in [the estimation run record](luke_full_session_medicine_run_20260909.md).

The rigid arm on huklaban1 is now requested as an automatic persistent queue;
see [the queue record](luke_medicine_rigid_queue_20260909.md). After the field
finishes, it publishes `rigid_application_contract_v1.json` in the shared
directory with the common channel set and operator policy. Reuse those for
the nonrigid arm; do not independently select channels or change interpolation.
This queue does not launch anything on huklaban5.

## Work to do after pulling

1. Check the working tree and preserve unrelated changes. Record hostname,
   repository commit, actual GPU/RAM/free local disk, current GPU jobs, and the
   locked RESCUE environment versions. Do not assume historical status files
   describe live processes.
2. Verify access to the accepted source:
   `/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/`.
   Use the recording and sort hashes in the governing plan. Check for an existing
   verified local copy before considering a 241 GB copy. Plan corrected voltage
   and sorter scratch under `/media/huklaban5/Data/`, not the shared server.
3. Inspect the existing external-field correction and managed-job code. Prepare
   a nonrigid runner that consumes a digest-identified external field, applies
   correction once to the accepted RESCUE voltage, and disables KS4 internal
   motion estimation/correction. Preserve thresholds 12/9 and the frozen
   baseline settings except for the declared correction intervention.
4. Agree the same operator/version, time mapping, output channels, edge policy,
   dtype and source bundle with huklaban1 before full sorting. Do not independently
   tune/refit the motion estimate. Do not use the old historical field or the
   five-minute MEDiCINe field as a full-session substitute.
5. Run cheap fixture checks for sign/coordinates, zero displacement, exact
   translations, fractional displacement, large movement and nonrigid geometry.
   Verify independent systemd launch survival and final exit receipt on a cheap
   dummy. Reuse existing validated machinery when its proof applies.
6. Publish a preflight acknowledgement and readiness evidence to the shared
   directory below. Work can proceed on runner/fixture preparation while the
   full-session field is being prepared on huklaban1. Full sorting waits for
   that field, the final common application contract, and resource readiness.

## Shared exchange contract

Directory:
`/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/`.

Create it additively when ready to publish. Use atomic status replacement and
host-owned filenames; do not edit huklaban1's status or older exchange folders.

Write `huklaban5_status.json` with:

- owner/actual hostname, UTC verification time, task ID when available;
- phase (`preflight`, `ready_waiting_for_field`, `sorting`, `qc`, `complete`,
  or `blocked`) and a concrete reason for any blocker;
- source identity, code/environment hashes, local output root and disk budget;
- job-manager proof, service name, current process state, log/receipt paths;
- field and common-contract digests once received;
- final exit status only when actually observed.

huklaban1 publishes `field_manifest.json`, field arrays and rigid reduction,
`run_contract.json`, the required code/source bundle, and its own status. These
names describe pending deliverables; none is claimed to exist yet. Treat an
incomplete package as unavailable; validate all listed file hashes before use.
Stage bulk shared reads in coordination with huklaban1, then run sorts on local
storage in parallel. Share compact receipts/QC/figures, not corrected binaries.

## Full-run behavior

Use a new run-specific service and output directory. Keep old HOLD markers and
failed-run evidence intact. Sorting, curation, QC and export must survive the
chat and terminal closing. No automatic restart. Installed KS4 has no validated
within-sort checkpoint; interruption requires restarting that entire sort arm
after investigation, even when completed field/input stages can be reused.

Produce full-session sorting, curation, QC and exports plus complete provenance
and terminal receipts. Report edge/support restrictions and all unmeasurable
endpoint intervals. The requested endpoint is the complete 10,473.55 s session;
a smoke sort or five-minute result does not complete this assignment.
