# Overnight motion diagnostics

Continue from the retained compensated broadband 3σ amplitude-sum baseline. No new global conditioning sweep or production sort is authorized by this diagnostic continuation. The full-sort hold remains intact.

The two bounded work streams are:

1. Qualify shallow references using quiet data, an unchanged-gate comparator, held-out observations and injection controls. Test one spatial weighting hypothesis and at most three additional nearby candidates. Only qualified references proceed to the disputed transition. Failures and sparse intervals remain explicit gaps.
2. Apply the frozen baseline to separate early, middle and late 100-second intervals. Reuse cached early/late waveform observations descriptively; these candidates have not passed identity qualification. The middle interval has no biological reference. Motion rasters alone cannot establish transfer validity.

All main findings get PNG/PDF figures and numerical source artifacts. The previous-round overview and detailed figure links are indexed in `docs/luke_motion_overnight_figure_index_20260908.md`.

## Persistence and restart behavior

Launch through `testing/launch_luke_motion_overnight.py`, which creates an independent systemd user service with a fresh launch directory, exact command, stdout/stderr, managed receipt and final service result (including termination). At most two heavy services; each is limited to four CPUs, with one numerical thread per worker. The coordinator launches serially and verifies actual process state after launcher exit.

The lifecycle dummy `luke-overnight-lifecycle-probe-v1` was observed running after its launcher exited, then completed successfully with both managed receipt and independent service exit result. Evidence is in `testing/outputs/luke_overnight_lifecycle_probe_v1/`.

Completed stages are reusable only after settings/input/output hash validation. Interrupted attempts remain preserved and restart from their stage boundary. A partial localization chunk or DREDGE solve is not resumable internally. Restarts require investigation and a new service/receipt name; there is no blind automatic restart. No full recording sort or giant uncheckpointed scan is queued.

An RTX A5000 is available outside the sandbox. These bounded comparisons retain the established single-thread CPU DREDGE backend; any GPU substitution must first demonstrate numerical equivalence.

## Current status

Transfer job `luke-motion-transfer-overnight-v1.service` launched and independently verified active after launcher exit (MainPID1818448). Its command, logs and receipt are in `testing/outputs/luke_motion_transfer_overnight_v1_job/`. Frozen epochs:940–1040,6000–6100,9480–9580s. Five prelaunch tests passed, including abrupt process termination/restart, changed-input and corrupt-checkpoint rejection, and missing-reference figure generation. Reference job `luke-shallow-reference-overnight-v1.service` also launched and independently verified active after launcher exit (MainPID1822153). Its exact command, logs and separate core/report receipts are in `testing/outputs/luke_shallow_reference_overnight_v1_job/`. The same service runs the cached-data comparison renderer only after successful core completion. Both services have independent final-status capture. First transfer checkpoint(s940) committed successfully before handoff; subsequent work remains running.


Reference stage documentation: [reference experiment](luke_shallow_reference_overnight_v1.md). Transfer stage documentation: [transfer experiment](luke_motion_transfer_overnight_20260908.md). [Figure index](luke_motion_overnight_figure_index_20260908.md).

Reference/report test evidence was copied out of temporary directories to `testing/outputs/luke_motion_overnight_v1/checkpoint_validation/`. Restart after an interruption only after inspecting the failure: use a fresh service name/receipt and unchanged scientific source/settings to reuse validated completed stages. CPU jobs are bounded; runtime is not a promise to fill the night with additional experiments. These jobs finish the prespecified diagnostic batch without autonomous parameter tuning or production sorting.


## Completed results, checked after overnight execution

Both services are inactive with MainPID0 and successful persisted exit statuses. Transfer completed in56.3minutes; shallow qualification and report completed in4.7minutes. No job remains running. All15transfer detection/localization checkpoints committed, covering early940–1040, middle6000–6100 and late9480–9580s (1,124,035;1,524,705;1,084,798peaks). All strict search-bound checks and final source-hash checks passed. This is computational completion, not biological validation.

No shallow candidate passed the full frozen gate, so transition tracking was correctly skipped. Unit80 improved substantially under contextual weighting: quiet recall47.7%→80.5%, unique injected shift recovery55.3%→92.7%. Worst-rival false acceptance remained3.33%(one acceptance in30trials for several rivals), above the1%gate. Thus sensitivity improved but specificity is still unresolved. Units98,125,126 also failed qualification; no new reference was promoted.

Initial transfer review shows substantial pairwise inconsistency and disagreements with provisional cached footprints. For example, late candidate near2900µm at9532.5s changes−1.23µm in its cached centroid versus−36.78µm event-matched DREDGE. These candidates have unqualified identity and measurement sensitivity; this disagreement does not establish physical error. Sampled supported triangle-residual95thpercentiles are large across depths in allthreeepochs. The frozen baseline has not demonstrated trustworthy recording-wide transfer. Next diagnosis should inspect specific competing registrations and qualify local biological evidence in those intervals, without immediately expanding to a full scan or sort.
