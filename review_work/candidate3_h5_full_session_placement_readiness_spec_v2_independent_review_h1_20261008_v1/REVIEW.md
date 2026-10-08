# Focused independent rereview: candidate-3 H5 placement/readiness v2 repair

## Verdict

`GO_REPAIRED_CHECKLIST_SPECIFICATION_ONLY_H5_NOT_READY_NO_LAUNCH`

Reviewed repair packet:

- `/mnt/NPX/Luke/DARTsort_motion_experiments/candidate3_h5_full_session_placement_readiness_spec_h1_20261008_v2`
- manifest SHA-256 `86f23f9a04b18b1a3ef60aeb13a4cf7bd85867fc57d205292d17108194ed7d37`
- complete SHA-256 `175dae90c577621eb607af9eaf025c82e505a6b4e2e2b0f1e97504713174c70e`

V1 remains immutable blocked evidence. The effective checklist is v1 plus the
four exact replacements in v2 `CORRECTIONS.json`. V2 repairs both blocking
specification defects without claiming that either implementation or H5 itself
has passed the repaired gates.

## Findings

1. **Seal and chronology pass.** All five manifested v2 members match recorded
   sizes and hashes; `COMPLETE.json` binds the exact manifest. Local birth times
   place payload and validation before the manifest and `COMPLETE.json` after
   the manifest. Shared birth times independently show `COMPLETE.json` copied
   last.

2. **Correction pointers are exact and old values match v1.** The only effective
   pointer replacements are:

   - `/storage_envelope/monitor_stop_conditions/2`
   - `/storage_envelope/monitor_stop_conditions/3`
   - `/fresh_h5_checks/5` (`H5-06-storage`)
   - `/fresh_h5_checks/10` (`H5-11-checkpoint`)

   Both old stop-condition strings and both old gate IDs match immutable v1.
   Applying the overlay leaves 12 unique H5 checks, preserves the original
   `SPECIFICATION_COMPLETE_H5_NOT_CURRENTLY_PROVEN_READY` status and preserves
   the original release rule requiring all checks against one later immutable
   run contract before one coordinator-dispatched launch.

3. **H5-06 now closes the machine-accounting defect.** The repaired gate binds
   `available_bytes=f_bavail*f_frsize`, linked allocated bytes
   (`st_blocks*512`), linked apparent bytes (`st_size`), device/inode
   deduplication, no filesystem crossing, symlink/mount-escape rejection, and
   live managed-cgroup FD enumeration for open-unlinked regular files. Both
   total allocated and total apparent bytes must remain at or below
   482,618,717,184, while available bytes remain at or above 120,000,000,000.
   The external monitor starts before science, samples pre-attempt, every 60
   seconds, at stage transitions, before promotion and in the terminal
   finalizer, and writes atomic timestamped receipts.

   The failure rule stops the whole managed cgroup, preserves last/terminal
   accounting and the attempt, records the exact breach, and prohibits deletion
   or automatic retry. The cheap future fixture must exercise linked, sparse,
   open-unlinked, threshold and escape branches. Any additional full-recording
   copy/materialization is now an immediate breach even if sparse or below the
   cap. This makes the conservative cap enforceable in specification; it still
   is not an expected-usage estimate or a guarantee of completion.

4. **H5-11 now closes the abrupt managed-kill evidence gap.** The repaired gate
   requires both a manager timeout and external control-group kill during a
   nonzero peel append through the exact installed production interpreter,
   runner, wrapper, service and finalizer. Each must retain the original partial
   HDF5 and manager receipts, publish no stage completion or promotion, and make
   the runner refuse reopen/stock resume. A distinctly named fresh attempt must
   revalidate byte-identical input/source/effective-config/chunk-schedule
   bindings, preserve the failed original, recompute the whole stage, and match
   an independent clean baseline in event/lineage arrays and committed HDF5.
   Receipts bind cgroup membership, kill cause/time, attempt identities, hashes
   and zero automatic restarts. Caught-exception evidence is explicitly barred
   as a substitute.

5. **No unrelated effective semantics changed.** V2 is an overlay rather than a
   rewritten readiness document. It binds exact v1 and the blocking review,
   says all other v1 content is inherited, keeps the capacity values and H5
   preference unchanged, and adds no alternate launch path, parameter change,
   materialization allowance or readiness shortcut.

6. **No current-readiness or launch claim exists.** V2 status says H5 is still
   not currently ready; the report says no H5 check was executed; provenance
   records no H5 contact, payload action, fixture, service or sort; and the
   effective release rule explicitly says the repair packet does not authorize
   or start a run. The repaired monitor and kill fixtures remain future hard
   gates whose implementations and results require their own evidence and
   review.

No blocking defect remains in the two repaired specification areas. This GO
accepts the checklist text only. It does not release a run contract, authorize
fixture execution, establish current H5 readiness, or validate any future
monitor/runner implementation.

## Implementation checks

- Done: v2 packet seal and COMPLETE-last chronology -> exact and valid.
- Done: v1/review bindings and four correction pointers -> exact; all old values
  match immutable v1.
- Done: effective overlay -> 12 unique gates, unchanged capacity numbers,
  unchanged not-ready status and unchanged no-launch release rule.
- Done: H5-06 semantics -> linked/apparent/open-unlinked accounting, dedup,
  statvfs reserve, cadence, external monitor, fixtures and breaches specified.
- Done: H5-11 semantics -> installed-path timeout and external-cgroup-kill
  preservation/no-promotion/no-reopen plus fresh baseline-equal restart specified.
- Not done: monitor or runner implementation, either fixture, any H5 check,
  payload read/hash, transfer/materialization, service/launch/sort or RF/holdout
  -> excluded and still required later where applicable.
- Can establish: the two v1 checklist defects are repaired without unrelated
  semantic change and the effective specification is ready to govern later
  implementation/prestart evidence.
- Cannot establish: implementation correctness, current H5 readiness, wrapper
  equality, resource sufficiency, launch authorization, full-session completion
  or scientific performance.
