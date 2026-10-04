# Independent H1 review: covariance/W smoke relaunch candidate

Review date: 2026-10-01.

## Verdict

**GO for byte-identical installation, full installed-path validation, and
exactly one normal-tool start, conditional on every prerequisite in
`INSTALL_PREFLIGHT_ONE_START_GO.json`.** This review does not install or start a
service, open a recording, read voltage, or access RF/holdout material.

The fresh candidate packet is complete and hash-valid:

- packet manifest: `99c6250b527bef1ef5e81c9b9f48e49e67e8d7236a088499cf4592fe9c5479c2`;
- packet COMPLETE: `17d171d972c10bbebd3540f1380d2a3df0804e8ee3d70d63acfeb501d19fbfcc`.

The candidate source differs from the previously reviewed enabled source by
one exact policy line: run-root suffix `v1` becomes `v2`. The config changes
only the matching `run_root`, `launch_evidence_dir`, `results_dir`, and candidate
source hash. The service changes only its config path/hash and fresh-root
condition. Launcher, support helper and candidate tests are byte-identical to
the reviewed dependencies. Therefore the recording, crop
`[208498882,226498783)`, 29,999.835983263598 Hz clock, geometry, motion field,
preprocessing, 12-batch schedule, covariance gate, W construction and
pre-training stop are unchanged.

The frozen source reserves the v2 run root atomically with `exist_ok=False`
before native execution, creates separate evidence/results children, journals
each sampled covariance batch, preserves validation/failure evidence, and
raises the intentional stop only after durable passing C validation and a
finite hash-bound W. The service is a retained oneshot with `Restart=no` and a
fresh-root condition. Kilosort is explicitly non-resumable within this smoke.

The unchanged candidate suite passed 18/18 independently on H1 using the
packet's frozen source and test bytes in an isolated temporary package. The
helper source hash `9c981047...42560f`, candidate-test hash
`786c77bd...ec3667d`, launcher hash `59e4c776...0d60c`, and frozen Kilosort
module hashes match the previously reviewed 34/34 chain. The 16 helper tests
were not rerun on H1 because part of that suite consumes H5-only saved ops and
recording-derived artifacts; their unchanged bytes and the historical 34/34
receipt are verified, not misreported as a fresh H1 execution.

The prior v1 launch decision was consumed by a pre-execution normal-tool
denial, preserved under
`en_minimum_training_support_mask_covariance_smoke_result_20261002_v2_h5`
with manifest `51e5203054bb6b03d2c7671a712e199daf6703cf5ac8095d022036e0e1d907c6`.
It records an absent unit/root/reservation and zero recording opens/read bytes.
The user's subsequent direct standing authorization covers this comparable
bounded covariance/W smoke and resolves the approval-provenance issue for a
fresh normal-tool request.

Runtime, CPU, RAM and GPU-utilization caps remain unspecified, as in the
reviewed contract; the documented resource condition is CUDA on the reviewed
RTX A5000. The hard data/evidence bounds remain 554,084,352 padded input bytes
and 33,554,432 retained bytes. No automatic or manual retry is authorized.

## Implementation checks

- Done: candidate manifest and COMPLETE -> exact expected hashes; all ten
  manifest members verify.
- Done: source/config/service diff -> only the required v1-to-v2 fresh-root,
  binding and config-hash changes; all scientific inputs and stop logic are
  unchanged.
- Done: source boundary and failure preservation -> atomic fresh-root claim,
  per-batch journal, durable C/W validation, pre-training stop and failure
  receipt paths inspected.
- Done: unchanged candidate dependencies -> independent 18/18 candidate tests
  pass; helper/test/launcher hashes match the prior 34/34-tested chain.
- Done: prior denial linkage -> exact result manifest and zero-execution state.
- Not done: H5 installation, current collision/CUDA check, full validator
  against installed absolute paths, post-install 34/34 suite, voltage read or
  real C/W -> mandatory post-GO prelaunch/execution work.
- Can establish: the fresh v2-root candidate is scientifically unchanged and
  ready for conditional byte-identical installation and one-start dispatch.
- Cannot establish: current mutable H5 state, installed binding correctness,
  actual covariance rank/condition, W finiteness, successful stop, or
  scientific benefit.
