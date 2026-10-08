# Candidate-3 mini transition: amended resource contract

Status: `FROZEN_AWAITING_INDEPENDENT_REVIEW_DO_NOT_START`.

- Milestone: complete the missing transition arm of the synthetic Candidate-3
  native-stage capability comparison.
- Decision changed: whether ordinary and one-cell transition inputs both traverse
  the same frozen DARTsort stages, reload, and support stage/accounting inspection.
- Cheapest adequate test: reuse the verified ordinary native outputs and run only
  the already-generated transition input. Do not rerun ordinary and do not alter
  the scientific window, synthetic input, preprocessing boundary, or sorter
  settings to meet the former 120 MB limit.
- Completion: managed transition service exits zero; native final NPZ reloads;
  detection/matching/refinement artifacts are present; compact accounting/report
  is generated; hashes and final manager status are preserved. Any failed attempt
  remains immutable and no automatic retry occurs.

## Frozen bindings

- Exact executed source:
  `/mnt/NPX/Luke/DARTsort_motion_experiments/candidate3_mini_e2e_executed_source_recovery_h1_20261008_v1/source/candidate3_mini_e2e_stage_capability.py`,
  SHA-256 `5d55bb1e0257afbb87b60d9d0d9dd5fcb4290a49ce2f67f8656969929e18801a`.
- DARTsort interpreter: `/home/huklab/Documents/DARTsort/.venv/bin/python`.
- DARTsort commit: `edcfe1b51d672b4136eb13cc78c0875da804b851`.
- Transition input:
  `/tmp/candidate3_mini_e2e_20261008_v2/inputs/transition_prewhitening.npy`,
  SHA-256 `dc78516734e154e45bec8f29f8f757bff904bbd32938fe0a2e0d1767f41a46f2`.
- Input receipt SHA-256:
  `5f4dd3354c0257449cef69493c5b669a6390b7283080316c9445df2a77622fa3`.
- Reused ordinary root:
  `/tmp/candidate3_mini_e2e_20261008_v3/runs/ordinary`, measured
  `128,223,957` bytes. Reuse is bound to the exact-source recovery and the
  independent mini-stage review; it is not a performance claim.
- Fresh transition output:
  `/tmp/candidate3_mini_e2e_20261008_v4/runs/transition`. It must be absent,
  nonsymlinked, and contained by the nonsymlinked v4 attempt root before launch.

The resolved sorter configuration remains exactly the recovered source's
configuration, including `detection_type="threshold"`, `preprocessing="none"`,
CPU execution, one matching iteration, saved intermediate features/labels, and
the resolved consequential defaults `subsampling_spikes_per_channel=5000`,
`subsampling_presence=0.025`, and `chunk_sampling="kmeanspp"`. Those defaults
are accepted for this synthetic capability comparison only and do not define a
real/full-recording detection contract.

## Amended resource derivation

All values are decimal bytes except explicitly stated binary units.

- retained ordinary output: `128,223,957`;
- retained input set: `57,085,352`;
- transition projection: `ceil(1.25 * 128,223,957) = 160,279,947`;
- transient/log allowance: `64 MiB = 67,108,864`;
- subtotal: `412,698,120`;
- explicit 25% safety headroom: `103,174,530`;
- derived need: `515,872,650`;
- rounded total cap: `512 MiB = 536,870,912`, leaving `20,998,262` bytes
  beyond the derived need.

At freeze time `/tmp` had `488,483,151,872` bytes available. This is a
documented storage condition, not permission to consume that free space. If the
combined retained inputs plus ordinary plus v4 transition/log artifacts exceed
`536,870,912` bytes, classify the run as resource-failed, preserve it, stop the
service if still active, and do not retry or tune settings.

The managed service additionally freezes `RuntimeMaxSec=2700`,
`MemoryMax=16G`, `CPUQuota=800%`, `TasksMax=128`, `KillMode=control-group`,
`TimeoutStopSec=30`, zero GPU request, and `Restart=no`. The exact service argv
must be copied into the prelaunch receipt. The existing user-systemd method has
prior success/failure dummy-disconnection evidence under
`testing/outputs/group2_preparation/`; this contract introduces no new manager.

## Stop and recovery rules

Stop on nonzero exit, timeout, memory limit, cap breach, missing/mismatched input
or source hash, changed DARTsort commit, pre-existing output/receipt, symlink or
containment failure, malformed resolved config, or user cancellation. There is
no within-sort checkpoint. An interruption requires a whole transition-arm
restart in a new attempt namespace after investigation; never reopen or overwrite
the failed output.

The service must not start until an independent review binds this contract, the
source-recovery packet, the ordinary review, the exact launch argv, the fresh
output checks, and the amended resource arithmetic.

Implementation checks
- Done: exact executed source recovered and hash-matched to the prelaunch receipt.
- Done: ordinary native output reuse checked against exact producer source and retained resolved config/output hashes.
- Done: cap derived from measured ordinary/input footprints, 25% transition projection margin, 64 MiB transient allowance, and 25% explicit headroom.
- Done: fresh namespace, full-stage restart, manager, runtime, memory, CPU, task, and stop conditions frozen.
- Not done: independent contract/launch review -> required before start.
- Not done: transition run/reload/accounting -> this contract authorizes only after the review gate passes.
- Can establish: a bounded, one-arm synthetic transition capability result when all completion checks pass.
- Cannot establish: real-input readiness, scientific benefit, biological identity, purity, causal attribution, or full-session behavior.
