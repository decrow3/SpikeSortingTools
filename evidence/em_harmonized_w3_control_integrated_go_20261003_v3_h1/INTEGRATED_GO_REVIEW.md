# H1 final integrated GO — input-harmonized W3 control v3

## Decision

**GO_TO_COORDINATOR_RELEASE for exactly one input-harmonized W3 control.**

This is not a launch. The separate coordinator release and the packet's
immediate H5 release preflight remain mandatory. Any failed release check stops
the run without retry.

V3 repairs the v2 lifecycle blocker with one disconnect-safe managed controller.
The controller launches the existing managed sort, observes the sort service
until terminal success, independently validates `receipt.json`, the sort
`COMPLETE.json`, and the H sorting hash, and only then invokes the frozen
score/seal worker synchronously once. Sort failure never reaches score dispatch;
evaluator failure and controller failure are preserved; existing state/run/score
namespaces block rerun and overwrite.

H5's real systemd fixture passes. H1 independently reran the packet fixture on
huklaban1 using the absolute controller path and reproduced all required states:
score absent while sorting was active, one score after terminal success, no
score after sort failure, preserved evaluator failure, and blocked rerun. Two
earlier H1 attempts are recorded rather than overwritten: sandboxed systemd
access failed, then a relative controller path failed before the absolute-path
fixture passed. No attempt used recording data, GPU, or scientific sorting.

## Frozen scope and release conditions

- Exact implementation packet manifest:
  `b501f06501c16d636f5071ff4d52d253440e4e92c8430f3b433ed231dc2ecc2c`.
- One fresh H sort only on global frames `[239998073,250197991)`, local
  `[0,10199918)`, 182 channels, float32, using the exact full-session cache view.
- No refilter, rereference, rescale, motion refit, raw reconstruction, second
  crop, channel reorder, learned-model reuse, resume, retry, or cleanup.
- Immediate preflight must verify packet-bound H1/coordinator receipts, absent
  run/score/workflow outputs, cache lineage, >=25 GiB disk, >=160 GiB available
  RAM, exactly one RTX A5000 at `cuda:0`, and zero retry.
- The scientific sort service is capped at 3,480 seconds and <=1 GPU. The
  4,500-second controller wall includes the subsequent CPU-only score/seal stage;
  it does not extend the sorter/GPU service cap.
- Primary H-F uses the frozen four-category joint common-block rule at +/-0.05.
  H-L remains secondary/non-decompositional. Yield and three-domain segment-safe
  short-interval guardrails must pass as frozen; missing/undefined is
  non-advancing.
- No RF, outer holdout, target A, long/full work, parameter search, production
  promotion, or biological identity/purity claim.

Success requires terminal sort and managed workflow receipts plus an immutable
score packet with `COMPLETE.json` written last. Failure must remain preserved.
No outcome automatically authorizes another experiment or promotion.

## Implementation checks

- Done: packet integrity -> all 36 v3 manifest members verified; COMPLETE binds
  manifest and is later than all members.
- Done: lifecycle source -> controller ordering, terminal-state semantics,
  receipt/hash validation, synchronous one-shot scoring, failure preservation,
  timeout, and no-overwrite behavior traced.
- Done: real managed behavior -> H5 fixture inspected and H1 independently
  reproduced success, sort failure, evaluator failure, active-state exclusion,
  and rerun rejection with transient dummy services.
- Done: v2-to-v3 diff -> cache descriptor/preflight, frozen evaluator and helper
  hashes are identical; run config changes only packet provenance; scorecard
  changes only the fresh H output path; scientific/resource rules are unchanged.
- Done: release controls -> 25-GiB disk, 160-GiB RAM, exact GPU/device,
  <=3,480-second sorter wall, manifest-bound receipts, fresh namespaces, and
  zero retry are fail-closed.
- Not done: current H5 release resources and production unit state -> intentionally
  deferred to the mandatory immediate coordinator release preflight.
- Not done: H scientific outcome -> H does not exist and no score ran.
- Can establish: the exact v3 package is implementation-reviewed and ready for
  the separate coordinator release gate.
- Cannot establish: future resource availability, H result, biological identity
  or purity, or suitability for long/full/production progression.
