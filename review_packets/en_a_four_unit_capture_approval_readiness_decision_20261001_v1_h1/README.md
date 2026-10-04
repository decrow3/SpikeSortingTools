# Final H1 decision: A four-unit physical-signal capture readiness

Verdict: **ready to request the explicitly bounded A capture approval**.

This is an approval-readiness decision, not approval to capture and not voltage
authorization. No recording binary was opened, no voltage was read, and no
sort, matcher, RF, holdout, or covariance job ran during this review.

The H5 v3 metadata preserves the original H1 20-slot design without changing a
unit, role, physical state, selection predicate, or window rule. The unchanged
H1 verifier ran on H5 against the hash-bound saved event/model arrays. Its 20
selected rows agree with the producer receipt on every common field. A separate
H1 regeneration from the included 512,838-event metadata independently recovers
the same earliest pairs and controls. Unit/template distributions and all
comparable scalar footprint fields also agree exactly.

The bounded future request remains: eight pair windows and twelve single-event
windows from each of the original and arm-A-corrected int16 representations,
all 384 channels, plus one duplicate positive-control read. Maximums are 41
reads, 32,653,824 logical input bytes under a 33 MiB cap, 4,073,472 retained
int16 bytes under a 6 MiB cap, 2 GiB RAM, four CPU threads, and 300 seconds.
Transient float stages are processed one representation/window at a time.
Numeric snippets remain H5-local.

An explicit human approval is still required before either recording is opened.
That approval must remain confined to the hashes, selections, windows, resource
limits, processing stages, retention rules, and prohibitions in `DECISION.json`.

Implementation checks

- Done: verified all v3 packet/product hashes and completion receipts.
- Done: reconciled 20/20 slot identities against the original frozen H1 CSV.
- Done: independently regenerated earliest original-adjacency selections from
  the complete four-unit metadata and matched the producer and H1-verifier rows.
- Done: matched exact unit/template distributions and comparable scalar template
  footprints; retained their model-derived interpretation limits.
- Done: recomputed the 41-read, 32,653,824-byte, retention, and RAM envelope.
- Not done: voltage capture or physical-signal measurement -> requires explicit
  human approval and remains disabled.
- Can establish: immutable metadata selections, provenance, and a bounded safe
  request suitable for explicit approval.
- Cannot establish: physical waveform behavior, biological identity or purity,
  preprocessing causality, or scientific benefit.
