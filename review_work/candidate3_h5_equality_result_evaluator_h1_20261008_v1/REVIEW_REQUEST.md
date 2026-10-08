# Candidate 3 equality result evaluator review

Review the frozen evaluator as a consumer of the unchanged accepted v3 runner
packet.  It must validate the seal, exact execution contract and implementation
hashes before interpreting any result.  It uses exact zero-tolerance equality,
requires both context positive controls, validates the no-copy DARTsort boundary
and resource counters, and emits only the minimum coordinator decision.

The real-pass category is unavailable to synthetic provenance.  A valid sealed
runner failure is classified as execution failure and stopped without automatic
retry; it is never converted to an equality failure or pass.

Implementation checks
- Done: accepted v3 synthetic packet independently classified as
  `PASS_SYNTHETIC_FIXTURE_ONLY_NO_H5_DECISION`.
- Done: six controls cover success, seal tampering, handled failure, primary
  mismatch, noninformative positive control, and resource excess.
- Done: expected request frames/shapes are reconstructed from the hash-bound v3
  contract rather than copied from RESULT.
- Done: exact source hashes, runner hash, mode/status agreement, real release
  hashes and real argv/interpreter bindings are checked before a real pass.
- Not done: real H5 output inspection; prohibited for this task and unavailable.
- Can establish: deterministic interpretation and fail-closed behavior for the
  frozen packet schema.
- Cannot establish: real equality, sorting benefit, biological identity, purity
  or authorization for any later experiment.

Requested focused verdict: GO only for use on the later exact v3 result packet;
otherwise preserve the precise interpreter defect.  Do not contact H5.
