# H5 re-review: workstream-B temporal-domain repair

Verdict: `GO_FOR_MEASUREMENT_MODULE_CHANGED_PATH_EXECUTION_REMAINS_DISABLED`.

The requested repair packet and every manifest member verify. The delta is
limited to validation order, amplitude-table domain checks, and bounded coverage
construction plus their synthetic tests. The accepted matcher, execution-disabled
configuration, pipeline config, R/DeltaR definitions, paired unit bootstrap, and
strict one-sided `lower95 > 0.05` gate retain their reviewed bytes or semantics.

H5 reran all 30 packet tests from a writable byte-identical stage. Independent
adversarial checks additionally established that:

- non-finite/non-positive duration fails;
- malformed, overlapping, non-finite, and out-of-domain `finite_interior`
  intervals fail, including an invalid row outside the selected cohort;
- exact `[0,duration_s]` support is accepted and every coverage fraction is in
  `[0,1]`;
- complete REF and candidate streams are checked against the half-open cell
  domain before correspondence;
- exact-stop and negative events fail, including candidate and non-cohort REF
  primary-competition attacks;
- a matcher sentinel is not invoked for invalid complete-sort input;
- valid events at zero, an interior cell boundary, and `stop-1` preserve the
  intended boundary and R semantics.

This GO is narrowly for the repaired in-memory measurement-module changed path.
It is not authorization or evidence for real scoring, a job, a cross-arm effect,
power, biological identity, novel-event recovery, or advancement.

Implementation checks
- Done: verified repair MANIFEST `61758790...`, COMPLETE `9cbe0887...`, every
  member, preserved original/review bindings, source delta, tests, matcher, and
  configuration.
- Done: reran 30 packet tests and independent duration/interval/coverage/domain/
  boundary/matcher-order/R/DeltaR/bootstrap/gate adversarial checks.
- Not done: no real inputs, scores, evaluator, recording, voltage, sorter,
  waveform, RF, holdout, power, or scientific outcome was accessed or run.
- Can establish: H5-B1 and H5-B2 are repaired for bounded in-memory inputs
  without changing the frozen estimand or decision rule.
- Cannot establish: any real effect or prospective experiment readiness beyond
  this reviewed measurement-module path.
