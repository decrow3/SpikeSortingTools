# H5 independent changed-path review: H1 workstream B

Verdict: `TARGETED_REPAIR_REQUIRED_EXECUTION_REMAINS_DISABLED`.

The event-retention redesign is correct on the reviewed synthetic domain. It
computes an equal-REF-unit mean, uses the accepted exclusive correspondence
implementation, limits credit to a unique reciprocal primary, assigns exact
zero to unmatched cohort units, preserves pairing in the unit bootstrap, and
uses the strict one-sided lower-bound `> 0.05` replacement gate while retaining
the original proposal as a diagnostic. Packaged and independent split/merge,
boundary, exclusivity, and margin known answers passed.

Two related domain-validation failures block GO:

1. `conditional_amplitude_diagnostic` accepts finite windows outside
   `[0, duration_s]`. A bounded synthetic input containing `[0,60)` and
   `[60,120)` with `duration_s=60` produced REF, candidate, and common support
   fractions of `2.0`. These are not valid full-cohort-time coverage fractions.
   Non-finite `duration_s` is also not rejected.
2. When diagnostic cell edges are supplied, `reference_event_retention` checks
   cohort REF events only during cell assignment. Candidate events, non-cohort
   REF events, and therefore reciprocal-primary competition are not required to
   lie in the same half-open domain. In the independent fixture, three candidate
   events beyond the declared stop changed candidate retention from `1.0` to
   `0.4`, removed the primary match, and changed `R` from `1.0` to `0.0` without
   raising an error.

Required repair: fail closed on finite positive `duration_s`; reject every
included amplitude interval with `start_s < 0` or `end_s > duration_s`; and,
when cell edges are provided, validate all events from both complete supplied
sorts against `[edges[0], edges[-1])` before correspondence. Add negative
known-answer fixtures for each case, republish under a fresh immutable
namespace, and keep execution disabled pending re-review.

No recording, voltage, real score, evaluator, sorter, waveform, RF, holdout, or
real-arm array was opened or run.

Implementation checks
- Done: verified requested MANIFEST and COMPLETE hashes and every manifest
  member; inspected exact source, contract, tests, and accepted matcher bytes.
- Done: reran the 9 packet tests from a writable byte-identical stage; all
  passed. Independent dominant split/merge, strict-margin, half-open-boundary,
  hidden-I/O AST, amplitude-domain, and sort-domain fixtures were bounded and
  synthetic only.
- Done: confirmed frozen 60-second edges equal ceiling-converted edges at
  `29999.835983263598 Hz`, ending at the exact local stop `17999901`.
- Not done: no real inputs or outcomes were accessed, so no scientific effect,
  power, biological identity, novel-event recovery, or advancement claim was
  assessed.
- Can establish: the core event-retention and margin semantics are correct on
  in-domain in-memory inputs, but the public measurement functions do not yet
  enforce their declared temporal domains.
- Cannot establish: safe prospective execution or valid reported support and
  retention if malformed or clock-misaligned supplied artifacts escape a future
  loader.
