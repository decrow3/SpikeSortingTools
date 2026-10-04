# H1 Phase-1 REF-only postexecution review

Verdict: `PASS_PHASE1_POSTREVIEW`.

The exact one-shot Phase-1 result and service packets verify, and H1's
independent REF-only recomputation meets the frozen acceptance conditions:
61 units, all three R controls exactly 1, 227,895 cohort events independently
summed and reported credited once, correct half-open boundaries and final-stop
rejection, invariant REF-self support union, matching request/source provenance,
and no cross-arm scalar.

No B-arm file or voltage was opened. The only prior-result table read was the
explicitly cited REF-only `unit_metrics_baseline.csv` at sha256
`890fe7c1301001a11b62308b7d2954fc204d49bb56e33c281b288f3bd8be1842`.
It independently reproduces the inclusive-depth cohort, cohort digest, and
227,895-event denominator.

Correctness passes for the frozen diagnostic scope. One-shot reliability passes
from the managed-service receipt, zero restarts, `Restart=no`, persistent exact
token claim, and failed post-completion freshness conditions. Cross-host
portability is not established because the real REF event and QC arrays remain
H5-local.

The next H1 item is `H1-WAVEFORM-REVIEW`, but it is dependency-waiting until H5
publishes the actual-host waveform-method candidate. H1's ready slot remains
empty. Phase 2 is not started or authorized by this review.

## Implementation checks

- Done: verified the five result and three service manifest members plus all
  four delegated MANIFEST/COMPLETE identities.
- Done: traced exact request, contract, managed unit, persistent token claim,
  executed source, matching/counting semantics, frames, domains, provenance,
  resource receipt, and completion order (`SOURCE_TRACE.json`).
- Done: independently recomputed the REF cohort, digest, event denominator,
  cell edges, boundary assignments, binding equality, and report invariants
  (`INDEPENDENT_RECOMPUTE.json`).
- Not done: no B, voltage, Phase 2, RF, holdout, real sorting/training, full
  cross-host replay, or scientific-effect evaluation.
- Can establish: the exact REF-only diagnostic run completed once and satisfies
  the frozen Phase-1 acceptance rule.
- Cannot establish: waveform preservation, repaired-preprocessing benefit,
  repeatability, identity, purity, portability, or scientific advancement.
