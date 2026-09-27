# CB integration guards for the BZ helper

## Verdict

BZ helper v3 is ready for direct H5 integration. It keeps all v2 training-state
and common-domain repairs, and closes two final integration gaps without adding
a scientific threshold.

- `min_events` now qualifies the original common-domain point only. Bootstrap
  draws retain every sampled common block and use `min_events=1`, avoiding a
  biased interval formed by selectively dropping sparse draws.
- Final labels require all requested block draws to be accepted and every
  interval used by the relevant branch to be finite, ordered and based on the
  expected replicate count. Incomplete accounting or invalid required evidence
  is explicitly `unresolved`.

Compatibility can still qualify when signed-difference cosine is degenerate,
because that metric is irrelevant to the compatibility branch. Stable
difference can qualify when the compatibility-gap interval is unavailable if
its own reliability, deficit and difference intervals are valid. If neither
branch can be evaluated completely, the result is unresolved. Original BX
margins remain 0.90, +/-0.03, 0.10 and 0.80.

Nine focused tests pass. New regressions show that a point exactly at the event
minimum retains all 40 sparse block draws, incomplete replicate accounting
cannot label, nonfinite intervals cannot label, and an irrelevant degenerate
difference metric does not suppress otherwise complete compatibility evidence.

The shared v3 packet includes a direct `from bz_identity_helper import ...`
example and a self-contained event/row/block/channel/noise/support contract.
Intervals are explicitly conditional on the already-qualified common-domain
point; the helper makes no population-prevalence or biological-identity claim.
No extraction, raw read, GPU work or scoring of real data occurred.
