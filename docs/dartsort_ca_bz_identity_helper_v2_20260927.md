# CA repair: BZ identity helper v2

## Verdict

BZ v1 had material leakage between training and evaluation and could issue a
final-looking label without the dependence-aware interval required by BX. V2
repairs those issues without changing BX's prespecified numerical margins.
Seven targeted regressions pass.

## Repairs

- `FrozenPairState` is learned from rest half 1 only. Its declared/training-
  finite support, fixed noise weights, lag and B-to-A gain are reused for held-
  out, bootstrap and episode arrays. Missing values on frozen held-out support
  make the state unresolved; they never shrink support after inspection.
- Point estimates and bootstrap draws use the identical intersection of A/B
  5-second blocks in each half. Counts are explicit, and each half requires at
  least 10 common blocks after exclusions.
- Bootstrap draws resample whole common blocks and call the frozen-state scorer;
  lag and gain are not fitted inside draws.
- Point estimates are `descriptive_metrics_only`. Final limited-compatibility or
  stable-difference labels require supplied block-bootstrap intervals.
- The original BX margins remain unchanged: within reliability 0.90;
  cross-minus-within interval contained in +/-0.03 for compatibility; lower
  cross-deficit bound above 0.10 and lower signed-difference bound at least 0.80
  for stable difference. Stable difference now also requires both within-unit
  lower bounds at least 0.90.
- Same-parent pairs contribute that parent only once before incident-pair
  medians and equal-parent aggregation.

## Deliberate scope

This remains a bounded scoring helper, not a general framework, extractor,
classifier of biological identity or replacement for H5's BV worker. It reads
no voltage and performs no GPU work. Rest remains primary; episode transfer is
separate and uses the rest-frozen state.

## Validation and accounting

The seven focused regressions cover: no final label without intervals; rejection
of an arbitrarily negative compatibility gap; reliability-qualified stable
difference; rest-to-episode state locking; held-out missing support; identical
point/bootstrap common domains; the 10-block minimum; no bootstrap refitting;
fixed-noise behavior; and no duplicate same-parent vote. Result:
`7 passed in 0.09 s`.

V1 remains immutable in the root of the shared BZ packet. V2 is published in
the packet's `v2/` subdirectory with its own manifest, hashes and API example.
