# 0001 — KS4 on the unwarped frozen graph is the production sorter

**Status:** Adopted (2026-08-30, cross-probe control 2026-08-31).
**Qualified 2026-09-02 by [0008](0008-amplitude-completeness-gates-promotion.md)
and [0009](0009-cross-sort-comparisons-must-be-unit-matched.md)** — read both
before citing any yield result below.
**Further qualified 2026-09-03 by [0011](0011-cross-sort-event-matching-and-detection-evidence.md)
and [0015](0015-corrected-cross-sort-audits-do-not-establish-equivalence.md):**
the empirical matched-unit completeness result is retracted. Corrected Phase A
found no confirmed new/lost detection but left seven units unresolved; it does
not establish equivalence or that rescue performs better.
**Applies to:** every production sort

> **The interpretation in this record is narrowed by
> [0008](0008-amplitude-completeness-gates-promotion.md), as corrected by
> [0009](0009-cross-sort-comparisons-must-be-unit-matched.md).** The
> measurements here stand. The inference that yield gains alongside *fewer*
> assigned spikes demonstrate the improvement is not detection inflation remains
> insufficient — aggregate spike count is a contamination-side check, not a
> per-unit recall measure. But the completeness deficit 0008 originally reported
> was composition-confounded and is retracted; the first matched-neuron equality
> result was also retracted because its matcher reused target events. This record is
> still not by itself a basis for promotion; the failed gates are.

## Decision

Sort with Kilosort 4 on a frozen upstream graph that never spatially resamples
voltage. The graph is:

1. Neuropixels phase (ADC) correction.
2. Samplewise bilateral blanking at 500 µV — see [0003](0003-saturation-blanking-and-artifact-sidecar.md).
3. Interpolate physical channel 191 and include it in sorting — see [0004](0004-bad-channel-191-interpolation.md).
4. One internal Kilosort high-pass / CAR / whitening stage.
5. Disable *all* of: external motion correction, Kilosort internal motion
   correction (`do_correction=false`, native `nblocks=0`), the cross-peel claim
   mask, and Kilosort's batch artifact threshold.
6. Write raw samples over 500 µV to a separate artifact sidecar and exclude
   nearby detections from artifact-sensitive claims.

This is the frozen **engineering** reference, not a claimed biological optimum.

## Evidence

Both probes of Luke 2025-08-04, full session (10,473.6 s):

| | imec1 | imec0 |
|---|---|---|
| Units | 583 | 727 |
| KS-good units | 216 (+43.0% vs best prior full-probe) | 301 (+15.8% vs legacy) |
| Assigned spikes | 43.7 M (−6.7% vs `pipeline_an5`) | 30.5 M (−12.2% vs legacy) |
| Median KS-good contamination | 3.55% | 2.50% (from 4.05%) |
| Median 1.5 ms refractory violation | 0.125% | 0.113% (from 0.203%) |

The yield gain comes with *fewer* assigned spikes on both probes, so it is not
explained by indiscriminate *aggregate* detection inflation.

**This was originally described as the load-bearing part of the result. It is
not sufficient.** Fewer total spikes with more good units is still compatible
with a population whose individual units are each less completely detected.
That is precisely what the post-curation truncation analysis later measured —
see [0008](0008-amplitude-completeness-gates-promotion.md). Aggregate spike
count is a contamination-side check; it says nothing about per-unit recall.

## Important qualification

The prespecified imec0 evaluator returned `reject_universal_default` and that
verdict stands — it was **not** waived retrospectively. Three frozen gates
failed: stable-good-unit fraction 73.75% vs a 75% threshold, edge burden 2.004%
vs 2.000%, and 37 nearby similar good–good template pairs against a 6% pair-count
limit.

Follow-up reduced those 37 broad candidates to one strong and one partial
duplicate hypothesis (units 184/191 and 164/165), all four heavily
artifact-associated. Conservatively discounting all four still leaves 297
KS-good units, +14.2% over legacy.

So: the rescue graph is the **locked operational downstream reference**, and that
is a different and weaker claim than universal adoption. Do not cite this record
as evidence that the frozen gates passed.

The post-curation evaluation initially appeared to add a second, independent
ground for that verdict — worse amplitude completeness than either comparator —
but that population inference is confounded and its first matched correction is
also retracted ([0011](0011-cross-sort-event-matching-and-detection-evidence.md)). The verdict rests
on the failed gates alone. The evaluation did reframe the pair evidence: after curation, 27 similar good–good pairs remain (against 8 legacy
and 11 claim-mask) while the artifact-aware audit finds *zero* strong duplicate
hypotheses. The pre-curation 184/191 and 164/165 hypotheses recorded above were
a different screen at a different stage; both results stand as measured. See
[0008](0008-amplitude-completeness-gates-promotion.md).

Neither does this establish Yates parity. That comparison is confounded by
anatomy, depth, preprocessing, duration, and the high contamination of the
available Yates sort.

## Rejected alternatives

- **External nonrigid voltage warping** — damages or relocates otherwise
  sortable spikes; see [0002](0002-motion-is-estimated-never-applied.md).
- **Kilosort internal motion correction** — same class of risk, and the accepted
  `ops.npy` independently records that it was off.
- **Cross-peel claim mask** — tested and measured as a source of loss/duplication.
  *Qualified 2026-09-02, then largely restored:* the claim-mask configuration
  appeared to have the best amplitude completeness of the three, but on matched
  neurons it is indistinguishable from rescue (0.73% vs 0.75%, p = 0.19). Its
  apparent advantage came from keeping only larger units at a large yield cost
  (191 KS-good against 301). The original rejection stands; what remains true is
  that the yield-versus-selectivity trade-off was never measured. See
  [0009](0009-cross-sort-comparisons-must-be-unit-matched.md).
- **Kilosort batch artifact threshold** — superseded by the explicit sidecar.
- **Pinned AIND preprocessing** — a competent independent comparator that ties
  rescue on aggregate sealed-event recovery (470/720) and improves
  refractory/coincidence diagnostics, but is not a full-session finalist; rescue
  retains better yield and normalized similar-template burden.

## Reopening conditions

Broad preprocessing searches are **paused** — preprocessing is in
diminishing-returns territory. Reopen only for a demonstrated, *stage-local*
failure, per [0007](0007-stage-local-validation.md). A better final sort is not
sufficient justification.

## Evidence pointers

- `docs/luke_20250804_full_probe_rescue_result.md` (imec1)
- `docs/luke_20250804_imec0_rescue_result.md` (imec0 control)
- `docs/luke_20250804_rescue_status_and_test_plan.md` (frozen reference, matched tests)
- Frozen gates: `configs/rescue/imec0_legacy_acceptance_criteria.json`
