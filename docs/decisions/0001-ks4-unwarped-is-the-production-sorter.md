# 0001 — KS4 on the unwarped frozen graph is the production sorter

**Status:** Adopted (2026-08-30, cross-probe control 2026-08-31).
**Qualified 2026-09-02 by [0008](0008-amplitude-completeness-gates-promotion.md)** —
read that record before citing any yield result below.
**Applies to:** every production sort

> **The interpretation in this record is narrowed by
> [0008](0008-amplitude-completeness-gates-promotion.md).** The measurements
> here stand. The inference that yield gains alongside *fewer* assigned spikes
> demonstrate the improvement is not detection inflation has since been shown
> insufficient: post-curation amplitude-truncation analysis finds rescue units
> are typically *less* completely detected than both comparators. This record is
> not the basis for promoting the configuration.

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

The post-curation evaluation added a second, independent ground for that verdict
— worse amplitude completeness than either comparator — and reframed the pair
evidence: after curation, 27 similar good–good pairs remain (against 8 legacy
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
  *Qualified 2026-09-02:* on the post-curation imec0 comparison the claim-mask
  configuration has the **best** amplitude completeness of the three (0.82%
  median missingness in the >1 Hz cohort, 91.8% of units below 10%), at a large
  yield cost (191 KS-good against 301). The rejection was reasonable on the
  evidence then available, but the trade-off it represents was never measured.
  See [0008](0008-amplitude-completeness-gates-promotion.md).
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
