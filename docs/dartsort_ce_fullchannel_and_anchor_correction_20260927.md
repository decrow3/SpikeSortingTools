# CE full-channel review and residual-anchor provenance correction

## Verdict

The CD full-channel pair packet provides useful **worker-reported, limited
waveform compatibility** for two of 18 pairs, but the bounded two-row audit
cannot independently certify its final interval accounting or per-pair support.
Separately, the earlier 48-anchor residual-motion proposal is invalid because
its reliability IDs came from a different sort. The 48 count and lists are
withdrawn. No residual update may be fitted from them.

### 2026-09-27 supplement

The later sibling packet
`luke0804-imec1-cd-fullchannel-v1-supplement-v1/` closes the narrow
reproducibility caveat below for pairs 1 and 5. Its hashes match the original
selection, raw cache, metrics and intervals. Both pairs have exact saved
support/noise/frozen state, four unique 100-event banks, and 1,000/1,000 finite
block draws; independently reapplying helper v3 returns
`limited_waveform_compatibility` for both. The cross-sort 48-anchor withdrawal
and all limits on identity/merge interpretation remain unchanged. Independent
details: `docs/dartsort_ch_cf_packet_review_20260927.md`.

## Full-channel packet: bounded two-row review

Packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-cd-fullchannel-v1/`.
It is complete and hash-manifested, uses authoritative helper v3
(`f0e245...`), and records a selection frozen before voltage. Common 5-s blocks
were intersected before the deterministic cap. Sixteen declared pairs have
exactly 100 A/B events in each half; pairs 11 and 13 are explicitly descriptive
shortfalls. The cache header is `[6400,121,182] float32`.

The two saved waveform rows inspected are finite `[121,182]` arrays, with ranges
[-21.84,15.52] and [-20.10,11.07] in recorded scale. Their first two manifest
rows have unique row IDs, monotonic session times and declared common-block
count 29. Global geometry contains 182 rows with ordered channel IDs and source
AP IDs. This validates serialization for the inspected rows, not all snippets.

The two reported compatible rows are:

- pair 1, units 6/9, parent 91: 22 support sites, 20/23 common blocks; within
  lower bounds 0.956/0.957 and gap interval [-0.0107,0.0097];
- pair 5, units 535/538, parent 12: 30 support sites, 29/34 common blocks; within
  lower bounds 0.929/0.923 and gap interval [-0.0023,0.0267].

Those values meet the fixed compatibility margins. However, the packet does not
save accepted/requested replicate counts, per-pair support indices, frozen
noise vectors or serialized frozen states. `SUPPORT_SLICE_EQUIVALENCE` checks
pair 0 only, not pairs 1/5. Thus the two labels are plausible and useful but not
independently reproducible from the allowed two-row audit. They remain limited
waveform compatibility on recorded support, never identity or merge truth.

## Exact donor-source audit: why 48 anchors are withdrawn

`testing/luke_aw_full_probe_extract.py` (SHA-256 `f65eee...`) explicitly sets
`static = HANDOFF / "static_w2"` at lines 379–380, loads
`static_w2/dartsort_sorting.npz` at lines 390–394, and loads its sealed
`template_data.npz` at line 549. Exact source artifacts are:

- static sorting `be6106...`;
- static template bank `a99b12...`;
- static config `5600b5...`;
- static input manifest `0975b3...`;
- candidate measurements `692962...`;
- BC v2 reliability table `37aafc...`.

BQ instead uses
`luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/`, including `matching1.h5`
`6734ba...` and the prospective D2L stage/final maps. `depth_reorder` maps only
within that run. No artifact maps static final IDs to D2L stage units or final
components. Equal integers across these namespaces have no identity meaning.

Therefore the CD inverse mapping, force exclusions, 48-anchor count, unit lists,
56,658-row cache estimate and proposed folds are all invalid and withdrawn. The
original CD report is retained with a failure banner; it is not silently
rewritten as a pass.

## Correct replacement design — not yet run

First establish anchors in the **same D2L/BQ state** by either auditable
event-level correspondence to an independently qualified sort or fresh
same-state reliability. The latter is narrower and direct:

1. Freeze BQ event rows, D2L pre/final maps, corrected 3,000-sample construction
   bank, physical geometry and original stage-unit peak sites.
2. Exclude force-dependent groups in their native stage/pre-component namespace
   before looking at residual outcomes.
3. Read a bounded raw-snippet cache from the accepted 182-channel float32 source.
   Score depth hypotheses from raw snippets on fixed physical support around the
   original peak sites, carrying competitors and unmatched events. Do not use
   saved matching point localizations as the validation endpoint: they inherit
   accepted-template and neighborhood bias and are only a localization
   diagnostic after corrected construction.
4. Qualify same-state anchors with disjoint raw-snippet halves, explicit signal
   energy/support, event/bin counts and waveform identity ambiguity. Missing
   reliability is unresolved; it cannot inherit the static donor result.

For a future single residual update, distinguish evaluation claims:

- estimating `delta(t)` from fit units contemporaneously and scoring other
  units at the same times is **unit-heldout only**;
- a true contiguous-time holdout requires the correction rule to be frozen on
  the earlier segment and applied later without using any later anchor to fit
  `delta(t)`. If that prediction rule is not specified, make no time-heldout
  claim.

Controls must include zero update, sign reversal as a directional diagnostic,
and a genuine phase null. Prespecify all nonzero circular 5-s-bin phase shifts
of the candidate update (or a frozen seeded subset); these preserve its value
distribution and circular first-difference smoothness while breaking temporal
alignment. Success requires held-out-unit improvement over zero and the 95th
percentile phase-null result with a unit/block bootstrap interval excluding
zero. T8 must be rerun on label-free peaks with episode/pseudo-episode bootstrap
uncertainty and the existing 10-um reference quantization reported; its error
may not worsen beyond Q's frozen 3-um tolerance. Disagreement is inconclusive.

This design uses raw waveform evidence and explicit cross-run correspondence or
fresh same-state qualification. No localization-derived residual field, fit,
loop, voltage change or sort was run in CE.
