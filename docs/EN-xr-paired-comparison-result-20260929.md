# EN XR paired full-session comparison — 2026-09-29

## Verdict

The existing unrounded external-rigid arm (XR) is not a production candidate
on this completed paired comparison.  Relative to the motion-off 12/9
reference, XR has lower curated yield and assigned spikes, weak exclusive
correspondence, and worse available contamination, coincidence, and edge
guardrails.  The amplitude-completeness endpoint is infeasible because only 23
of 586 eligible reference units are measurable in both arms.

This result is supporting evidence for EN.  It does not replace the frozen raw
four-arm Tier-1 panel, decide whether rounding improves the XR field, or by
itself trigger Tier 2.

## Paired result

The completed comparison is
`/media/huklaban5/Data/luke_improved_rigid_comparison_20260928_v1` and binds XR
sort identity
`06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555`.

| Measure | Motion-off REF | XR | Reading |
|---|---:|---:|---|
| Curated units | 710 | 470 | lower in XR |
| Assigned spikes | 29,227,829 | 26,659,257 | lower in XR |
| Median 1.5 ms refractory-violation fraction | 0.00517 | 0.02004 | worse in XR |
| Chance-aware near-coincident excess | 0.14837 | 0.17134 | worse in XR |
| Edge-unit fraction | 0.05493 | 0.06170 | worse in XR |
| Edge-spike fraction | 0.05535 | 0.06369 | worse in XR |

The exclusive correspondence graph contains 16,294 qualifying edges but only
82 reciprocal primary matches, 77 of them in the common interior.  Across
primary matches, median reference-event retention is 0.7944, median XR-event
retention is 0.7386, and median Jaccard overlap is 0.5605.  The graph has
16,212 ambiguous edges, so these spike-train relationships are not biological
identity claims.

For amplitude completeness, 586 reference units are eligible, of which 509
are unmatched, 54 lack fit support, and 23 are measurable in both arms over
enough common time.  Coverage is 3.92% of the eligible reference cohort and
29.87% conditional on an interior primary match.  The prospective endpoint is
therefore `infeasible_insufficient_coverage`; the observed median
missingness change of -1.74 percentage points cannot support a population
completeness claim.

## Provenance

The persistent service terminal receipt records `success`, `exited`, and exit
status 0.  The result request records the controller, generic comparator, and
handoff verifier SHA-256 values; each matches the committed implementation at
the time of this review.  The comparison launched no sort and accessed no raw
voltage, RF result, or outer holdout.

Key artifact SHA-256 values:

- `comparison_summary.json`:
  `0b24c9044747e42467e0d17e0fb6ee27c68349448fec6c58bf61a983d35745b6`
- `request.json`:
  `828ff3d4a608ac722d6e052383a5bc1e9118bf7f43ac22c553903bb579d83042`
- XR handoff `MANIFEST.json`:
  `6dd940f99fe0805596dabf1fd387852425ec476f47ba143e37c3fa6310611a1a`
