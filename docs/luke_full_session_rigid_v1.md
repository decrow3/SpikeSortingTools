# Full Luke0804 imec0 native rigid comparison

Authorized by the user on 2026-09-06 after snippet endpoint feasibility failed.
This is a new full-session development experiment. All 10,473.553728 seconds and
384 channels are included, including intervals previously reserved in the Luke
snippet panel. Those observations are development evidence here; this run does
not meet historical held-out promotion requirements. Previous closed results
remain unchanged.

## Execution

Reference: the completed full-session rescue 12/9 sort and curated QC.
Candidate: the identical accepted rescue recording, preprocessing, locked
environment and KS4 parameters, with native rigid correction enabled
(`do_correction=True`, effective `nblocks=1`). No external field or voltage crop.
Existing legacy full-session curated results provide historical context.
One candidate, no sweep, no automatic production adoption.

Reuse the existing sorter, curation and QC implementations. Verify recording
content, baseline recording identity and parameter equality before execution;
verify effective candidate settings afterward. Write to a new local directory:
`testing/outputs/luke_full_session_rigid_v1/`. The accepted 241 GB recording is
read directly, without making another binary copy. The runner and saved request
are `testing/luke_full_session_rigid.py` and the output's `request.json`.

## Descriptive comparison specified before candidate output

- Keep nominal 1,000-spike historical QC windows and report all fit statuses.
  Do not pool amplitudes across clusters or infer recovery from unmatched
  population medians. Exact-index sensitivity is a follow-up if an apparent
  improvement depends on the fit estimate.
- Establish descriptive train correspondence from exclusive per-cluster event
  matches within 0.5 ms. Enumerate temporal-coincidence candidates first, then
  compute exclusive matches for every pair whose possible overlap is at least
  10% of the smaller train. Retain the full qualifying edge table for split/merge
  inspection. This avoids pooled events competing across unrelated clusters.
- A primary correspondence is reciprocal best Jaccard overlap with at least
  50% of each train matched. Include all cluster labels when finding partners;
  do not hide a baseline-good to candidate-MUA transition. These are train
  correspondences, not assertions of neuronal identity.
- For each matched pair, intersect valid fit intervals in physical recording
  time. Report duration-weighted baseline minus candidate missing percentage
  on that common support, each arm's valid-fit coverage, and common coverage.
  At least two valid windows per arm are required for a paired summary. Also
  report coverage failures; they are not improvement or zero missingness.
- Report full-train short-ISI fractions, event retention in both directions,
  spike counts, labels, and unmatched baseline-good and candidate-good counts.
  Report effects across the full baseline-good denominator and stratified by
  common-time coverage. Unit count alone is not success.
- Plot time-resolved missingness for the paired good cohort and named baseline
  diagnostic clusters 37, 553, 452 and 36 when correspondence exists. Keep
  nonfinite/boundary fits and unmeasured time distinguishable.

This first full-session comparison is descriptive; it has no retrospectively
chosen pass margin. A promising result needs waveform/identity review of the
actual differences and independent-session validation. A poor result does not
trigger automatic parameter tuning. Report what improved, what regressed, and
what remains unmeasurable.
