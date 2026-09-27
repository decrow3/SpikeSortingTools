# BZ bounded waveform-pair scoring helper

## Verdict first

The small array-in/array-out helper is ready for the BV worker. It is a scoring
component, not an extractor or identity classifier. It preserves the BX design
constraints: rest is primary, episodes are a separate eligibility stratum,
half-1 lag and gain are locked before held-out scoring, uncertainty resamples
5-second temporal blocks rather than events, and aggregation gives each parent
one vote. H5 continues to own all real waveform/cache extraction and BW.2 GPU
work.

## Contract

`testing/bz_identity_helper.py` accepts four finite waveform banks with shape
`[event, time, channel]`: A/B in half 1 and A/B in half 2. The caller supplies a
fixed per-channel noise standard deviation and optional support mask. A positive
lag means B is earlier than A, so `A[lag:]` is compared with `B[:-lag]`.

The scorer:

1. forms median templates and time-centres each supported channel;
2. applies the caller's fixed noise scale;
3. searches the inclusive half-1 lag range (default -2 through +2 samples) and
   fits one positive least-squares B-to-A gain;
4. locks both values for all half-2 scores;
5. reports within-A/B reliability, held-out cross cosine, the cross deficit
   relative to the weaker within-parent reliability, signed-difference
   stability, residual RMS, support and event counts;
6. uses deliberately limited descriptive labels: `limited_waveform_compatibility`,
   `stable_waveform_difference`, or `inconclusive`. None is identity truth.

The convenience wrapper always scores rest first. Episode arrays are scored
separately when supplied; missing or under-supported episodes remain explicitly
unresolved. The bootstrap samples shared temporal-block identifiers with
replacement, never individual events. Production callers must use 5-second
block identifiers. Parent aggregation median-combines incident pair values per
parent and then gives each parent equal weight; the 18 fixed pairs must not be
treated as 18 independent observations.

## Fixed descriptive thresholds

Limited waveform compatibility requires both within-parent reliabilities to
support a minimum of 0.90 and held-out cross deficit no larger than 0.03. Stable
waveform difference requires a deficit greater than 0.10 and signed-difference
cosine at least 0.80. Everything else is inconclusive. These thresholds are
bounded diagnostics for the fixed BV pairs and are not a general identity
framework.

## Validation

One targeted test module covers known-equal and known-different waveforms,
inclusive lag-boundary recovery, a changed held-out gain (showing that gain is
not refit), a high-noise-channel control, insufficient support, independently
eligible episode handling, deterministic block bootstrap and parent-equal
aggregation. Final result: `7 passed in 0.10 s`.

No voltage was read, no scientific cache was opened, no sort or GPU fit ran,
and no H5 worker implementation was duplicated. The helper's bootstrap is
small-array CPU work only.

## Handoff

The scoped packet is published at
`/mnt/NPX/Luke/DARTsort_motion_experiments/bz_identity_helper_20260927/`.
`MANIFEST.json` and `SHA256SUMS` identify its contents. The packet intentionally
contains no held BH payload and no real waveform data.
