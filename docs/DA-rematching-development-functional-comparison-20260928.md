# DA rematching development functional comparison

## Frozen purpose

Compare the saved final W2 outputs for the original accepted D2L arm,
REMATCH0, and CD1_FULL with the already-validated CP development-only RF
evaluator. This is a functional screen of three total pipelines. It is not an
identity-preserving unit comparison: the rematching arms use new row and label
namespaces, and their adaptive localization encoder states differ even though
their common TPCA/waveform basis is exact.

## Frozen support and evaluator

- Probe: imec1.
- Source AP frames: `[26999783, 37199701)` at
  `29999.759166666667` Hz; local sorting times are offset by the first frame.
- Original stimulus universe: 81 trials, split before cropping into 61
  development and 20 sealed outer-holdout trials.
- W2 complete development trials: 12, with the original 7/5 inner folds and
  6,052 valid gaze/lag-safe frames.
- W2 complete outer-holdout trials: 5; they remain unopened and unscored.
- Eligibility: at least 500 development spikes total and at least 200 in each
  inner fold.
- Gaze calibration, lag, response binning, stimulus crop, thresholds, and
  count calibration are reused unchanged. Nothing is refit.

The accepted hashes are the CP hashes: config `e23297fb...`, stimulus manifest
`d5241af5...`, stimulus frames `92b46015...`, trial split `399858bb...`, trial
table `6a950b31...`, and gaze CSV `3a726727...`. The interval is re-derived
verbatim from the authoritative accepted W2 `input-manifest.json`, rather than
from a rounded 340-second duration.

## Inputs

- D2L: original accepted W2 final, SHA-256 `03c339b4...`.
- REMATCH0: CY `REMATCH0_v3` final, SHA-256 `82210d21...`.
- CD1_FULL: CY `CD1_FULL_v1` final, SHA-256 `ba69b03c...`.

No legacy label joins are valid or used. Whole-population eligibility failures,
coverage, and scores are reported for each arm. Any paired summary is optional
and may use only an RF-independent correspondence with ambiguity preserved; it
cannot block the main evaluation.

## Resource and interpretation contract

At most 1,200 seconds of new all-work CPU, two numerical threads, one reader,
20 GB memory, 1 GB scratch, 500 MB final output, at least 30 GB free, zero raw
voltage reads, zero GPU, and no sorting. H1 cumulative accounting starts at
17,333.22 seconds and has a 20,500-second ceiling. The expected elapsed time is
12--18 minutes, though the previously measured evaluator completed three arms
in 26.6 seconds; actual elapsed time will be reported.

The outcome can compare functional RF support at this saved boundary. Because
adaptive localization differs after matching, final-output differences are
total-pipeline effects rather than pure coordinate-descent mediation.
