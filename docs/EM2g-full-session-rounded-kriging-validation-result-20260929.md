# EM.2g full-session rounded-kriging validation result

## Verdict

**The full-session rounded-kriging pipeline completed and is operationally
usable, but it does not yet pass as the production replacement.** Both frozen
window comparisons are `mixed_or_inconclusive`. Yield and short-ISI directional
checks pass, while yield equivalence fails because the full-session model has
roughly twice as many units. The rate-rank continuity proxy is lower than the
independently trained exact-lattice reference in both windows, with the larger
and clearly non-equivalent decrease in W3. Cached-event overlap is moderate and
contains substantial ambiguity. These results support retaining the run for
identity-aware review, not promoting it or rejecting kriging solely from raw
yield.

No RF, outer holdout, or voltage data was used.

## Completed full-session run

- Run: `/home/huklaban5/DARTsort_experiment_scratch/em2f_full_20260929/rounded_kriging_v1`
- Terminal status: complete, exit 0
- Duration: 10,473.554 s; 182 channels
- Final sorting SHA-256: `3b324bbc4ec3c2bcfcc8bc694f0cf9b0237ac0d4f87d1c13ef7b12facfade660`
- 21,822,563 total events; 21,747,060 assigned; 75,503 negative labels
- 984 assigned units
- DARTsort 0.5.23.post4+g6c0566b2; SpikeInterface 0.104.8

The original service was killed by `systemd-oomd` after matching. The complete
10,474-chunk, 21,822,563-event matching checkpoint was reused; clustering and
refinement restarted under explicit memory protection. The recovered sort and
QC completed successfully. DARTsort's final per-event shifts produced 287,811
local time-order inversions, all at most two samples. Validation stably ordered
times, labels, and channels together and recorded that normalization.

## Frozen window scorecards

| Window | Full-session units | Exact-window units | Full rho | Exact rho | Delta rho (95% CI) | Result |
|---|---:|---:|---:|---:|---:|---|
| W2 | 969 | 477 | 0.6770 | 0.6963 | -0.0193 (-0.0429, -0.0073) | mixed/inconclusive |
| W3 | 972 | 485 | 0.6762 | 0.7412 | -0.0650 (-0.0960, -0.0474) | mixed/inconclusive |

For both windows, directional yield and short-ISI checks pass. Short-ISI
equivalence passes. Yield equivalence fails because the full-session and
independently trained window models partition the population very differently.
The rate correlation is a continuity proxy and does not establish identity or
purity.

## Cached-event correspondence

At the frozen ±2-sample tolerance:

| Window | Reciprocal pairs | Median F1 | P90 F1 | Pairs F1 ≥ 0.5 | Matched fraction, full-session | Matched fraction, reference |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 356 | 0.426 | 0.827 | 154 | 0.327 | 0.318 |
| W3 | 356 | 0.377 | 0.755 | 133 | 0.303 | 0.277 |

Only 43 W2 and 22 W3 reciprocal pairs reach F1 ≥ 0.8. Comparator-side best
matches are ambiguous for 385 W2 units and 414 W3 units. The exact-sample
sensitivity is lower still. Training-context effects contribute to these
differences, so the overlap result cannot isolate the remap operator.

## Whole-session descriptive result

The rounded-kriging sorting has rho 0.716 across negative-excursion versus flat
domains among 962 eligible units. Rescue Kilosort has rho 0.637 for all clusters
(532 eligible) and 0.159 for KS-good units (194 eligible). Event rates and unit
counts are not directly comparable because the sorters use different detection,
training, curation, and noise semantics. This descriptive result therefore does
not identify a winner.

The simplest useful next check is cached, identity-aware waveform review of the
high-confidence reciprocal pairs and the large candidate-only population across
flat and negative-excursion domains. It can test whether the added units reflect
stable waveform families or fragmentation without a new sort, RF fit, outer
holdout access, or voltage read. Full promotion would still require evidence
that the W3 continuity decrease and correspondence ambiguity do not represent
longitudinal fragmentation.

## Provenance

- Terminal validation receipt:
  `/home/huklaban5/DARTsort_experiment_scratch/em2g_full_validation_20260929/v4/RECEIPT.json`
- Validation status: complete, exit 0
- Validation wall interval: 269.5 s
- Whole-session descriptive resources: 73.7 CPU s, 81.6 wall s, 8.35 GiB peak RSS
- Voltage bytes read: 0; GPU used by validation: false
- Event-order adapter commit: `ecdd2bc`
- Direct-invocation repair commit: `deef7b5`
