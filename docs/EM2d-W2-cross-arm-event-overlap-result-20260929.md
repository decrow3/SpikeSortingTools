# EM.2d W2 cross-arm event-overlap result — 2026-09-29

## Verdict

**Rounded kriging preserves the exact-DD event populations more closely than
the prespecified unrounded-field contrast.** At the frozen primary tolerance
of ±2 samples, rounded kriging has 329 reciprocal-best unit pairs versus 292
for unrounded kriging. Its reciprocal-pair median event F1 is 0.534 versus
0.402, and those pairs contain 37.5% versus 32.4% of the exact-DD reference
events.

This is supporting evidence for the selected production-order remap. It is a
descriptive association of cached sorter events, not proof of biological unit
identity or purity. Integer unit IDs were never treated as identities, and
ambiguous, nonreciprocal, and unmatched results remain in the packet.

## Frozen comparison

The reference is the rounded exact-DD W2 sort. The selected rounded-kriging
sort and the unrounded-kriging field contrast are compared using maximum
one-to-one ordered event matches. Candidate pairs require at least five
matches. Pair score is event F1, and reported pairs are reciprocal best by F1;
ties prefer more matched events and then the lower integer unit ID. A best
match is marked ambiguous when its F1 lead over the second best is at most
0.05.

| Primary ±2-sample result | Rounded kriging | Unrounded kriging |
|---|---:|---:|
| Reference / comparator units | 477 / 474 | 477 / 486 |
| Reciprocal-best pairs | 329 | 292 |
| Reciprocal F1, p10 / median / p90 | 0.187 / 0.534 / 0.911 | 0.137 / 0.402 / 0.829 |
| Reciprocal pairs with F1 ≥ 0.5 | 174 | 115 |
| Reciprocal pairs with F1 ≥ 0.8 | 71 | 34 |
| Reference events in reciprocal pairs | 37.51% | 32.44% |
| Comparator events in reciprocal pairs | 38.19% | 36.52% |
| Ambiguous reference best matches | 113 | 150 |
| Ambiguous comparator best matches | 110 | 164 |
| Reference units without a candidate | 2 | 4 |
| Comparator units without a candidate | 0 | 11 |

The exact-sample sensitivity check points the same way. Rounded kriging has
328 reciprocal pairs, median F1 0.353, and 27.83% reference-event coverage;
unrounded kriging has 296 pairs, median F1 0.267, and 23.54% coverage. Rounded
kriging also has fewer ambiguous best matches in both directions.

These summaries should not be converted into a matched-cell yield. Event
coincidence can arise from splits, merges, shared detections, or nearby cells,
and the candidate graph is deliberately permissive so that uncertainty is not
discarded.

## Reproducibility and resources

The clean packet is
`testing/outputs/em2d_w2_event_overlap_v2_20260929`. Its manifest SHA-256 is
`6c113865aef908e4add31d07e7808afd63775aa54c1d020fdfb4758270881de3`,
and `RESULT.json` has SHA-256
`640802b180b032f2f5582ab4ec23d243ef33e78837b62bc30ffd710c05e67e3b`.
The rerun reproduced the earlier scientific result byte for byte. It used
114.7 CPU seconds, 115.1 wall seconds, and a 354.1 MiB peak resident set. It
read only the three cached sorting NPZ files: no voltage, RF, outer holdout, or
new sort was used.

The first output directory is retained with `SUPERSEDED.json`. Two launcher
attempts overlapped there after the first tool invocation yielded without a
session identifier, causing its manifest to include stale bookkeeping files.
That packet fails manifest validation and is not evidence; the fresh versioned
packet validates all seven declared products and its completion receipt.

## Pipeline implication

The result closes the cheapest unit-level cross-arm check that was missing
from the arm-local continuity scorecard. Together, the W2 practical
equivalence result, cached W3 lattice transfer, and this cross-arm association
support keeping rounded production-order SpikeInterface kriging as the imec1
remap. The remaining gap is direct waveform-based identity tracking and
full-session behavior, with ambiguity and dropout preserved as required by the
lighthouse policy.
