# Kilosort versus DARTsort unit comparison

Date: 2026-09-10

## Answer

DARTsort recovers clear counterparts for most KS-good units in the unwarped,
AP-rigid, and DARTsort-motion Kilosort arms, but it represents those neurons
with substantially fewer spikes and often multiple DARTsort fragments. The two
LFP-warped Kilosort arms agree much less with DARTsort at the unit level.

This does not establish that either sorter is ground truth. The matched-pair
asymmetry is consistent with DARTsort being more selective or fragmented, with
Kilosort merging or adding contaminating events, or both. Waveform and CCG
inspection of the discrepant families is needed to distinguish those cases.

## Exclusive unit matches

The baseline requires at least 20 spikes per unit, at least 20 exclusive event
coincidences within 0.5 ms and 100 micrometers, at least 50% coverage of the
smaller unit, and a mutual-best unit assignment. Each Kilosort spike depth was
mapped back to its original observed-depth coordinate before comparison with
DARTsort's raw point-source localization.

| Kilosort arm | Eligible KS units matched | KS-good matched | Matched pairs | Median smaller-unit agreement | Shifted-null matches |
|---|---:|---:|---:|---:|---:|
| Unwarped | 105/171 (61.4%) | 28/37 (75.7%) | 105 | 85.4% | 0 |
| AP rigid | 109/172 (63.4%) | **29/36 (80.6%)** | **109** | 84.9% | 0 |
| DARTsort-motion KS | **108/167 (64.7%)** | 29/40 (72.5%) | 108 | **86.4%** | 0 |
| LFP native | 39/183 (21.3%) | 14/44 (31.8%) | 39 | 68.8% | 0 |
| LFP SG25 | 39/170 (22.9%) | 19/43 (44.2%) | 39 | 67.8% | 0 |

The AP-rigid, DARTsort-motion, and unwarped counts are stable across 0.25–1.0
ms tolerances and 0.3–0.7 smaller-unit coverage thresholds. At the baseline
50% threshold they retain 108–109, 107–109, and 105–107 pairs respectively.
Both LFP arms retain 39–40. Three circular shifts (17, 31, and 47 s) produce
zero baseline-qualified matches for every arm.

The LFP gap is not caused by the 100 micrometer gate. At 60, 100, and 200
micrometers, native LFP gives 39, 39, and 35 pairs and SG25 gives 39, 39, and
37. The three other arms remain near 102–113 pairs. Very wide gates eventually
reduce matches because unrelated simultaneous events compete for exclusive
assignment.

## What the matched pairs contain

| Kilosort arm | Median KS event coverage | Median DARTsort event coverage | Median DART/KS rate ratio | KS refractory fraction | DARTsort refractory fraction |
|---|---:|---:|---:|---:|---:|
| Unwarped | 50.2% | 80.8% | 0.70 | 2.72% | 0.52% |
| AP rigid | 35.3% | 80.8% | 0.50 | 2.93% | 0.61% |
| DARTsort-motion KS | 44.2% | **82.8%** | 0.50 | 2.79% | 0.61% |
| LFP native | 51.9% | 50.0% | 1.13 | 1.67% | 0.61% |
| LFP SG25 | 52.0% | 46.2% | 1.17 | 3.09% | 0.30% |

For the strongest three comparisons, approximately 81–83% of a matched
DARTsort unit's events occur in its Kilosort counterpart, but only 35–50% of
the Kilosort unit's events occur in that single DARTsort unit. DARTsort's median
rate is half the Kilosort rate for the AP-rigid and DARTsort-motion pairs. This
is direct evidence that DARTsort usually supplies a selective subset, not an
equivalent full spike train.

DARTsort has a much lower paired refractory burden in 70–74% of the strongest
arm matches. That advantage cannot be interpreted alone: discarding spikes or
splitting one neuron across units can reduce refractory violations without
improving biological accuracy.

## Splits, merges, and unmatched yield

Within 930–1030 s and the AP26–AP373 depth range, DARTsort has 362,418 assigned
spikes and 1,020 units with at least 20 spikes, versus 167–183 Kilosort units.
Only about 10.3–10.7% of eligible DARTsort units enter baseline mutual-best
matches with the strongest three Kilosort arms.

The broader qualified-edge graph is dominated by one-Kilosort-to-several-
DARTsort families. For example:

| Kilosort arm | 1 KS : 1 DART | 1 KS : 2 DART | 1 KS : 3 DART | DART-only |
|---|---:|---:|---:|---:|
| Unwarped | 66 | 38 | 15 | 729 |
| AP rigid | 55 | 44 | 14 | 736 |
| DARTsort-motion KS | 46 | 36 | 19 | 715 |

The 911–915 DARTsort units unmatched by the baseline in those comparisons are
smaller than matched DARTsort units: median 143–147 versus 259–308 spikes.
They are yield differences, not automatically additional neurons. Conversely,
59–66 Kilosort units remain unmatched and have substantial median rates of
8.6–10.0 Hz, so the discrepancy is not confined to tiny Kilosort units.

## Scope and limitations

- DARTsort was run on 384 channels for 300 s; this comparison uses its first
  100 s and retains units by median localization within the matched 348-channel
  depth range.
- Final DARTsort labels come from later refinement than `matching1.h5`.
  Localization rows were accepted only after confirming identical row channels
  and at most one sample of final template realignment.
- DARTsort has no label directly equivalent to `KSLabel=good`.
- Spike-time/depth agreement is strong identity evidence, but waveform and CCG
  arbitration remain necessary for split/merge decisions.

## Artifacts

- Output: `testing/outputs/luke_kilosort_dartsort_unit_comparison_v2`
- Summary: `summary.json`
- Baseline matches: `matched_unit_pairs.csv`
- Qualified and unqualified unit edges: `unit_pair_edges.csv`
- Time/coverage sensitivity: `threshold_sensitivity.csv`
- Depth-gate sensitivity: `depth_sensitivity.csv`
- DARTsort unit metrics: `dartsort_unit_metrics.csv`
- Figure: `01_kilosort_dartsort_unit_comparison.png`
- Reproducible analysis: `testing/luke_kilosort_dartsort_unit_comparison.py`
- Preserved failed integrity-gate attempt: `testing/outputs/luke_kilosort_dartsort_unit_comparison_v1/failure.json`
