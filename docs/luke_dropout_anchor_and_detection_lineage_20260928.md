# Independent dropout anchors and detection-stage lineage

Date: 2026-09-28. Status: **complete; cluster 553 is the current development target.**

## Decision

The cached rescue-good amplitude census contains two cases with material loss
against an independent legacy event anchor: clusters 21 and 553. Exact lineage
places both deficits at or before Kilosort's saved `full_st` universal-detection
table, not in retained-output identity handling or final curation.

Cluster 553 is the stronger pipeline-development target because it is an
interior unit near 3,020 um, has clean existing contamination evidence, and
loses 37.2 percentage points of independent-anchor coverage at `full_st`.
Sparse voltage confirms that the missed anchor times retain a recognizable but
weaker waveform. This supports developing one bounded detection/preprocessing
candidate. It does not reopen the closed 8/8 or 9/9 Kilosort threshold arms and
does not authorize a new sort by itself.

## Seven-case independent-anchor screen

The screen included every rescue `KSLabel=good` row in the hash-frozen baseline
recovery census. For each target, candidate labels were fixed by template peak
depth within 100 um. A legacy anchor was selected using only the reference
phase and had to pass 20 events, 25% observed correspondence, 10 percentage
points above fixed 137/431/997 s circular-shift nulls, and target-label
dominance. Maximum-cardinality one-to-one pairing used a 0.5 ms tolerance.

| cluster | fitted missingness change | anchor passes | reference match | failing match | independent loss | advances |
|---:|---:|:---:|---:|---:|---:|:---:|
| 21 | +19.52 pp | yes | 99.31% | 88.63% | **10.68 pp** | yes |
| 36 | +13.40 pp | yes | 99.60% | 99.81% | -0.21 pp | no |
| 91 | +34.84 pp | yes | 90.82% | 80.85% | 9.97 pp | no; frozen near miss |
| 452 | +19.04 pp | yes | 97.17% | 94.96% | 2.21 pp | no |
| 472 | +25.32 pp | no | 90.30% | 92.99% | -2.68 pp | no |
| 553 | +31.57 pp | yes | 93.54% | 57.00% | **36.55 pp** | yes |
| 652 | +20.35 pp | no | 97.66% | 92.79% | 4.86 pp | no |

Five of seven cases have a qualified independent anchor. Only 21 and 553 pass
the frozen 10-point material-loss rule. Cluster 91 remains a failed threshold,
not a discretionary promotion.

## Cluster 21 localization

Legacy cluster 12 supplies 2,042 reference and 2,437 failing-phase events. The
fraction assigned to rescue 21 falls from 74.44% to 55.40%, but no alternate
rescue label reaches the frozen redistribution rules. The largest alternate
fractions are only 5.4--5.9%, with template cosines near zero or negative.

Exact final-curation lineage maps raw cluster 21 directly to final cluster 21,
with no merge. Duplicate removal is 9/2,008 events (0.45%) in the reference
phase and 8/2,007 (0.40%) in the failing phase.

The corrected detection-template audit uses
`ops.yc[ops.iU[detection_template_id]]`. Anchor coverage is 99.22% in reference
and 88.35% in failure at `full_st`; the exact same fractions appear in raw
sorter and curated outputs. The 10.87-point deficit is therefore already
present at or before universal detection. Cluster 21 lies at the 60 um probe
edge, so it is retained as corroboration rather than the first general-purpose
candidate target.

The first detection-lineage execution is invalid and preserved as a setup
failure: it mapped 1,332 detection-template IDs through the 731-row final
template bank. V2 uses the saved `iU` detection-template-to-channel map; two
known-answer tests cover the mapping and stage decision.

## Cluster 553 localization

Legacy cluster 425 supplies 2,215 reference and 4,467 failing-phase events.
Using the curated target depth of 3,020 um, independent-anchor coverage is:

| stage | reference | failing | loss |
|---|---:|---:|---:|
| `full_st` spatial detections | 97.02% | 59.86% | **37.16 pp** |
| raw retained sorter output | 97.02% | 59.79% | **37.23 pp** |
| final curated output | 93.63% | 57.82% | **35.81 pp** |

The target is raw cluster 577 with no merge. Final duplicate removal is stable:
116/2,115 (5.48%) in reference and 115/2,114 (5.44%) in failure. Neither identity
redistribution nor final curation explains the loss.

The first 553 stage execution is invalid and preserved: it looked up final label
553 in the raw sorter template namespace, obtaining 2,840 um. V2 takes target
depth from the curated template namespace and detection-template depth from
the raw `iU` mapping.

## Sparse voltage at missed anchor times

The waveform audit used the already preprocessed rescue recording, fixed
channels 294--309, 61 samples, and at most 200 evenly spaced events in each
partition. It requested 650,016 scalar samples and did not scan or copy a
continuous voltage interval.

| partition | available / sampled | median waveform PTP | cosine to reference |
|---|---:|---:|---:|
| reference, matched by `full_st` | 2,149 / 200 | 112.5 µV | 1.000 |
| reference, missed by `full_st` | 66 / 66 | 96.1 µV | 0.895 |
| failing, matched by `full_st` | 2,674 / 200 | 91.4 µV | 0.927 |
| failing, missed by `full_st` | 1,793 / 200 | 77.3 µV | **0.865** |

The failing unmatched waveform retains the frozen minimum cosine of 0.80 and
has 68.75% of reference PTP, below the frozen 90% attenuation boundary. The
result is `subthreshold_like_waveform_supported`: a recognizable waveform is
weaker at times when the current universal detector misses the independent
anchor. Median similarity does not establish every unmatched event as the same
neuron, and the legacy anchor is not ground truth, but the population result is
strong enough to motivate one bounded detection/preprocessing candidate.

## Reproducibility and next action

- Screen config:
  [`luke_dropout_anchor_screen.v1.json`](../configs/luke_dropout_anchor_screen.v1.json),
  SHA-256 `3eda37c45db05d00c1848bce90dc8d975e6ec49960a0989afd6172c1a5019235`.
- Cluster 21 corrected detection config:
  [`v2`](../configs/luke_cluster21_detection_lineage.v2.json), SHA-256
  `4b6bead2ccc84d0e3a0f498b0c8dc533934a9027792d04cf464184f659dd1274`.
- Cluster 553 corrected detection config:
  [`v2`](../configs/luke_cluster553_detection_lineage.v2.json), SHA-256
  `e9f23467f18f8c383f0ebb3e38d7f264c693795ab6d8fd0ae6be9f9d40b14534`.
- Missing-waveform config:
  [`luke_cluster553_missing_anchor_waveforms.v1.json`](../configs/luke_cluster553_missing_anchor_waveforms.v1.json),
  SHA-256 `180912933c3202d28344ed9b49bc327c7130bed99dceca14292b9f5ed0972e15`.
- Implementations:
  [`luke_dropout_anchor_screen.py`](../testing/luke_dropout_anchor_screen.py),
  [`luke_cluster21_detection_lineage.py`](../testing/luke_cluster21_detection_lineage.py), and
  [`luke_cluster553_missing_anchor_waveforms.py`](../testing/luke_cluster553_missing_anchor_waveforms.py).

The simplest next check is to reuse the extracted partitions to compare local
baseline noise and artifact proximity for matched and missed events. That can
distinguish pure attenuation from a noise/artifact-driven SNR loss; it still
cannot select a production detector. Only after that direct check should one
specific preprocessing or detection operation receive a frozen bounded
candidate contract on cluster 553 plus healthy controls.

No sort was launched and no production output, threshold, or label changed.
