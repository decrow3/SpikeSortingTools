# Luke 2025-08-04 motion-correction candidate and spike-recovery test

## Technical summary

The best-supported non-rigid estimate for the 8,160--8,280 s pathological
window currently uses **300 µm spatial windows sampled every 200 µm**. Its
residual field is reproducible across random peak halves, invariant to denser
100 µm sampling, supported by the independent decentralized estimator, and
more consistent across the simultaneous probes than the 600 µm family.

Applying this field did **not** improve recovery of the prespecified missed
spikes. With identical conditioning, Kilosort settings, and disabled claim
mask, no external correction recovered 23/27 reviewed neural events (85.2%),
the current 150/100 DREDGE correction recovered 19/27 (70.4%), and the selected
300/200 field recovered 18/27 (66.7%). Every event recovered after the selected
correction was also recovered without correction.

The rigid-only test separates the two main failure modes. It recovered 20/27
events (74.1%): better than either non-rigid correction, but worse than no
correction, with all three losses coming from events recovered in the
no-correction condition and no compensating gains. Rigid-only also returned
learned detections and cross-unit coincidence almost exactly to baseline
(550k and 36.6%). The non-rigid residual warp is therefore the leading driver
of the detection/collision explosion, while external resampling or the rigid
trace itself still reduces sensitivity to real events.

The zero-displacement identity control now isolates the application path. It
passed the same float interpolation, `force_zeros` border handling, and int16
export as the motion-corrected recordings, but its 2.764 GB trace binary was
byte-for-byte identical to the no-correction input. Its Kilosort outputs were
also byte-for-byte identical for spike times, clusters, positions, learned
detections, KS labels, and contamination. Thus the application machinery does
not damage the data at zero motion; the rigid losses require nonzero
displacement, and the non-rigid explosion requires depth-dependent
displacement.

Kilosort's internal rigid correction on the untouched binary is intermediate:
22/27 events recovered, 549k learned detections, and 36.8% coincidence. It
avoids overdecomposition and outperforms the full external rigid correction,
but remains worse than no correction, with 53.1% median contamination and 75
KS-good units. Its estimated displacement broadly agrees with the inverted
DREDGE rigid trajectory (r=0.80), yet has 3.9× the temporal variance and ten
adjacent 2 s jumps larger than 50 µm. It is not production-ready without
stabilizing its motion estimate.

A rigid-gain sweep finds the first correction that improves the fixed-event
result: 0.25× and 0.5× both recover 24/27, compared with 23/27 at zero gain.
The 0.25× condition preserves 94 KS-good units and has no collision inflation,
whereas 0.5× collapses to 75 good units and 63.7% median contamination. At
0.75× recovery returns to baseline and only 60 units remain good; at full gain
recovery falls to 20/27. Thus motion information can help, but the current
displacement amplitude is much too aggressive for voltage resampling.

That apparent 0.25× benefit did **not** replicate in the independent 240 s
shared-template window. No correction recovered 33/35 visually neural missed
events and 12/12 automatically neural-like missed events; 0.25× rigid recovered
32/35 and 12/12, respectively. It rescued E0068 but lost E0032 and E0143. The
corrected sort also fell from 130 to 115 KS-good units while median
contamination rose from 26.4% to 36.7%. No correction is therefore the current
production baseline; reduced rigid correction remains mechanistically
interesting, but is no longer a production finalist.

The voltage-level audit identifies a direct application cost. Across all 126
reviewed events, 0.25× rigid resampling retained a median 96.6% of local peak
magnitude; visual neural misses retained 95.7%, and the automatic neural-like
subset retained 97.7%. Peak retention decreased strongly with applied shift
(Spearman r=-0.83), even though the median applied displacement was only 1.5
µm. This systematic attenuation provides an upstream mechanism for the loss of
template quality and marginal events.

That cost is strongly geometry- and kernel-dependent. With SpikeInterface's
default 20 µm kriging scale, a 2 µm shift leaves a maximum interpolation weight
of 0.914 on Luke's four-column, 20 µm-pitch geometry, versus 0.962 on a Yates
35 µm-pitch shank. Reducing the scale to 10 µm improved median reviewed-event
peak retention from 96.6% to 97.7% and RMS retention from 98.4% to 99.2%.
In the independent 240 s window this restored 34/35 visual-event recovery and
127 KS-good units, compared with 33/35 and 130 without correction and 32/35
and 115 at the default scale. However, contamination remained 9.9 percentage
points above baseline and there was no gain on the stricter 12-event cohort.
The narrower kernel validates interpolation attenuation as a mechanism but is
not yet a production winner.

The failure also generalizes beyond DREDGE. Applying the independently
estimated MEDiCINe non-rigid field with the narrower 10 µm kernel recreated the
same pathological signature: 1.231M learned detections, 72.1% coincidence,
68.3% median contamination, only 41 KS-good units, and 19/27 reviewed events
recovered. The common causal factor is therefore depth-dependent voltage
warping, not a DREDGE-specific optimizer failure.

Matched 120 s comparisons nevertheless show that Luke genuinely moves more
than the known-good Yates session. Across non-overlapping Yates windows,
MEDiCINe's median rigid excursion was about 3.0 µm per shank, compared with
13.1 and 14.7 µm on Luke's two probes. Kilosort-style estimates gave 0.75--1.0
µm median Yates excursions versus 5.8 and 10.1 µm for Luke. Luke therefore has
a real motion problem; the present voltage-resampling remedies are simply
more damaging than leaving that motion uncorrected.

The selected full field recreated the pathological sorting signature: 1.293M
learned-template detections, 73.2% cross-unit coincidence, 50.1% median
contamination, and 79 KS-good units. Rigid-only had 62.1% median contamination
and 71 KS-good units despite baseline-like spike counts and coincidence. This
suggests a second, subtler clustering/template-quality degradation that cannot
be explained by overdecomposition alone.

A direct geometry audit now substantially downgrades channel mapping as the
cause. All 384 raw channel IDs and coordinates agree row-for-row with the saved
processed metadata, and all 384 coordinates agree exactly with Kilosort's
`channel_positions.npy`. The acquisition's saved-channel subset is AP0--AP383
in order, and its `snsGeomMap` independently reproduces the same four-column
lateral stagger and 20 µm depth progression.

The matched Luke--Yates detection audit changes the interpretation of the
original cross-session deficit. Luke has 3.41 times as many contacts per
millimetre of sampled depth as Yates, so per-channel normalization made Luke
look artificially sparse. Per physical millimetre, the current no-motion Luke
sort has approximately the same unit density as Yates (63--76 versus 66
units/mm), higher learned and final spike density, and essentially the same
cross-unit coincidence. Luke does contain about half as many strong negative
input events per millimetre, but Kilosort recovers 94--96% of them in the Luke
pathological/current conditions versus 97% in Yates. Thus the large residual
"missed-spike deficit" is not a broad failure of Kilosort to claim obvious
events once external motion correction is removed.

Upstream reference and CAR ablations are also close to null. Removing local
reference, substituting a global reference, or disabling Kilosort's second CAR
does not improve the fixed-event result. Letting Kilosort perform the only
high-pass/CAR pass is more interesting: in an independent 240 s window it
reduces learned detections by 10.9%, raises KS-good units from 130 to 168,
reduces coincidence, and recovers 120/126 rather than 119/126 reviewed events.
However, nearby highly similar template pairs increase from 62 to 170; about
two thirds of the raw unit-count increase is explainable as added similarity-
graph redundancy. Single-pass conditioning is therefore a useful full-session
candidate, not a verified production winner.

## The original per-channel Luke--Yates deficit was denominator-confounded

The comparison below replays each saved Kilosort input through the exact saved
Kilosort high-pass, CAR, and whitening transform. Events are deduplicated in
physical time and depth and all density metrics use sampled probe depth, not
channel count. Luke pathological is the original 120 s audit window; Luke
current and single-pass use 60 evenly sampled 2 s batches from the independent
240 s window; Yates uses 60 evenly sampled batches across its known-good run.

| Metric | Luke pathological | Luke current | Luke single-pass | Yates known-good |
|---|---:|---:|---:|---:|
| Contacts/mm | 100.5 | 100.5 | 100.5 | 29.5 |
| Strong negative events/mm/s | 126.8 | 118.8 | 99.7 | **238.0** |
| Strong-event final recovery | 93.7% | 96.1% | 93.2% | **96.9%** |
| Learned detections/mm/s | 1,192.9 | **1,647.7** | 1,456.8 | 667.6 |
| Final spikes/mm/s | 1,144.0 | **1,602.3** | 1,434.9 | 666.1 |
| Units/mm | 63.4 | 76.4 | **102.9** | 66.4 |
| Cross-unit coincidence | 36.5% | 37.2% | **35.1%** | 36.0% |

The remaining two-fold difference in strong negative event density is real in
the Kilosort input and persists under both Luke conditioning graphs. It may
reflect acquisition, probe geometry, laminar sampling, biology, or an earlier
shared preprocessing step; this audit does not identify which. Crucially, it
is a density difference, not evidence that the final Luke sorter discards most
of the high-amplitude events it receives.

A subsequent matched **raw-voltage** audit now narrows this statement. Before
referencing, Luke has more rather than fewer large 300--6000 Hz extrema than
Yates, together with a substantially larger shared high-frequency component.
After a 100 µm local median reference and a fixed 75 µV threshold, imec0
approaches Yates's negative-event density, while imec1 retains a negative
deficit paired with a six- to ten-fold positive-event excess. The Kilosort-input
gap therefore does not pre-exist as a simple lack of large raw AP events. It is
strongly shaped by common-mode removal, probe geometry, and polarity.

These raw results are an interim diagnostic, not a completed biological
comparison. The fixed-radius reference contains different contact counts on
the two probe types, the shank-median sensitivity run was paused before
completion, and corrected spatial-compactness metrics still need to be rerun.
The authoritative paused handoff is
`docs/luke_yates_raw_voltage_audit_notes.md`.

Luke's Kilosort inputs also contain many more positive-polarity extrema and
roughly three quarters of Luke templates are positive-dominant, versus about
30% in Yates. Reference/CAR ablations do not remove this polarity difference,
and Kilosort assigns 97--99% of the strong positive events to final spikes.
The sign/morphology contrast is therefore an acquisition/session clue rather
than the cause of the reviewed misses.

## Single-pass conditioning improves yield but also increases redundancy

The 120 s causal ablations show no benefit from changing the reference graph.
The single-pass condition retains saturation blanking and bad-channel
interpolation, but omits the upstream 300--6000 Hz bandpass and reference so
Kilosort performs the only high-pass/CAR operation.

| Pathological-window condition | Reviewed recovery | Learned detections | KS-good | Coincidence | Median contamination |
|---|---:|---:|---:|---:|---:|
| Current no motion | **23/27** | 562,417 | 95 | 36.5% | **38.7%** |
| Bandpass, no reference | **23/27** | 579,298 | 90 | 36.4% | 44.1% |
| Global reference | 22/27 | 574,693 | 91 | 36.3% | 47.4% |
| Local reference, no KS CAR | **23/27** | 561,947 | 96 | 36.2% | 40.1% |
| Single KS preprocessing | 22/27 | **517,187** | **114** | **35.1%** | 38.4% |

The independent 240 s replication is more favorable to single-pass
conditioning, but the unit-structure audit shows why raw unit count cannot be
taken at face value.

| Shared-window metric | Current conditioning | Single KS preprocessing |
|---|---:|---:|
| Visual-neural recovery | 33/35 | **34/35** |
| Strict automatic-neural recovery | **12/12** | **12/12** |
| All reviewed recovery | 119/126 | **120/126** |
| Learned detections | 1,408,807 | **1,255,885** |
| KS-good units | 130 | **168** |
| Cross-unit coincidence | 37.1% | **35.0%** |
| Median contamination | **26.4%** | 29.9% |
| Nearby template pairs, similarity >=0.8 | **62** | 170 |
| Similarity-graph components | 240 | **273** |

The extra 101 raw units come with 68 additional redundant nodes in the nearby
template-similarity graph. Effective graph components rise by only 14%, while
good-unit components rise by 23%. This is consistent with a mixture of better
separation and over-splitting. A production choice requires matched merging,
unit-family continuity, and full-session validation rather than choosing the
condition with the largest unmerged unit count.

## The supported non-rigid scale is approximately 300--400 µm

For the selected 300/200 field, residual-field split-half correlations were
0.953 on imec0 and 0.928 on imec1. Changing only the sampling step from 100 to
200 µm gave correlations of 0.986 and 0.999, showing that 100 µm sampling adds
little independent structure. Agreement with decentralized 300/200 was 0.790
and 0.893. The simultaneous-probe residual correlation was 0.609.

The 600 µm family was also internally reproducible, but its residual structure
changed substantially on imec0 relative to the 300 µm family and its
simultaneous-probe residual correlation was -0.129. It appears to smooth away
supported local structure rather than providing a better common estimate.

The selected field has a residual spatial correlation length near 400 µm. On
imec1 its median and P95 non-rigid spreads were 23.8 and 46.8 µm; on imec0 they
were 17.4 and 44.2 µm. These values describe the fitted field and should not be
interpreted as ground-truth tissue displacement.

## Channel-to-geometry mapping is exact across all three representations

The raw SpikeGLX extractor, saved preprocessed metadata, and Kilosort geometry
each contain 384 neural channels. Raw and processed channel-ID sequences are
identical. Every row has the same `(x, y)` coordinate in all three
representations: the maximum absolute coordinate difference is 0 µm, the best
geometric assignment is the identity for 384/384 rows, and the recording time
origins are identical at 3,057.678 s.

The `.meta` file provides an independent acquisition-level check rather than
merely comparing copies of the same table. `snsSaveChanSubset` is
`0:383,768`, meaning the 384 AP channels are saved sequentially followed by the
sync channel. `snsGeomMap` cycles through the expected staggered x positions
with y increasing by 20 µm; after SpikeInterface's harmless x-origin shift,
these are exactly the stored 0/16/32/48 µm coordinates.

The attached script's raw-waveform alarm is not valid evidence for a mapping
error on this dataset. Its first pass uses unconditioned raw voltage and is
dominated by row 216. After bandpass and local referencing it is dominated by
row 191, which is precisely the single channel flagged as bad upstream
(similarity -0.853) and interpolated before sorting. After reproducing that
interpolation, several high-amplitude “KS-good” units from the pathological
sort remain noncompact, but the same recurring row-216 signal also affects the
no-correction control. These units are therefore not an independent biological
ground truth; the result is more consistent with common/artifactual events or
bad unit labels than with a hidden geometry permutation.

## Full-amplitude motion correction does not improve event recovery

The downstream comparison used the identical 120-second current-conditioned
imec1 recording. The selected field was applied with float interpolation,
`force_zeros`, and an `int16` saved result, matching the production application
path. Kilosort parameters and probe geometry were held fixed, internal drift
correction was disabled except in the explicitly labelled KS-internal condition,
and the claim mask was off in every condition.

| Metric | No correction | Zero identity | KS rigid | Current DREDGE | External rigid 1× | DREDGE 300/200 | MEDiCINe NR, σ10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reviewed neural misses recovered | **23/27** | **23/27** | 22/27 | 19/27 | 20/27 | 18/27 | 19/27 |
| Learned-template detections | 562,417 | 562,417 | **548,693** | 1,298,050 | 550,252 | 1,293,487 | 1,230,786 |
| Final spikes | 544,553 | 544,553 | **535,103** | 905,605 | 538,563 | 847,162 | 802,094 |
| Cross-unit coincidence | **36.5%** | **36.5%** | 36.8% | 77.0% | 36.6% | 73.2% | 72.1% |
| Median contamination | **38.6%** | **38.6%** | 53.1% | 51.5% | 62.1% | 50.1% | 68.3% |
| KS-good units | **95** | **95** | 75 | 74 | 71 | 79 | 41 |

The identity result is an unusually strong negative control: all reported
metrics match because the underlying trace binary and checked sorter outputs
are exact byte matches, not merely statistically similar reruns.

Rigid-only removes 96--98% of the non-rigid condition's excess learned and
final detections and restores coincidence to within 0.05 percentage points of
the no-correction value. Yet it loses E0009, E0011, and E0163 relative to no
correction and rescues no event that no correction missed. The no-correction
condition therefore Pareto-dominates rigid-only on the reviewed-event cohort
and the main quality guardrails.

## Reduced rigid gain reveals useful motion but a narrow safety margin

The same DREDGE depth-mean trace was applied at gains from zero to one, with
every other processing and sorting choice fixed.

| Rigid gain | Recovery | Learned detections | Coincidence | Median contamination | KS-good |
|---:|---:|---:|---:|---:|---:|
| 0 | 23/27 | 562,417 | 36.5% | **38.6%** | **95** |
| 0.25 | **24/27** | 570,501 | 36.1% | 49.1% | 94 |
| 0.50 | **24/27** | 560,861 | 36.3% | 63.7% | 75 |
| 0.75 | 23/27 | 554,208 | **35.9%** | 61.9% | 60 |
| 1.00 | 20/27 | 550,252 | 36.6% | 62.1% | 71 |

At 0.25×, every event recovered without correction remains recovered and
E0103 is additionally rescued. The same paired-event benefit persists at
0.5×, but the unit-quality safeguards deteriorate sharply. The lack of a
collision increase at any rigid gain confirms that spatial gradients, not
rigid interpolation alone, cause the peeling explosion. The gain-dependent
contamination and good-unit loss reveal a separate template/clustering cost of
nonzero resampling that aggregate spike counts do not expose.

## Narrower kriging reduces damage but does not beat no correction safely

The independent replication used Luke imec1 from 7,095--7,335 s, twice the
duration of the selection window. Motion was re-estimated on that window with
the selected 300/200 DREDGE configuration. The current-conditioned recording
was then sorted either unchanged or after applying 0.25× its depth-mean rigid
trace. Both sorts used the same claim-off Kilosort settings and recovery rules.

| Metric | No correction | 0.25× rigid, σ20 | 0.25× rigid, σ10 |
|---|---:|---:|---:|
| Visual neural misses recovered | 33/35 (94.3%) | 32/35 (91.4%) | **34/35 (97.1%)** |
| Automatic neural-like misses recovered | **12/12 (100%)** | **12/12 (100%)** | **12/12 (100%)** |
| All reviewed events recovered | 119/126 (94.4%) | 118/126 (93.7%) | **121/126 (96.0%)** |
| Learned-template detections | 1,408,807 | 1,404,047 | 1,407,028 |
| Final spikes | 1,382,654 | 1,384,197 | 1,388,478 |
| Cross-unit coincidence | **37.1%** | 37.3% | 37.4% |
| Median contamination | **26.4%** | 36.7% | 36.3% |
| KS-good units | **130** | 115 | 127 |

At σ10, every visual event recovered without correction remained recovered and
E0068 was added; across all reviewed events E0197 was also gained and none were
lost. Both gains were ambiguous or low-SNR and failed the strict automatic
neural-like screen. The improved recovery and 12-unit restoration relative to
σ20 are real evidence for the attenuation mechanism, while the unchanged
automatic cohort and elevated contamination prevent a production claim.

## A fixed interpolation scale is not probe-neutral

SpikeInterface's default kriging kernel uses `sigma_um=20`. Direct evaluation
of its interpolation matrix shows that the same sub-channel displacement mixes
Luke's dense four-column geometry more aggressively than a Yates shank.

| Shift | Luke max weight, σ20 | Yates max weight, σ20 | Luke max weight, σ10 |
|---:|---:|---:|---:|
| 1 µm | 0.957 | 0.981 | 0.973 |
| 2 µm | 0.914 | 0.962 | 0.944 |
| 4 µm | 0.828 | 0.918 | 0.877 |
| 6 µm | 0.743 | 0.869 | 0.801 |

This makes the default spatial bandwidth a probe-dependent conditioning
choice, not a neutral implementation detail. A fair cross-probe correction
must calibrate the kernel against actual site pitch and stagger, then verify
waveform retention after integer export.

## Luke motion exceeds Yates motion under two estimator families

The comparison used the same 120 s duration: the prespecified Luke pathological
window and non-overlapping Yates windows summarized by their median and maximum.

| Estimator | Recording | Rigid excursion | Median nonrigid spread | P95 spread | P99 speed |
|---|---|---:|---:|---:|---:|
| MEDiCINe | Luke imec0 | 13.1 µm | 3.4 µm | 8.8 µm | 0.87 µm/s |
| MEDiCINe | Luke imec1 | 14.7 µm | 8.8 µm | 19.0 µm | 1.30 µm/s |
| MEDiCINe | Yates shank 1, median | 3.0 µm | 2.8 µm | 7.5 µm | 0.36 µm/s |
| MEDiCINe | Yates shank 2, median | 3.1 µm | 4.9 µm | 8.3 µm | 0.40 µm/s |
| Kilosort-style | Luke imec0 | 5.8 µm | 4.0 µm | 10.2 µm | 3.41 µm/s |
| Kilosort-style | Luke imec1 | 10.1 µm | 6.0 µm | 9.1 µm | 3.53 µm/s |
| Kilosort-style | Yates shank 1, median | 0.75 µm | 0.0 µm | 1.0 µm | 0.25 µm/s |
| Kilosort-style | Yates shank 2, median | 1.0 µm | 1.0 µm | 1.0 µm | 0.25 µm/s |

Yates occasionally shows sizeable MEDiCINe differential spread, especially on
shank 2, so nonrigid motion is not uniquely Luke-like. The robust distinction
is Luke's much larger rigid excursion and speed. Estimator settings and source
preprocessing are not perfectly matched across legacy sessions, so these are
same-family descriptive comparisons rather than calibrated ground truth.

The paired comparison is more informative than the one-event aggregate
difference. Thirty-one of 35 visual-neural events were recovered by both
conditions and one by neither. Correction rescued E0068, but caused losses at
E0032 and E0143. None of these three passed the stricter automatic neural-like
screen, whose 12 events were recovered in both conditions. Thus the small
selection-window gain was not stable across events or windows, while the
unit-quality cost reproduced. There is still no collision explosion under
rigid correction; learned detections, final spikes, and coincidence remain
near baseline.

Direct voltage inspection of the three switching events shows attenuation in
all cases, not selective enhancement of the rescued event. Local peak magnitude
changed from 23 to 22 counts for E0032, 250 to 233 for E0068, and 26 to 22 for
E0143; local RMS fell by 4--8%. E0068 nevertheless switched from absent to
present, so sorter recovery is not a monotonic readout of raw amplitude. It is
also a particularly diffuse event (134 active channels, 9.9% local energy,
common-mode ratio 2.22) that failed the automatic neural-like screen. All three
switching events failed that stricter screen. The event-level exchange is best
treated as marginal assignment instability; the replicated quality-cost signal
is the 15-unit KS-good loss and 10.3-point contamination increase.

The effect is systematic beyond those three events. For all 126 reviewed
events, corrected/uncorrected local peak magnitude had a median of 96.6%; the
35 visual neural misses had a median of 95.7%, and the stricter 12-event subset
had a median of 97.7%. Median local RMS retention was 98.4%, 97.9%, and 98.9%,
respectively. Absolute applied displacement predicted peak attenuation with
Spearman r=-0.83 (descriptive p=3.6e-33), across a modest 0--6 µm range. The
comparison takes the maximum across a ±120 µm local footprint, so this is not
merely energy moving from the nominal peak channel to its neighbor. Spatial
resampling itself smooths the voltage enough to matter for marginal templates.

## Kilosort's internal rigid estimate is motion-like but discontinuous

After sign alignment and median centering, Kilosort's internal trace correlates
0.80 with the DREDGE depth mean, so it is not unrelated noise. However, its
standard deviation is 37.0 µm versus 9.45 µm for DREDGE, and its 128.5 µm
peak-to-peak range is three times DREDGE's 41.6 µm. Ten of 59 adjacent 2 s
steps exceed 50 µm; the largest is 102.5 µm. DREDGE has no step above 50 µm
and only one above 20 µm on the same sampling grid.

Inspection of the local Kilosort code explains why the nominal
`drift_smoothing` parameter may not solve this by itself: temporal Gaussian
smoothing is applied only to the final fine-correlation tensor, after nine
coarse iterative alignments whose discrete shifts are accumulated without
temporal regularization. Luke's weak or repetitive spatial fingerprint can
therefore jump between coarse alignment modes before the smoothing stage.

## Misses are concentrated at larger fitted displacement and gradient

Among the 27 reviewed events, the nine missed after selected correction had a
median absolute fitted displacement of 5.45 µm versus 4.26 µm for recovered
events. Their median absolute spatial gradient was 0.016 versus 0.008
µm/µm. One-sided Mann--Whitney comparisons gave descriptive p-values of
0.021 and 0.014, respectively.

This association is consistent with correction damage increasing where the
warp is larger or more differential. It is not a confirmatory test: the cohort
is small, the comparison was motivated after observing the recovery result,
and motion magnitude and gradient are correlated.

## Scope and metric definitions

- **Window:** Luke 2025-08-04 imec1, frame-relative 8,160--8,280 s.
- **Independent replication window:** imec1, 7,095--7,335 s, with 35 visual
  neural unmatched events and a stricter 12-event automatic neural-like subset.
- **Fixed cohort:** 27 blinded-review events labelled neural that were locally
  unmatched to the original patched sorting.
- **Recovery:** a final sorted spike within 0.5 ms and 100 µm of the fixed raw
  event.
- **Residual non-rigid field:** displacement remaining after least-squares
  removal of rigid and linear-with-depth components on a common 2 s by 200 µm
  grid.
- **Split-half reproducibility:** correlation between fields independently
  estimated from a deterministic random half of the detected AP peaks.
- **Cross-unit coincidence:** fraction of final spikes participating in a pair
  from different clusters within 0.5 ms and 75 µm.

## Methodology

`testing/luke_motion_scale_sweep.py` performs cache-safe motion estimation and
now includes a dedicated 300/200 split-half validation candidate.
`testing/luke_motion_candidate_sort.py` applies the selected imec1 field, its
depth-mean rigid component at configurable gain, or an all-zero field to the fixed
current-conditioned recording and runs the unchanged claim-off Kilosort
diagnostic. The rigid field repeats the arithmetic mean across the 18 native
depth bins at each time bin, retaining the same temporal trace while setting
every spatial derivative to zero. The identity field retains the identical
motion time/depth grid while setting every displacement value to zero.
The internal-rigid condition instead reads the untouched no-correction binary
and enables Kilosort `do_correction=True, nblocks=1`.
`testing/luke_motion_candidate_results.py` produces the spatial
validation, paired-event recovery, event-gradient analysis, and figures.
`testing/luke_motion_replication_sort.py` independently estimates, applies,
sorts, and scores the shared-window no-correction/0.25× contrast;
`testing/luke_motion_replication_results.py` renders its paired diagnostics.
`testing/luke_motion_replication_event_traces.py` traces the three switching
events through the saved uncorrected and corrected voltage.
`testing/luke_motion_replication_voltage_effects.py` applies the same audit to
all 126 reviewed events and relates attenuation to applied displacement.
`testing/luke_yates_interpolation_kernel_audit.py` evaluates the exact kriging
weights on Luke and Yates probe geometries and measures event-level sigma
sensitivity. `testing/luke_yates_motion_comparison.py` compares non-overlapping
120 s windows under MEDiCINe and Kilosort-style estimators. The candidate sorter
also applies the cached MEDiCINe native nonrigid field with a 10 µm kernel.
`testing/luke_yates_detection_stage_audit.py` replays the exact saved Kilosort
input transform and measures physically deduplicated events, final-event
recovery, and sorter outputs per sampled millimetre. The source datasets retain
the per-batch numerators, denominators, depth exposure, and polarity split.
`testing/luke_upstream_sorter_ablation.py` changes one conditioning stage at a
time on the fixed 120 s recording. `testing/luke_motion_replication_sort.py`
also materializes and sorts the single-pass condition in the independent 240 s
window. `testing/luke_preprocessing_unit_structure_audit.py` compares temporal
presence and nearby-template similarity graphs for the current and single-pass
sorts.

Production defaults were not modified.

## Limitations and robustness

- Selection is based on one prespecified pathological window. The spatial
  scale should be checked in the shared-motion and quiet windows.
- Cross-probe agreement supports a shared component but the probes sample
  different tissue, so perfect non-rigid agreement is neither expected nor a
  ground-truth criterion.
- The identity control rules out the application path at exactly zero
  displacement, but does not prove that interpolation is harmless at nonzero
  shifts. Rigid-only still jointly evaluates actual displacement and
  nontrivial spatial interpolation.
- The 0.25× gain result was one event on a 27-event descriptive cohort and did
  not replicate in a second window. The new result is still a single-window,
  single-probe replication, but it is larger and agrees with the unit-quality
  warning from the selection window.
- The internal Kilosort/DREDGE trace comparison aligns sign and removes each
  estimator's median only for shape comparison. Kilosort applies its stored raw
  `dshift`, including its registration gauge.
- Claim masking was intentionally disabled to expose pathological peeling; a
  production run may use a safeguard, but it should not be used to hide an
  upstream failure.
- The fixed-event cohort is descriptive and small. Unit count and KS labels are
  not ground truth, which is why recovery and collision safeguards are reported
  together.
- The legacy Yates preprocessed binary is strongly transformed relative to its
  raw binary, and its timestamps indicate preprocessing followed motion
  estimation, but the exact preprocessing implementation is unavailable. The
  Luke--Yates comparison therefore establishes robust relative motion scale,
  not a perfectly graph-matched application benchmark.
- Luke and Yates use different probe geometries and do not necessarily sample
  identical laminar populations. Per-depth normalization removes the largest
  channel-density confound but does not turn the sessions into biological
  replicates.
- Kilosort `good` labels and contamination are internal diagnostics, not
  ground truth. The Yates run itself has only 35 active KS-good units and high
  median contamination, so the report prioritizes matched event recovery,
  coincidence, and physically normalized counts over any one label.
- The template-similarity graph uses a declared 0.8 correlation and 100 µm
  neighborhood. It quantifies likely redundancy but does not replace manual or
  model-based merging.
- MEDiCINe settings differ between the cached Luke and Yates runs. Agreement
  with the independent Kilosort-style contrast is the main robustness check.

## Recommended next steps

1. Keep **no external voltage correction** as the production baseline for
   Luke while motion estimation and application are separated further. The
   0.25× correction failed its independent imec1 replication.
2. Advance **single Kilosort preprocessing** to a matched full-session
   current-versus-single-pass sort, but require identical post-sort merging,
   unit-family continuity, strict event recovery, and contamination safeguards.
   Do not select it on raw or KS-good unit count alone.
3. Treat the 10 µm kriging result as a mechanistic control, not a finalist. If
   external correction is revisited, tune spatial bandwidth relative to probe
   pitch and require both strict event recovery and contamination safeguards.
4. Do not apply either DREDGE or MEDiCINe full nonrigid voltage warps in
   production: two independent fields recreate the same overdecomposition.
5. Stabilize or reject Kilosort internal estimates with coarse-step limits or
   post-estimation temporal regularization; changing only the documented fine
   `drift_smoothing` parameter is unlikely to remove the observed mode jumps.
6. Only after a rigid application passes recovery safeguards, add a
   gradient-limited residual field with clipped derivative and gain. The full
   residual should not be used as the starting point.
7. Replace the original per-channel Luke--Yates deficit with physical-depth
   metrics. The first raw-voltage audit shows that the Kilosort-input deficit
   is not a simple raw-amplitude shortage: Luke has excess shared high-frequency
   voltage and imec1 has a large polarity imbalance. Complete the reference and
   compact-footprint sensitivity controls before making a biological density
   claim.
8. Consider using the motion estimate for coordinate/template tracking or
   downstream merging without resampling the raw voltage, if sorter-internal
   correction preserves detection better.

## Further questions

- Why does contamination rise at 0.25× despite preserved KS-good yield and
  baseline-like collision counts?
- Why does 0.25× exchange two visual-neural events for one in the replication
  rather than providing a stable direction of benefit?
- What property of Luke's binned spike fingerprint makes Kilosort's coarse
registration switch between modes separated by 70--103 µm?
- How much rigid gain is needed to improve depth continuity before reviewed
  events begin to disappear?
- Why do multiple high-amplitude clusters share a dominant raw contact even
  after the known bad channel is interpolated?
- Do correction-specific event losses cluster at particular depth bands or
  template families outside this reviewed cohort?
- Which part of the legacy Yates preprocessing graph differs materially from
  Luke's 12th-order bandpass, local reference, bad-channel interpolation, and
  subsequent Kilosort high-pass/CAR pass?
- Does the single-pass condition retain its 23% increase in good similarity-
  graph components after identical automated and manual merging over the full
  session?
- Is Luke's lower strong-negative event density already present in matched raw
  AP voltage, and does it persist within comparable laminar depth bands?
