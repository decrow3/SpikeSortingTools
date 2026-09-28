# Imec1 lighthouse motion evaluation: method audit and implementation plan

Prepared 13 September 2026. This document freezes the execution contract
supplied for the imec1 analysis. The implementation may document corrections
required by actual coordinate semantics, but it must not tune identity tracking
to any evaluated motion field.

## Recommendation and scope

Reuse two complementary evaluations: independent waveform-displacement
prediction and lighthouse-event concentration in existing sorter outputs. Score
large-displacement failures and quiet-period false motion separately. No full
new sort, motion refit, production promotion, or RF-holdout evaluation is in
scope. Begin with cached evidence; if it cannot meet the old three-increment
criterion, use bounded, managed extraction in deterministic extra windows.

## What the previous work measured

The huklaban1 motion-prediction analysis froze waveform identities and family
assignments, sampled motion at accepted-event times, made consecutive 5-second
family-median increments, and evaluated residual = lighthouse increment minus
predicted increment. Family-balanced RMSE is the square root of the mean of
family mean squared residuals. Historical skill against zero is
`1 - RMSE(candidate) / RMSE(no motion)`. MAE, signed error, correlation,
through-origin slope, and family bootstrap intervals are diagnostics; no fitted
slope, lag, gain, smoothing, or sign may recalibrate an evaluated field.

At least three adjacent supported increments are required per scored family and
at least three families for a population aggregate. Missing observations stay
missing. Strict, lower-score, ambiguous, and unmatched evidence remain
separate. Quiet means absolute lighthouse increment below 20 µm; movement means
at least 20 µm. The lattice-node analysis is a sensitivity, not an acceptance
repair.

The historical sorter comparison matched frozen reference events within 0.5 ms
and 100 µm, then reported recovery, dominant-cluster concentration,
fragmentation, and duplicate matches. Its aggregation averaged units within
families and then families. The new primary analysis pools unique family events;
the old aggregation is retained only as a reproduction sensitivity. A high
concentration can reflect overmerging, so cross-family mixing, loss, duplicate
matching, and raw refractory QC must accompany it.

Historical LFP sampling selected the nearest native-rate sample only within
10 ms and required supported, non-invalid, finite displacement. It did not
bridge gaps. Historical 5-second bins allowed one strict event and retained
event count and temporal span, so a point is not necessarily a precise
five-second neuronal-position average. Sparse-family bootstrap intervals are
descriptive rather than independent biological replication. The failed
lighthouse-derived RANSAC motion pilot remains excluded as an estimator.

### Imeс0 findings to carry forward without transferring conclusions

| Setting, 930–1030 s | No motion | AP rigid | LFP rigid |
|---|---:|---:|---:|
| Movement increment RMSE | 133.19 µm | 103.32 µm | 39.75 µm |
| Quiet increment RMSE | 1.44 µm | 8.28 µm | 48.86 µm |

At 1150–1200 s, quiet RMSE was 1.43, 4.92, and 21.86 µm respectively. A later
matched 348-channel comparison found movement concentration 0.409 unwarped,
0.439 AP-rigid, 0.498 native DARTsort, 0.480 native-rate LFP, and 0.567 SG25
LFP; quiet concentration favored unwarped (0.684 versus 0.355 for SG25).
An intermediate result confounded smoothing with changing from the exported
4-Hz trace to the native 250-Hz field. Never substitute a plotting CSV for the
field actually applied by a sorter.

## Frozen imec1 assessment population

| Tier | Candidate IDs | Use |
|---|---|---|
| Count/depth retained | p08_f025, p08_f033, p06_f010, p08_f029, p06_f045, p09_f020, p09_f024, p08_f044 | Initial primary proposals; only independently qualified strict observations are scored |
| Near threshold | p06_f047, p06_f008, p09_f006, p09_f038 | Separate expansion sensitivity; gates unchanged |
| Requires splitting | p09_f018, p08_f047 | Discovery only until residual cores independently qualify |
| Weak/ambiguous | p06_f038, p08_f010, p08_f017, p07_f021, p08_f039, p07_f008 | Preserve rejection evidence; exclude from primary aggregate |

The source contains 20 unique compact-core proposals and 47,013 qualification
detections in 310–320 s; those detections are not accepted lighthouse spikes.
Every identity is namespaced with its discovery run and template hash. Short
candidate IDs must never join different banks. Templates, global rivals and
decoys, strict gates, tiers, family dependencies, and event keys are frozen
before predictions are loaded. Count/depth retention is not cell certification.
Matching retains exact relative geometry, cross-phase competition, scores,
margins, energy/gain evidence, and unmatched states. It uses no motion, absolute
depth preference, or forced temporal continuity.

The 310–320 s interval is already-used qualification evidence. It can provide
at most one adjacent five-second increment and therefore cannot produce a
replicated ranking. New tracker versions require new validation windows.

## Motion inventory and coordinate contract

| Candidate | Frozen role |
|---|---|
| Zero | Required analytic control |
| Native DARTsort AP | Working baseline from the preserved native sort motion pickle |
| Fine MEDiCINe | Primary 4,634-time × 4-depth challenger, 0.25-s grid, 1-s kernel |
| MEDiCINe rigid projection | Equal-weight four-depth diagnostic, reported separately |
| LFP unsharpened 80/260 µm | Native-rate rigid candidates with their support masks |
| LFP derivative 80/260 µm | Inventory/unsupported diagnostics only when support is zero |

LFP producer:
`/home/huklaban5/Documents/SpikeSortingTools/NPX_preprocessing/outputs/dredge-lfp-imec1-dartsort-crop-v1/`.
MEDiCINe producer root:
`/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/`.
The compact LFP transfer contains only motion arrays, axes, support masks,
specification, completion/overlap receipts, and checksums—not the large LF
contexts.

Verify probe/session/stream, raw AP interval `[193737, 34942956)`, sampling rate
29,999.759166666667 Hz, geometry/channel order, and hashes. The canonical event
clock is raw AP sample index and
`t_crop = (raw_ap_frame - 193737) / fs`. AP recording-relative, crop-relative,
and acquisition-absolute time are distinct; the crop shift is about 6.458 s and
must not be subtracted twice. The older 3057.677546-s acquisition offset is used
only when confirmed by source metadata.

The common sign is `corrected depth = observed depth - displacement`. Synthetic
translations and the producer motion object's public sampler test sign and
axes. No silent endpoint clamp, depth extrapolation, or gap interpolation is
allowed in primary scores. Native support and out-of-domain events are retained.
For a forward reference-depth field, sample frozen family reference depth; for
an inverse observed-depth correction field, use observed event depth. Audit the
originating API and preserve both physical-prediction and deployed-sorter
operators when they differ. Use only the full-crop LFP result; never splice its
context-dependent standalone segment.

Engineering permission to trial MEDiCINe is distinct from lighthouse validation
or production acceptance; retain both dated receipts where their status differs.

## Stage A: cached scoring

Reproduce the archived method on archived inputs first. Export one row per
observation with immutable event key, namespaced candidate/family, tier, raw
frame and clocks, centroid and reference depth, detector/lattice phase, scores,
runner-up, margin, gain, common energy, and role. Missing source columns remain
explicitly unavailable.

Sample zero, native AP, MEDiCINe, LFP80, and LFP260 at every event. Build an
identical primary common-support set before reducing medians; unsupported
derivative traces cannot erase the comparison. Also report challenger/native
paired support and each field's total support. Make per-family five-second
medians and adjacent increments only within continuous windows. Save signed
residual, RMSE, MAE, slope, correlation, counts, P95 absolute error, largest
error, and family-balanced exceedance fractions at 40/80/120 µm. Require three
increments in each of three families before an aggregate rank.

Separate quiet and movement results and report quiet false-correction magnitude
and added residual relative to zero. Preserve exact-event plots and show support,
first/last event, and gaps in optional 1-s/0.25-s summaries. Do not carry values
forward. MEDiCINe's 0.25-s grid is four samples/s and its 1-s kernel further
limits bandwidth; neither a denser plot nor sparse events establish >4-Hz
fidelity. Rapid-transition comparisons use actual elapsed-time bands within the
same identity.

The 40-µm lattice sensitivity must retain synthetic translations at
0/10/20/30/40 µm and detector-phase dropout. No diagnostic fit changes a scored
field; any fitted variant becomes a new arm on separate observations.

## Stage B: independent temporal coverage

Reuse cached windows but label inspected evidence exploratory. If insufficient,
freeze six 30-second windows evenly across the common crop interior using only
duration, then deterministically shift forward past training, qualification,
reviewed discovery, and RF-holdout intervals. Freeze raw frame bounds before
reading voltage. Window choice cannot depend on field behavior or candidate
tracks. Do not open the RF response holdout.

Use a timed pilot before managed extraction. Require three supported families;
show leave-one-family and leave-one-window-out sensitivities. A nonrigid claim
also requires simultaneous separated-depth families. Sparse evidence is
reported as such rather than converted to a continuous trajectory.

## Stage C: existing-sort concentration

Verify and separately score completed native-AP and fine-MEDiCINe full-probe
sorts. Crop sorts remain a separate geometry comparison. Start with the old
0.5-ms timing tolerance and a stricter sensitivity. Assignment is deterministic
and one-to-one; competing and unmatched events are saved, and no sorter spike
can recover two references. Prefer raw acquisition-coordinate waveform support.
Any registered-coordinate gate must transform both sides identically and expose
field dependence.

For every family report recovery, overall concentration
`dominant_cluster / N_reference`, conditional concentration
`dominant_cluster / N_matched`, full cluster counts, fraction outside the
dominant cluster, and clusters covering 90%. Save family/cluster contingencies
and flag output labels receiving independent families. Compare loss, duplicates,
and the DARTsort evaluator's raw refractory definitions. Historical eligibility
is 20 reference events overall and 10 per regime, fixed before arm results.

Show output labels over time beside waveform residuals and locate fragmentation
before matching, after TMM, or after final agglomeration when saved labels exist.
Keep the 61-development-trial RF/QC evaluation alongside this work and leave 20
RF holdout trials closed. A new motion field would require restart before the
first motion-dependent feature/clustering stage; a final-merge restart is not a
valid motion test.

## Implementation and deliverables

- `testing/imec1_lighthouse_motion_manifest_v1.json`: frozen sources, hashes,
  clocks, geometry, identities, gates, windows, support and scoring policy.
- `testing/luke_imec1_lighthouse_motion_adapters_v1.py`: portable field loaders
  and samplers.
- `testing/luke_imec1_lighthouse_motion_comparison_v1.py`: common-support
  prediction, increments, family/regime/depth metrics, and figures.
- `testing/luke_imec1_lighthouse_sort_concentration_v1.py`: deterministic
  existing-sort matching and concentration/mixing metrics.
- `testing/test_imec1_lighthouse_motion_comparison.py`: sign, clock, coordinate,
  gap, support, weighting, and assignment regressions.

Primary output directory:
`testing/outputs/luke_imec1_lighthouse_motion_comparison_v1/`. Expected durable
tables include event predictions, support, increments, family/regime metrics,
sort matches, sort-family metrics, family/cluster contingency, summary, source
receipt, and a concise report. Figures show exact-event tracks, native-rate
predictions and gaps, residuals, movement-tail/quiet-penalty, support/depth, and
labels over time with every discovery/qualification/validation interval marked.

Sequence: source/coordinate preflight → cached scoring → freeze needed windows →
bounded tracking → field comparison → existing-sort concentration → outcome
decision. The first cached pass has a 20–30-minute allowance. Any larger
extraction gets a measured deadline and checkpointed managed execution. Preserve
all completed stages and failed-run evidence.

Favor fields that reduce large movement residuals and improve retained-spike
concentration without material quiet penalty, loss, or cross-family mixing. Do
not force a weighted single rank without agreed tradeoffs.

## Source trail

- `Assess LFP motion with lighthouses`, task `01a08b73-6a81-7281-9e88-3d3aaace19f6`.
- `Find Luke0804 lighthouse units`, task `01a09800-1eae-7d82-8616-1b86f513e0d1`.
- `Improve lighthouse unit discovery`, task `01a09974-d02c-7661-85c8-eb55c83ab5c4`.
- `Run DREDGE motion estimation`, task `01a09891-b80e-7a62-96f1-725a532233f6`.
- Current candidate assessment: `testing/outputs/luke_imec1_twenty_candidate_assessment_v1/`.
- Historical implementations: `testing/luke_lfp_lighthouse_validation_v1.py`,
  `testing/luke_lfp_ap_100s_matrix_analysis.py`, and
  `testing/luke_motion_candidate_lighthouse_comparison_v1.py`.

Fresh implementation checks include remote hashes, event-clock schema,
namespaced dependencies, exact nonrigid coordinate semantics, support outside
qualification, and completed-sort provenance. These are execution preflights,
not grounds for another conceptual planning round.
