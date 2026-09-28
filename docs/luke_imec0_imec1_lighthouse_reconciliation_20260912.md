# Luke0804 lighthouse-method reconciliation: imec0 versus imec1

## Bottom line

The current imec1 analysis has repeated the most important imec0 experiment:
whole-probe, depth-blind matching of relative 49-sample x 16-channel waveforms
on exact 40 um probe translations. It has also repeated the same failure modes:
sparse strict support, morphology lookalikes spanning implausibly large probe
distances, and lattice-phase dropout. It should not be extended or used to rank
motion estimators in its present form.

The imec1 work is not wasted. It improves on imec0 by discovering seed
waveforms without sorter labels, freezing 50 candidates before depth reveal,
retaining lower-score/ambiguous evidence, and adding time- and space-reversed
decoys. But its current family construction and matching are partitioned by
probe phase. This prevents cross-phase rivals from competing and can split one
biological waveform across phases. That is a regression from the imec0 global
247-template competition and leaves the known alternate-phase problem in the
identity step.

The cheapest defensible next action is not another motion comparison. Reuse the
imec0 method audit on the frozen imec1 cache, then rebuild identity competition
globally across phases. Promote only seed-localized candidates and calibrate
their node/half-step recovery and decoy false-match rates before reading more
voltage or fitting a >4 Hz motion field.

## Chronology of the imec0 work

### 1. Sorter-label signal screen and local controls

- `luke_lighthouse_screen_20260907.md`: 196 motion-off reference clusters were
  screened in 4080--4090 and 4100--4110 s. Nine passed waveform amplitude,
  noise, repeatability, locality, and event-shape gates. These were candidates,
  not independently verified identities.
- `luke_lighthouse_gentle_20260907.md`: 20 reviewed candidates entered local
  quiet controls; 11 passed, and nine spatially separated labels were followed
  in 10 s bins through 4160--4260 s. Several showed a shared rise/drop/recovery,
  but fixed local support and sorter-conditioned events could manufacture
  dropout or apparent stationarity.
- `luke_peak_population_and_relative_geometry_20260907.md`: all nine changed in
  the same direction at the 4185-to-4195 s step, with depth-dependent centroid
  changes. This was direct shared-movement evidence, not displacement
  calibration.

### 2. Fixed-template failure and translated local matching

- `luke_long_context_and_tracker_sensitivity_20260907.md`: exact +/-40 and
  +/-80 um shifts of four templates all failed the fixed-position matcher. This
  invalidated stationary-looking fixed-template centroids as evidence against
  a motion estimate.
- `luke_translated_matcher_validation_20260907.md`: a -80:40:+80 um translated
  bank recovered noiseless injected shifts, but real held-out matches occupied
  multiple offsets and confused nearby shapes. Spatial flexibility fixed
  sensitivity while exposing identity ambiguity.
- `luke_early_unit445_transfer_20260908.md` and
  `luke_early_unit445_depth_search_20260908.md`: unit 445 had sparse early
  transfer. The expanded +/-120 um search produced 42 unique matches but no
  ten-event single-shift group, so it could not define a trajectory.
- `luke_early_lighthouse_binning_20260908.md`: 0.25--5 s and fixed-event bins
  were tried on two high-rate candidates. At 0.25 s only 43/400 and 128/400
  bins were supported; 100-spike groups had median durations 6.01 and 2.52 s.
  Finer display resolution did not create finer independent evidence.

### 3. Whole-probe waveform-only matching: the direct predecessor of imec1

- `luke_waveform_only_global_20260908.md`: 247 complete seed templates from
  sorter labels were spatially recentered. Selection used 930--940 s; matching
  covered 930--1030 s. Every complete probe patch was eligible, every one of
  the 247 templates was a rival, and depth, motion, and trajectory continuity
  were excluded. Waveforms used 49 samples, 16 channels, exact 40 um
  translations, +/-3 sample alignment, strict cosine 0.86, lower display
  cosine 0.80, and identity margin 0.025.
- The initial 12 selected seeds recovered only 20/693 cached seed spikes. A
  training-only recovery gate left five identities. Held-out matching produced
  519 accepted events, including 84 beyond +/-120 um, but repeated separated
  depth bands raised lookalike concerns.
- `luke_waveform_only_expansion_20260908.md`: cached evidence expanded the panel
  to 17 candidates: five original, eight margin-passing additions, and four
  whose training recovery depended on ambiguous matches. These are the
  previously discussed approximately 17 waveform-traced labels, not 17 proven
  cells.
- `luke_lighthouse_extension_300s_20260909.md`: the frozen 17 were extended
  from 930--1030 to 930--1230 s without changing matching rules.

### 4. Audit of the 17-label tracker

- `luke_lighthouse_method_audit_20260909.md`: the exact 40 um lattice was shown
  to dominate acceptance. With recorded noise, strict acceptance was 44% at a
  node, 13% at 10 um, 0% at 20 um, 19% at 30 um, and 43% at 40 um. This can
  manufacture dropout and apparent stationarity.
- Median held-out temporal coverage was 11%; only unit 673 exceeded 50%.
  Units 125, 161, 557, 698, and 705 spanned more than 500 um, up to 2880 um,
  and were judged lookalike-dominated.
- For 82% of suprathreshold detections a runner-up identity also exceeded
  cosine 0.86. Candidate independence varied from 11 to 17 families as the
  template-cosine threshold moved from 0.80 to 0.95.
- The seed-recovery circular-shift null passed for all 16 testable candidates,
  showing nonrandom seed recovery, but no whole-probe false-positive calibration
  was available. Time-reversed and vertically flipped decoys were explicitly
  identified as the missing control.

### 5. Bounded depth-aware qualification

- `luke_depth_aware_lighthouse_policy_20260908.md` formalized moving waveform
  support with the depth hypothesis, full rival competition, unmatched and
  ambiguous states, and fixed-template tracks only as controls.
- `luke_shallow_identity_qualification_20260908.md`: dense labeled and unknown
  rivals plus real-background shift injections were used for units 80 and 154.
  Unit 80 failed sensitivity (65.3% injected recovery); unit 154 passed the
  frozen gate (92.7%) and alone proceeded to bounded tracking.
- `luke_motion_preconditioning_round_20260908.md`: unit 154's individual-event
  centroid compressed injected movement. The centroid of the median waveform
  behaved better, but still did not justify a physical scale correction.
- `luke_shallow_reference_overnight_v1.md`: four shallow candidates were tested
  with contextual weights and explicit rival injections. None passed all gates;
  no new shallow reference was created.

### 6. Direct replication, consensus, and the rejected lighthouse estimator

- `luke_lighthouse_direct_check_20260908.md`: cached accepted observations
  showed one exploratory agreeing pair, units 388/398, with 42 paired events
  and 6.9 um median absolute difference. Other regions were sparse or did not
  agree.
- `luke_lighthouse_consensus_reproduction_20260909.md`: 897 strict events from
  930--1030 s exactly reproduced the old 5 s consensus. Pooling dependent
  labels 673/675 yielded 16 family votes and preserved the result. Median
  pairwise correlation was 0.255; median leave-one-out correlation was 0.433,
  with circular-shift p <= 0.001. The result was reproducible but still
  conditional on ambiguous, lattice-biased accepted events.
- `luke_lighthouse_ransac_field_20260909.md`: a nonrigid affine RANSAC field
  failed family-held-out prediction (73.4 um median error versus 61.7 um for a
  rigid median) and was retained only as a negative control. Lighthouses were
  thereafter reserved for validation, not motion fitting.

### 7. Depth-blind Kilosort tracklet families

- `luke_kilosort_family_plausibility_20260909.md`: templates were linked by
  relative waveform similarity before revealing depth/time. At cosine 0.95,
  five of 24 families passed post-hoc coherence. Isolation from external
  templates was more useful than internal cosine alone.
- `luke_kilosort_family_threshold_sweep_20260909.md`: cosine 0.97 retained three
  coherent families; 0.98 retained none. Very high pair cosine did not prevent
  pathological families.
- `luke_kilosort_premerge_lighthouse_20260909.md`: using the checkpoint directly
  after `clustering_qr.run(mode="template")` and before final Kilosort merging
  exposed c097_F009, an 11-tracklet family spanning 240 um, with a clean merged
  refractory diagnostic. Its correlation with prior cells was suggestive, not
  significant after library selection. Final Kilosort merging had not merged
  these tracklets; the useful change was the finer premerge partition.
- `luke_premerge_prior_lighthouse_reconciliation_20260909.md`: all 742/742 old
  seed spikes survived premerge detection. Fifteen of the 17 old labels mapped
  mainly to singleton clusters that the family-only report omitted. Labels
  673/675 shared a dominant cluster, leaving 16 representations. Static-cluster
  centroids showed only weak shared movement, confirming that a cluster-local
  representation cannot by itself follow a moving identity.

### 8. Motion-estimator and preprocessing comparisons using provisional lighthouses

- The 100 s long-motion, waveform-screen, screen-sweep, filtering,
  preconditioning, AP-method, and LFP experiments tested estimator inputs and
  overlays. They did not create new lighthouse identities. The aggressive
  waveform screen selectively removed registration information and was rejected.
- `luke_motion_candidate_lighthouse_comparison_20260910.md`: on strict,
  lattice-controlled 5 s increments, LFP rigid followed large excursions best
  in 930--1030 s, but produced the worst false motion in quiet increments and
  in 1150--1200 s. No-motion was best in quiet periods. This established
  regime dependence, not a winning correction field.

## What has been done on imec1

| imec1 step | Result | imec0 precedent and interpretation |
| --- | --- | --- |
| Static DARTsort-label direct check | 25 of 56 eligible labels selected; broad fixed-label activity did not replicate consistently | Same limitation as the first imec0 sorter-label controls: clusters may split/merge under motion. |
| Cached DARTsort TPCA waveform pass | Only units 656, 1009, and 1116 passed seed recovery; all three later spanned >500 um | Repeats the imec0 global lookalike failure, with the additional dependence on DARTsort TPCA. |
| Raw matcher seeded by static labels | 56 targets, 1009 rival units/1150 phase templates; no unit passed the seed-only identity gate; only 16 strict events across seven units | A valid negative bounded test. Relaxing its thresholds would repeat an imec0 mistake. |
| Sorter-free waveform discovery v1/v2 | 108254 raw seed detections in 36.458--56.458 s; 192 morphology families; five strict-shape families; 50 frozen review candidates; 183 strict events in six motion-rich windows | Genuine improvement: no sorter labels. But morphology clusters are not cell identities, and per-phase construction recreates the known phase/lookalike problem. |
| Seed-depth reveal | Only 3/50 selected families have seed P90 span <=120 um; 46/50 exceed 250 um and 44/50 exceed 500 um | Most of the selected bank is already spatially implausible as a single seed identity. This is the same warning that disqualified five old imec0 tracks. A 250 um total held-out path can be plausible; a multi-kilometre-equivalent span inside a 20 s seed family is not. |
| Heavy-window post-selection audit | p06_f000 was the sole provisional candidate; the only evaluable localized pair, p06_f000/p06_f015, correlated -0.369 | No replicated shared motion, matching the imec0 rule that one attractive track is not enough. |
| Top-20 heavy-window estimator overlay | Only six top-20 families were active and five evaluable | Too sparse and too identity-ambiguous for estimator ranking. |
| Estimator-selected gentle windows | Six 10 s windows: 70--80, 110--120, 420--430, 560--570, 820--830, 850--860 s; 278 strict events from 18/50 families, but only five top-20 families were evaluable | The selection was independent of candidate tracks but conditional on the estimators. As in imec0, gentle agreement can diagnose gross incompatibility but cannot validate magnitude. |
| Gentle estimator comparison | DREDGE had the strongest median seed-reference correlation (0.809), while no motion had the lowest family-balanced local RMSE (2.055 um; MEDiCINe 2.049, DREDGE 2.185) | This is the same regime-dependent result seen on imec0: low-motion windows have little identifying power and often favor zero. |
| Temporal-density audit and exact 10 s snippets | Strict all-50 pooled rate was 4.63 events/s, but only 43.3% of 200 ms bins were occupied. Strict+lower all-50 reached 20.67 events/s and 90% occupancy. Individual plausible p06_f000 remained below 4 Hz | Pooling low-confidence families gives display density, not an identity-specific >4 Hz lighthouse measurement. Imec0's 0.25 s experiment reached the same conclusion. |

## Exact method overlap and regressions

| Design choice | imec0 whole-probe | imec1 sorter-free v2 | Assessment |
| --- | --- | --- | --- |
| Seed source | Sorter-derived templates | Raw both-sign 5-sigma detections clustered by waveform | Clear imec1 improvement |
| Seed duration | 930--940 s (10 s) | 36.458--56.458 s (20 s) | More seed data, but family membership is already spatially diffuse |
| Waveform support | 49 samples x 16 channels, relative geometry | Same | Direct reuse |
| Spatial search | Every complete exact 40 um patch | Same phase across all exact 40 um patches | Direct reuse with phase restriction |
| Temporal alignment | +/-3 samples | +/-3 samples | Direct reuse |
| Strict/lower gates | 0.86 / 0.80 cosine; 0.025 margin | Same | Direct reuse; not re-calibrated for imec1 |
| Rival scope | All 247 complete-support identities compete globally | Only families in the detection's phase compete | imec1 regression; cross-phase aliases cannot veto |
| False-match controls | Proposed but not run | Time-reversed and channel-reversed decoys included | Improvement, but winner counts are not yet a calibrated candidate-level false-match rate |
| Candidate independence | Threshold-sensitive 11--17 family curve | MiniBatchKMeans families treated as ranked units | imec1 needs the same family/dependence audit |
| Lattice audit | Explicit injected 0/10/20/30/40 um test; 0% noisy strict recovery at 20 um | Known in limitations, not rerun candidate-by-candidate | Missing critical control |
| Moving support | Later bounded tracker moved support with discrete depth hypothesis | Observed patch moves, but family identity is phase-limited and has no interpolated alternate-phase support | Incomplete implementation of the later imec0 policy |
| Motion use | Excluded from selection/matching; later comparison only | Same; gentle windows selected from estimators | Mostly correct, with conditional gentle-window inference |

The decoy issue is substantial rather than cosmetic: decoys won 46908/265416
(17.7%) detections in the six heavy windows and 53390/316052 (16.9%) in the six
gentle windows. Those aggregate rates do not directly estimate false acceptance
for any candidate, but they show that reversed shapes are highly competitive
and must be included in candidate-level calibration.

## Decision and reuse plan

1. Freeze the existing imec1 detections, seed families, event waveforms, and
   status tables. Do not read more voltage yet and do not use the current
   top-20 estimator correlations as a ranking result.
2. Rebuild a single global identity/decoy bank across all phases. Permit a seed
   family to be represented on compatible phase supports, and let every real
   and decoy identity compete before accepting an event.
3. Apply the cached imec0 method audit to every candidate: seed localization,
   node/10/20/30/40 um injected recovery with recorded noise, runner-up
   suprathreshold rate, candidate-level decoy acceptance, template-family
   dependence curve, held-out coverage, and post-hoc depth plausibility.
4. Start review with the three seed-localized candidates, especially p06_f000
   and p06_f015. Keep diffuse families plotted as negative controls, not as
   independent votes. A held-out span around 250 um can remain plausible if the
   seed identity is localized and the path is supported; the present
   millimetre-scale 500--3400 um seed-family spans cannot.
5. Require at least two spatially distinct localized candidates to reproduce a
   shared event in exact event-time plots before comparing estimator amplitudes.
   Preserve p06_f000 as a lead, not a lighthouse ground truth.
6. Only after identity/replication passes, refit candidate motion estimates at
   at least 5 Hz. Do not interpolate the saved 0.5--1 Hz fields and call that a
   >4 Hz comparison.
7. In parallel only if sorter-based corroboration is still desired, reuse the
   imec0 premerge checkpoint method on imec1. Emit singletons as well as linked
   families; the imec0 audit showed that family-only output hid 15/17 known
   candidates.

This sequence reuses the validated parts of the imec0 work and stops at the
first inexpensive failure. It does not require another sorter run, motion
estimator, or raw extraction for the first three steps.

## Primary evidence files

The imec0 lineage is documented in:

- `docs/luke_lighthouse_screen_20260907.md`
- `docs/luke_lighthouse_gentle_20260907.md`
- `docs/luke_peak_population_and_relative_geometry_20260907.md`
- `docs/luke_long_context_and_tracker_sensitivity_20260907.md`
- `docs/luke_translated_matcher_validation_20260907.md`
- `docs/luke_early_lighthouse_binning_20260908.md`
- `docs/luke_early_unit445_transfer_20260908.md`
- `docs/luke_early_unit445_depth_search_20260908.md`
- `docs/luke_waveform_only_global_20260908.md`
- `docs/luke_waveform_only_expansion_20260908.md`
- `docs/luke_lighthouse_extension_300s_20260909.md`
- `docs/luke_lighthouse_method_audit_20260909.md`
- `docs/luke_depth_aware_lighthouse_policy_20260908.md`
- `docs/luke_shallow_identity_qualification_20260908.md`
- `docs/luke_shallow_reference_overnight_v1.md`
- `docs/luke_lighthouse_direct_check_20260908.md`
- `docs/luke_lighthouse_consensus_reproduction_20260909.md`
- `docs/luke_lighthouse_ransac_field_20260909.md`
- `docs/luke_kilosort_family_plausibility_20260909.md`
- `docs/luke_kilosort_family_threshold_sweep_20260909.md`
- `docs/luke_kilosort_premerge_lighthouse_20260909.md`
- `docs/luke_premerge_prior_lighthouse_reconciliation_20260909.md`
- `docs/luke_motion_candidate_lighthouse_comparison_20260910.md`

The imec1 lineage is in `testing/luke_imec1_dots_lighthouse_direct_check.py`,
`testing/luke_imec1_dots_waveform_lighthouse_check.py`,
`testing/luke_imec1_dots_raw_lighthouse_check.py`,
`testing/luke_imec1_dots_sorterfree_waveform_discovery.py`, and the corresponding
frozen output directories under `testing/outputs/`.
