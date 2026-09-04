# Luke 2025-08-04 pre-sort conditioning and motion investigation

> **HISTORICAL HANDOFF — superseded 2026-09-03.** The measurements remain part
> of the investigation record, but the active sequence is now defined in
> [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md) and decisions
> 0011–0015. The accepted motion estimates and within-Luke dose-response design
> supersede the earlier estimator-priority language here. No statement in this
> handoff establishes that rescue is better than legacy.

## 2026-08-31 active handoff update

The conditioning question that motivated this handoff is now sufficiently
resolved to freeze preprocessing as the current reference. Full-session rescue
generalized across both Luke probes, and a bounded comparison against the
substantially different pinned-AIND branch produced similar overall KS4 results
with opposing tradeoffs rather than a general AIND win. Rescue remains the
production/downstream reference; AIND remains a fixed independent comparator.

The active load-bearing question is now motion estimation: **does any estimator
improve held-out raster residuals, independent recurrent trajectories and field
stability beyond the current DREDGE fit?** Select estimators without sorter
labels or voltage resampling. Motion application is a separate decision:
coordinate correction must demonstrate benefit before voltage interpolation is
tested, and a good field does not authorize a warp automatically.

All new work follows stage-local validation: each stage must pass an observable
the next stage cannot rescue or manufacture. See
[`luke_pipeline_stage_local_validation_strategy.md`](luke_pipeline_stage_local_validation_strategy.md)
for the current development order, gates and preprocessing reopening criteria.
The 2026-08-29 material below is retained as investigation history.

## 2026-08-29 pause update

The investigation is paused after a first matched raw-voltage Luke--Yates
audit. The important new result is that Luke's previously observed deficit at
the Kilosort input does **not** appear to be a simple shortage of large events
in the original AP voltage. Before referencing, Luke has many more large
high-frequency extrema than Yates, much of which is shared across nearby
contacts. After a 100 µm local median reference, imec0 approaches Yates's
fixed-amplitude negative-event density, whereas imec1 retains a negative-event
deficit together with a very large positive-event excess. This shifts the next
rescue step toward probe-specific common-mode, polarity, and waveform-shape
diagnostics rather than a global threshold reduction.

The audit's event-rate and channel-noise tables are usable. Its saved spatial
footprint tables and PNG are explicitly provisional: the first calculation
used a full-shank energy denominator and an overly broad temporal maximum. The
code has been corrected, but the corrected footprint and shank-median
reference sensitivity run were intentionally stopped before completion. See
[`luke_yates_raw_voltage_audit_notes.md`](luke_yates_raw_voltage_audit_notes.md)
for the exact results, limitations, and resume sequence.

## Handoff objective

Determine whether the weak Luke sorting yield is caused primarily by signal conditioning or motion correction before Kilosort, rather than by the sorter itself. Use Luke 2025-08-04 as the pilot, Bacon 2025-10-16 as the Neuropixels/halo reference, and Yates 2022-02-16 as the known-good foveal-V1 reference.

The first implementation should be an isolated diagnostic/sweep path in this repository. Do not change the production pipeline defaults until the pre-sort evidence identifies a better configuration.

## Bottom line from the pilot

The current evidence does not support broadly elevated raw electrical noise in Luke. It does support a combination of shared mechanical motion, unusually strong depth-dependent displacement estimates, and sensitivity to motion-registration/sorting choices, with imec1 consistently worse than imec0.

The clearest Luke–Bacon difference is not every measure of rigid movement. It is the spatial variation of estimated displacement along the probe:

| Kilosort-style registration metric | Luke probe mean | Bacon probe mean | Luke/Bacon |
|---|---:|---:|---:|
| Rigid excursion, P95−P5 | 9.50 µm | 5.25 µm | 1.81× |
| P99 abrupt rigid step | 6.91 µm | 5.81 µm | 1.19× |
| Median nonrigid spread | 6.30 µm | 2.00 µm | 3.15× |
| P95 nonrigid spread | 18.38 µm | 7.25 µm | 2.53× |

Here, nonrigid spread is the P95−P5 range of displacement across depth bins at each time point. It can reflect real differential tissue motion, estimator instability, or both.

Additional findings:

- Luke imec0 and imec1 DREDge rigid traces correlate at `r = 0.758`; adjacent one-second steps correlate at `r = 0.421`. The shared component is consistent with common mechanical movement, although it does not establish that the single headpost is causal.
- Luke imec1 contains a one-bin 107 µm Kilosort registration jump. DREDge detects a much smaller simultaneous displacement on both probes near that period, suggesting a real shared event plus an exaggerated imec1 registration failure.
- Mean robust AP-band noise after the same 300 Hz high-pass/common-median calculation is 11.68 µV for Luke imec0, 11.13 µV for Luke imec1, and 11.64 µV for Yates. Broad electrical noise is therefore not the leading explanation.
- The average Luke probe has 1,859 sorted spikes/s versus 1,698 spikes/s for Yates, but this comparison hides the channel denominator. Luke has 4.84 sorted spikes/s/channel versus 26.53 for Yates, an 81.8% deficit.
- Automated KS-good units in Luke fire at 1.87 Hz on average versus 10.41 Hz in Yates, an 82.1% deficit. The medians are 0.51 versus 2.73 Hz, an 81.2% deficit.
- In a matched 100-second imec1 test, disabling Kilosort's internal correction increased total units from 160 to 194 and KS-good units from 29 to 35. This proves configuration sensitivity, but the test is too short to select a production sorter configuration.

## Important temporal-resolution correction

The saved Luke DREDge result is **not** a 20-second-bin estimate. Both probes contain `10,474 × 40` displacement arrays with exactly 1.0-second temporal spacing. The 20-second value used in the analytical report is visualization-only downsampling.

With SpikeInterface 0.102.1, `dredge_ap` defaults to:

- `bin_s=1.0`
- `histogram_time_smooth_s=1`
- `time_horizon_s=1000`

The repository does not set these explicitly in `pipeline/motion.py`; they are inherited through `**dredge_motion_args`. The effective temporal response is still broader than a bare one-second sample because the spike histogram is smoothed and the displacement field is globally regularized. Subsecond or short multi-second mechanical events may therefore be attenuated or represented as a smaller, broader displacement.

## Relevant code paths and implementation risks

### Motion estimation and application

- `pipeline/motion.py:18` exposes `dredge_motion_args`, but uses mutable dictionary defaults.
- `pipeline/motion.py:140` specifies spatial windows but leaves DREDge temporal parameters implicit.
- `pipeline/motion.py:146-159` treats the existence of `motion.npy` as a valid cache regardless of the parameters used to create it.
- `extra_outputs=True` is requested, but the returned correlation/weight diagnostics are discarded rather than persisted.
- `pipeline/motion.py:248` interpolates the sorting branch with `border_mode='force_zeros'`, then casts to `int16`. Zero-filled boundary samples and interpolation behavior should be measured directly.

The cache behavior is a blocker for a trustworthy parameter sweep: changing `bin_s`, smoothing, window geometry, peak detection, or localization can silently reuse an old result when `recalc=False`.

### Signal conditioning

`pipeline/preprocess.py` currently performs:

1. inter-sample phase-shift correction;
2. saturation blanking;
3. bad-channel detection and interpolation;
4. a 300–6000 Hz sorting branch and 300–3000 Hz motion-estimation branch;
5. 12th-order forward-backward Butterworth filtering;
6. local median reference with a 40–140 µm radius.

All are plausible, defensible choices, but their interactions have not been isolated for Luke. In particular, saturation blanking followed by high-order zero-phase filtering, local referencing, bad-channel interpolation, and motion interpolation can each alter waveform shape or spatial coherence before Kilosort sees the data.

## Ranked hypotheses

1. **Under-resolved fast motion.** Luke contains subsecond-to-few-second motion that a one-second histogram plus temporal regularization attenuates. Prediction: shorter-bin estimates grow in amplitude or reveal coherent transients, especially on imec1, while Bacon changes less.
2. **Estimator instability from weak/sparse peak evidence.** Luke's low activity per channel leaves some time-depth bins underconstrained, producing noisy nonrigid displacement. Prediction: large spatial gradients align with low peak counts, low correlation weights, or empty histogram regions and are not reproducible across nearby parameterizations.
3. **Real depth-dependent tissue motion.** Luke's single-headpost preparation permits differential tissue displacement along the shank. Prediction: nonrigid structure is reproducible across conditioning variants and estimators and is temporally coherent across simultaneous probes.
4. **Conditioning-induced spatial distortion.** The motion-estimation branch or local reference changes peak localization enough to create apparent nonrigid motion. Prediction: displacement fields and peak trajectories change materially when reference/filter settings change, even before motion interpolation is applied.
5. **Motion interpolation damages the sorting branch.** The estimated warp may be reasonable, but `force_zeros`, nonrigid interpolation, or repeated casting introduces waveform loss. Prediction: pre/post-correction waveform and zero-fill metrics degrade around large estimated displacements.

These hypotheses are not mutually exclusive.

## Proposed implementation

### Phase 0: make motion runs reproducible and cache-safe

Implement this before running a sweep.

- Replace mutable dictionary defaults with `None` and construct dictionaries inside `correct_motion`.
- Resolve and save the complete effective parameter set, including inherited DREDge defaults, to `motion_params.json`.
- Include SpikeInterface version, repository commit, recording identity, probe/stream, frame slice, conditioning configuration, peak count, and array shapes.
- Hash the parameters and source slice into the cache key, or refuse to reuse a cache whose manifest does not match.
- Persist compact DREDge diagnostics needed to judge estimator support: time/depth bin edges, peak counts per bin, window centers, usable correlation/weight summaries, and rejected-pair fractions. Avoid storing prohibitively large matrices unless explicitly requested.
- Keep all experiment outputs in new directories. Never overwrite the canonical Luke or Bacon products.

Suggested files:

- `testing/luke_0804_presort_motion_sweep.py`
- `testing/luke_0804_presort_motion_sweep.json`
- `docs/luke_20250804_presort_motion_handoff.md` (this file)

### Phase 1: pre-sort diagnostic sweep on matched windows

Start with short, matched windows rather than full sessions:

- one quiet control window;
- the largest shared DREDge event near local session time 7,275 s;
- the imec1 Kilosort outlier near local session time 8,220 s;
- at least one additional high-nonrigid-spread interval;
- the same relative windows on Luke imec0 and imec1;
- duration-matched Bacon probeA and probeB controls.

Use 5–10 minutes per window where practical. Keep peak-detection and localization settings fixed during the first motion sweep.

Minimum motion grid:

| Dimension | Values |
|---|---|
| Correction | none; DREDge |
| DREDge temporal bin | 0.5, 1, 2, 5 s |
| Histogram temporal smoothing | 0, 1, 2 s, constrained to sensible combinations with `bin_s` |
| Spatial model | rigid; current nonrigid windows |

Only add a narrower `bin_s` if peak counts support it. If 0.5-second bins are sparse or unstable, record that result rather than forcing a subsecond correction.

After the temporal grid, test the most informative conditioning contrasts one at a time:

- current local median reference versus global median reference;
- current 300–3000 Hz motion branch versus a less restrictive upper cutoff;
- current local-reference radius versus one wider radius;
- saturation blanking on/off for windows without true saturation, and current threshold versus a higher threshold where saturation exists;
- interpolated versus excluded bad-channel evidence for motion estimation, while keeping the sorting channel geometry valid.

Do not combine all conditioning variables factorially at first. Advance only contrasts that materially change pre-sort diagnostics.

### Phase 2: evaluate before Kilosort

For every variant, compute diagnostics on the conditioning output, the displacement estimate, and the motion-corrected sorting branch:

- detected peaks per second and per time-depth bin;
- fraction of empty or low-support bins;
- rigid excursion, P99 step, median/P95 nonrigid spread;
- spatial derivative of displacement and frequency of implausibly steep or sign-changing warps;
- stability of the displacement field under small parameter changes;
- Luke imec0/imec1 cross-probe correlation for rigid traces and event times;
- pre/post-correction zero fraction, especially samples introduced by `force_zeros`;
- clipped/saturated fraction and filter-ring duration around blanked events;
- robust AP noise and common-mode ratio;
- template-free waveform stability from localized threshold crossings: amplitude, peak channel/depth, spatial footprint, and waveform correlation before/after high-motion events.

The decisive distinction is whether large Luke nonrigid estimates are well supported and reproducible. A large field that persists across reasonable parameters and aligns across probes supports real motion. A large field that tracks sparse bins or disappears with small parameter changes supports estimator instability.

### Phase 3: downstream sorting only for finalists

Advance at most two pre-sort variants plus the current baseline to Kilosort. Hold all Kilosort thresholds, geometry, curation, and QC settings fixed.

For external DREDge variants, keep `do_correction=False`. Include one control with no external correction and Kilosort internal correction enabled. Do not interpret unit count alone.

Compare:

- matched-unit recovery across variants;
- template correlation and amplitude/depth trajectories;
- presence ratio and fixed-bin firing stability;
- refractory violations and contamination;
- missing/truncated spike estimates;
- unit splitting/merging and near-zero-lag duplicate burden;
- stability across the predefined quiet and motion-event windows.

## Acceptance criteria

A preprocessing change is a credible improvement only if it satisfies all of the following:

1. Its motion estimate is reproducible under nearby parameter settings and supported by adequate peak counts/correlation weights.
2. It reduces pathological Luke nonrigid spread or abrupt registration failures without simply flattening every displacement trace.
3. It does not increase zero filling, saturation/filter ringing, AP noise, or waveform distortion.
4. It improves continuity for matched units across motion events, not merely the number of clusters returned.
5. Contamination, refractory violations, and duplicate/split burden do not worsen materially.
6. The direction of benefit appears on both Luke probes or has a documented probe-specific explanation.

A useful pilot target is at least a 25% reduction in Luke's median nonrigid spread or event-linked waveform instability, with no degradation in the safeguards above. This is a prioritization threshold, not a claim that Bacon's 2.0 µm value is a universal target.

## Data and evidence locations

Pilot source data:

- Luke raw and pipeline outputs: `/mnt/NPX/Luke/20250804`
- Bacon sorted reference: `/mnt/NPX/Bacon halo_declan/20251016`
- Bacon raw reference: `/mnt/MGS/Ephys/Raw/Bacon/20251016/Record Node 101/experiment1/recording1/continuous`
- Yates known-good session: `/media/huklab/Data/Yates_session_copy/processed/Allen_2022-02-16`

Saved analysis evidence:

- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/direct_comparisons.csv`
- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/motion_summary.csv`
- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/luke_cross_probe_motion_summary.csv`
- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/luke_shared_motion_events.csv`
- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/raw_probe_summary.csv`
- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/sorting_probe_summary.csv`
- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/sorting_unit_metrics.csv`
- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/luke_sorting_sensitivity.csv`

Reproduction scripts:

- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/scripts/diagnostics/ephys_quality_pilot.py`
- `/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/scripts/diagnostics/build_ephys_quality_report_payload.py`

## Definition of done for the coding-agent task

The task is complete when the repository contains:

1. a cache-safe, parameterized pre-sort motion experiment runner;
2. a machine-readable manifest for every run;
3. compact CSV summaries and diagnostic plots for the predefined Luke and Bacon windows;
4. an evidence-backed recommendation for zero, one, or two full-sort finalists;
5. tests covering parameter resolution, cache invalidation, output isolation, array alignment, and metric calculations;
6. no changes to production defaults unless separately reviewed after the pilot.

The current conclusion should remain **share with caveats**. The evidence is sufficient to prioritize pre-sort investigation, but not to attribute the deficit solely to the headpost, conditioning, DREDge, or Kilosort.
