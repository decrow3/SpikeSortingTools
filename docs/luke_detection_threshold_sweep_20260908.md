# Fresh 3σ, 4σ and 6σ detection versus 5σ

Completed the requested comparison on compensated voltage over4160–4260s. The3σ input has the best aggregate agreement with the provisional lighthouse tracks in this window;4σ gives a smaller overall improvement;6σ is worse. This supports investigating lower detection thresholds, but does not establish a full-session setting or the neural purity of added detections.

## Controlled methods

All four thresholds were freshly detected with negative locally-exclusive detection,50µm radius and the same frozen original per-channel MAD vector used in earlier5σ detection. The shared-response compensation and300–6000Hz preprocessing were unchanged. No aggressive waveform screen was applied. This is a detector-threshold comparison, not relaxation of a post-detection amplitude gate.

All thresholds exclude the first/last50samples of each20s chunk. The fresh5σ sample/channel keys exactly reproduce the cached5σ baseline on those common interiors, and the amplitudes agree within declared numerical tolerance. The common edge rule excludes six previously retained boundary peaks, giving216715rather than216721baseline events.

The union of fresh detections was localized once per chunk. Exact sample/channel matches reuse their cached5σ localizations, while newly detected events use the identical monopolar-triangulation method and75µm radius. Each threshold selects locations from that union. DREDGE settings remain unchanged, including explicit enforcement of±80µm pairwise search bounds. No gain, sign, lag or lighthouse selection was fitted.

## Results

Scores use motion sampled at the unchanged accepted lighthouse event times, summarized in matching10s bins. Overall difference is equal-cell mean absolute difference in supported nonbaseline bins. Movement difference separately scores the4185→4195drop and4195→4205recovery. Neither is a ground-truth displacement error.

| Threshold | Peaks | Overall difference,µm | Drop/recovery difference,µm | Median across cells of95th-percentile1s increment,µm |
|---|---:|---:|---:|---:|
|3σ|1299549|1.073|1.270|1.570|
|4σ|437358|1.214|1.444|1.534|
|5σ|216715|1.353|1.433|2.444|
|6σ|122885|1.666|1.480|3.607|

Compared with5σ,3σ reduces overall difference by approximately21% and movement difference by11%, while detecting about6times as many peaks.4σ roughly doubles the peak count and reduces overall difference by10%; its movement score is essentially unchanged.6σ detects57% as many peaks and increases overall difference by23%.

The individual trajectories show reduced central instability at3σ/4σ, particularly relative to6σ around units445and463. Improvement is not uniform across cells. Large shallow excursions around4245–4250s persist, although some are smaller at3σ; the largest absolute field remains about57µm. Thus the lower threshold does not resolve the outstanding motion-verification problem.

Added3σ examples contain small ambiguous waveforms and neighboring activity. Their appearance does not establish that all additional detections are neural. Better population registration can arise without each event being an isolated, confidently identifiable neuron. Conversely, aggregate improvement on this already-inspected interval can reflect sampling or registration biases; a separate interval is needed before adoption.

## Figures and artifacts

Outputs: `testing/outputs/luke_detection_threshold_sweep_v1/`.

- `01_threshold_summary.png/.pdf`: aggregate agreement and peak counts.
- `02_lighthouse_overlay.png/.pdf`: event-time-matched trajectories for all nine lighthouse units.
- `05_continuous_overlay.png/.pdf`: continuous1s estimates, retaining visibility of brief excursions.
- `03_peak_histograms.png/.pdf`: depth–time counts and amplitude mass at all thresholds, shared color scales.
- `04_added_3to4sigma.png/.pdf`, `04_added_4to5sigma.png/.pdf`: deterministic examples of added peaks from each probe quarter in the first20s; not prevalence estimates.
- Settings and noise/model hashes, all chunk and full-interval peaks/locations, all fields and pairwise constraints, localization-reuse records, per-cell and aggregate scores, event-matched predictions and audit receipt.

All threshold-eligibility, index ordering, baseline correspondence, finite-localization/field and pairwise-bound checks pass. Approximately0.79% of3σ and0.51% of5σ locations lie outside physical probe depth; these are recorded, not silently removed with an arm-specific rule. Bootstrap bars show within-bin centroid sampling uncertainty, excluding baseline and identity uncertainty.

Scripts: `testing/luke_detection_threshold_sweep.py` and `testing/luke_detection_threshold_sweep_review.py`. The shared sweep scoring helper is reused for identical metric definitions; threshold-specific summary figures replace its generic labels. Independent service `luke-detection-threshold-sweep-v1` completed with exit0 and final actual state inactive/dead. Commands, logs and exit receipt persist under `testing/outputs/luke_detection_threshold_sweep_job_v1/`. Completed stage artifacts persist, but there is no automatic or within-localization checkpoint/resume. No sort or production motion application was run.

## Implication

3σ is the strongest candidate from this particular detection sweep. Keep5σ as the established comparison arm and test transfer before promoting a new full-recording configuration. The result favors retaining useful low-amplitude population structure after compensation rather than assuming stronger peaks always make a better motion input.
