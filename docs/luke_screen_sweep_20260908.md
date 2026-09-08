# Why the waveform screen worsens motion estimation

Completed21 input variants on the same4160–4260s interval. The evidence points to selective loss of registration information, with both reduced sampling and changes in which waveform populations remain contributing. The amplitude cutoff and neighboring-waveform rule matter most among the tested gates. No tested threshold combination meaningfully improves on compensation alone.

## Experiment

All arms retain the same shared-response compensation, cached peak locations, and bounded DREDGE settings. Features were computed once across five20s chunks. Reconstructed full-screen masks exactly match the previous masks in all five chunks.

Variants include compensation-only and full-screen baselines; omission of each of six gates; amplitude-only, center-energy-only and neighbor-only screens; amplitude thresholds5/6/8 residual sigma, neighbor cosine0.6/0.7/0.8, center-energy fractions0.5/0.55/0.65; one combined relaxation; two uniform random masks with exactly the same total retained count as the full screen; and one random mask matching its retained count within every1s×100µm detector-depth stratum. This is a small set of one-at-a-time changes and controls, not an exhaustive factorial search.

Here, omitting the shared gate means omitting extra peak rejection *after* compensation. The shared-response compensation itself stays enabled throughout this sweep.

Scores use the unchanged accepted lighthouse spike times, not screened spike times. Each estimate is sampled at those actual frames and summarized into the same10s bins as the centroid. Only a first-bin offset is removed; no gain, sign, lag or cell selection is fitted. Overall score is equal-cell mean absolute difference across supported nonbaseline bins. Movement score separately evaluates the4185→4195 drop and4195→4205 recovery with at least10accepted events at both ends. These are differences from provisional centroids, not true displacement errors. A constant-zero field is included solely as a diagnostic control.

## Results

| Input | Retained fraction | Overall difference,µm | Drop/recovery difference,µm |
|---|---:|---:|---:|
| Compensation only |1.000|1.353|1.433|
| Full screen |0.241|3.081|2.362|
| Full screen without amplitude cutoff |0.487|1.740|2.566|
| Full screen without neighbor rule |0.386|2.671|1.928|
| Amplitude cutoff only |0.435|2.405|1.652|
| Neighbor rule only |0.595|1.741|2.527|
| Center-energy rule only |0.848|1.352|1.420|
| Combined threshold relaxation |0.562|1.612|2.359|
| Uniform random thinning,seed14 |0.241|1.982|1.591|
| Uniform random thinning,seed29 |0.241|1.699|1.572|
| Random matching depth/time counts |0.241|2.467|1.972|
| Flat-zero diagnostic |—|1.587|3.754|

- Removing the amplitude cutoff reduces overall mismatch substantially but does not improve the main drop/recovery score. Removing the neighbor rule improves the movement score but does not restore baseline agreement. Therefore there is no single gate removal that repairs the whole screen.
- Width and broad-residual exclusions change only148 and159retained events, respectively; their removal barely changes the scores. Extra shared-explained rejection is redundant with other gates in this full mask.
- Lowering the neighbor cosine threshold alone actually worsens overall agreement. Loosening is not monotonically beneficial because it changes the population presented to registration.
- Both uniform random controls outperform the full screen at exactly the same count. Thus total count loss alone does not explain the deterioration. The depth/time-matched control is intermediate, consistent with both distribution changes and selective waveform loss mattering. Only two uniform seeds and one stratified seed were run; this is an exploratory control, not a precise variance decomposition.
- Center-energy screening alone excludes approximately15% with essentially unchanged agreement. Its tiny numerical improvement is not evidence of an established better setting or cleaner neural population.
- The flat control beats several candidates on overall difference while failing the movement score. Selecting thresholds only by average agreement would reward suppression of real motion.

## Mechanism in the problematic central region

At the2210µm estimation window, examine pairs linking4225–4240s with earlier times before4220s. This region was chosen to diagnose the already-observed screened excursion, not to define the sweep score.

Compensation-only assigns4.88% of constraint weight to pairwise shifts larger than15µm; full screening assigns38.53%. The excess weight concentrates on a competing approximately55–65µm alignment. Median active-pair correlation is nearly unchanged:0.555versus0.561. The screened raster therefore promotes large alternative alignments without an obvious loss of correlation confidence. DREDGE's resulting central field has the inflated excursions seen in the lighthouse overlay.

Without the amplitude cutoff, large-shift weight drops to23.84%; the depth/time-matched random control gives26.89%. These controls reduce but do not eliminate the competing alignment. This supports loss/change of useful profile structure as an explanation. It does not yet identify which individual neural or artifact waveform families create the competing features. Motion-dependent dropout from the selection rules remains a plausible contributor rather than a proved mechanism.

## Figures and artifacts

`testing/outputs/luke_screen_sweep_v1/`:

- `01_sweep_summary.png/.pdf`: all21input variants and the flat control.
- `02_key_controls.png/.pdf`: compact ablation and random-control comparison.
- `03_pairwise_mechanism.png/.pdf`: competing displacement weights and correlation distributions.
- `04_central_trajectories.png/.pdf`: units445and463 with compensation, full screen, amplitude-cut omission, and center-energy-only screening.
- `settings.json`: variants and scoring definitions saved before results.
- `features.npz`, per-chunk features, all masks and their SHA256 manifest, all motion fields and pairwise constraints, coverage arrays, per-cell and aggregate scores, event-matched predictions.
- `audit.json`: exact full-mask reconstruction, mask counts, finite bounded fields, and exact random-control count/stratum checks passed.

Scripts: `testing/luke_screen_sweep.py` and `testing/luke_screen_sweep_review.py`. Independent service `luke-screen-sweep-v1` exited0; actual final state checked inactive/dead. Launch command, logs and exit receipt reside in `testing/outputs/luke_screen_sweep_job_v1/`. No estimator parameters changed and no sort or production motion application occurred.

## Implication

Keep compensation-only as the reference. Do not adopt a winner from this same-window tuning exercise. The useful next diagnosis is to inspect the waveform/depth-profile families lost by the amplitude and neighbor cuts at the competing55–65µm alignment, then test targeted rejection or retention on a separate interval. The existing lighthouse limitations, including unresolved shallow excursions, still apply.
