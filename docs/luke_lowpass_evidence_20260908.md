# Independent low-pass comparison evidence audit

The six-arm experiment remains exploratory on the previously inspected 4160–4260 s interval. Its nine fixed-template lighthouse tracks are biological corroboration, not ground-truth motion. In particular, identity selection can favor stationary matches during movement, and the ten-second centroid summaries cannot resolve the brief 4245–4250 s shallow excursions.

The independent audit is implemented in `testing/luke_lowpass_evidence_v2.py`. Run it after the experiment finishes:

```bash
python -m testing.luke_lowpass_evidence_v2 --replay-curves
```

It writes `testing/outputs/luke_3sigma_lowpass_screen_v2/independent_evidence/`:

- Independently sampled predictions at actual accepted lighthouse event times, with fixed first-bin offsets and no fitted gains, signs, or lags. Both first-bin support and evaluated-bin support must reach ten events.
- Per-cell, depth-band, and time-bin agreement, separate drop/recovery changes, a flat-zero control, and continuous-field excursion diagnostics. Missing support remains explicit.
- Exact D/C/U plots and active-edge antisymmetry/cycle consistency at approximately 410 and 2210 µm. Cycles require all three edges to have positive solver weight.
- Optional exact bounded correlation replay from cached peaks and locations, validating saved D/C in both windows before interpreting alternatives. Alternative lags are at least 15 µm from the winner and within ±80 µm. All solver-supported pairs in the three diagnostic intervals are reported, alongside the central cross-epoch pairs (rows 4225–4240 s, columns 4160–4220 s) and representative largest-shift curves.

No voltage is read, no new motion is fitted, and no lighthouse identity is changed by this audit. D/C/U alone cannot reveal runner-up correlation margins; those require the optional replay. High correlation, consistency, or low motion do not independently demonstrate accuracy.

The six-arm experiment and independent audit are complete. Exact correlation replay reproduced every D and C entry in both selected windows for all six arms. The results below retain the compensated 3σ baseline; no new arm is promoted.

Validation: the cached compensated 3σ baseline smoke audit completed successfully and reproduced equal-cell mean absolute disagreement of 1.0731962735 µm (flat-zero control: 1.5872009820 µm). The complete cached-baseline correlation replay also passed with one Torch/BLAS thread: all 10,000 D entries and C values matched exactly in each inspected window, including 1,560 pairs beyond the 60-second solver horizon. The directed central cross-epoch table contains 720 positive-weight pairs; beyond-horizon pairs have zero solver weight and are retained only in the full correlation arrays. A four-thread smoke attempt changed one shallow winner by 1 µm at a correlation difference of 2.98×10⁻⁷; matching the original one-thread computation restored exact identity. The audit therefore fixes Torch to one thread and retains strict winning-lag equality. The subsequent six-arm replay also matched all winning lags and C values exactly in both windows.


## Completed comparison and interpretation

**Keep the compensated broadband 3σ baseline. The tested 3 kHz low-pass creates a predominantly shallow alignment failure; gentle screening does not offer a consistent improvement.**

| Arm | Peaks | Equal-cell disagreement (µm) | Drop/recovery disagreement (µm) |
|---|---:|---:|---:|
| Broadband 3σ | 1,299,549 | 1.073 | 1.270 |
| Broadband + screen | 804,882 | 1.044 | 1.331 |
| Low-pass fixed threshold | 412,662 | 2.197 | 1.701 |
| Low-pass fixed + screen | 330,169 | 2.390 | 1.800 |
| Low-pass adjusted threshold | 867,571 | 1.943 | 1.826 |
| Low-pass adjusted + screen | 627,689 | 2.155 | 1.787 |

The aggregate low-pass deterioration is highly localized. Unit 80 at 220 µm changes from 2.272 to 9.966 µm disagreement under adjusted low-pass, accounting for approximately 98% of that arm's aggregate excess. Excluding this cell, baseline and adjusted-low-pass averages are 0.923 and 0.940 µm. Several deeper cells improve slightly, while others worsen. This is evidence against treating the whole-probe aggregate as a uniform loss of performance. Screening's small aggregate gain likewise comes from unit 80; the other eight cells average 0.923 µm before screening versus 0.947 afterward, and drop/recovery agreement worsens.

Exact pairwise replay provides corroborating failure evidence independent of the lighthouse centroids. At 410 µm, across the selected late-versus-earlier epochs, solver weight on |D|≥50 µm rises from **0% to 14.6%** under adjusted low-pass (34.7% fixed; 27.2% adjusted+screen; 43.9% fixed+screen). The median correlation margin over an alternative at least 15 µm away decreases from **0.149 to 0.077**. Full-window supported-triangle cycle-error p95 rises from **9 to 60 µm**. A representative adjusted-low-pass pair (4227.5 versus 4195.5 s) prefers a boundary-adjacent large shift while retaining a nearly comparable small-shift alternative. Low-pass changes which plausible alignments win, beyond merely smoothing the final trajectory.

At 2210 µm, the baseline and adjusted-low-pass cross-epoch constraints have **0%** weight at |D|≥50 µm and full-window cycle-error p95 of **3 µm**. Thus the earlier aggressive-screen 55–65 µm central pathology is not the dominant failure in this six-arm comparison. Fixed-threshold low-pass does create some central inconsistency, but tracing every failed arm is not the highest-priority next step.

The peak histograms show substantial time/depth-dependent population changes. Fixed-threshold low-pass retains only 31.8% of baseline detections; noise adjustment raises this to 66.8% but does not restore shallow alignment. Representative waveform overlays visibly attenuate narrow components and alter event shape. These examples are illustrative, not a quantitative neural-preservation assay: neither exact-key novelty nor visual smoothing establishes removal of noise, loss of a particular cell, or causality of localization changes.

The [completed common-event comparison](luke_lowpass_common_events_20260908.md) shows that both amplitude reweighting and localization changes perturb shallow registration on identical events; their combination nearly reproduces the full low-pass discrepancy. The next step is local reference qualification rather than further filtering sweeps. A later sum-versus-mean-amplitude raster test is technically feasible, but should target **depth-dependent rate/composition changes**: normalized correlation already cancels uniform scaling of a profile. Mean amplitude per occupied bin also amplifies the influence of sparsely occupied bins and does not remove population-composition bias. No representation change or session-wide lighthouse search was launched by this audit.

The remaining shallow 4245–4250 s baseline excursion still requires the local reference limitations below to be respected. The low-pass outcome does not validate the baseline there, and does not justify promoting any configuration to the entire recording.

## What existing evidence can resolve at 4245–4250 s

Cached accepted-event timestamps show that the shallow references are present, but their support is uneven. Unit 80 (220 µm) has 60 accepted events during these five seconds, distributed **21, 5, 7, 13, 14** per second. Unit 154 (620 µm) has 49, distributed **27, 18, 2, 1, 1**. Thus unit 154 contributes almost no accepted evidence over the final three seconds; a ten-second median can conceal this loss of support. Unit 80 retains events, but two seconds fall below the current ten-event support rule.

| Unit | Depth (µm) | Accepted events, 4245–4250 s |
|---|---:|---:|
| 80 | 220 | 60 |
| 154 | 620 | 49 |
| 246 | 1360 | 19 |
| 317 | 1740 | 0 |
| 445 | 2260 | 16 |
| 463 | 2380 | 32 |
| 510 | 2740 | 41 |
| 549 | 2960 | 152 |
| 587 | 3100 | 51 |

The two shallow templates use fixed 14-channel footprints: 160–280 µm for unit 80 and 560–680 µm for unit 154. Detection searches only channels within ±20 µm of each original peak. Matching allows temporal alignment but no spatial translation. There is no lighthouse centered near the 410 µm DREDGE window; interpolation between two distant cells does not establish that window's motion. Loss of events can reflect firing variation, spatial movement, or failed matching; these cached records cannot distinguish them.

The 4245 s centroid point pools **4240–4250 s**, mixing before-excursion events with the excursion. Its subsequent 4255 s point pools 4250–4260 s. Their small differences (unit 80: +1.81 µm; unit 154: +2.08 µm) therefore neither rule out a brief large movement nor prove the DREDGE excursion wrong. Bootstrap intervals quantify resampling of the accepted, fixed-support population only.

Existing waveform assets do not repair this temporal limitation: `luke_lighthouse_gentle_v1/templates.npz` stores training templates from 4080–4090 s, and `luke_common_residual_review_v1/lighthouse_waveforms.npz` stores aggregate before/after-compensation waveforms for the two halves of **4180–4200 s**. The target-event table stores frame, score, and gain, but no per-event target waveforms or target competitor margins. The gentle-track script did not persist the target waveform arrays. Consequently this cached audit can expose unsupported timing and alignment ambiguity, but cannot independently validate displacement during 4245–4250 s. If that excursion remains decisive after the six-arm comparison, the bounded next step is local waveform/competitor verification around this transition, before adding session-wide tracking.

This section used cached CSV/NPZ metadata and source inspection only; no recording voltage was read and no identities were changed.
