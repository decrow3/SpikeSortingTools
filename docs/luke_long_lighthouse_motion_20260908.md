# Longer motion estimates with lighthouse verification

Completed the100s interval4160–4260s with all nine existing lighthouse tracks. Shared-response compensation recovers central motion that is absent from the original input, but agreement is incomplete. The aggressive waveform screen is markedly unstable in the longer comparison and should not be promoted.

## Inputs and controls

Original240989 peaks; compensated216721; screened52267. Identical DREDGE settings in all three arms, including explicitly enforced±80µm pairwise search. Frozen shared-response model and unchanged v2 waveform rules. Detection/localization recomputed only for surrounding4160–4180,4200–4220,4220–4240 windows; previously completed4180–4200 and4240–4260 arrays reused. The central20s original, compensated and screened peak/location arrays match the earlier comparison exactly. Screen refactor reproduces the existing central mask exactly. Clock uses29999.835983263598Hz and absolute sample offsets, avoiding accumulated chunk rounding.

## Figures and interpretation

`01_lighthouse_overlay.png/.pdf`: continuous1s estimates against10s lighthouse centroid summaries, all nine units across220–3100µm. Each estimator is offset by its first10s temporal median; each cell by its first centroid. No fitted sign, gain or lag. Bootstrap bars quantify bin centroid sampling only; shared baseline and identity uncertainties are not included. Panel y-ranges differ to show each depth clearly.

`03_event_matched_overlay.png/.pdf`: the estimator is instead interpolated at each actual accepted lighthouse event frame, then summarized in the same10s bins as the centroids. This is the fairer comparison when accepted spikes sample time unevenly. Each series is offset by its own first-bin value. The CSV event time_s field labels bins; actual timestamps are computed from frame/fs. Empty bins are left missing. Sampling alignment uses the existing reference events, not the screened subset.

Compensation often follows the slower lighthouse trends, including the central drop and recovery. Unit445 agreement looks substantially better when the motion field is sampled at the actual accepted spike times than when uniformly averaging each10s interval. Unit317 still shows smaller displacement than its centroid at the main drop; other mismatches persist.

The aggressive screen produces large central reversals and excursions, including roughly−30µm around4230–4240s that are not corroborated by the available cell summaries. This longer run strengthens the case against that particular broad inclusion mask.

There are unresolved brief shallow excursions around4245–4250s in all inputs. Compensated motion reaches roughly−60µm near unit80 and−50µm near154 in the continuous plot. The accepted lighthouse stream has62(unit80) and49(unit154) events at times when interpolated compensated displacement is below−20µm, so lack of events alone cannot dismiss the disagreement. Event-time matching reduces some averaging mismatch but does not resolve it. Fixed-template selection/identity errors and registration errors remain possible; these excursions are not verified physical movement by this audit.

`02_depth_fields.png/.pdf`: full-probe fields on a common color scale, offset by first10s at each depth. Color limits use the largest per-arm99th percentile absolute displacement and therefore saturate the most extreme excursions; consult the trajectories/NPZ for their full size.

## Verification and provenance

Scripts: `testing/luke_long_lighthouse_motion.py`, `testing/luke_long_lighthouse_motion_audit.py`, `testing/luke_long_lighthouse_event_sampling.py`.

Outputs: `testing/outputs/luke_long_lighthouse_motion_v1/` includes settings, summary, audit, chunk arrays and masks, all three fields and pairwise constraints, binned and event-matched comparisons, shallow event-support counts, and figures inPNG/PDF. Exact mask/array reuse, index ordering and bounds, field dimensions, finite values and pairwise displacement bounds all pass. The descriptive median differences from centroids are not ground-truth motion errors and can favor a flat estimate in mostly stationary bins.

Independent service `luke-long-lighthouse-motion-v1` completed with exit0; final actual state checked inactive/dead. Launch, stdout/stderr and exit receipt persist under `testing/outputs/luke_long_lighthouse_motion_job_v1/`. No sort or motion application to source voltage was performed. This is a100s motion-verification experiment, not an amplitude-completeness or full-session validation.
