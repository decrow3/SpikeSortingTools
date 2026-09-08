# Upper amplitude cutoff on the 3σ input

Rejecting detected peaks above8σ worsens the same100s motion comparison. The uncapped3σ input remains the better candidate for this interval.

## Test and results

Existing compensated3σ negative detections, unchanged localized positions, noise vector, DREDGE settings and unchanged event-matched lighthouse references. The upper cutoff uses absolute detected negative peak amplitude divided by the frozen original detector-channel noise. It rejects events; it does not clip their histogram weights or screen the maximum voltage anywhere in the full waveform.

1299549input events;1249758retained;49791excluded. This removes3.83% of events and10.54% of summed absolute detected amplitude.

| Input | Overall lighthouse difference,µm | Drop/recovery difference,µm |
|---|---:|---:|
|Uncapped3σ|1.073|1.270|
|3–8σ|1.540|2.288|

Overall difference increases44%, and movement difference increases80%. The effect is depth dependent. Notably, the cap largely loses the observed drop around unit246and greatly attenuates it around317. Some other cells show little change or localized improvements. The aggregate result argues against this upper event cutoff. It suggests that at least some strong-peak populations contribute essential registration information; it does not prove every excluded event is neural.

## Verification and artifacts

Script `testing/luke_3sigma_upper_cap.py`; output `testing/outputs/luke_3sigma_upper_cap_v1/` includes settings, exact keep mask, sigma values, retained/excluded peaks and positions, both motion fields and constraints, score tables, depth-wise exclusion, summary, audit and figures:

- `02_lighthouse_overlay.png/.pdf`: all nine event-matched cell comparisons.
- `01_summary.png/.pdf`: agreement metrics and retention.
- `03_histograms.png/.pdf`: uncapped, capped and excluded populations.

Exact baseline field reuse, sigma<=8 mask, retained/excluded arrays, location alignment, finite motion values and explicit±80µm pairwise bounds all pass. Independent service `luke-3sigma-upper-cap-v1` exited0; actual final state inactive/dead. Launch/log/receipt under `testing/outputs/luke_3sigma_upper_cap_job_v1/`.

The user's pause on the low-pass/screening experiment remains respected: that job was not restarted. The subsequent upper-cap request authorized this separate bounded diagnostic only. Other agents and their jobs were not interrupted. No sort or production motion application was performed. Lighthouse estimates remain provisional, and this same-window comparison is not independent full-session validation.
