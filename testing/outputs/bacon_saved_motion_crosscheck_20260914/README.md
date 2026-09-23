# Bacon: saved Kilosort versus Fast MEDiCINe

The previous Bacon comparison used Kilosort 4 dshift arrays from probeA/ops.npy and probeB/ops.npy. No DREDge field was found in the original Bacon output tree searched. The saved KS fields have 3227 batches at 2 s spacing and nine depth centers.

Both methods report small motion in the three pilot windows. MEDiCINe is higher, not lower, on the matched summary metrics; this is agreement in overall magnitude class, not numerical equivalence or demonstrated trajectory agreement.

| Probe | KS rigid excursion | MED rigid excursion | KS nonrigid residual excursion | MED nonrigid residual excursion |
|---|---:|---:|---:|---:|
| Bacon A | 1.00 µm | 2.33 µm | 1.00 µm | 2.33 µm |
| Bacon B | 1.00 µm | 1.85 µm | 1.00 µm | 1.90 µm |

Values are medians of three window-level P95−P5 excursions. Comparison uses identical complete KS batches inside each pilot window (118 s), averages eight native MED samples per 2 s batch, and interpolates MED spatially onto the nine KS depth centers within shared depth coverage. This does not increase MED's four-depth model resolution. Remove each depth's temporal median before defining rigid=across-depth median and nonrigid=residual. No signed-trace agreement is asserted.

The full-session historical KS rigid P95−P5 excursions reproduce 7.5 µm (A), 3.0 µm (B); median raw depth spreads reproduce 2.1 and 1.9 µm. These full-session metrics are not equivalent to a 120 s window summary. Full-session rigid min-to-max range is 43.5 µm on each probe. Largest single 2 s rigid steps are 32.5 µm (A, around 102 s) and 37.5 µm (B, around 6438 s), outside the pilot windows. Thus the pilot does miss some larger Bacon episodes.

Native 250 ms MED metrics remain separately saved. Averaging onto 2 s batches markedly reduces its excursions/speeds; this comparison cannot validate its extra fast variation or rule out motion missed by both methods. The earlier preprocessing difference remains: KS and MED use different detection/preprocessing pipelines.

The cheapest next targeted experiment, if needed, is Fast MEDiCINe on windows covering the two saved KS largest-step events, not a full-session rerun. That would test whether MED captures already-flagged Bacon episodes but still would not establish detection of every fast transient. No new extraction, fit or sort was launched for this cross-check.

[Comparison figure](comparison.png) · [Matched metrics](matched_window_metrics.csv) · [Full-session metrics](historical_full_session_metrics.csv) · [Extrema](full_session_extrema.json) · [Provenance](provenance.json)
