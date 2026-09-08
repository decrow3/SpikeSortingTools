# Luke lighthouse signal screen, 2026-09-07

Nine of 196 eligible current-reference clusters pass the explicit signal screen. This completes the first candidate screen and the waveform/repeatability portion of quiet-period verification. Independent identity discrimination from neighboring cells remains outstanding; none are yet validated motion anchors.

## Data and screen

Current accepted motion-off imec0 reference, sorted event times from cur/cur_output, all labels eligible. Read 4080–4090 and 4100–4110 seconds from recording frame zero (20 seconds total, plus filter margins). These are previously inspected quiet intervals, not a fresh identity holdout. Apply actual manifest gain, 300–6000 Hz third-order forward/backward Butterworth with 50 ms margins, then global median reference. No new sort or motion estimator run.

Require at least 15 sorted events in each interval; uniformly subsample at most 150 per interval for waveforms. Screen thresholds fixed before reading: median-waveform absolute peak >=150 µV, peak/local robust noise >=10, average local waveform cosine >=0.9 across intervals, >=70% median-waveform energy within ±60 µm, and >=80% repeat events with peak-channel waveform cosine >=0.8. Noise is per-channel MAD/0.67448975 of conditioned voltage. Screen amplitude and energy are from the first interval. Single-event repeatability is conditional on existing sorted assignments, not independent detection precision.

Passing IDs: 5, 341, 554, 587, 612, 628, 657, 666, 667. Only 5, 341 and 587 carry KS-good labels; other passes are MUA. Labels alone neither qualify nor reject a neural source. Coverage: 20 µm, 1820 µm, and seven candidates at 3000–3540 µm. Two candidates share 3540 µm and must not be counted as independent sources without discrimination.

## Figure review

Figures exported as PNG and PDF in testing/outputs/luke_lighthouse_screen_v1. Contact sheets intentionally include near misses and the earlier stationary candidate, not just passes. Individual repeat spikes are sampled uniformly without selecting good shape. Spatial images retain individual channel rows, including paired depths; they are not interpolated physical-depth maps. Gray envelope is ±3 robust noise sigma, not a waveform confidence interval.

- Unit 5 (20 µm, 164 µV, 13.8 noise sigma): clear negative trough/rebound, reproducible localized waveform. Probe-edge position limits two-sided footprint tracking.
- Unit 341 (1820 µm, 177 µV, 19.3 sigma): clear negative trough/rebound, spatially localized; useful central candidate. Earlier broad-search specificity failure remains unresolved.
- Unit 587 (3100 µm, 289 µV, 36.2 sigma): exceptionally reproducible localized positive-leading waveform. Polarity does not prove artifact or neuron; retain as morphology candidate, not proven neural identity. Earlier expanded-search failure still applies.
- Unit 628 (3220 µm, 289 µV, 29.9 sigma): strong negative trough/rebound and local footprint. Existing MUA label means mixed events/identity separation need attention.
- Unit 445 (2260 µm): convincing-looking negative trough/rebound but 69% local energy falls just below the preset 70% gate. This is a screening boundary, not evidence of non-neural origin.
- Unit 478 (2520 µm): 168 µV but only 6.4 local noise sigma; fails SNR. The previous stationary example must not serve as our lighthouse evidence.

Main limitation: signal quality can be excellent while a translated matcher confuses identities. This screen does not reverse earlier specificity failures. Next bounded work is neighbor/competing-event discrimination at a small spatial search appropriate for gentle drift, on fresh quiet data. Only after that should displacement be estimated. No amplitude completeness inference comes from these short intervals.

## Execution and QA

Managed service luke-lighthouse-screen-v1 completed: MainPID=0, ExecMainStatus=0, active/exited. Launch command, settings, service log and exit receipt persisted under testing/outputs. Script syntax checked; all four exported PNG figures visually inspected. Full-sort hold preserved. No biological motion conclusion.
