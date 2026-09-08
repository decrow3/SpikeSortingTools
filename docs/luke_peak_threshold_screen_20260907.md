# Fresh peak thresholds and provisional morphology screen

Fresh 4/5/6/7σ detection does not resolve the central stationary registration-input pattern in the first paired epoch. Raising thresholds removes many peaks and loses some lighthouse recovery. A simple broad-synchrony/brief-impulse screen also does not resolve it and visibly flags some convincing neural waveforms, so it is not suitable for automatic exclusion.

## Scope and controlled comparison

First bounded pair: 4180–4190 versus 4190–4200 s, encompassing a shared lighthouse decrease. Actual reference fs 29999.835983263598 Hz, 384 channels; accepted conditioned int16 voltage converted with manifest gain, 300–6000 Hz third-order forward/backward Butterworth with 50 ms margins, then global median reference. This matches the lighthouse voltage branch, not a verified replay of historical preprocessing.

Fresh SpikeInterface locally-exclusive detections were independently run at 4, 5, 6, 7σ, negative polarity, 50 µm exclusion radius. The same channel MAD noise vector was supplied for all thresholds. Their union contains 118,198 events. Localize that union once with monopolar triangulation, radius 75 µm, ±0.5 ms waveform window, installed defaults: ptp feature, enforce_decrease=True, minimize_with_log_penality optimizer, max_distance_um=150. Map identical event/channel keys back into each threshold branch. No amplitude-subsetting proxy for detector execution was used.

| Threshold | Detected peaks | Provisional flags | Median lighthouse coincidence |
|---|---:|---:|---:|
| 4σ | 118,198 | 2,331 | 100.0% |
| 5σ | 60,719 | 1,227 | 99.2% |
| 6σ | 35,723 | 762 | 95.4% |
| 7σ | 22,120 | 495 | 86.1% |

Coincidence means a peak within ±0.8 ms and detection-channel depth ±60 µm of one of the twenty candidate reference events. It is not confirmed one-to-one identity, precision, or full-population recall. Negative-only detection can disadvantage positive-leading waveforms. The detector polarity is constant across thresholds. Values are conditional on current sorted reference trains.

## What changes and what does not

- 7σ removes 81.3% of the 4σ detections; they cannot all be described as noise.
- Many clear negative-leading candidates remain well recovered. Others lose most coincidences (e.g. candidate 510: 79.6% at 4σ, 4.4% at 7σ).
- Fresh localized peaks near units 341, 445, 463 and 317 show approximate decreases of 6.4, 6.2, 6.1 and 9.5 µm at 4σ. These are nearest coincident-peak medians, not identity-validated physical displacement.
- Yet the full fresh amplitude-weighted peak profiles in both central windows, 1960–2560 and 2260–2860 µm, still prefer zero shift at every threshold. Zero-shift correlations span approximately 0.992–0.998. The diagnostic uses identical 1 µm summed-amplitude profiles/smoothing as the preceding audit, not the complete DREDGE objective.
- Removing provisionally flagged peaks leaves every preferred shift at zero. Therefore neither tested change supplies a reason to rerun production motion estimation yet.

## Non-neural screening review

Two deliberately inspectable provisional flags were tested: at least twenty channels more than 200 µm away exceeding five noise σ within ±0.1 ms of the detection; or negative-detection absolute half-height width below 0.06 ms. Edge events without full waveforms are excluded from provisional masks rather than called noise. Of fully inspected union events, 2,227 carry the broad flag and 105 the brief flag; overlaps are possible.

Visual review rejects using these flags as automatic non-neural labels. Broad examples include clear neural trough/rebound waveforms coinciding with other activity. Very brief negative extrema can accompany plausible positive-leading waveforms. Unflagged examples also need review: absence of either flag does not establish neural origin.

Next screening design should use repeatable full spatiotemporal waveform shape and spatial footprint, explicitly allow both polarities, and distinguish coherent artifacts from independent coincident spikes. Inspect the stationary, high-contribution source populations directly. Stationarity itself must not be used as the artifact label, since that would make the motion comparison circular. Do not lower the validation standard simply to obtain fewer peaks.

An exploratory phase check on detection channels 238/251/252 did not show strong 50/60/100/120 Hz locking (phase resultant magnitudes <=0.056). This small check does not establish neural origin or exclude other periodic artifacts.

## Artifacts and validation

Outputs: testing/outputs/luke_peak_threshold_screen_v1. Full peak arrays, locations, common noise vector, provisional masks, per-peak morphology, per-candidate coincidence and localized-step tables, registration-objective scores, and three PNG/PDF figure pairs are retained. Morphology example sheets display strongest examples per category/depth band and are deliberately not prevalence samples.

Service luke-peak-threshold-screen-v1 completed: MainPID=0, ExecMainStatus=0, active/exited. Launch, settings, log, and final exit receipt are saved outside chat. Detection-key correspondence asserted; figures inspected. No sort, full-recording run, motion-estimator rerun or production rejection rule was applied. Full-sort hold preserved.
