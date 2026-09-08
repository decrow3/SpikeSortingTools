# Existing motion estimator failure audit: first localization of the disagreement

New continuous lighthouse estimation is deferred. The first audit points to the historical peak representation as a plausible source of the central stationary estimate, without yet establishing whether preprocessing, detection, localization, or actual source mixture is responsible. Lighthouse centroid changes remain uncalibrated, so disagreement alone does not prove the estimator wrong.

## Evidence

In the already reviewed 4160–4260 s interval, historical DREDGE displacement range is only 0.007–0.074 µm at its 2060–2860 µm depth centers. This is not a plotting/interpolation artifact: the saved motion columns themselves are nearly flat. The same region is unusually stationary across the full recording (approximately 1.1–1.5 µm range). Decentralized estimates also have a nearly stationary central band in this epoch. Historical KS sidecar is distinct from the current native rigid run and should not be conflated with it.

The historical peak cache contains 332,702 detections during the interval; 83,223 localize to 2000–2900 µm. Thus this band is not simply empty. Channels 252 and 238 contribute 22.73% of its events and 43.12% of its summed absolute saved amplitude. Summed amplitude is descriptive, not a reconstruction of the complete registration weighting (which may normalize, average bins and smooth). Channel 252 supplies 11,229 events, median saved amplitude 68 extractor units, median localized depth 2514.61 µm, depth IQR 1.59 µm. This concentration does not by itself prove artifact.

A bounded alignment-objective diagnostic compared 4180–4190 with 4190–4200 s, when several lighthouse footprints decrease. We summed absolute peak amplitudes in 1 µm depth bins, Gaussian-smoothed at 1 µm, and evaluated centered profile correlation over ±20 µm shifts:

- 320–920 µm: best shift −4 µm; best correlation 0.454 versus 0.327 at zero.
- 1960–2560 µm: best shift zero, correlation 0.995.
- 2260–2860 µm: best shift zero, correlation 0.996.
- Removing channel 252, or channels 238/251/252 together, leaves the central best shift at zero. After removing all three, zero-shift correlation remains 0.826 and 0.895 for the two central bands.

This is a diagnostic of the existing peak representation, NOT a replay of DREDGE/decentralized with their historical exact settings. It nevertheless demonstrates how a stationary solution can be supported strongly by those inputs, despite changing selected waveform footprints. A simple dominant-channel deletion is not sufficient in this diagnostic.

## Reuse of existing tests

Reviewed the completed imec0 input-factorial summary. Prior synchronous-event exclusions generally retain very high correlations with the full input, and removing the globally most frequent detection channel usually changes the rigid field little. Those tests used global aggregate correlations and a single count-dominant channel; they do not settle local sensitivity in this exact interval. The high-amplitude subset has larger effects in some regimes, especially decentralized support dropout, so amplitude selection is not innocuous. No broad factorial was rerun.

Installed SpikeInterface DREDGE code explicitly builds an amplitude-weighted histogram and allows averaging within bins. Exact original parameters are not stored alongside the three historical field arrays. Current installed defaults must not be presented as recovered historical settings.

## Next focused diagnostic

Trace matched lighthouse events through the historical pipeline stages: conditioned voltage waveform → selected peak and polarity → detected channel → localized position → histogram contribution. In particular determine whether the stationary peak ridges are neuronal sources that really remain fixed, artifacts, or neural spikes whose localization is insensitive/biased under the historical conditioning. Compare actual examples and coordinate conventions before changing filters or estimation settings.

Keep the native KS large-jump issue separate: it uses different conditioning/detections, and its full event table remains on huklaban5. Its aggregate shared field cannot establish the same cause. Reuse the existing narrow-band and LFP work rather than launch new branches now.

## Artifacts and scope

Outputs in testing/outputs/luke_existing_motion_failure_audit_v1: field ranges, channel support, single-pair objective scores, and two PNG/PDF figures. Both figures inspected. Scripts are testing/luke_existing_motion_failure_audit.py and testing/luke_registration_objective_diagnostic.py. These read only cached peaks/fields; no voltage, sort, correction, or production estimator rerun. The full-sort hold remains in place.
