# Distant motion validation exposes failure of compensated DREDGE

> Interpretation update: the fixed-position waveform matcher rejects exact ±40 µm translations of its own templates. Small centroids of accepted events therefore cannot decisively falsify the larger motion excursions. See `docs/luke_long_context_and_tracker_sensitivity_20260907.md`; earlier failure wording below must be read with this limitation.

The frozen compensation model preserves the available early/late candidate waveforms, but the resulting DREDGE fields show large excursions inconsistent with those provisional observations. The successful central development comparisons do not generalize to these intervals. This is evidence against adopting the current compensated DREDGE configuration across the recording.

## Controlled comparison

Fresh negative locally exclusive 5-sigma detections used identical frozen per-channel noise, 50 µm radius, and 75 µm monopolar localization on original and compensated voltage at 970–990 and 9520–9540 s. The shared-response model and estimator settings were unchanged. Both arms enforced the requested ±80 µm lag bound. Exact reconstruction of the installed implementation's unrestricted D/C matrices was checked before applying the bound for every window/arm; resulting fields and localizations were finite.

Independent positive-dominant candidate waveform observations came from the preceding held-out template search. They were not used in detection, localization, DREDGE fitting or parameter selection. Three candidates lack nearby competitor templates and remain provisional; deep-only coverage is incomplete.

## Outcome

At early depths 2920 and 3380 µm, compensated DREDGE produces approximately 42 and 40 µm excursions between five-second medians. Available candidate centroid ranges are approximately 1.3 and 1.7 µm. The early 2920 µm candidate has a gap during the largest excursion, but the 3380 µm candidate has a valid waveform bin there (21 accepted events), supporting a real discrepancy rather than one inferred through a gap.

In the late interval, compensated DREDGE at 2900 µm drops approximately 20 µm while the candidate centroid range is approximately 1.2 µm. The 3300 µm candidate has only two valid late bins; its roughly 0.7 µm centroid increase contrasts with an approximately 5 µm DREDGE decrease between those same bins. Earlier low-count bins at that depth remain uncorroborated.

Original-input DREDGE is largely pinned near zero at these depths, except for a smaller early drift near 3380 µm. Apparent closeness of a stationary field in a low-motion interval does not validate the original contaminated input or undo the earlier demonstrated central failure.

All 13 available fixed-event waveform-preservation checks passed the existing cosine/amplitude-ratio gates. Thus the model can preserve these selected local waveforms while the population registration remains unreliable. Conditional waveform preservation is not enough to establish a valid motion estimator.

## Interpretation and next action

The uncertainty is now broader than the shallow 4240–4260 s pair. Removing a dominant stationary contribution can expose unstable population registration elsewhere. Possible explanations include changing population profiles, remaining artifacts, pairwise ambiguity and short-interval estimator context. The current evidence does not isolate their relative contributions, and provisional identity checks remain a limitation.

Do not tune DREDGE toward the four centroid traces. Next compare an established alternative registration method on the same cached inputs, and examine longer contiguous context where independent waveform support exists. Reuse existing narrowband/LFP work before new input sweeps. Retain the demonstrated shared-response mechanism as a useful conditioning result while treating the motion field as unvalidated.

## Artifacts and execution

Scripts: `testing/luke_distant_motion_validation.py` and diagnostic helper `testing/luke_dredge_bounded.py`.
Outputs: `testing/outputs/luke_distant_motion_validation_v1/`: all peak/location arrays, strict-bound fields, legacy fields and D/C/U constraints, settings, preservation checks, binwise waveform/DREDGE comparison, descriptive ranges and `01_distant_comparison.png` / `.pdf`. The figure was visually inspected; gaps represent missing waveform observations, not interpolated measurements.

Independent service `luke-distant-motion-validation-v1` was verified live after launcher exit and completed with zero exit status. Launch command, logs and receipt are persisted alongside the output directory. No production correction or spike sort was launched. The full goal remains incomplete, and the full-sort hold remains in force.
