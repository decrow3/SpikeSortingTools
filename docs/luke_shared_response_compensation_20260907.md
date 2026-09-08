# Shared-response compensation restores bounded DREDGE motion sensitivity

On Luke 2025-08-04 imec0, 4180–4200 s, a probe-wide shared-response compensation followed by fresh peak detection and localization changed central DREDGE estimates from essentially zero to downward movement corroborated in direction by multiple lighthouse cells. This supports stationary input contamination as a cause of the flat solution in this interval. It does not establish full-recording accuracy or justify releasing the full-sort hold.

## Intervention and controls

The preceding stage audit found a large common waveform and channel-dependent residuals already present in acquisition voltage. Global median referencing leaves some residuals looking spatially localized. Conservative event rejection alone removed shared-explained events but left the population alignment at zero: a local residual protects an event even when the original detected position still describes the common artifact.

We reused a 31-tap per-channel response model of the pre-reference common median, fitted on 4180–4190 s only. Subtracting the predicted channel responses permits detection and localization of remaining local signals. Neither channel blacklists, lighthouse agreement, nor motion estimates enter the fit. Its hardware origin remains unknown, and a common biological signal could also be removed.

Three bounded stages completed:

1. **Residual review and waveform protection.** Of 742 retained shared-explained events with local residuals, two had both template and timing support; 740 remained unresolved. An unmatched event cannot be called noise given the sparse template library. On nine lighthouse cells across two halves, all 18 median-waveform checks passed: cosine similarity 0.997976–0.999874 and peak-amplitude ratio 0.993253–1.007201, using the same matched event times before and after compensation.
2. **Fresh detection and localization.** Negative locally exclusive detection at 5 sigma used the original per-channel noise vector in both arms, preserving absolute thresholds. Radius 50 µm detection and 75 µm monopolar localization were unchanged. Counts changed from 60,719 to 55,912. Lighthouse coincidence totals were 1,940 originally and 1,939 after compensation among 2,501 reference events. These are aggregate recovery counts, not a claim that precisely one original event was lost. Both central population-profile comparisons changed their preferred shift from zero to −5 µm.
3. **Actual DREDGE comparison.** Identical estimator settings were applied to original and compensated peak inputs. Both returned finite 20-time-bin × 20-depth-window fields. The metadata-only zero recording supplies clock and geometry; cached peaks and localizations supply all estimation observations.

## DREDGE result

Changes compare the second 10 seconds with the first 10 seconds. DREDGE values use medians across time bins; lighthouse values use the difference between the corresponding median-waveform centroids.

| Lighthouse | Depth (µm) | Original DREDGE (µm) | Compensated DREDGE (µm) | Lighthouse centroid (µm) |
|---|---:|---:|---:|---:|
| 317 | 1740 | −0.002 | −6.62 | −9.60 |
| 445 | 2260 | approximately 0 | −4.64 | −7.24 |
| 463 | 2380 | approximately 0 | −3.99 | −5.55 |
| 510 | 2740 | approximately 0 | −1.82 | −1.13 |

The central stationary solution is sensitive to this input intervention while the selected neural waveforms remain closely preserved. The centroid changes provide independent waveform corroboration but are not calibrated physical displacement ground truth. The experiment does not isolate compensation from its necessary redetection and relocalization.

DREDGE settings: nonrigid Gaussian windows, step 200 µm, scale 300 µm, margin 50 µm; spatial bins 1 µm, temporal bins 1 s; histogram smoothing 1 µm and 1 s; time horizon 60 s, maximum displacement 80 µm, minimum correlation 0.1, CPU. These are identical controlled settings across arms, not a reconstruction of the historical full-recording configuration.

## Scope and next step

This is the same development interval, including the model-fitting half. The preservation cohort is selected and small; coincidence is not a precision estimate. Twenty seconds is suitable for this bounded waveform/motion comparison, not amplitude-completeness estimation. Nothing here establishes a cause for the native Kilosort jumps elsewhere in the recording.

The next bounded validation should freeze the model and thresholds and apply them to a separate gentle-drift interval, checking waveform preservation, residual shared contamination, and DREDGE agreement with multiple lighthouse cells. Model stability across time must be demonstrated before broader use. No production motion correction or full sort was launched.

## Reproducible artifacts

- `testing/luke_common_residual_review.py` and `testing/outputs/luke_common_residual_review_v1/`: residual attribution, waveform checks, PNG/PDF figures.
- `testing/luke_compensated_peak_trial.py` and `testing/outputs/luke_compensated_peak_trial_v1/`: new peaks/locations, recovery table, population comparison figure. `trial_settings.json` describes fresh detection; `settings.json` includes inherited context.
- `testing/luke_compensated_dredge_trial.py` and `testing/outputs/luke_compensated_dredge_trial_v1/`: estimator settings, both motion fields, comparison CSV, summary, and `01_dredge_comparison.png` / `.pdf`.
- Model: `testing/outputs/luke_common_event_screen_v1/shared_response_model.npz`.

The stages ran as independent systemd jobs; final process state and zero exit status were checked. Saved field finiteness, comparison values, and the main DREDGE figure were reviewed.
