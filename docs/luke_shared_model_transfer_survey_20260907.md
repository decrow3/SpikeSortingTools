# Frozen shared-response model: recording-wide reconnaissance

Applied the model fitted at 4180–4190 s without refitting to twelve evenly spaced two-second voltage samples spanning 10–10461.55 s of the approximately 174.6-minute recording. Sample times were fixed before output calculation. Filtering, gain conversion and model coefficients were unchanged; predictions used padded voltage. No peak selection, lighthouse measurements or motion estimates entered the survey.

## Findings

The median compensated/original voltage MAD across channels was 0.9914–0.9950 over the twelve samples. No sampled channel/window showed a MAD increase above 20%. This threshold is a descriptive alarm count, not a validated neural-safety criterion.

At the previously identified dominant residual channels, compensation reduced conditional energy throughout the sampled recording. During each sample's largest 1% absolute common-voltage samples, compensated/original residual-energy ratios were:

| Channel | Minimum ratio | Maximum ratio |
|---|---:|---:|
| 238 | 0.0188 | 0.0307 |
| 251 | 0.0089 | 0.0563 |
| 252 | 0.0055 | 0.0118 |

These channels are reported because prior independent diagnosis identified strong shared residuals; the model is fitted and applied probe-wide without a channel blacklist. The full channel-by-time table and heatmap are saved. Most channels have much smaller changes, as expected when ordinary referencing already removes most common voltage there. Across channels, median conditional energy ratios were 0.954–0.984; the 95th percentile was 1.014–1.065. Some channels therefore increase in conditional energy even though the median decreases.

## Interpretation and limits

The model's suppression of the identified residual pattern transfers to widely separated sampled times, supporting broader validation. This is sparse sampling (24 seconds total), not continuous coverage or evidence about every severe-motion interval. Lower MAD or conditional energy can also reflect removal of biological signal; these summaries do not prove neural preservation. None of these measurements establishes DREDGE displacement accuracy or downstream sorting benefit.

Next validate clear neural waveforms at widely separated epochs using fixed waveform-quality criteria, then extend compensated peak/motion analysis with independent multi-depth corroboration and explicit uncertainty in the shallow region. Do not release the full-sort hold on voltage-summary evidence.

## Reproducibility

Script: `testing/luke_shared_model_transfer_survey.py`.
Artifacts: `testing/outputs/luke_shared_model_transfer_survey_v1/`: settings with model hash and exact times, per-channel metrics, summary CSV, completion file and `01_transfer_survey.png` / `.pdf`. The figure was visually inspected. Heatmap colors clip outside 0.8–1.2, explicitly labeled; exact values remain in CSV.

Independent service `luke-shared-model-transfer-survey-v1` was verified running after launcher disconnection, then completed with zero exit status. Launch command, log and receipt are persisted alongside the output directory. No production recording, library or sorting configuration was modified.
