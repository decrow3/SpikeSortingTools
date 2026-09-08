# Same-event lighthouse waveform audit: 3 kHz low-pass

The added 3 kHz low-pass has **small filter-induced energy-centroid changes on the accepted lighthouse events, but strongly heterogeneous waveform attenuation**. It cannot be described as preserving every cell's waveform equally. This audit supports testing a change in detection population as a mechanism for altered DREDGE behavior; it does not establish motion accuracy or justify adopting the filtered input on its own.

## Evidence and interpretation

We sampled 2,807 of 4,392 previously accepted events from nine units over 4180–4200 and 4240–4260 s, at most 100 uniformly spaced accepted events per unit/10 s bin. Broadband and low-pass waveforms use exactly the same frames and spatial support. Each of 36 unit/bin combinations was evaluated both on the original template channels and an expanded ±120 µm support. Thus the 5,614 event-metric rows represent two supports of 2,807 events, not 5,614 independent events.

Across original-support median templates, the median multichannel cosine is **0.9538** (minimum 0.8187), and the median low-pass/broadband peak-amplitude ratio is **0.7465**. Amplitude ratios vary considerably by unit:

| Unit | Median amplitude ratio | Median multichannel cosine | Maximum change in filter-induced centroid offset across four bins (µm) |
|---|---:|---:|---:|
| 80 | 0.555 | 0.907 | 0.600 |
| 154 | 0.707 | 0.950 | 0.732 |
| 246 | 0.784 | 0.977 | 0.344 |
| 317 | 0.827 | 0.954 | 0.891 |
| 445 | 0.768 | 0.951 | 0.425 |
| 463 | 0.682 | 0.953 | 0.785 |
| 510 | 0.444 | 0.833 | 0.719 |
| 549 | 0.774 | 0.969 | 0.129 |
| 587 | 0.965 | 0.989 | 0.079 |

The all-unit contact sheet shows that unit 510's sharp multiphasic structure is substantially smoothed: its narrow central feature is largely removed, and its peak amplitude falls to about 44% of baseline. Unit 80 also loses nearly half its amplitude. Unit 587 is much less affected. These are real waveform transformations, despite moderate-to-high pooled cosine. Fixed absolute thresholds can therefore disproportionately lose some cells; noise-adjusted thresholds do not automatically restore the original cell mixture.

The median absolute **median-template energy-centroid offset** is 0.440 µm on original support, with maximum 2.373 µm. Expanded support gives median 0.562 µm and maximum 2.737 µm. Those absolute offsets include stable cell-specific changes. For relative motion, the more relevant diagnostic is the change in each cell's filter-induced offset between bins: the largest pairwise difference across the four bins is **0.891 µm on original support**, and **1.503 µm on expanded support**. These are descriptive differences on the audited samples, **not calibrated motion error, uncertainty bounds, or monopolar localization bias**.

## Limits and next use

- Identity and frame selection are inherited from the existing lighthouse matcher. Fixed support can preferentially retain stationary or smaller-displacement events. No new identity validation or low-pass detection-recall measurement was performed.
- Some late bins have only 10–14 accepted events. Their median waveforms are visibly noisier; no confidence interval for preservation metrics was calculated.
- The contact sheet fixes one strongest broadband channel per unit across all four bins and both filters. Per-panel amplitude/cosine annotations summarize all original support channels, not just that displayed channel. Voltage scales are shared within each unit's row, not across units.
- The result narrows the plausible mechanism: large changes in a population motion estimate need not imply comparable spatial displacement of the retained lighthouse waveforms. Investigate differential detection/weighting and the matched-event localization comparison before declaring waveform fidelity sufficient.

## Reproducible artifacts

- Input provenance, metrics, median templates, and initial figure: `testing/outputs/luke_lowpass_waveform_preservation_v2/`.
- All-unit contact sheet: `02_all_unit_waveform_contact_sheet.png` and `.pdf` in that directory; deterministic selected channels in `contact_sheet_selection.csv`.
- Numerical audit: `testing/luke_lowpass_waveform_preservation_v2.py` (managed run completed in 39.47 s, exit 0).
- Independent renderer: `testing/luke_lowpass_waveform_contact_sheet_v2.py`; reads saved templates only. Run with `environments/rescue-production/.venv/bin/python -m testing.luke_lowpass_waveform_contact_sheet_v2`.

Validation: same-shape amplitude-scaling synthetic check passed; all nine units and four bins present; exported contact sheet visually inspected for scales, labels, selection consistency, and waveform differences. No voltage reread or modification of the launched six-arm comparison was needed for the contact sheet.
