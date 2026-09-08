# Candidate footprint audit: background energy biased the compactness screen

Audited all 12 previously rejected direct-detection cohorts using full-probe original-voltage median waveforms from each five-second half. Candidate events and thresholds were unchanged. The audit includes a matched-count random-time background reference, cross-half waveform dot products, and peak-channel alignment over ±3 samples.

## Finding

The original compactness statistic divides squared median-waveform energy within ±60 µm by squared median-waveform energy across all 384 channels. With limited event counts, non-repeating background across hundreds of channels contributes substantially to the denominator. This can reject a clear localized waveform without establishing a broad coherent source.

Four large, repeatable positive-dominant cohorts illustrate the issue:

| Cohort | Original local energy fraction | Signed cross-half local fraction | Far energy / random-reference energy | Event-shape fraction |
|---|---:|---:|---:|---:|
| Early ch293 / 2920 µm | 0.481 | 0.962 | 1.31 | 0.970 |
| Early ch338 / 3380 µm | 0.588 | 0.974 | 0.98 | 0.990 |
| Late ch290 / 2900 µm | 0.538 | 0.956 | 1.04 | 0.920 |
| Late ch330 / 3300 µm | 0.674 | 0.983 | 1.07 | 0.938 |

Cross-half energy is the signed sum of products of median waveforms from independent event halves, calculated channel by channel. Non-repeating background largely cancels. It is a diagnostic, not a bounded probability or calibrated acceptance statistic; ratios can exceed one when far products are negative. A single random-time reference is descriptive and does not provide a significance test.

The strongest distant channels generally coincide with the previously identified residual region around 2380–2520 µm, but their median waveforms repeat poorly across halves. Local signals and unrelated common/background contamination can coexist. Prereference common coherence alone should therefore not be used to classify an entire candidate cohort as non-neural.

## Alignment and mixed-event issues remain separate

For the early channel-322 negative cohort, ±3-sample alignment leaves the event-shape pass fraction at 0.74. Its positive cohort remains at 0.673. Their consistency failure is not explained by this small timing jitter alone. The late channel-322 positive cohort remains at 0.122 and should not be promoted by the footprint result.

Late channel225 improves from 0.733 to 0.900 with alignment, but other repeatability limitations remain. Edge cohorts remain unsuitable as full position guides. A static waveform-repeatability gate can also reject a moving unit; identity must ultimately be tested with spatially varying templates, rather than requiring fixed position before tracking.

## Consequence for next work

The four cohorts above are defensible leads for independent identity validation because they are large, repeatable, and have concentrated repeatable waveform structure. Their successful compensation-preservation observations remain available from the preceding review. They are not yet validated lighthouse cells or motion ground truth. No candidate should be selected for agreement with DREDGE.

Use independent event detection and template competition on a separate time interval to qualify these leads, with explicit uncertainty and coverage limitations. Improve the compactness assessment to account for finite-sample background before scaling candidate screening; do not silently lower the old threshold or retroactively relabel all rejected events.

## Artifacts and execution

Script: `testing/luke_candidate_footprint_audit.py`.
Outputs: `testing/outputs/luke_candidate_footprint_audit_v1/`: full-probe median and random-reference waveforms, per-cohort metrics, settings and three PNG/PDF footprint sheets. The first sheet was visually reviewed, and the numerical summary covers all 12 cohorts.

Independent service `luke-candidate-footprint-audit-v1` completed with zero exit status and terminal state was checked. Launch, logs and receipt are persisted. No production correction, classifier exclusions, motion settings or spike sort was changed. Recording-wide motion reliability and downstream sorting/amplitude-completeness benefit remain unverified.
