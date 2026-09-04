# Luke imec0 frozen rescue control

> **COMPLETED HISTORICAL PLAN — interpretation updated 2026-09-03.** The frozen
> gate result and raw measurements remain valid. Higher yield and favorable
> conventional metrics do not establish detection or biological superiority.
> The later completeness comparison and its first matched-unit correction were
> both retracted; see [`validation-summary.md`](validation-summary.md) and
> [decision 0015](decisions/0015-corrected-cross-sort-audits-do-not-establish-equivalence.md).

## Purpose

imec0 is the prespecified negative control for the full-probe rescue pipeline.
It was not the failing stream and already has a strong legacy full-session
result. The control asks whether the same pipeline policy can rescue imec1
without materially degrading imec0. It does not require identical results
between probes.

## Re-audited legacy baseline

The existing imec0 pipeline output was re-scored with the same diagnostic code
used for the accepted imec1 rescue. Over 10,473.6 s it contains 34,721,074
spikes, 602 units and 260 KS-good units. Median KS-good contamination is 4.05%,
median 1.5 ms refractory violations are 0.203%, median 300 s presence is 100%,
and 205/260 good units are present in at least 90% of bins. The median
holdout-window coincidence excess is 0.162.

The sealed automatic raw-event holdout recovers 60.65% overall versus a 21.82%
jitter-null mean. As on imec1, the middle depth third is the weakest stratum:
25.0% recovery versus 72.9% and 84.0% in the outer thirds. This recurring
middle-depth pattern predates the rescue run and should not be attributed to
the new pipeline without a paired decline.

## Frozen acceptance rule

The primary target is at least 260 KS-good units while passing every quality
gate. A result with 234--259 KS-good units is only a qualified outcome and
requires at least two material quality improvements. Fewer than 234 good units,
or failure of any hard quality gate, rejects the rescue settings as a universal
default and retains them as an imec1-specific policy pending diagnosis.

The machine-readable thresholds are frozen in
`testing/outputs/luke_full_probe_rescue_diagnostics_imec0_legacy/acceptance_criteria.json`.
They cover spike-count inflation, good-unit contamination and refractory
violations, longitudinal presence, coincidence excess, overall and
middle-depth holdout recovery, edge burden, and similar good-template burden.

## Run configuration

The control uses `SpikeGLX_ext_ref_rescue.py` with explicit stream
`imec0.ap`, the same preprocessing policy and frozen Kilosort settings as
imec1, and automatic channel selection from the frozen similarity/noise
thresholds. It does not copy imec1's AP191 repair mask. The output is
`/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0`.
The raw-over-500-uV artifact sidecar is diagnostic and does not alter sorter
input.

## Result

The completed rescue increased KS-good yield from 260 to 301 (+15.8%) while
reducing assigned spikes by 12.2% and improving contamination, refractory,
depth-continuity, holdout-recovery and coincidence endpoints. The frozen
universal-adoption evaluator remains formally negative because three gates
failed, principally the broad similar-good-pair screen. Exact artifact/CCG
follow-up reduces 37 broad pairs to one strong and one partial duplicate
hypothesis, both artifact-associated. Residual and waveform review finds no
outside-artifact subset for any of the four units, extreme positive-dominant
morphology/raw threshold footprints, and no evidence that two distinct
templates improve the coincident-event residual. The units remain unmerged and
are conservatively discounted for sensitivity accounting. This leaves 297 good
units (+14.2% versus legacy). The historical frozen decision remains negative,
but the rescue graph is locked as the current downstream Luke reference for
bounded challengers. See `docs/luke_20250804_imec0_rescue_result.md`.
