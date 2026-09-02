# Luke imec0 frozen-rescue control result

> **Superseded in interpretation, 2026-09-02.** These pre-curation measurements
> stand. The post-curation evaluation in
> [`luke_20250804_imec0_postcuration_evaluation.md`](luke_20250804_imec0_postcuration_evaluation.md)
> adds an amplitude-completeness analysis showing rescue units are typically
> *less* completely detected than both the legacy and claim-mask comparators,
> and reports 0 strong duplicate hypotheses among the 27 similar pairs surviving
> curation. Read it before citing the yield results below.

The cross-probe control supports the frozen rescue graph as a substantial
improvement on imec0. It is now locked operationally as the downstream Luke
reference graph, while the prespecified universal-adoption result remains the
original `reject_universal_default` rather than being waived retrospectively.

The rescue produced 30,494,981 spikes, 727 units and 301 KS-good units, versus
34,721,074 spikes, 602 units and 260 KS-good units in the legacy imec0 pipeline.
This is 41 more KS-good units (+15.8%) with 12.2% fewer assigned spikes. Median
KS-good contamination improved from 4.05% to 2.50%, median 1.5 ms refractory
violations improved from 0.203% to 0.113%, median depth excursion fell from
25.9 to 20.2 µm, sealed-event recovery rose from 60.6% to 62.3%, and median
coincidence excess fell from 16.2% to 14.0%.

Three frozen gates failed. The stable-good-unit fraction was 73.75% versus a
75% threshold, although the absolute number increased from 205 to 222. Edge
burden was 2.004% versus a 2.000% threshold. The substantive failure was 37
nearby similar good--good template pairs versus a six-percent pair-count limit.

The exact >500 µV artifact sidecar completed in 2 h 55 min using the validated
20-worker implementation. It contains 3,921,905 threshold points and 397,839
claim-active samples, excluding AP191-only threshold events. CCG, waveform,
refractory and artifact-proximity refinement reduces the 37 broad candidates
to one strong duplicate hypothesis and one additional partial hypothesis. Both
are heavily artifact-associated. The strong pair is units 184/191; units
164/165 form the partial hypothesis. Conservatively discounting all four units
would leave 297 KS-good units, still 37 (+14.2%) above the legacy baseline.

The residual and waveform review is complete. Every spike from all four units
is within 0.5 ms of a sidecar claim sample, so an outside-artifact counterfactual
does not exist. Their templates are unusually large and positive-dominant; unit
164 is also at the 100th percentile for active-channel count and the 99.7th
percentile for active depth span among KS-good units. In deterministic samples,
the median maximum number of channels simultaneously over 500 µV was 15, 19,
269 and 75 for units 184, 191, 164 and 165, respectively; all four were above
the 92nd percentile of good units and their 90th-percentile footprints exceeded
333 channels.

At 128 reconstructed coincident events, adding a second template produced
negligible residual improvement, but neither saved template fit the empirical
waveforms well: median template cosines were only 0.18--0.29 and residual-energy
fractions were 0.97--0.98. This does not authorize a biological merge. Together
with the morphology and raw-channel footprint it supports classifying these as
four artifact-associated questionable units for sensitivity accounting.
Proximity alone is not causal evidence because the units themselves may
contribute to the threshold crossings.

The screens remain diagnostic and do not merge or relabel units. The frozen
evaluator therefore remains `reject_universal_default` as originally specified,
but the completed follow-up supports a localized artifact/template failure
rather than global unit inflation. Conservatively discounting all four units
still leaves the rescue at 297 KS-good units (+14.2% versus legacy). The rescue
graph is consequently the locked downstream reference for bounded challengers;
this does not convert the historical frozen decision into an acceptance.

Evidence:

- `testing/outputs/luke_full_probe_rescue_diagnostics_imec0_rescue/`
- `testing/outputs/luke_imec0_similar_pair_audit/`
- `testing/outputs/luke_imec0_artifact_pair_residual_review/`
- `testing/outputs/luke_imec0_artifact_pair_review/`
- `/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/artifacts/raw_over_500uv.h5`
