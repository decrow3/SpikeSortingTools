# Validation summary

One durable record of what the production pipeline has actually been validated
on. This is the document a production user should read to know what is and is not
established. The reasoning behind each choice is in
[`decisions/`](decisions/README.md); the full investigative record stays in
this research repository.

Research repository baseline for this summary: commit `efc873c`.

## Datasets

| Session | Probe | Duration | Role |
|---|---|---|---|
| Luke 2025-08-04 | imec1 | 10,473.6 s (full) | Primary full-session rescue reference |
| Luke 2025-08-04 | imec0 | 10,473.6 s (full) | Cross-probe control against a legacy pipeline |
| Luke 2025-08-05 | — | bounded | Cross-session recurrence check |
| Yates sessions | — | sampled | External comparator (confounded — see limits) |
| Rocky 2024-07-07 | imec1 | — | Legacy pipeline history only |

## Outcomes against the frozen criteria

The prespecified gates are tracked in
`configs/rescue/imec0_legacy_acceptance_criteria.json` (11 hard gates, a primary
target of 260 KS-good, and a written decision rule).

**imec0 verdict: `reject_universal_default`.** This stands and was not waived.
Three gates failed:

| Gate | Threshold | Observed |
|---|---|---|
| Stable-good-unit fraction | ≥ 0.75 | 0.7375 |
| Edge burden (40 µm) | ≤ 2.000% | 2.004% |
| Nearby similar good–good pairs per good unit | ≤ 0.06 | 37 pairs (over limit) |

Follow-up reduced the 37 broad pair candidates to one strong (184/191) and one
partial (164/165) duplicate hypothesis, both heavily artifact-associated.
Conservatively discounting all four units still leaves **297 KS-good units,
+14.2% over legacy**. This supports a localized artifact/template failure rather
than global unit inflation — but it does not convert the frozen verdict into an
acceptance.

**Operational status:** the rescue graph is the locked downstream reference for
bounded challengers. That is a weaker claim than universal adoption, and the two
should not be conflated.

**Second, independent ground for the verdict (2026-09-02).** The post-curation
evaluation added a concern the frozen gate set never measured: per-unit
amplitude completeness. See *Amplitude completeness* below and
[0008](decisions/0008-amplitude-completeness-gates-promotion.md). The
post-curation edge-spike fraction also rose to approximately 2.066%, from the
2.004% recorded pre-curation against a 2.000% threshold.

## Headline results

| Endpoint | imec1 | imec0 |
|---|---|---|
| Units | 583 | 727 |
| KS-good units | 216 | 301 |
| Change in KS-good | +43.0% vs best prior full-probe | +15.8% vs legacy |
| Assigned spikes | 43,669,711 (−6.7%) | 30,494,981 (−12.2%) |
| Median KS-good contamination | 3.55% | 2.50% (from 4.05%) |
| Median 1.5 ms refractory violation | 0.125% | 0.113% (from 0.203%) |
| Median KS-good presence (300 s bins) | 100% | — |
| Sealed automatic raw-event recovery | 74.54% (jitter null 22.12%, p < 0.004) | 62.3% (from 60.6%) |

Yield rose while assigned spikes fell, on both probes. **That is a
contamination-side observation and does not establish per-unit recall** — see
below.

## Post-curation comparison, imec0 (2026-09-02)

Matched on the same recording, against the legacy and claim-mask configurations.

| Metric | Rescue | Legacy | Claim-mask |
|---|---:|---:|---:|
| Curated spikes | 29,227,829 | 33,099,200 | 17,905,055 |
| Curated units | 710 | 563 | 494 |
| KS-good units | **301** | 228 | 191 |
| Stable good units | **211** | 182 | 136 |
| Stable fraction of good units | 70.1% | **79.8%** | 71.2% |
| Good units >1 Hz | **95** | 73 | 62 |
| Good units >5 Hz | 21 | **24** | 18 |
| Median good contamination | **2.9%** | 4.45% | 4.2% |
| Similar good–good pairs | 27 | **8** | 11 |

The +73 KS-good units (+32%) over legacy become only +29 stable good units
(+16%), the stable *fraction* falls by 9.7 points, and there is no gain at
>5 Hz or >10 Hz. The increase is concentrated in low-rate units.

## Amplitude completeness — reported deficit, since retracted

Estimated percentage of spikes missing below the detection boundary, summarized
per unit (median across 1,000-spike windows) then compared across methods.

| Good units >1 Hz | Rescue | Legacy | Claim-mask |
|---|---:|---:|---:|
| Median estimated missingness | 3.07% | 1.16% | **0.82%** |
| Fraction below 10% missingness | 68.8% | 77.9% | **91.8%** |

**These population figures are confounded and do not show a detection-quality
difference.** They reproduce exactly, but the three configurations admit
different unit populations, and estimated missingness depends strongly on unit
amplitude (within-method Spearman −0.44 rescue, −0.56 legacy).

Matched on the same neurons, the configurations are indistinguishable:

| Matched pair | n | A | B | Paired Δ | p |
|---|---:|---:|---:|---:|---:|
| rescue vs legacy | 43 | 0.63% | 0.63% | −0.02 pp | 0.797 |
| rescue vs claim-mask | 47 | 0.75% | 0.73% | +0.03 pp | 0.193 |
| legacy vs claim-mask | 42 | 0.60% | 0.76% | +0.01 pp | 1.000 |

The population gap decomposes entirely into the units each sort uniquely finds:
rescue is 43 shared units at 0.63% plus **50 unique units at 9.45%**; legacy is
the same 43 at 0.63% plus 25 unique at 3.01%. Rescue's extra units are smaller
(amplitude 15.8 vs 19.0) and small units sit closer to the detection floor in
every pipeline. See
[0009](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md).

**Estimator status:** validated. Unbiased to under 0.5 pp for true missing
fractions of 0.5–40%, which covers every KS-good cohort (0.6–3%). Hard-censored
at 50% — a window reported at exactly 50.0 is boundary-pinned, not measured, and
must be filtered. That affects 54.9%/56.3%/16.4% of all windows but ~0% of
KS-good windows.

## Established limits — do not overstate these results

1. **Yates parity is not proven.** The comparison is confounded by anatomy,
   depth, preprocessing, duration, and the unusually high contamination of the
   available Yates sort.
2. **Sealed-holdout recovery is not spike recall.** The holdout was selected
   automatically from raw extrema and carries no manual neural labels. It is a
   preservation guardrail.
3. **A localized middle-depth deficit exists.** imec1 automatic recovery is
   47.22% in the middle depth third vs 82.64% and 93.75% in the outer thirds.
   Recovery is also lower for negative events (64.81%) and 50–75 µV events (62.50%).
4. **The imec1 polarity signature persists.** 59.0% of all templates and 49.1% of
   KS-good templates are positive-dominant, against roughly 30% in the sampled
   Yates result. This blocks a clean biological density interpretation.
5. **Single session.** All full-session validation is Luke 2025-08-04. No
   multi-session replication cohort has been run.
6. **No manual curation** was performed in the accepted runs.
7. **Higher yield still does not by itself mean better detection.**
   Contamination and refractory improvements do not measure completeness, and
   firing-rate-bin occupancy ("stable" units) is not a substitute for
   amplitude-based missing-spike estimation. But the specific claim that rescue
   detects units *less* completely is retracted — see above.
8. **Retracted pending corrected rerun.** The earlier claim that the yield
   difference was relabelling rather than detection is unsupported. None of the 200
   KS-good units rescue has and legacy does not is a new detection; every one is
   built from spikes legacy already detected (80 legacy-MUA promotions, 85
   dispersed across many legacy clusters, 35 splits of legacy good units). The
   headline +73 is really +200/−127 — rescue also fails to reproduce 127 legacy
   good units, which no gate measures. See
   [0010](decisions/0010-rescue-yield-is-relabelling-not-detection.md), retracted
   by [0011](decisions/0011-cross-sort-event-matching-and-detection-evidence.md).
   Its matcher reused target events, while “found anywhere” had an 87–89%
   whole-probe chance baseline. Treat every number in this item as historical
   until the exclusive, spatial, null-controlled v2 output exists.
9. **A tension is unresolved:** the artifact-aware audit finds zero strong
   duplicate hypotheses, yet 27 similar good–good pairs survive curation against
   8 and 11 for the comparators. The 27 are not attributable to the extra units
   — involvement is 20.0% for splits, 14.9% for units shared with legacy, 13.8%
   for MUA promotions, 10.6% for dispersed units (odds ratio 0.66, p = 0.36).
   The cause is still unknown.

## Not yet validated

- Multi-session replication cohort
- Cross-session generalization of the frozen graph
- Selective rigid voltage correction (unimplemented, unauthorized — [0002](decisions/0002-motion-is-estimated-never-applied.md))
- Any challenger sorter in production ([0005](decisions/0005-dartsort-kiasort-deferred.md))
- MUA family promotion ([0006](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md))
- Second-machine installation and bounded smoke test
- A formal amplitude-completeness acceptance gate ([0008](decisions/0008-amplitude-completeness-gates-promotion.md)).
  The fitter audit is now complete, but the gate must be specified on
  unit-matched or amplitude-stratified comparisons, never on population medians
  ([0009](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md))
- The yield-versus-completeness trade-off curve across intermediate
  detection/artifact/claim settings

## Reproduction entry points

- Production run sheet: `SpikeGLX_ext_ref_rescue.py`
- Diagnostics: `testing/luke_full_probe_rescue_diagnostics.py`
- Acceptance evaluator: `testing/luke_imec0_rescue_acceptance.py`
- Pair audit: `testing/luke_imec0_similar_pair_audit.py`
- Post-curation evaluation: `docs/luke_20250804_imec0_postcuration_evaluation.md`
- Estimator audit and unit-matched re-analysis: `docs/luke_20250804_truncation_fitter_audit.md`
  (`testing/luke_truncation_fitter_audit.py`, `testing/luke_truncation_matched_units.py`)
- Composition of the extra units: `docs/luke_20250804_rescue_unique_units_audit.md`
  (`testing/luke_rescue_unique_units_audit.py`)
- Accepted run receipts: `rescue_pipeline_results_Luke0804_V2V1_g0_imec{0,1}/kilosort4/rescue_sort_manifest.json`

Note: the three `testing/` modules above are imported by `pipeline/downstream.py`
and are therefore production dependencies, not research scripts. They must be
ported alongside the pipeline.

## Test suite

307 tests pass under the locked runtime (`environments/rescue-production`,
Python 3.12.4).
