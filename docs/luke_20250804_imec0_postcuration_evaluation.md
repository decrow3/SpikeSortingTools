# Luke0804 imec0 rescue pipeline evaluation (post-curation)

**Recording:** Luke0804_V2V1_g0, imec0.ap
**Duration:** 10,473.55 s
**Rescue sort identity:** `22ded4d503b6de8edf4851a08797ae4e594fe41118b913451365727ffbd616ac`
**Assessment:** Do not promote the rescue configuration as a universal production default.

> **CURRENT CORRECTION — 2026-09-03.** This report does not establish an
> amplitude-completeness deficit or equality between pipelines. The original
> population comparison was composition-confounded; the first correction below
> then used a non-exclusive matcher that could reuse target events. Its
> 43/47/42 matched-unit counts, paired estimates, and “50 additional units”
> decomposition are withdrawn under
> [decision 0011](decisions/0011-cross-sort-event-matching-and-detection-evidence.md).
> The exclusive v2 completeness comparison has not yet been run. The assessment
> above remains justified only by the original frozen gates that failed. The
> body below is retained as historical audit context, not as a current finding.

> **CORRECTION, 2026-09-02 (same day).** The amplitude-completeness finding
> below is a unit-composition artifact and is retracted. Matched on the same
> neurons the three configurations are indistinguishable (rescue 0.63% vs
> legacy 0.63% on 43 shared units, p = 0.80); the population gap is entirely
> the 50 additional, smaller units rescue recovers. All numbers below
> reproduce exactly — the arithmetic is right, the inference is not. See
> [`luke_20250804_truncation_fitter_audit.md`](luke_20250804_truncation_fitter_audit.md)
> and [`decisions/0009`](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md).
> The assessment above still stands, on the prespecified gates that failed.

> Recorded 2026-09-02. This is the post-curation evaluation. It supersedes the
> yield-centred reading of
> [`luke_20250804_imec0_rescue_result.md`](luke_20250804_imec0_rescue_result.md)
> without contradicting its measurements: that document reported pre-curation
> diagnostics, this one reports matched post-curation comparisons plus a new
> amplitude-completeness analysis that was not previously performed.
>
> Decision consequences are recorded in
> [`decisions/0008-amplitude-completeness-gates-promotion.md`](decisions/0008-amplitude-completeness-gates-promotion.md).

## Executive summary

> **Historical interpretation, withdrawn.** This section predates the matcher
> correction above.

The new rescue pipeline increases nominal unit yield and improves several
conventional quality metrics, including median contamination and refractory
violations. However, the amplitude-truncation analysis raises a major concern:
good units from the rescue sort do not appear more completely detected within
supported temporal windows.

Among good units firing above 1 Hz, the typical rescue unit has an estimated
3.07% of spikes missing below the detection boundary, compared with 1.16% for
the legacy sort and 0.82% for the claim-mask sort. Only 68.8% of eligible
rescue units have median estimated missingness below 10%, versus 77.9% for
legacy and 91.8% for claim-mask.

This materially changes the interpretation of the increased yield. The rescue
finds more nominally good units, but the additional population is not
demonstrably more complete and may contain partially detected, fragmented, or
noise-contaminated units. The current evidence does not establish which
mechanism is responsible, but it is sufficient to reject universal adoption
pending further diagnosis.

The existing formal decision — `reject_universal_default` — should remain in
force.

## Pipeline configurations compared

1. **New rescue:** external-reference rescue pipeline without voltage-domain
   motion correction.
2. **Legacy:** previous `pipelineold`-style processing and curation.
3. **Claim-mask:** patched pipeline using the prior artifact-claim masking
   strategy.

All comparisons cover the same 10,473.55 s imec0 recording. The new rescue
outputs are bound to the frozen sort identity above.

## Post-curation population results

| Metric | New rescue | Legacy | Claim-mask |
|---|---:|---:|---:|
| Total curated spikes | 29,227,829 | 33,099,200 | 17,905,055 |
| Total curated units | 710 | 563 | 494 |
| KS-good units | **301** | 228 | 191 |
| Stable good units | **211** | 182 | 136 |
| Stable fraction of good units | 70.1% | **79.8%** | 71.2% |
| Median good-unit rate | 0.338 Hz | 0.443 Hz | **0.495 Hz** |
| Good units >1 Hz | **95** | 73 | 62 |
| Good units >5 Hz | 21 | **24** | 18 |
| Good units >10 Hz | 4 | **6** | 1 |
| Median good contamination | **2.9%** | 4.45% | 4.2% |
| Similar good–good pairs | 27 | **8** | 11 |

Relative to legacy, the rescue produces 73 more KS-good units, a 32% increase.
However:

- Stable-good yield increases by only 29 units, or 16%.
- The fraction of good units classified as stable decreases by 9.7 percentage
  points.
- The rescue does not increase the number of >5 Hz or >10 Hz good units.
- Much of the yield increase is therefore concentrated among low-rate units.

The rescue's lower median contamination and refractory-violation rates are
favorable, but they do not establish detection completeness.

## Amplitude-truncation analysis

### Metric definition

The truncation analysis divides sufficiently supported continuous activity into
1,000-spike windows. Within each window it fits the observed amplitude
distribution and estimates the percentage of spikes missing below the detection
boundary.

The relevant result is `mpcts`, not the duration or number of `valid_blocks`.

For the comparisons below:

1. `mpcts` was summarized across windows for each unit using the median.
2. Unit-level medians were then compared across methods.
3. This prevents high-rate units from dominating simply because they contribute
   more windows.
4. Lower estimated missingness is better.

### Unit-balanced missingness

| Good-unit cohort | New rescue | Legacy | Claim-mask |
|---|---:|---:|---:|
| All eligible good units | 2.97% | 1.04% | **0.89%** |
| Good units >1 Hz | 3.07% | 1.16% | **0.82%** |
| Good units 1–5 Hz | 3.09% | 0.85% | **lowest** |
| Good units >5 Hz | 3.02% | 2.47% | **0.98%** |

The rescue has greater typical estimated missingness in every reported cohort.

### Units below a 10% median-missingness threshold

| Good-unit cohort | New rescue | Legacy | Claim-mask |
|---|---:|---:|---:|
| All eligible units | 71.8% | 78.2% | **87.5%** |
| Good units >1 Hz | 68.8% | 77.9% | **91.8%** |
| Good units >5 Hz | 76.2% | 66.7% | **94.4%** |

The >5 Hz comparison between rescue and legacy is mixed:

- The rescue has slightly higher typical missingness: 3.02% versus 2.47%.
- A larger fraction of rescue units remain below 10% missingness: 76.2% versus
  66.7%.
- Legacy has more units with severe or ceiling-level estimates, which raises its
  mean despite a better median.

Claim-mask is consistently strongest across both median missingness and the
fraction of well-captured units.

### Eligibility

The 1,000-spike support requirement excludes many low-rate good units:

- Rescue: 110 of 301 good units had fitted windows.
- Legacy: 78 of 228.
- Claim-mask: 72 of 191.

This coverage is similar proportionally across methods. In the >1 Hz cohort
nearly every unit is eligible, making that comparison the most informative
current population-level result.

### Interpretation

The amplitude-truncation results do not support the claim that rescue units are
more reliably or completely detected. They suggest instead that:

- The rescue's increased nominal yield comes with poorer typical amplitude
  completeness.
- The added units may include partially observed units whose low-amplitude
  spikes are missed.
- The additional units could also reflect fragmentation or threshold-related
  splitting, although the current analysis does not prove that mechanism.
- Claim masking sacrifices substantial yield but produces the strongest
  amplitude-completeness profile.

## Similar-unit and artifact evidence

Before curation the rescue produced 32 similar good–good pairs across 52 unique
good units. After curation 27 similar good–good pairs remained, compared with 8
for legacy and 11 for claim-mask.

The artifact-aware audit found:

- 0 strong duplicate hypotheses
- 0 partial-or-strong duplicate hypotheses
- 0 artifact-enriched coincident pairs
- 10 artifact-associated pairs
- 0 pairs both strong and artifact-associated

This is reassuring with respect to obvious duplicate units and direct
artifact-driven coincidence. It does not resolve the amplitude-truncation
problem.

The evidence therefore contains an important tension:

- Refined CCG/template analysis does not identify strong duplicate pairs.
- The rescue nevertheless retains substantially more similar pairs than either
  comparator.
- Amplitude completeness is worse despite the higher nominal yield.

One plausible explanation is subtler fragmentation or partial detection that
does not satisfy the current duplicate-pair criteria. Other explanations —
including differences in amplitude scaling, fit behavior, or unit composition —
must remain open.

### Relation to the earlier pre-curation pair review

[`luke_20250804_imec0_rescue_result.md`](luke_20250804_imec0_rescue_result.md)
reported a pre-curation screen of 37 broad similar-pair candidates that refined
to one strong (units 184/191) and one partial (units 164/165) duplicate
hypothesis, both heavily artifact-associated.

The present post-curation artifact-aware audit reports **0** strong and **0**
partial-or-strong duplicate hypotheses among the 27 surviving pairs. These are
different stages and different screens, not a corrected number: the earlier
result is pre-curation with its own criteria, this one is post-curation. Both
stand as recorded. Neither resolves the completeness question, which is the
reason this evaluation reaches a different overall reading.

## Other acceptance-gate results

The pre-curation rescue passed the gates for:

- Minimum KS-good yield
- Maximum total spikes
- Median good-unit contamination
- Median refractory-violation rate
- Median fixed-bin presence
- Holdout coincidence excess
- Automatic raw-event preservation
- Middle-depth event recovery

It failed or marginally exceeded:

- Similar good–good pairs per good unit
- Edge-spike fraction
- In an earlier run, the fraction of good units present in ≥90% of fixed bins

The latest pre-curation edge-spike fraction was 2.004%, narrowly above the 2.0%
threshold. After curation it increased to approximately 2.066%.

These failures already required `reject_universal_default`. The
amplitude-truncation comparison is not an additional independent reason: both
its original population inference and its first matched-unit correction are
withdrawn pending an exclusive rerun.

## Historical decision section

### Durable decision: reject universal promotion on the original failed gates

The rescue pipeline should not become the universal default based on this imec0
result.

This does not mean the experiment was unsuccessful. It establishes several
useful findings:

- Disabling the previous suppression mechanism can recover additional candidate
  units.
- Conventional contamination and refractory metrics can improve at the same
  time.
- Increased good-unit yield alone can conceal poorer amplitude completeness.
- Firing-rate-bin occupancy is not an adequate substitute for amplitude-based
  missing-spike estimation.
- Claim masking remains the strongest configuration for detection completeness
  in this comparison, though at a substantial yield cost.

The production question should now be reframed from "Does rescue increase
good-unit yield?" to:

> Can the rescue configuration retain its yield advantage while matching legacy
> or claim-mask amplitude completeness and controlling similar-unit
> proliferation?

This report cannot answer that question; an exclusive matched-unit
completeness rerun and known-truth pipeline comparison are still required.

## Historical follow-up list — superseded by the active plan

1. **Make amplitude completeness a formal acceptance gate.** Add a
   unit-balanced truncation criterion, preferably within the >1 Hz good-unit
   cohort. Candidate gates include median unit missingness and the fraction of
   units below a prespecified missingness threshold.

2. **Recompute all three truncation analyses with one frozen implementation.**
   The stored legacy, claim-mask, and rescue QC results were generated at
   different times. Although their schemas and apparent algorithms match, legacy
   runs lack the new identity-bound receipts. A frozen matched recomputation is
   needed before treating small differences as definitive.

3. **Validate the truncation fitter.** Many fitted windows reach the estimator's
   hard 50% ceiling. This may correctly represent severe truncation, but the
   ceiling behavior, fallback fits, amplitude units, and parameter bounds should
   be audited before formalizing thresholds.

4. **Inspect discordant units manually.** Prioritize:
   - Rescue good units with median estimated missingness ≥10%
   - Rescue >5 Hz units with severe or ceiling-level windows
   - Similar-pair units with poor amplitude completeness
   - Units near the repaired/bad-channel depth
   - Units counted as stable by time-bin occupancy but poorly captured by
     amplitude fitting

5. **Test the fragmentation hypothesis directly.** Determine whether
   low-amplitude tails from rescue units are being assigned to nearby units,
   discarded, or absorbed into noise. Template similarity alone is insufficient;
   waveform continuity, amplitude distributions, CCG structure, spatial overlap,
   and raw-trace recovery should be considered together.

6. **Develop a yield-versus-completeness trade-off curve.** Evaluate whether
   intermediate detection, artifact, or claim settings recover some rescue yield
   without producing the observed truncation penalty.

7. **Replicate before changing policy.** Any revised configuration should be
   evaluated on additional probes or sessions using the same frozen gates.
   Luke0804 imec0 should remain a diagnostic case, not the sole basis for a
   production-wide policy.

## Confidence and limitations

**Confidence is high** that the current rescue configuration should not be
promoted: it fails existing prespecified gates, and the amplitude-completeness
analysis supplies an additional material concern.

**Confidence is moderate** in the exact ranking and magnitude of truncation
differences because:

- Only units with supported 1,000-spike windows receive estimates.
- The fitter is capped at 50% estimated missingness.
- Stored comparator results were generated at different times and are not all
  protected by equivalent provenance receipts.
- KS-good labels are automated quality classifications, not manually adjudicated
  biological-unit identities.
- The analyses diagnose association and consistency, not the causal mechanism
  producing incomplete amplitude distributions.

These caveats do not remove the red flag. They define the validation needed
before assigning a precise cause or setting a final numerical production
threshold.

## Source record

- Formal decision:
  `/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/decision/formal_decision.json`
- Post-curation comparison:
  `.../rescue_pipeline_results_Luke0804_V2V1_g0_imec0/diagnostics/postcuration_comparison/postcuration_comparison.json`
- Artifact-aware pair audit:
  `.../rescue_pipeline_results_Luke0804_V2V1_g0_imec0/diagnostics/artifact_pair_audit/summary.json`
- Rescue truncation analysis:
  `.../rescue_pipeline_results_Luke0804_V2V1_g0_imec0/qc/amp_truncation/truncation_qc.npz`
- Legacy truncation analysis:
  `/mnt/NPX/Luke/20250804/pipeline_results_Luke0804_V2V1_g0_imec0/qc/amp_truncation/truncation_qc.npz`
- Claim-mask truncation analysis:
  `/mnt/NPX/Luke/20250804/patched_pipeline_results_Luke0804_V2V1_g0_imec0/qc/amp_truncation/truncation_qc.npz`
- Prespecified acceptance criteria: `configs/rescue/imec0_legacy_acceptance_criteria.json`
  (tracked; previously
  `testing/outputs/luke_full_probe_rescue_diagnostics_imec0_legacy/acceptance_criteria.json`,
  moved when generated outputs were untracked — the content is byte-identical)

No repository or pipeline files were changed while preparing this report.
