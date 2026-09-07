# Luke0804 method selection — 2026-09-07

**Provisional recommendation: retain rescue 12/9, motion off, as the operational
reference and first candidate if a full-probe choice must be made now. Do not
treat this as a proven biological winner or commit another full run yet.**
Native rigid at 12/9 is the strongest unresolved challenger: it has positive
known-truth motion evidence, but its real-data implementation is not qualified.
The immediate decision is to finish the existing-motion-output audit, not to
launch a full sort. The explicit hold on the cancelled full-session rigid job
remains in force.

**Decision correction published:** the comparison implementation now reports
`motion_diagnostics_required` rather than deriving closure/promotion from
directional overlap. The executed corrected notebook is
`testing/outputs/luke_native_rigid_comparison_v2/native_rigid_vs_motion_off.ipynb`.
Its v2 summary attests the copied data and original v1 summary; all numerical
QC is unchanged. V1 remains intact as historical evidence. Seven focused tests
passed, including perfect-recovery/pure-dropout counterexamples; all seven
notebook code cells executed and the rendered figure was visually checked.

## Comparable completed full-duration strip methods

These nine arms concern Luke0804 imec0, 10,473.554 s, 100 processing contacts
at 1400–2380 µm, with the 1600–2180 µm scoring interior. Counts below are all
exported curated clusters/events, not neuron counts or interior-only yields.
Threshold notation is universal/learned.

| Method | Clusters | Spikes, millions | Strength | Weakness / selection |
|---|---:|---:|---|---|
| **12/9 off** | 230 | 7.772 | Established reference; 141 interior clusters and 4.237M interior spikes; lower median refractory burden than 12/10 | Motion tracking remains unproven; more coincidence and edge burden than 12/10. Provisional operational pick only. |
| **12/10 off** | 236 | 5.875 | 148 interior clusters; less coincidence and edge burden than 12/9 | 3.320M interior spikes; stricter detection may discard real low-amplitude events. Main conservative threshold alternative, not a proven completeness winner. |
| 10/10 off | 213 | 5.977 | Lower coincidence than 12/10 | Only 120 interior clusters; higher boundary and refractory burden than 12/10; little total event gain. |
| 10/9 off | 240 | 7.860 | Lower coincidence than 12/9; more total clusters | Only 1.13% more total spikes, fewer interior spikes (4.118M), and greater boundary/refractory burden. No demonstrated recovery gain. |
| 9/9 off | 222 | 7.849 | Similar spike yield to 10/9 | Fewer interior clusters (124); little case for the additional universal-threshold reduction. |
| 10/8 off | 239 | 10.495 | 33.5% more spikes than 10/9, including interior events | Coincidence excess rises 1.99 percentage points; true recovery versus extra assignments unresolved. |
| 9/8 off | 232 | 10.530 | Highest event yield; 34.2% more spikes than 9/9 | Coincidence excess rises 2.26 points; 9 rather than 10 adds only 0.33% spikes at learned 8. |
| **12/9 native rigid** | 164 | 7.084 | Known-truth benefit in exact 40 µm motion control; modest positive real-data excursion screen; less edge burden | Higher aggregate refractory burden, major correspondence changes, unknown validity of this run's motion estimate. Strongest unresolved motion challenger. |
| 12/9 native nonrigid | 156 | 6.966 | Can model depth-dependent movement | No obvious aggregate rescue; 34 interior matches and 4/141 completeness coverage. Detailed arrays/motion field absent from local handoff. |

The measured tradeoff is not a total order. For example, 12/10 → 12/9 adds
32.3% total events but also 1.46 points of coincidence excess and 2.53 points
of edge-spike burden. The same guardrail concern used against learned 8
therefore also applies to learned 9 relative to 10. Retaining 12/9 is a
continuity-of-operating-point decision under uncertainty, not evidence that it
dominates 12/10. Coincident events may include physiological synchrony; this
metric is a contamination warning, not a count of proven duplicate spikes.

Lowering universal 12 → 10 adds only about 1–2% total spikes at fixed learned
threshold, with more boundary burden. Lowering learned threshold consistently
adds about one-third. The 50-family threshold diagnostic finds most added
events in the lower amplitude tail, with elevated coincidence; no family meets
its clean-recovery signature. Those signatures are descriptive and do not prove
the extra events are false. The cohort is selected for survival across all
seven arms and cannot evaluate lost/new families well.

## Why amplitude completeness cannot settle this

The prospective common-time endpoint covers only 2.0–4.8% of eligible units
across threshold edges and 4/141 for each motion arm. An exploratory all-seven
comparison has 19 amplitude-measurable families, but relaxes the common-time
criterion and selects high-count survivors. Its roughly 99% completeness
medians do not validate full-session completeness or rank the whole population.

There is direct known-truth evidence of this blind spot. In the valid exact
40 µm C2 control, **13/14 uncorrected donors fragmented**; the best single
cluster missed a median **50.51%** of their full train while phase-specific
truncation estimated only **0.615%** missing. Native rigid's whole-train median
missingness in that control was **0.728%**. This supports a motion-tracking
benefit in that control, not a claim about Luke-scale efficacy. The 5/11 µm
synthetic arms were invalidated because their forward model attenuated rather
than faithfully displaced the waveform. The earlier report's contrary
Luke-scale conclusion is withdrawn.

Truncation measures the observed amplitude tail during estimable epochs. It
cannot by itself detect a neuron changing cluster identity or disappearing
entirely. Likewise, the current candidate-minus-reference overlap score is
directionally invalid: extra true candidate spikes can make it negative.
See the [metric audit](luke_native_rigid_metric_validity_audit_20260907.md).

## New motion-conditioned check on existing real outputs

Executed `testing/luke_motion_conditioned_screen.py`, using the existing 40
interior primary pairs and three accepted full-probe motion estimates. It
subtracts the recording's attested acquisition-clock origin, 3057.677050 s,
before assigning motion to zero-origin sorted spike times.

The exploratory rule was saved before execution: nonoverlapping 120 s bins;
upper/lower motion quartiles; nearest unused quiet bin within 600 s; assess
within-bin P95–P5 excursion and absolute displacement from session median
separately. Zero-event count bins remain included. Amplitude comparisons are
within-arm ratios on nonempty paired bins, not cross-arm microvolt estimates.

The rate statistic below is the per-family `(rigid high/quiet)/(off high/quiet)`,
then median across families. One means no differential motion association.

| Independent estimate | Excursion time pairs | Median rate ratio | Displacement time pairs | Median rate ratio |
|---|---:|---:|---:|---:|
| DREDGE | 7 | 1.0215 | 6 | 1.0005 |
| Decentralized | 5 | 1.0289 | 4 | 0.9940 |
| KS motion sidecar | 7 | 1.0205 | 23 | 0.9908 |

This is a small, consistent descriptive excursion advantage, without a
corresponding displacement advantage or consistent amplitude improvement.
It does not demonstrate a large hidden benefit in these survivors. It also
cannot exclude such a benefit among unmatched neurons or on the full probe.
There are few independent time pairs, reused across estimators; firing-state
changes remain confounded with motion, and the matched cohort is selected.
No significance or promotion threshold is assigned to this new endpoint.

Outputs: `testing/outputs/luke_motion_conditioned_screen_v1/` contains the
selected physical-time pairs, 240 family/exposure/estimator rows, summary,
and input hashes. Source arrays and the seven threshold-arm provenance were
revalidated; the receipt is `testing/outputs/luke_method_selection_audit_v1/provenance.json`.

## Older approaches do not supply a better qualified winner

- **Legacy / claim-mask pipelines:** historical full-probe runs are useful
  references but change preprocessing, detection and/or masking together.
  Population completeness is composition-confounded and the first matched
  analysis was retracted. They do not isolate a motion benefit.
- **External voltage registration:** plausible architecture with an exact
  synthetic inverse control, but historical warp failures and geometry-invalid
  synthetic claims cannot qualify a new full-data operating point. No comparable
  current-ladder winner exists for it.
- **Motion-aware identity stitching:** plausible way to preserve unwarped
  voltage, but no validated current-ladder winner. Historical clean-merge claims
  were retracted; pooled fragment trains were often not refractory-clean.
- **8/8 and earlier 9/9 train-stability tests:** neither separated from 12/9
  under the frozen donor-level uncertainty analysis. They did not establish
  equality, and did not qualify a threshold replacement.

## Decision needed before committing GPU time

The shortlist for the motion decision is **12/9 off versus 12/9 native rigid**.
Full-data nomination remains provisional until the saved native motion estimate
is checked. The 12/10 off arm remains a threshold sensitivity alternative if
raw-event review favors conservative detection; it is not an automatic third
full-probe run. Keep the existing maximum of two full-probe candidates.

The next required evidence is rigid/nonrigid `ops.npy` or motion exports and
the associated settings from huklaban5. Its hostname failed resolution both
inside and outside the sandbox; an address has been requested. No job was
restarted. Once accessible, inspect the motion fields and qualify suspicious
correspondence links using spatial/waveform evidence before pooling families.
If strip estimation is implausible or unrepresentative, a full-probe
registration-only diagnostic is more informative than another blind full sort.

If a decision must be made with the present evidence, my pick is **12/9 off
for confirmation**, with low confidence in its biological superiority. My
recommendation now is to finish the bounded motion diagnostic first. Neither
the new screen nor the failed original efficacy gate warrants declaring motion
correction unhelpful.

Key source records: `docs/luke_20250804_c2_v4_result.md`,
`docs/luke_20250804_c2_v4_truncation_diagnostic.md`,
`docs/luke_c2_train_stability_stage2_result.md`,
`testing/outputs/luke_rescue_c2_drift_challenge_v4/truncation_diagnostic/summary.json`,
`/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/threshold_grid_analysis_v1/summary.json`,
and `testing/outputs/luke_threshold_family_diagnostic_v3/`.
