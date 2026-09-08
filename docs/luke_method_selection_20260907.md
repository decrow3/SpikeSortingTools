# Luke0804 method selection — 2026-09-07

**Nomination: rescue 12/9 with motion off is the operational winner among the
tested configurations for full-probe confirmation.** The newly supplied native
motion audit resolves the missing-file blocker: both saved native fields contain
large unsupported excursions. Do not promote either native configuration
unchanged. Confidence is moderate in that operational exclusion and low in
motion-off's biological superiority. This is a recommendation, not a claim that
motion correction is unhelpful or that completeness has been validated. No sort
was launched; the explicit hold on the cancelled full-session rigid job remains.

**Decision correction published:** the comparison implementation now reports
`motion_diagnostics_required` rather than deriving closure/promotion from
directional overlap. The executed corrected notebook is
`testing/outputs/luke_native_rigid_comparison_v2/native_rigid_vs_motion_off.ipynb`.
Its v2 summary attests the copied data and original v1 summary; all numerical
QC is unchanged. V1 remains intact as historical evidence. Seven focused tests
passed, including perfect-recovery/pure-dropout counterexamples; all seven
notebook code cells executed and the rendered figure was visually checked.
That v2 notebook remains the numerical-comparison audit; the later field-based
nomination below supersedes its pending-file status without reinstating the
invalid overlap efficacy rule.

## Comparable completed full-duration strip methods

These nine arms concern Luke0804 imec0, 10,473.554 s, 100 processing contacts
at 1400–2380 µm, with the 1600–2180 µm scoring interior. Counts below are all
exported curated clusters/events, not neuron counts or interior-only yields.
Threshold notation is universal/learned.

| Method | Clusters | Spikes, millions | Strength | Weakness / selection |
|---|---:|---:|---|---|
| **12/9 off** | 230 | 7.772 | Established reference; 141 interior clusters and 4.237M interior spikes; lower median refractory burden than 12/10; avoids observed native-field instability | Motion tracking remains unproven; more coincidence and edge burden than 12/10. Nominated for operational confirmation, not proven biological superiority. |
| **12/10 off** | 236 | 5.875 | 148 interior clusters; less coincidence and edge burden than 12/9 | 3.320M interior spikes; stricter detection may discard real low-amplitude events. Main conservative threshold alternative, not a proven completeness winner. |
| 10/10 off | 213 | 5.977 | Lower coincidence than 12/10 | Only 120 interior clusters; higher boundary and refractory burden than 12/10; little total event gain. |
| 10/9 off | 240 | 7.860 | Lower coincidence than 12/9; more total clusters | Only 1.13% more total spikes, fewer interior spikes (4.118M), and greater boundary/refractory burden. No demonstrated recovery gain. |
| 9/9 off | 222 | 7.849 | Similar spike yield to 10/9 | Fewer interior clusters (124); little case for the additional universal-threshold reduction. |
| 10/8 off | 239 | 10.495 | 33.5% more spikes than 10/9, including interior events | Coincidence excess rises 1.99 percentage points; true recovery versus extra assignments unresolved. |
| 9/8 off | 232 | 10.530 | Highest event yield; 34.2% more spikes than 9/9 | Coincidence excess rises 2.26 points; 9 rather than 10 adds only 0.33% spikes at learned 8. |
| **12/9 native rigid** | 164 | 7.084 | Known-truth benefit in exact 40 µm motion control; modest positive real-data excursion screen; less edge burden | Higher aggregate refractory burden, major correspondence changes, and 560 adjacent batch jumps >100 µm. Reject unchanged for promotion. |
| 12/9 native nonrigid | 156 | 6.966 | Can model depth-dependent movement | No obvious aggregate rescue; 34 interior matches and 4/141 completeness coverage. Fields repeat rigid excursions (610 transitions >100 µm); reject unchanged. Curated trains remain absent from the handoff. |

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

## Native motion handoff verified locally

The user supplied
`/mnt/NPX/Luke/20250804/shared_analysis/luke_motion_audit_20260907_v1/`.
All 53 inventoried files (16,603,153 bytes) passed local size/SHA-256 checks.
The off and rigid candidate sort identities match the previously analyzed
handoff. The field-audit code was inspected and replayed locally using the
production environment: Kilosort 4.0.27, SpikeInterface 0.102.1. Every field
summary metric and the complete agreement CSV reproduced exactly. The CPU
application-sign check also reproduced, and the resulting figure was inspected.

| Diagnostic | Rigid | Nonrigid |
|---|---:|---:|
| Saved physical displacement range | −309.5 to +315.5 µm | −325.5 to +329.5 µm |
| Adjacent batch transitions with any field jump >100 µm | 560/5,236 | 610/5,236 |
| Median absolute adjacent step | 2.5 µm | 3.5 µm |
| P95 absolute adjacent step | 146.125 µm | 149.0 µm |
| Batches sampling any scoring-interior coordinate outside the strip | 76 | 75 |

Rigid and nonrigid median trajectories correlate at 0.9972. At the same depth
and physical times, native rigid RMS displacement is about 45.7 µm versus
3.39 µm for DREDGE and 3.55 µm for the independent KS sidecar; their correlations
with native rigid are only 0.213 and 0.271. Decentralized is almost flat at this
depth (0.370 µm RMS), so it should not be described as an equally informative
third consensus vote. The physical sign was fixed from the interpolation
operator, not selected to maximize correlation. Independent estimates are not
ground truth and differ in preprocessing, but they do not support the native
hundreds-of-micrometers excursions.

The supplied raster diagnostic is corroborative only and was not rerun locally:
it is post-selection, mixes neurons and shares detections with the estimators.
The nomination does not depend on its modest profile-correlation result.

Local reproduction: `testing/outputs/luke_saved_motion_field_audit_local_v1/`.
The source handoff contains scripts and the detailed source audit report.
The scientific conclusion is an unstable saved native estimate, not a proven
root cause for every lost event and not a general failure of motion correction.

## Full-data recommendation and remaining scientific scope

Choose **rescue 12/9, native motion disabled (`nblocks=0`)**, with the accepted
preprocessing and unchanged common curation profile, for full-probe confirmation
among currently tested settings. The 12/10 off arm is the conservative threshold
alternative, but no new evidence establishes that its removed events are false;
it does not displace 12/9. Do not spend on another unchanged native rigid or
nonrigid full sort based on these strip outputs.

The strip has fewer channels than the full probe. Consequently this finding
does not show that full-probe native motion estimation will fail. If the next
priority is a corrected-motion method, first estimate motion on the full probe
without template learning or sorting and qualify that field against independent
evidence. This is a proposed new diagnostic, not a completed experiment or an
authorization to remove the full-sort hold.

Biological recovery remains unresolved: amplitude coverage is sparse, the
correspondence graph cannot be pooled naively, and raw-event family validation
has not been completed. Those limit claims of superiority; they do not justify
promoting a demonstrated unstable field over the existing operational reference.
The requested method comparison and operational nomination are now complete;
no new sort, full-probe registration job, or hold removal was performed.

Key source records: `docs/luke_20250804_c2_v4_result.md`,
`docs/luke_20250804_c2_v4_truncation_diagnostic.md`,
`docs/luke_c2_train_stability_stage2_result.md`,
`testing/outputs/luke_rescue_c2_drift_challenge_v4/truncation_diagnostic/summary.json`,
`/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/threshold_grid_analysis_v1/summary.json`,
and `testing/outputs/luke_threshold_family_diagnostic_v3/`.
