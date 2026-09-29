# SpikeSortingTools: experiments, rationale, and conclusions

**Compiled:** 23 September 2026. **Evidence reviewed:** repository decision and
experiment records through 14 September 2026, including the completed
Bacon–Luke–Allen pilot. This is a synthesis of documented results, not a fresh
audit of every stored array or a live job-status report. It covers the major
research threads; linked reports retain individual ablations, figures, settings,
and corrections. A launch record alone is not a completed experiment.
The [23 September saved-array audit](../testing/outputs/motion_saved_array_audit_20260923/README.md)
adds subsequent field comparisons, centring corrections, and export provenance;
its qualifications supersede the pilot-only interpretation below.

## What we are trying to improve

The goal is to recover interpretable neuronal identities and spike trains
throughout a recording, including periods of motion and amplitude loss. A larger
number of clusters, a cleaner-looking voltage trace, or a plausible motion field
is not sufficient. We need evidence that real events remain associated with the
same neuron without added contamination, duplication, fragmentation, or loss in
previously healthy periods.

Most experiments use **Luke 2025-08-04**, with imec0 and imec1 providing two
probe views of the same session. **Bacon halo 2025-10-16** and **Allen/Yates
2022-02-16** provide external comparisons, not controlled tests of head fixation:
anatomy, probe geometry, acquisition, neural activity, and behavior also differ.

## What we can currently conclude

1. **Some historical processing choices caused demonstrable local failures.**
   Removing one tested external DREDGE warp improved sorting and reviewed-event
   recovery in a problematic Luke interval. This does not imply that motion
   correction in general is harmful.
2. **The rescue pipeline is an operational reference, not a proven universal
   winner.** More KS-good units and better contamination metrics do not establish
   more recovered neurons. Its full-session imec0 acceptance gates were not all
   met, and corrected cross-sort audits do not establish detection equivalence.
3. **Motion estimation can seriously underrepresent movement when its inputs or
   settings are unsuitable.** Shared contamination, bounds, smoothing, and peak
   selection have all mattered in bounded Luke experiments. Conversely, a large
   returned field can be unsupported estimator behavior.
4. **The fast-estimator pilot finds a large Luke–Bacon difference.** Luke's early
   and late sampled windows have much larger shared and depth-dependent
   displacement than Bacon's. Referenced AP voltage variability is much closer
   between datasets. Allen's sparse-depth estimates prevent a trustworthy
   three-way all-depth motion ranking.
5. **Better estimated motion has not yet produced a universally better sorting
   configuration in the documented comparisons.** Correction, sorter choice,
   interpolation, and identity tracking have regime-specific tradeoffs.
6. **The evaluator itself needs validation.** Event matching, synthetic motion,
   amplitude completeness, and lighthouse identity tracking have each exposed
   defects or limits that changed earlier conclusions.

The documented production reference remains the frozen rescue graph with
Kilosort thresholds **12/9**, without applied motion correction. Later external
correction and DARTsort runs are development experiments, not automatic amendments
to that policy. See [decisions](decisions/README.md), the
[validation summary](validation-summary.md), and the
[development plan](pipeline_improvement_plan.md).

## 1. Did preprocessing and motion application explain the original Luke failure?

**Why:** Luke produced poor-looking or unstable sorting despite many detected
events. We needed to distinguish voltage conditioning, motion application,
artifact handling, and downstream clustering effects.

**What we did:** a matched 120 s imec1 ablation at 8160–8280 s varied external
DREDGE correction and intermediate numerical precision. Full-session rescue
runs and a bounded pinned-AIND preprocessing challenger followed.

**What we found:** removing the tested external warp reduced learned detections
by about 57%, improved reviewed-neural unmatched-event recovery from 70.4% to
85.2%, and increased KS-good units from 74 to 95. Floating-point conditioning
alone did not resolve that failure. This identified the tested motion branch,
but did not separate field error from interpolation error. In the AIND
comparison, total sealed-event recovery tied rescue at 470/720; improvements in
some nuisance metrics came with opposing yield, continuity, and duplicate-burden
tradeoffs.

Later [motion-application controls](luke_20250804_motion_candidate_report.md)
partially separated mechanisms: zero-displacement interpolation was byte-identical
to unwarped input; rigid-only application removed most detection inflation but
still lost reviewed events; changing the interpolation kernel reduced measured
attenuation. These establish a nonzero-resampling cost without establishing that
the displacement fields were physically correct.

**Conclusion:** retain rescue as the comparator; do not advance that AIND
configuration or treat every external warp as beneficial. This is a bounded
intervention result, not a general rejection of motion correction.

Sources: [upstream ablation](luke_20250804_upstream_ablation_report.md),
[bounded AIND result](luke_20250804_aind_downstream_bounded_result.md),
[full-probe rescue](luke_20250804_full_probe_rescue_result.md),
[imec0 control](luke_20250804_imec0_rescue_result.md).

## 2. Did rescue actually recover more neurons than legacy?

**Why:** increases in KS-good yield could reflect improved detection, different
partitioning of the same events, or altered curation labels.

**What we did:** compare full-session outputs, inspect apparent rescue-only and
legacy-only units, and revise cross-sort matching to enforce exclusive event
assignments and use null-controlled spatial evidence.

**What we found:** the full-session reports recorded 216 KS-good units on imec1
and 301 on imec0, while assigned-spike counts fell. On imec0, the frozen
acceptance verdict remained `reject_universal_default`: stable-good fraction,
edge burden, and nearby similar-pair count missed their gates. In the corrected
exclusive comparison, 208/210 rescue-side and 132/137 legacy-side apparent
differences had spatially plausible overlap with the other complete sort.
Seven units remained unresolved.

**Conclusion:** much of the visible difference is expressed in clustering and
curation. We have neither established a confirmed detection difference nor
proved that the pipelines detect the same neurons. “Rescue detects nothing new,”
“no neuron is lost,” and “the entire difference is relabeling” are too strong.

Sources: [validation summary](validation-summary.md),
[corrected interpretation, decision 0015](decisions/0015-corrected-cross-sort-audits-do-not-establish-equivalence.md),
[unique-unit audit](luke_20250804_rescue_unique_units_audit.md),
[lost-unit audit](luke_20250804_rescue_lost_units_audit.md).

## 3. Can our benchmarks measure recovery and motion damage correctly?

**Why:** an apparently successful pipeline comparison is uninformative if the
scorer rewards pooled detections, the injected motion is unrealistic, or the
quality metric compares different neuron populations.

**What we did and learned:**

| Experiment | Result | Consequence |
|---|---|---|
| Truncated-amplitude fitter audit | Accurate in its tested model regime; values pinned at the 50% boundary are censored, not measured missingness. | Preserve fit coverage, gaps, and boundary status. Do not compare unmatched population medians as pipeline efficacy. |
| C2 scorer validation | Earlier pooled-event scoring could confuse any nearby event with recovery of one neuron. | Score injected truth per output cluster; earlier invalid scorer results are superseded. |
| C2 v4 motion challenge | Exact 40 µm staircase motion produced a strong recovery penalty; 13/14 donors fragmented. Small fractional-offset arms suffered a defective forward model that mostly attenuated rather than displaced waveforms. | The staircase validates a mechanism; flat 5/11 µm results do not show motion tolerance. |
| Threshold Stage 2 | All 588 cells completed. Neither 8/8 nor 9/9 separated from 12/9 under the frozen paired confidence-interval rule. | `no_threshold_change`, not proof of equivalence or 12/9 superiority. |
| Within-Luke dose–response | Primary qualified-unit-yield endpoint was null; several secondary quality endpoints worsened with estimated motion. | Descriptive associations do not establish causality or show that motion is harmless. |

Sources: [fitter audit](luke_20250804_truncation_fitter_audit.md),
[scorer correction](decisions/0014-injected-truth-scoring-is-per-cluster.md),
[corrected C2 v4 result](luke_20250804_c2_v4_result.md),
[threshold Stage 2](luke_c2_train_stability_stage2_result.md),
[dose–response](luke_within_rigid_motion_dose_response_result.md).

## 4. Can post-sort identity reconciliation repair dropout?

**Why:** if events survive detection but move between clusters, identity
reconciliation could help without resampling voltage or rerunning a sorter.

**What we did:** an amplitude-completeness audit selected bounded failure cases
from permitted development intervals and reproduced all 24 selected cached fits.
Cluster 37 became the first intervention case. Candidate v1 used epoch linking;
v2 enforced preservation of whole original clusters.

**What we found:** v1 split the target across epochs and failed identity and
contamination gates. V2 preserved cluster 37 exactly but improved completeness
by **0.0 percentage points**. Its other cross-cluster merges were not validated
as biological identities. Healthy-cluster preservation passed, but completeness
coverage there was too low for a strong safety conclusion. A later baseline
census found other measurable deterioration cases, including 553 and 452.

**Conclusion:** close this candidate on cluster 37. A failed intervention on one
case does not show that no useful cases exist. The census nominates follow-up
cases; it is not a completed recovery intervention on those cases.

Sources: [dropout audit](luke_amplitude_dropout_audit_result.md),
[candidate v1](luke_first_pipeline_candidate_v1_result.md),
[candidate v2](luke_first_pipeline_candidate_v2_result.md),
[recovery census](luke_baseline_recovery_census_v1.md).

## 5. Why did some motion estimates look nearly stationary?

**Why:** observed waveform trajectories and apparent sorting failures did not
always agree with small population-based displacement estimates.

**What we did:** inspect acquisition voltage, common signals, detected peak
populations, pairwise registration, search bounds, thresholds, waveform screens,
and alternative filters before developing a new estimator.

**What we found:** a broad shared signal with unequal channel responses left
stationary-looking residual peak populations after ordinary median referencing.
In a controlled 20 s imec0 interval, a frozen 31-tap shared-response compensation
model followed by redetection/relocalization changed central DREDGE estimates
from approximately zero to movement corroborated in direction by several
waveform candidates. Selected waveform-preservation checks were favorable.
Hardware origin and general neural preservation were not established.

Aggressive waveform screening was not reliably better: it could discard useful
population structure and favor a competing alignment. Threshold effects depended
on the interval. A diagnostic DREDGE search-bound issue was also identified and
bounded explicitly; fixing it did not solve every failure. Separate audits found
large unsupported excursions in saved native Kilosort fields, including after
expansion to the full probe.

**Conclusion:** estimator input quality and constraints matter at least as much
as the algorithm name. Neither a flat nor a large field is sufficient evidence
of the actual motion. Bounded compensation results do not authorize a universal
preprocessing change.

Sources: [shared-response intervention](luke_shared_response_compensation_20260907.md),
[audit sequence](luke_motion_audit_pause_summary_20260908.md),
[screen sweep](luke_screen_sweep_20260908.md),
[bounds audit](luke_dredge_limits_sigma_audit_20260908.md),
[method-selection correction](luke_method_selection_20260907.md).

## 6. Can waveform “lighthouses” independently validate the motion?

**Why:** the motion field should be checked against recognizable waveform
identities without using that same field, or original depth, to choose matches.

**What we did:** develop whole-probe, depth-blind waveform competition that
retains relative multichannel geometry. Separate strict, lower-score, ambiguous,
and unmatched observations; determine absolute depth after matching. Audit
lattice effects, competing identities, seed replication, compact templates, and
agreement between simultaneously observed candidates.

**What we found:** fixed-template support can lose a moving cell and create
apparent stationarity. Whole-probe matching revealed larger trajectories, but
the 40 µm template lattice biases acceptance and lookalike switching remains a
serious risk. An affine RANSAC field derived from lighthouse observations lost
to a simpler rigid median in leave-one-family-out validation and was rejected.

On imec1, an independently seeded candidate pair provided encouraging **local**
replication: relative-depth correlation 0.923 and centered median difference
7.8 µm, with no shared raw detections. That did not provide a probe-wide or
independent >4 Hz reference. The subsequent eight-proposal expansion failed the
shared-movement premise: simultaneous candidate changes were inconsistent,
sometimes opposite. Native AP and fine MEDiCINe traded small metric advantages
against large candidate-specific residuals.

**Conclusion:** lighthouse discovery is useful, but candidate tracks are not
automatically verified neurons or ground truth. The September 13 imec1 analysis
does **not** establish that fine MEDiCINe is more accurate than native AP. Keep
the rejected RANSAC prototype as a negative control, not an estimation method.

Sources: [discovery policy](lighthouse_candidate_discovery.md),
[tracker audit](luke_lighthouse_method_audit_20260909.md),
[RANSAC rejection](luke_lighthouse_ransac_field_20260909.md),
[cross-seed replication](luke_imec1_cross_seed_lighthouse_replication_20260912.md),
[compact alignment](luke_imec1_compact_alignment_pilot_20260913.md),
[imec1 comparison](luke_imec1_lighthouse_motion_comparison_20260913.md).

## 7. Does improved motion estimation improve the resulting sort?

**Why:** measurement, voltage correction, and recovery of continuous neuronal
identities are separate questions. A better field can still be applied badly,
and a selected movement endpoint can improve while quiet periods deteriorate.

**What we did:** shortlist screened AP MEDiCINe configurations, extend their
temporal support, compare rigid/nonrigid voltage correction, evaluate AP/LFP
fields and interpolation variants, and compare Kilosort with native DARTsort.

**What we found:**

- The screened AP shortlist favored 5σ/relaxed input initially, with 6σ/full as
  a sensitivity arm. Rankings changed on the extension. This configuration is
  distinct from later MEDiCINe fits using DARTsort-derived peak populations.
- The seven-arm **930–1230 s** correction matrix produced no clear improvement
  over unwarped 12/9 across concentration, contamination, duplicates, and yield.
- The later **930–1030 s, 348-channel** matrix had no universal winner. SG25 LFP
  led the movement concentration endpoint (0.567); native DARTsort had lower
  movement duplication and a more balanced profile. Unwarped led quiet-event
  concentration (0.684). These are different contracts, not interchangeable
  entries in one leaderboard.
- Detailed Kilosort–DARTsort comparisons found fragmentation and selectivity
  tradeoffs. Grouping corresponding DARTsort fragments reduced, but did not
  eliminate, rate differences. LFP-warped Kilosort outputs had adverse waveform
  and correspondence evidence despite some favorable movement scores.

**Conclusion:** retain regime-specific findings without promoting one scalar
winner. These results do not establish a generally superior full-session
correction or sorter. Full-session estimation and correction queue documents
also exist, but launch-time reports alone cannot establish their final outcome.

Sources: [screened workflow](screened_medicine_workflow.md),
[five-minute matrix](luke_medicine_5min_voltage_correction_result_20260909.md),
[matched 348-channel matrix](luke_motion_348ch_candidate_matrix_result_20260910.md),
[LFP filter ablation](luke_lfp_native_sg_348ch_ablation_result_20260910.md),
[detailed sorter QC](luke_kilosort_dartsort_detailed_qc_20260910.md),
[full-session estimate run record](luke_full_session_medicine_run_20260909.md),
[rigid queue record](luke_medicine_rigid_queue_20260909.md).

## 8. How do Bacon, Luke, and Allen differ under Fast MEDiCINe?

**Why:** earlier estimates might have hidden large/fast Luke movement. The
question was the between-dataset difference, not simply how much an old method
missed. The hypothesis was that Luke has the greatest motion burden.

**What we did:** 18 fits: 120 s windows centered at 10%, 50%, and 90% of each
session, across two Luke probes, two Bacon probes, and two Allen shanks. Use
DARTsort initial detection/localization, then MEDiCINe with **0.25 s bins, 1 s
kernel, four depths, 10,000 steps, seed 0**. No clustering or voltage correction
was performed. Four samples/s does not imply 4 Hz physical-motion bandwidth.

The first batch stopped when an absolute high-frequency PSD cutoff rejected all
Bacon channels. A direct audit isolated that gate. V2 disabled it uniformly
while retaining coherence-based channel checks, preserved the failed evidence,
and reran all 18 fits. V2 completed successfully on September 13. Thus its
frontend is a documented modification of `ibllikecmr`, not the untouched default.

### Motion result

Values below are medians across three window-level summaries, in µm:

| Probe | Shared/rigid excursion | Nonrigid residual excursion | Interpretation |
|---|---:|---:|---|
| Luke imec0 | 190.6 | 58.1 | Large common shifts plus substantial depth dependence |
| Luke imec1 | 164.4 | 38.0 | Same broad pattern, with different depth dependence |
| Bacon A | 4.2 | 8.8 | Small fields; residual fluctuations may include estimator noise |
| Bacon B | 3.3 | 6.9 | Small fields; residual fluctuations may include estimator noise |
| Allen shank 1 | 7.0 | 18.0 | Unreliable all-depth summary because of sparse-depth instability |
| Allen shank 2 | 21.1 | 32.0 | Same limitation; not verified fast tissue movement |

For this decomposition, remove each depth's temporal median; shared motion is
the median across four depths, and nonrigid motion is the remaining residual.
Excursion is temporal P95−P5; residual excursions are summarized across depths.
These ranges do **not** add or define a percentage of total motion. Different
probe spans also mean different spatial coverage of nonrigidity.

Luke's early/late median-depth excursions were **186–196 µm**, versus **7–11 µm**
across Bacon windows. Luke's middle window was quieter (**14/31 µm**). Allen's
largest model excursions occurred at poorly supported depths; inspected dense
peak bands were much more stationary. The pilot supports a substantial,
time-dependent Luke–Bacon separation, but does not establish session-wide
prevalence, a causal headpost effect, or Allen's all-depth ranking.

### Noise result

Median channel robust sigma, then median across three windows, in µV:

| Dataset | No reference, probe/shank range | Shank-median reference, range |
|---|---:|---:|
| Luke | 32.6–40.5 | 8.8–9.6 |
| Bacon | 35.2–40.0 | 10.0–11.4 |
| Allen | about 17.3 | 9.8–9.9 |

Each window used twelve 1 s samples with identical 300–6000 Hz filtering across
three reference schemes. Spikes remain in these voltage-variability measures.
Local-reference results and neighborhood counts are also saved. Noise after
referencing differs much less than the observed Luke–Bacon motion. Historical
high-pass-only noise and the earlier quiet-window/local-reference audit use
different definitions and must not be pooled with this table.

### Did a Luke-favored estimator miss Bacon motion?

The saved Bacon comparator was **Kilosort 4**, with 2 s batches and nine depth
centers. On identical complete batches within the pilot windows, averaging MED
to 2 s and sampling it at KS depth centers gave rigid excursions **2.3/1.9 µm**
(A/B), versus **1.0/1.0 µm** for KS. Residual excursions were **2.3/1.9 µm** for
MED versus **1.0/1.0 µm** for KS. This is agreement on small motion, not exact
numerical agreement or trajectory validation; MED was somewhat higher.

Full-session KS has larger episodes outside those windows: maximum rigid steps
**32.5 µm near 102 s** on A and **37.5 µm near 6438 s** on B. Full-session rigid
P95−P5 was 7.5/3.0 µm, and full min-to-max range was 43.5 µm on each probe.
The pilot therefore misses some larger Bacon episodes. Coarse KS agreement also
cannot exclude faster motion missed by both methods.

**Subsequent saved-array qualification (23 September):** the imec1 p10 pilot's
depth-centred rigid excursion is 164.38 µm, versus 48.26 µm for the completed
shared-recovery full-session fit and 46.68 µm for the merge-bias full-session fit
on the same interval. The pilot magnitude is therefore not established physical
motion. The audit also distinguishes depth-centred spread from spread retaining
static offsets, and finds no coincident strict assignments between p06_f010 and
p08_f044, but cannot establish their independence. See the
[audit and numerical tables](../testing/outputs/motion_saved_array_audit_20260923/README.md).

Sources: [experiment history](bacon_luke_allen_motion_noise_20260913.md),
[reviewed pilot results](../testing/outputs/cross_dataset_fast_motion_v2/REVIEW.md),
[rigid/nonrigid measurements](../testing/outputs/cross_dataset_fast_motion_v2/rigid_nonrigid_decomposition.csv),
[Bacon cross-check](../testing/outputs/bacon_saved_motion_crosscheck_20260914/README.md),
[historical quiet-window noise](luke_yates_quiet_window_noise_compare.md).
Output links refer to local, potentially gitignored artifacts; the source reports
retain the rationale and provenance paths.

## 9. Engineering and execution lessons

Long jobs must be independent of chat or terminal lifetime. The interrupted and
subsequently user-cancelled native-rigid full-session attempt remains a distinct
run with a hold; later experiment authorization does not erase that history.
Independent systemd execution, launcher-disconnection checks, immutable requests,
logs, exit receipts, and actual process inspection are now part of the protocol.
Reusable completed stages are not optimizer or within-sort checkpointing.

Cache identity, sample clocks, geometry, voltage units, and physical channel
order are scientific requirements: errors can silently compare different
recordings or intervals. Identity-bound downstream execution and the additive
standard per-unit QC bundle improve auditability without themselves proving
better sorting. The standard QC bundle can backfill completed compatible sorts
without raw voltage reads or changes to historical curation labels.

Sources: [interruption investigation](luke_full_session_interruption_20260906.md),
[held native-rigid run](luke_full_session_rigid_v1.md),
[development workflow](development_comparison_workflow.md),
[standard unit QC](standard_unit_quality.md), [repository instructions](../AGENTS.md).

## Corrections that must survive future summaries

| Earlier interpretation | Supported replacement |
|---|---|
| More KS-good units proves better neuronal recovery. | Yield alone does not distinguish detection, partitioning, curation, or contamination. |
| Rescue has worse completeness, or all differences are relabeling. | Unmatched populations and a nonexclusive matcher invalidated those empirical claims; use decisions 0011/0015. |
| Luke imec0 has negligible motion. | The historical 1.28 µm sidecar claim was withdrawn in [decision 0013](decisions/0013-luke-imec0-has-appreciable-rigid-motion.md); newer measurements show substantial movement in some intervals. |
| Flat small-motion synthetic results prove tolerance. | The fractional-motion forward model did not faithfully displace waveforms. |
| Native-rigid overlap asymmetry establishes benefit or closes the motion question. | The endpoint was invalid for efficacy; inspect fields and identity-aware outcomes separately. |
| A lighthouse track is an independent verified neuron. | Lattice bias, lookalikes, sparse events, and failed shared-movement replication limit that inference. |
| The old DREDge field had 20 s temporal bins. | The checked Luke saved field had 1 s spacing; 20 s referred to a summary export. |
| Fast MEDiCINe has been proven best everywhere. | It is a selected candidate with bounded supporting evidence; the imec1 lighthouse comparison did not establish superiority. |

## What is still open, and the smallest useful next checks

| Open question | Cheaper next check using existing evidence | What would still need a larger experiment |
|---|---|---|
| Are Allen's unstable depths real motion? | Inspect cached depth bands, waveform rivals, and local support around those model depths. | A support-appropriate estimate and independent trajectory validation for an all-depth ranking. |
| Does MED capture Bacon's occasional larger events? | Fit only windows covering the two largest saved KS steps, reusing the existing runner. | Broader coverage to quantify prevalence and fast events not found by KS. |
| Which lighthouse identities are trustworthy? | Audit cached rivals, lattice/phase controls, and repeated shared movement before smoothing or consensus. | More independent identities and enough event density for fast validation. |
| Which correction improves neuronal recovery? | Reconcile existing matched outcomes by movement/quiet regime and inspect the waveform disagreements. | A frozen longer-duration comparison with completeness coverage and independent-case confirmation. |
| Can a real dropout case be improved? | The cached 553 review found no operation to replay. The completed 452 identity/curation replay found a strong independent anchor but no waveform-compatible label transfer and no differential curation exclusion. Keep both as negative localization evidence. | One evidence-motivated detection or waveform-integrity intervention, with identity, contamination, and healthy-period checks; neither 553 nor 452 currently nominates one. |
| What happened to a launch-only full-session experiment? | Read its final receipts, artifacts, and actual manager state before describing an outcome. | No new run unless the existing evidence and run-specific authorization justify it. |

These are remaining decisions, not newly authorized runs. Prefer a cached
inspection or bounded control before another broad sweep. A mechanistic
uncertainty need not prevent a well-controlled development comparison, but a
successful pilot does not waive production acceptance or independent validation.

## Keeping this document current

For each substantive completed experiment, update the relevant question above
with its scope, observed result, interpretation, and unresolved limitation; link
the detailed result and artifacts. Mark proposals and launch-only records as
such. Preserve failed experiments and correction chains. Record new production
policy in [decision records](decisions/README.md), not implicitly in this summary.
Change the evidence date only when later results have actually been reviewed.
