# Native-rigid metric validity audit — 2026-09-07

**Assessment: the evidence does not support closing the motion question.**
The completed strip comparison contains adverse results, but the directional
continuity score is not a valid improvement endpoint and amplitude completeness
is largely unmeasured. This audit neither establishes a motion-correction benefit
nor authorizes a full sort. No sort or registration job was launched.

**Implemented correction:** `testing/luke_native_rigid_comparison.py` now
separates overlap asymmetry from efficacy. The corrected, executed notebook and
summary are in `testing/outputs/luke_native_rigid_comparison_v2/`, retaining all
original numerical QC and hashes of copied inputs. V1 is preserved, but its
`close_native_rigid` interpretation is superseded. Reproduce the correction
with `--reassess-from testing/outputs/luke_native_rigid_comparison_v1` and a
fresh `--output-root` when invoking the comparison module.

## Sources and reproduction

- Current saved comparison: `testing/outputs/luke_native_rigid_comparison_v1/`.
- Comparator: `testing/sort_comparison.py`; decision wrapper:
  `testing/luke_native_rigid_comparison.py`.
- Recompute: `python testing/luke_native_rigid_metric_audit.py`.
- Saved recomputation: `testing/outputs/luke_native_rigid_metric_audit_v1.json`,
  including SHA-256 hashes of each CSV read.
- Compact source handoff:
  `/mnt/NPX/Luke/20250804/shared_analysis/luke_group1_handoff_v1`.

The comparison covers 10,473.554 s at 29,999.836 Hz, processing depths
1400–2380 µm and scoring depths 1600–2180 µm. Both arms use thresholds 12/9.
This audit rechecks saved tables and source semantics; it does not independently
recompute their upstream spike matching or raw-voltage amplitudes.

## 1. Directional continuity score fails a simple counterexample

For shared event count M, off count B, and rigid count C, the pair statistics
are `baseline_retention = M/B` and `candidate_retention = M/C`. The wrapper
calls `median(M/C) - median(M/B)` a continuity change, with positive values
treated as efficacy.

- Perfect rescue: off detects 80 real spikes, rigid detects those plus 20 real
  spikes. Score = 80/100 − 80/80 = **−0.20**, interpreted as worse.
- Pure dropout: off detects 100 real spikes, rigid retains only 80. Score =
  80/80 − 80/100 = **+0.20**, interpreted as better.

Reversing the sign would not fix validity: additional spikes could also be
contamination. The two directional fractions describe disagreement with a
reference of unknown completeness, not a directional biological improvement.

In the actual 40 interior primary pairs, **27 have more rigid events** and
the median rigid/off event-count ratio is **1.05235**. Thus the published
−0.06829 score cannot establish worsening continuity. These matched survivors
also cannot represent the unmatched population.

## 2. Completeness has severe selection and observation gaps

The saved coverage is 4/141 eligible baseline units (2.84%): 101 have no
interior primary match, 36 have insufficient fit support, and 4 are measurable.
Only 40/141 reach the interior matched cohort. The common-time requirement is
50% of recording duration with at least two valid windows in each arm.

The window table contains 5,001 off / 4,284 rigid boundary-pinned windows,
1,895 off / 1,940 rigid finite-interior windows, and 160 off / 71 rigid
no-fit cluster placeholder rows. These counts cover the full exported cohort,
not just the 141 eligible units, and windows are not independent observations.
Boundary-pinned fits are excluded from the finite-interior endpoint. They
must remain visible as censored/model-boundary outcomes, not be counted as
zero missingness or automatically assigned a biological missing percentage.

The fitter uses 1,000-event windows inside continuous spike blocks. Dropout can
remove the very events needed to make a window; complete silence cannot produce
an amplitude distribution. Conditioning on valid fits in both sorts excludes
some of the episodes most relevant to detection failure. A valid fit on observed
spikes is not evidence that all physical-time epochs were complete.

Presence uses 300 s bins: **36/40 matched pairs have presence 1 in both arms**.
It cannot discriminate partial spike loss while each bin still contains events.
Amplitude CV uses sorter-native amplitudes, whose template/normalization context
may change between arms; it is not a calibrated raw-voltage completeness measure.

## 3. Component pooling needs a better graph first

The 5,863 saved edges form a giant component of **228 off and 163 rigid
clusters**, one 1:1 component, and one isolated off cluster. The matcher uses
10% overlap relative to the smaller train without a spatial or shift-null gate.
Event matching is exclusive within each pair, not globally across all edges.

Consequently the pasted proposal to pool connected components directly is unsafe:
this graph would pool many neurons. Do not sum edge overlap counts as exclusive
shared event mass. Establish plausible links using depth/waveform compatibility
and a time-shift null, then match events exclusively within validated families.
Temporal complementarity alone is not proof of a single neuron.

## 4. Adverse evidence remains, but needs its proper scope

Off has 230 clusters / 7,772,209 spikes; rigid has 164 / 7,084,042. Those are
cluster and event losses, not measured neuron losses. Aggregate median 1.5 ms
refractory violation fraction increases by 0.007515 (0.7515 percentage points).
Within the 40 matched interior pairs, the median paired increase is much smaller:
0.0001551 (0.01551 percentage points), with 22/40 worse. These are different
cohorts and estimands; the paired result does not clear the unmatched units.

The existing nonrigid benchmark summary reports 156 clusters / 6,966,375 spikes,
34 interior pairs, and the same 4/141 completeness coverage. It does not show an
obvious aggregate rescue. Its unit arrays and motion field are absent from this
compact handoff, so motion-conditioned benefit and matched refractory burden
have not been audited here.

## 5. Historical and external evidence

The original population truncation comparison remains composition-confounded;
the first matched numerical interpretation was explicitly retracted for target
event reuse. See `docs/luke_20250804_truncation_fitter_audit.md`. That does not
retract the observation that individual units can have amplitude instability;
it retracts its use as an identified cross-pipeline motion effect.

Separately, the earlier within-Luke dose-response report found descriptive
associations of motion with waveform stability (rho −0.66), qualified firing
rate (−0.59), and fragmentation (+0.42), despite a null yield endpoint.
Those came from 24 short-window sorts, not the current arm comparison, and
do not establish causality or efficacy. They illustrate why yield alone is
insufficient. See `docs/luke_within_rigid_motion_dose_response_result.md`.

The [Kilosort4 ablation study](https://www.nature.com/articles/s41592-024-02232-7)
supports drift correction's importance in its simulations. The
[official parameter guidance](https://kilosort.readthedocs.io/en/latest/parameters.html)
also warns about unreliable estimates with small/sparse probes. Neither
establishes that Luke's particular estimated field and application are correct.

## Required diagnostic before a full-run decision

1. Retrieve existing strip rigid/nonrigid `ops.npy` or exported motion fields
   with settings, time base, geometry, and provenance. These are absent from the
   compact handoff. Compare against qualified independent motion estimates,
   verifying physical sign and time origin before examining correlation,
   excursion, steps, and lag. Correlation after freely choosing sign is not an
   application-sign validation.
2. On validated matched families, compare off and corrected event recovery in
   identical physical-time windows selected from independent motion. Include
   zero-event windows and explicitly report missing fits. Compare within-family
   high-versus-nearby-quiet changes between arms; use matched raw-waveform
   measurements to distinguish genuine recovery from contamination. Firing-state
   confounding and amplitude truncation can defeat a simple rate/amplitude ratio.
3. Use family reconciliation only after qualifying graph links. Audit pooled
   refractory burden and exclusive events; do not equate consolidation with
   successful merging without waveform evidence.
4. If strip motion estimation is questionable, prepare a full-probe
   registration-only comparison before spending on another full sort. That
   computation is outstanding, not an already-run diagnostic. Any launch must
   follow AGENTS.md's durable-manager and interruption requirements.

Retain the historical artifacts. Treat `close_native_rigid` as an unsupported
scientific closure pending this metric correction and diagnostic pass. A
failure to demonstrate benefit with an infeasible endpoint is not evidence of
equivalence or evidence that motion is unimportant.
