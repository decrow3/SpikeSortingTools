# Luke0804 imec0 MEDiCINe-aware clustering and merge plan

**Status:** bounded v1 implemented and executed; exploratory, no
motion-specific merge accepted  
**Date:** 2026-09-09  
**Primary window:** Luke0804 imec0, 930--1230 s relative to the selected
recording  
**Production baseline:** accepted Kilosort4 12/9 sort with motion correction
off

The measured displacement, speed distribution, estimator bandwidth, AP-peak
support, and probe-scale velocity limits for this window are summarized in
[the Luke motion-regime reference](luke_expected_motion_regime_20260910.md).

**Revision after implementation review:** the first draft incorrectly treated
20 um as an exact same-column translation and assumed that the saved MEDiCINe
field already carried support/confidence arrays. Luke imec0 alternates x-columns
between adjacent 20 um rows; identical x-columns recur every 40 um. The saved
five-minute field contains only `time_s`, `depth_um`, and `displacement_um`.
This revision uses 40 um same-column translations, defines the spike-depth
measurement, removes fixed-channel Kilosort similarity as the motion-arm
candidate gate, preregisters state-to-cluster aggregation, and makes field
qualification an explicit prerequisite.

## Bounded v1 implementation result

The implementation is in `pipeline/motion_aware_merge.py`,
`pipeline/motion_merge_gates.py`, and the `testing/luke_medicine_*` replay
drivers. Auditable outputs and atomic receipts are under
`testing/outputs/luke_medicine_motion_aware_merge_v1`.

- The 930--1230 s replay contains 782,910 saved detections and 343 freshly
  reconstructed premerge clusters. It reads no AP voltage and changes no AP
  voltage.
- Exact same-column geometry has zero overlap at 20 um, 380 channels at 40 um,
  and 376 channels at 80 um. Synthetic 40 and 80 um translations recover a
  similarity of 1.0 at the correct offset.
- Split-half field qualification marks 1,961 of 4,800 field knots eligible
  (40.85%). All-corner interpolation leaves 140,958 spikes eligible (18.00%).
- The correctly signed motion arm produces one provisional state-aggregate
  edge. Opposite-sign and circular 60 s time-misalignment controls produce
  none.
- The provisional edge, bounded clusters 128 and 136, is rejected by the rival
  margin, by absent independent coherent-family support, and by the pinned
  Kilosort CCG/refractory criteria.
- Static Kilosort replay performs 20 ordinary merges (343 to 323 clusters).
  The exact-zero-field adapter reproduces static labels, templates, and
  refractory flags bit-for-bit. The delivered motion-aware labels preserve
  those 20 baseline merges and add zero motion-specific merges.

This run is exploratory rather than confirmatory because the 0.05 rival margin
was not preregistered before the eligible motion scores were opened. The output
therefore disables motion-specific merge authorization even if an edge were to
pass the substantive gates. A v2 must calibrate and freeze that margin from the
static/synthetic and sign/time null controls before opening its motion arm.

## Decision this plan is designed to support

Determine whether the qualified MEDiCINe motion field can improve neuronal
identity continuity when it is consumed after detection and template learning,
without spatially resampling the AP voltage.

The first experiment is deliberately narrow:

> Can motion-aware spatial alignment reconnect independently recognizable
> Kilosort4 fragments that the static final merge leaves separate?

This is a mechanistic diagnostic, not a proposed production sorter. A positive
result motivates testing an established drift-tracking architecture, especially
DARTsort, before extending a custom Kilosort4 fork. A negative result stops the
custom merge path unless a specific, observed failure suggests one bounded
revision.

## Motivation

The five-minute MEDiCINe estimate appears useful enough to test as a correction
coordinate. Most adjacent 250 ms estimates imply less than one 20 um probe-row
movement, although a small set of rapid epochs exceeds that regime. Estimator
support does not establish that spatially interpolating AP voltage is safe.

Existing Luke evidence separates those questions:

- A zero-displacement interpolation control is byte-identical to the
  no-correction input, localizing earlier failures to nonzero spatial warping.
- DREDGE and MEDiCINe nonrigid voltage warps produced pathological
  overdecomposition.
- Rigid MEDiCINe voltage correction also degraded full-session sorting yield and
  quality despite preserving event timing.
- A conservative rigid-gain voltage arm did not survive its supported-crop
  confirmation.

The correction target should therefore be motion-invariant unit identity, not a
synthetic stationary voltage recording.

This direction has a close precedent. DARTsort registers spike coordinates,
clusters in registered location space, learns drift-position-specific templates,
and aligns those templates during drift-aware matching and merging. It explicitly
avoids making trace interpolation the central correction operation. The proposed
Kilosort4 replay is a minimal test of one part of that design, not a claim of a
new sorter architecture.

## Kilosort4 insertion point

In the pinned Kilosort4 implementation, two adjacent operations need to be
distinguished:

1. `clustering_qr.run(..., mode="template")` performs the final clustering. It
   assigns each learned detection template to a fixed probe-centered spatial
   partition and embeds its `tF` features in absolute physical-channel slots.
2. `template_matching.merging_function(...)` ranks the resulting clusters using
   `Wall` waveform correlation over temporal lag, then applies a CCG gate.
   Spatial specificity is implicit in the fixed-channel support of `Wall`; the
   function does not search spatial offsets.

Adding a corrected depth only to candidate selection would therefore be
incomplete. Fragments could be routed to the same anatomical neighborhood while
their feature vectors and `Wall` templates remained displaced over physical
channels.

The first experiment avoids that larger refactor. It starts from the preserved
premerge state and aligns cluster templates only for the purpose of replaying
the final merge.

## Scientific invariants

- Use the accepted, unwarped AP recording. Do not interpolate or rewrite traces.
- Do not rerun detection or template learning for the first replay.
- Freeze the MEDiCINe field before inspecting motion-aware merge results.
- Use the physical-displacement convention
  `corrected_depth = observed_depth - displacement(time, observed_depth)`.
- Validate the time origin against the selected recording duration; acquisition
  clock values must not be silently treated as selected-recording time.
- Apply no time or depth extrapolation outside the qualified field domain and
  eligible mask.
- Preserve ineligible events and merge candidates explicitly. Do not silently
  assign zero motion or nearest-boundary motion.
- Preserve the unmodified premerge clusters, static replay, ambiguous matches,
  unmatched events, and every newly introduced or removed merge edge.
- Unit count alone cannot advance a candidate.
- Do not use MEDiCINe, corrected absolute depth, or a motion prior to select or
  identify the lighthouse families that will evaluate MEDiCINe-aware merging.

## Independence of lighthouse discovery

Lighthouse discovery and motion-aware correction must remain separate branches.

### Lighthouse branch

- Start with the unmodified premerge clusters.
- Search across the whole supported probe using relative multichannel waveform
  geometry.
- Exclude absolute depth and MEDiCINe from candidate ranking, identity matching,
  and competition between hypotheses.
- Select candidates from the seed interval before reviewing their later support.
- Preserve strict, lower-score, ambiguous, and unmatched alternatives.
- Reveal absolute depth and MEDiCINe only after identity hypotheses are frozen.

### Correction branch

- Use the frozen MEDiCINe field to align templates and propose merge evidence.
- Evaluate new merge edges against the independently frozen lighthouse families.
- Do not promote a MEDiCINe-created family as independent validation of the same
  motion field.

See [lighthouse candidate discovery](lighthouse_candidate_discovery.md) and the
[depth-aware lighthouse policy](luke_depth_aware_lighthouse_policy_20260908.md).

## Phase 0: freeze and validate inputs

### Required artifacts

- Accepted no-motion Kilosort4 recording and sort manifests.
- Exact 930--1230 s sample bounds and sampling frequency.
- Premerge `st`, `tF`, `clu`, `Wall`, and `ops`, or an equivalent complete
  checkpoint immediately before `merging_function`.
- Mapping from every `tF` feature column to its physical probe channel.
- The saved five-minute MEDiCINe `time_s`, `depth_um`, and `displacement_um`
  arrays plus their fit settings, input peak/localization arrays, and provenance.
- A separately derived and qualified field-eligibility sidecar as specified
  below. The current three-array field is not a `qualified-motion-field-v1`
  artifact and must not be passed to the generic qualified-field loader.
- A frozen per-spike observed-depth sidecar computed as specified below.
- Frozen independent lighthouse candidate registry and all competing identities.

### Integrity checks

- Hash every input and write a request manifest before computation.
- Confirm that `st`, `tF`, and `clu` contain the same number and ordering of
  spikes.
- Confirm contiguous premerge cluster IDs and one `Wall` row per cluster.
- Confirm that channel locations, `iCC`, `iU`, and feature support agree.
- Confirm that MEDiCINe uses probe-y micrometers, observed-depth-offset sign, and
  selected-recording time.
- Verify the displacement sign with a known geometry translation before making
  biological interpretations.
- Confirm directly from `channel_positions.npy` that adjacent 20 um rows switch
  between x-column sets and that exact same-column translations recur at 40 um.
- Construct and qualify support/eligibility rather than treating missing arrays
  as present. Report the ineligible fraction by time and depth.

If a complete premerge checkpoint is unavailable, first add a read-only save
hook immediately before the existing merge call. Do not reconstruct a supposed
premerge state from final labels.

### Frozen observed-depth definition

For this experiment, `z_observed` means the per-spike squared-PC-feature energy
centroid computed from the saved premerge `tF` using the pinned Kilosort4
`compute_spike_positions` convention:

1. Sum `tF**2` over PCs for each local feature channel.
2. Normalize those nonnegative channel weights within each spike.
3. Map local feature columns through `iCC[:, iU[st[:, 1]]]` to physical channels.
4. Take the weighted mean of the corresponding `ops['yc']` coordinates.

This is neither template peak depth nor a later exported depth chosen by name.
Save the resulting sidecar, formula/version, channel mapping, and source hashes.
A spike with zero or nonfinite total feature energy is ineligible; do not fill or
clip its depth. The centroid is bounded by its real feature-channel support, and
the field sampler performs no depth extrapolation at probe edges.

Validate the estimator before use with synthetic `tF` observations translated
by exact 40 and 80 um same-column shifts on the actual imec0 geometry. The
centroid must move with the known sign and distance within a frozen tolerance.

### MEDiCINe support and eligibility qualification

The saved five-minute field has no native support or confidence arrays. Do not
manufacture them with constants or relabel field-domain membership as
confidence.

Create a separate qualification sidecar from the exact screened peaks and the
frozen MEDiCINe kernels/settings:

- Define local support at every time-depth knot as the Kish effective sample
  size `(sum(w))**2 / sum(w**2)` of contributing peaks, using the estimator's
  actual frozen temporal and spatial weights `w`.
- Record raw contributing-peak count and total weight alongside effective sample
  size so the statistic is auditable.
- Run a deterministic amplitude/depth/time-stratified split-half refit with the
  same hyperparameters. Record absolute full-versus-half and half-versus-half
  displacement differences at every knot.
- Derive a binary `eligible` mask from preregistered minimum effective support
  and maximum split-half discrepancy. Choose those numerical thresholds using
  synthetic known-shift recovery and the estimator's quiet-period repeatability,
  then freeze them before inspecting merge outcomes.
- A spike-time/depth query is eligible only when every interpolation corner with
  nonzero weight passes the mask. Do not extrapolate across an ineligible corner.

The first exact-translation premise check below may run before the split-half
refit, but it is descriptive only and may not accept biological merges. The full
merge replay is blocked until the sidecar and its qualification receipt exist.

## Phase 1: exact static replay and controls

Before adding motion, replay the pinned Kilosort4 merging function from the
frozen checkpoint.

Required controls:

1. **Static replay:** original `Wall`, original merge code, and original
   thresholds.
2. **Zero-field identity:** run the motion-aware adapter with a qualified field
   containing exactly zero displacement.
3. **Sign control:** evaluate the opposite displacement sign as a diagnostic,
   never as an eligible scientific arm.
4. **Time-misalignment control:** shift or permute the displacement trajectory by
   a preregistered amount that destroys its alignment to the recording while
   preserving its marginal displacement distribution.

The static replay must reproduce the original merge labels and templates, apart
from explicitly documented nondeterminism. The zero-field adapter must reproduce
the static replay exactly. Failure of either identity check blocks the motion
arm.

### Quick representation premise check

Before building state aggregation, select a few cluster pairs using frozen
waveform-only evidence. Reuse their saved `tF`, `ops`, and actual imec0 geometry
to compare exact spatial offsets of 0, 40, and 80 um in both directions.

- Translate only between contacts with identical x coordinate and a real target
  channel.
- Compare each intended fragment pair against frozen rival clusters.
- Report intended-pair score, best-rival score, and their separation at every
  offset; do not report only the maximizing offset.
- Include translated synthetic templates as positive controls and similar
  same-peak-channel neurons as negative controls.

This check can establish whether `tF`/`Wall` retains useful waveform identity
under exact same-column translation and whether alignment improves separation
from rivals. It cannot establish merge safety, unit identity, or continuity. If
the representation fails this check, stop before field qualification and the
full merge replay.

## Phase 2: minimal MEDiCINe-aware merge replay

### 2.1 Motion sampling

For each spike, bilinearly sample MEDiCINe at its time and the frozen
squared-PC-feature centroid defined in Phase 0. Apply the derived eligibility
mask at every active interpolation corner. The corrected anatomical depth is

```text
z_corrected = z_observed - displacement_um(t, z_observed)
```

The field may be evaluated continuously in time, but interpolation between its
250 ms samples does not create finer estimator bandwidth. Keep the original
effective support, split-half discrepancy, eligibility, and source-bin
identifiers attached to each evaluation.

### 2.2 Motion-state templates

A single cluster-level median displacement is insufficient when a cluster spans
multiple motion states. For every premerge cluster:

- Divide eligible spikes into 40 um displacement bins. This matches the smallest
  exact same-column recurrence on the staggered imec0 geometry, not its 20 um
  row spacing.
- Require a preregistered minimum number of spikes to construct a state template.
- Recompute each state template directly from the saved `tF` observations.
- Retain the cluster's ineligible spikes as an explicit unmatched state.
- Record time span, spike count, displacement median/range, corrected-depth
  distribution, amplitude distribution, effective support, split-half
  discrepancy, and physical channel support for every state template.

Do not smooth state templates across ineligible intervals or force a cluster to
occupy every intermediate state.

### 2.3 Exact same-column spatial alignment

For a candidate pair of state templates, translate their channel-coordinate
labels by the MEDiCINe-predicted relative displacement, rounded to the nearest
40 um same-column step. Preserve shank and x-column identity and compare only
real coordinate-matched channels. This is a two-row translation on imec0; it is
not described as an exact 20 um shift.

The eligible primary score uses only the predicted quantized offset and does not
maximize over residual offsets. A separately labeled sensitivity analysis may
also score the adjacent same-column offsets, ±40 um from the prediction. Save all
three scores. Correct its inference for the three tested offsets using a
max-statistic null derived from the frozen sign/time-misalignment controls. The
sensitivity maximum cannot accept a merge or advance the method.

No stagger-changing 20 um operator, kriging, fractional spatial interpolation,
raw-voltage interpolation, or extrapolated channel values are allowed in this
phase. A 20 um arm would require its own explicitly defined and synthetically
validated operator and is deferred.

### 2.4 Candidate pairs and merge evidence

Do not use the original fixed-channel Kilosort4 `cmax` threshold or early-stop
ordering to select motion-arm candidates: that would exclude the spatially
nonoverlapping fragments the experiment is intended to recover.

For the small premise cohort, score every intended pair and every frozen rival.
For the full replay, give every unordered premerge cluster pair an aligned score.
If that proves computationally excessive, replace it only with a separately
frozen position-invariant waveform shortlist that compares relative
multichannel geometry over all feasible exact 40 um translations without using
absolute depth or MEDiCINe. Validate shortlist recall against exhaustive scoring
on the premise cohort before allowing it. Corrected depth may restrict which
scored edges proceed to expensive evaluation, but it is not merge evidence by
itself.

For every candidate pair save:

- Unaligned and aligned multichannel waveform similarity.
- Predicted, residual, and total spatial offset.
- Number and geometry of overlapping channels.
- Template residual energy after alignment.
- Spike counts and temporal support of both clusters.
- Corrected-depth distributions and overlap.
- Amplitude distributions and transition continuity.
- Existing Kilosort4 CCG decision.
- Refractory and near-coincident burden before and after the proposed merge.
- Whether independently frozen lighthouse evidence supports, contradicts, or is
  ambiguous about the edge.

The existing CCG gate is necessary but insufficient. Temporally disjoint units
can pass it trivially. A disjoint-support merge must additionally have strong
aligned waveform evidence and either an independently supported bridge through
intermediate motion states or remain an uncommitted candidate edge.

### 2.5 State-to-cluster aggregation

Freeze waveform thresholds from the static/synthetic controls before inspecting
the eligible motion arm. Do not select the best single state pair.

For each cluster pair:

1. A state pair is eligible only when both templates meet the frozen spike-count
   rule, the field mask is eligible, and the predicted exact translation leaves
   the frozen minimum number of real coordinate-matched channels.
2. Compute the primary aligned score only at the predicted 40 um-quantized
   offset. Temporal-lag maximization remains exactly the pinned Kilosort4
   operation and is included when calibrating its static/null threshold.
3. Form reciprocal state matches: state `a` must be the highest-scoring eligible
   counterpart of `b`, and `b` the highest-scoring counterpart of `a`. Ties are
   retained as ambiguous.
4. Give each reciprocal state pair equal weight, not weight proportional to
   spike count. Summarize support with the median aligned score and the lower
   quartile; never use the maximum across states.
5. Require at least two reciprocal supported state pairs and require them to
   cover a frozen minimum fraction of eligible spikes in each cluster. Set the
   numerical count, channel-overlap, score, and coverage thresholds from the
   static/synthetic controls and freeze them before opening the motion arm.
6. Veto the cluster edge if any well-supported eligible state has a reciprocal
   comparison below the frozen incompatibility threshold, prefers a frozen rival
   by the required separation margin, contradicts an independent lighthouse
   identity, or produces an overlapping-time duplicate/refractory conflict.
7. Clusters with disjoint temporal support additionally require an independent
   waveform-supported intermediate state or track piece bridging them. Without
   that bridge, retain the edge as an uncommitted candidate regardless of its
   aggregate score.
8. Apply the standard CCG check to the union only after the state aggregation and
   veto rules pass. Passing CCG cannot override a state contradiction.

Save every eligible and ineligible state comparison so the coverage denominator
and vetoes are reproducible. The sign, time, offset-sensitivity, and rival
controls define the multiple-comparison null; none may be pooled into the
eligible arm after results are inspected.

### 2.6 Graph resolution

- Preserve pairwise edges before resolving connected components.
- Do not allow one weak edge to transitively merge several otherwise incompatible
  clusters.
- Recompute aggregate state templates and all relevant guardrails after each
  accepted greedy merge, matching Kilosort4's iterative behavior.
- Mark conflicting components as ambiguous rather than selecting the largest
  cluster automatically.
- Emit original-to-final label maps and a reason code for every accepted,
  rejected, and ambiguous edge.

## Evaluation

### Primary question

Among independently selected, waveform-supported lighthouse families, does the
MEDiCINe-aware replay recover credible fragment links missed by static Kilosort4
without introducing contradicted links?

### Primary measurements

- Number and fraction of eligible lighthouse fragment links recovered by static
  and motion-aware replay.
- Number of lighthouse-supported, contradicted, and ambiguous new merge edges.
- Family continuity through quiet--motion--quiet transitions.
- Aligned raw-waveform and `tF` similarity at each accepted edge.
- Ineligible or missing intermediate states.

### Global guardrails

- Total spikes and clusters before and after merging.
- Near-coincident duplicate burden.
- Median and tail refractory-violation fractions.
- Contamination estimates.
- Presence and endpoint-presence distributions.
- Number and size of one-to-many connected components.
- Amplitude discontinuity across newly joined pieces.
- Merge-edge sensitivity to sign reversal, time misalignment, and the separately
  labeled adjacent-offset sensitivity analysis.

### Interpretation rules

- A lower unit count is not evidence of improvement.
- Motion agreement is not identity evidence when motion created the identity.
- Static-template dropout does not establish MEDiCINe error.
- Failure during ineligible high-speed epochs does not invalidate eligible
  quiet or moderate-speed results; report the regimes separately.
- Improvement confined to a large residual spatial search, rather than the
  MEDiCINe prediction, does not support MEDiCINe-aware correction.

## Advancement and stop rules

Freeze numerical tolerances after the static controls and independent lighthouse
registry are complete but before opening the motion-arm results.

Advance beyond the diagnostic only if all of the following hold:

- Static and zero-field identity controls pass.
- At least several independent lighthouse families provide eligible fragment
  tests across motion states; do not advance from a single exemplar.
- Motion alignment yields a clear positive change in supported reconnections
  relative to static replay.
- No independently contradicted merge is accepted.
- Refractory, coincidence, contamination, and component-size guardrails do not
  show a material global regression.
- The effect is specific to the correctly signed and temporally aligned motion
  field.
- Results are not driven only by ineligible bins, and the eligible conclusion
  does not depend on the adjacent-offset sensitivity maximum.

Stop the custom merge path if:

- The identity controls fail.
- Eligible independent lighthouse families are too sparse to distinguish the
  arms.
- Correctly aligned MEDiCINe performs no better than the sign/time controls.
- Apparent recovery requires fractional interpolation or a wide unconstrained
  spatial search before the integer-row premise is supported.
- New merge edges increase known identity conflicts, duplicate burden, or
  refractory violations.
- The required premerge evidence is not preserved well enough for an exact
  paired replay.

One bounded revision is allowed only when the first result identifies a specific
implementation or support failure. Do not begin an open-ended parameter sweep.

## Phase 3: decision after the merge replay

### If the replay is negative

- Preserve the result as evidence that a final-merge-only correction is too late
  or that this representation cannot exploit the field.
- Continue using the premerge clusters for independent lighthouse discovery.
- Do not add fractional feature interpolation or deeper Kilosort4 modifications
  without a specific falsifiable reason.

### If the replay is positive

Run a same-window architecture comparison:

1. Accepted Kilosort4 12/9, motion off.
2. Kilosort4 with the minimal MEDiCINe-aware final merge.
3. A current, separately pinned DARTsort challenger on the same unwarped window
   with a validated preprocessing/standardization scale.
4. If the current DARTsort interfaces support a clean external-field handoff,
   DARTsort with frozen MEDiCINe motion in addition to its native-motion arm.

Native DARTsort changes both motion estimation and correction architecture. It
therefore tests a complete sorter, not MEDiCINe consumption in isolation. The
external-motion API and sign/time conventions must be audited before that arm is
prespecified.

The historical DARTsort 0.5.16 imec1 120 s result is integration evidence only.
It used another probe/window and had unresolved preprocessing-scale and temporal
sampling limitations. It is not a verdict on current DARTsort or this imec0
five-minute experiment. See [sorter architecture bake-off](sorter_architecture_bakeoff.md).

Only after the minimal replay beats its static control should we consider moving
MEDiCINe upstream into Kilosort4 final clustering. That larger intervention
would require per-spike corrected spatial routing and motion-aligned `tF`
embedding; changing the routing coordinate alone is not sufficient.

## Deliverables

The bounded replay should produce:

- A fingerprinted input and parameter manifest.
- Static, zero-field, sign, time-misaligned, and eligible MEDiCINe receipts.
- Per-spike raw depth, corrected depth, displacement, effective support,
  split-half discrepancy, and eligibility sidecars.
- Motion-state template metadata and arrays.
- Complete pairwise merge-edge tables before graph resolution.
- Original-to-final label maps for every arm.
- A lighthouse-family comparison table.
- Global guardrail summaries.
- A compact figure showing raw peak depth/time, frozen lighthouse observations,
  MEDiCINe added after matching, state-template support, and accepted/ambiguous
  merge edges.
- A final decision JSON and short Markdown result note stating advance, revise,
  or stop.

All outputs must be written atomically through a `.partial` directory, retain
failed-run evidence, and refuse a mismatched pre-existing request digest.

## Testing requirements

Add unit tests for:

- Motion sign and time-origin mapping.
- Bilinear field sampling and rejection of ineligible corners.
- Exact zero-field identity.
- Exact 40/80 um same-column geometry translation on the actual staggered probe
  layout.
- Preservation of x-column and shank identity.
- Overlap-only waveform comparison with no extrapolation.
- State-template construction and minimum-support handling.
- Deterministic pair ordering and graph resolution.
- Transitive-conflict preservation.
- Manifest mismatch, partial output, and tampered input refusal.

Use synthetic templates translated by known 40 and 80 um offsets before applying
the method to Luke. Include two distinct neurons with similar peak-channel
waveforms as a negative merge control.

## Operational requirements

The replay should be inexpensive and may not require a new spike sort. If any
bounded or full sorter run is subsequently launched, it must use an independent
job manager such as a systemd user service. Persist the resolved command,
settings, job identifier, stdout/stderr, and final exit status. Verify launcher
disconnection survival with a cheap dummy job before trusting a new launch path,
and state explicitly whether interruption requires restarting the sort.

## External references

- [DARTsort repository](https://github.com/cwindolf/dartsort)
- [DARTsort preprint](https://doi.org/10.1101/2023.08.11.553023)
- [DREDGE and modular drift-correction benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897502/)
- [NP Ultra application using registered spike positions and DARTsort](https://pmc.ncbi.nlm.nih.gov/articles/PMC10473688/)
