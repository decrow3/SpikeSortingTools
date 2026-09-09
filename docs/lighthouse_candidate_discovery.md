# Preferred lighthouse candidate discovery: waveform-only, whole-probe matching

Promoted by user direction on 2026-09-08. Use this as the default starting method
for finding lighthouse candidates: **remove absolute depth from identity matching
while retaining the waveform's relative shape across neighboring channels**.

The promoted workflow finds and exposes candidate identities through large
possible displacements. It does not certify each template as an independent cell,
make waveform centroids calibrated motion ground truth, or change the production
sorting pipeline. The numerical gates below describe the Luke pilot, not universal
thresholds for every recording.

## Workflow

1. **Start with cached evidence.** Inspect seed waveforms and their closest
   waveform lookalikes across the entire available template inventory. Reuse
   existing whole-probe scores before extracting voltage again. This first check
   establishes whether candidates are distinctive and detectable; it does not
   establish movement.
2. **Preserve relative spatial shape, remove absolute position.** Recenter each
   multichannel footprint in a common relative geometry and align waveform
   timing. Compare temporal shape and relative amplitudes across neighboring
   channels. Keep physical channel geometry to assemble translated support;
   do not use seed-depth proximity, depth quotas, or agreement with a motion
   estimate to choose identities. This is not single-channel shape matching.
3. **Test seed recovery and freeze selection.** Use a declared seed interval to
   assess waveform repeatability, SNR, locality, global rival distinctness, and
   recovery of seed spike times. Save selection and thresholds before reviewing
   later observations. Record any already-inspected intervals: a time-separated
   evaluation interval is not automatically an untouched confirmation of the
   overall method-selection process. Existing sorted labels may propose seeds;
   they do not establish later identity.
4. **Search across the available probe.** Compare detections against all available
   complete-support identities, including nonselected rivals. Give no preference
   to the original depth, expected displacement, DREDGE, or a continuous path.
   Keep gain and timing rules explicit. Record incomplete edge support, geometry
   restrictions, unmatched detections, rival scores, and location alternatives.
   A deliberately bounded diagnostic is allowed as a labeled control, not a
   substitute for whole-probe candidate discovery.
5. **Expand visibly when strict recovery is sparse.** Examine the seed-recovery
   bottleneck first. Lower-score or identity-ambiguous recovery can nominate
   additional exploratory candidates when the expansion rule is stated and
   selected on training data. Preserve original strict decisions and report
   which candidates require the relaxed evidence. Do not silently turn an
   ambiguous or lower-score match into a strict match.
6. **Reveal depth after matching.** Measure waveform centroids on the observed,
   translated channel support. Plot absolute depth against localized peak
   depth/time scatters. Keep accepted, lower-score, and ambiguous marks visible,
   with seed intervals, event counts, and gaps. Compare original and compensated
   backgrounds with identical waveform observations where useful; the background
   is for inspection after matching, not selection or identity scoring.
7. **Inspect replication and obvious artifacts before estimating consensus.**
   Compare seed and held-out waveform examples, inspect competing depth bands,
   and look for shared movement across independently supported identities.
   Repeated separated bands may be waveform lookalikes. Preserve mixed identities,
   ambiguous depths and dropout; do not force a path through them. Only then
   consider adaptive summaries, interpolation, or depth-resolved consensus.

## Luke reference configuration

The completed pilot searched **930–1,030 s**, using **930–940 s** for seed selection
and **940–1,030 s** for later support. It compared **247 complete seed templates**
with exact relative geometry on **16-channel patches translated in 40-µm steps**
across the available probe. Absolute lateral geometry was preserved; no spatial
interpolation or reflection was used. Waveforms used 49 samples centered on the
peak, normalized multichannel cosine, and ±3 samples of timing alignment.

The detector used both signs on every eligible channel, a threshold of
max(30 µV, 3 noise sigma), 0.8-ms per-channel spacing, and the strongest local
channel within ±40 µm at the peak time. These are relative-support/detection
choices, not an absolute-depth identity prior. Waveform preprocessing was
300–6,000 Hz third-order zero-phase Butterworth filtering and global median
reference on the original input. Save these choices for every new run.

Seed eligibility required repeatability ≥0.85, SNR ≥5, local energy fraction
≥0.35, far-peak ratio ≤0.8, and closest global rival cosine <0.95. Candidate
recovery required at least five and 20% of cached seed spike times, within
±0.3 ms. The original five candidates met recovery under strict acceptance.
Expansion allowed cosine ≥0.80, including ambiguous winners: eight additions
also met recovery with the identity margin, and four additions required
ambiguity to meet recovery. No candidate was selected for its later displacement.

| Evidence class | Pilot rule | Plot encoding |
|---|---|---|
| Strict accepted | Original accepted status: cosine ≥0.86, winner–rival margin ≥0.025, gain 0.35–3 | Filled blue circle |
| Lower-score support | Cosine 0.80–<0.86, margin ≥0.025, gain 0.35–3 | Open blue circle |
| Identity ambiguous | Cosine ≥0.80, margin <0.025, gain 0.35–3 | Orange cross |
| Unmatched or below display gates | Remaining detections retained in source results | Retained in data; not assigned a trace |

These thresholds are exploratory and have not been calibrated to a whole-probe
false-positive rate. The expanded figures contain **17 candidate templates,
12 more than the original five**. The additions have **248 later matches passing
the unchanged strict rule**. This demonstrates useful additional candidate
support, not 17 verified independent neurons or validated shared motion.

The plot's short lines connect consecutive strict observations only within the
same 40-µm support patch and with gaps ≤2 s. This is a display rule, not a motion
constraint used in matching. Points across patch changes remain visible but
unconnected. Peak localization and waveform energy centroid are different
position summaries; exact overlap is not required.

## Reuse and reproducibility

The reference scripts are currently specific to Luke and the stated interval;
version their inputs/settings/output paths before adapting them. Do not rerun
an extraction merely to change plots or candidate inclusion.

| Stage | Implementation | Saved evidence |
|---|---|---|
| Seed comparison and global extraction | [Global waveform test](../testing/luke_waveform_only_global_v1.py) | [Settings, scores, chunk receipts, seed comparison](../testing/outputs/luke_waveform_only_global_v1/) |
| Original seed recovery | [Training qualification](../testing/luke_waveform_only_seed_qualification_v1.py) | `training_qualified_candidates.csv` in the global output |
| Sensitivity and waveform inspection | [Recovery review](../testing/luke_waveform_only_global_review_v1.py), [direct examples](../testing/luke_waveform_only_examples_v1.py) | Sensitivity, observed support, conflict audit, and waveform examples in the global output |
| Expanded candidate cohort and plots | [Cached expansion](../testing/luke_waveform_only_expansion_v1.py) | [Candidate audit, settings, event classes, counts and validation](../testing/outputs/luke_waveform_only_expansion_v1/) |

Start visual review with the [17-candidate overview](../testing/outputs/luke_waveform_only_expansion_v1/01_candidate_overview.pdf)
and [individual original/compensated peak-scatter overlays](../testing/outputs/luke_waveform_only_expansion_v1/02_all_17_candidate_overlays.pdf).
The [five-cell overlays](../testing/outputs/luke_waveform_only_peak_overlay_v1/02_individual_cell_overlays.pdf)
and [original expansion record](luke_waveform_only_expansion_20260908.md) remain
available as controls and history.

For substantial new extraction, follow [the managed-job requirements](../AGENTS.md#running-spike-sorts)
and [depth-aware measurement policy](luke_depth_aware_lighthouse_policy_20260908.md).
The reference extraction ran in an independently managed systemd service. It
saved full per-identity score/gain/timing alternatives and hash-validated completed
five-second chunks. Interrupted chunks have no internal checkpoint and must be
investigated before restarting; completed-stage reuse is not within-stage resume.
Keep source hashes, resolved settings, launch command, job identity, logs, and
final exit status outside the chat.

Exact 40-µm translations can miss intermediate waveform changes, incomplete edge
patches are excluded, unknown cells and collisions can resemble seeds, and seed
labels inherit the original sorting. Preserve fixed-template and bounded-search
results as historical controls. Their dropout or apparent stationarity does not
establish that neurons were stationary or that a motion estimator was wrong.
