# Next step: turn amplitude-completeness failures into one pipeline decision

Date: 2026-09-05. Status: proposed implementation prescription, not a result.

**Delivery sequencing update:** the
[first-version delivery sequence](pipeline_improvement_plan.md#delivery-sequence-a-first-updated-pipeline-we-can-actually-use)
governs what happens after this audit. Its effort and case caps remain intact.
An inconclusive audit closes the diagnostic checkpoint and uses that sequence's
explicit candidate-selection fallback; it does not require another exploratory
round before a pipeline can be integrated. The audit implementation contract and
all frozen scientific gates remain unchanged.

## Purpose and justification

The practical goal is to recover a neuron's spike train throughout the recording,
including periods when its waveform becomes difficult to detect or assign.
The lab's useful diagnostic is estimated missing-spike percentage from truncated
amplitude fits in successive 1,000-spike windows. Rate alone cannot distinguish
reduced neural activity from incomplete capture. Use this existing diagnostic to
choose a real failure and test one improvement.

This is a bounded diagnostic checkpoint inside the existing development plan,
not a third pipeline-development branch. It should determine which existing
intervention deserves the next experiment. The latest threshold Stage 2 result
does not support replacing 12/9; this proposal does not reopen the 8/8 or 9/9
promotion gates. External registration and unwarped identity handling remain
the two development directions. Neither requires this audit to finish before
already-authorized implementation work proceeds.

Why spend any time here?

- The current benchmarks have exposed real failures, but their relationship to
  the dropouts visible in production QC remains insufficiently demonstrated.
- Existing sort and QC outputs can identify relevant periods without a new sort.
- A same-unit temporal comparison avoids the population-composition problem
  that invalidated earlier cross-sort completeness claims.
- Separating amplitude loss, reassignment, and curation loss selects different
  remedies. Building a remedy before distinguishing these costs more than a
  small audit if it targets the wrong stage.

The risk is another open-ended investigation. The caps, exits and deliverables
below are part of the implementation contract. Ambiguity is a permitted outcome;
it is not permission to add more donors, panels, metrics or sorter variants.

## Scope and decision budget

1. Start with existing Luke 2025-08-04 imec0 rescue and legacy curated outputs,
   because these already support the comparison and amplitude-QC work. Paths
   must be explicit inputs; no new recording or session search.
2. Inventory cached QC across these two sorts. Select at most four failure
   cases (two per sort) and two stable controls (one per sort).
3. Inspect existing arrays for all six; perform voltage review on at most the
   two highest-ranked failure cases and their corresponding controls.
4. Spend at most one working day on initial implementation and cached-output
   triage, then one working day on bounded evidence review. These are effort
   caps, not claims about measured runtime. At each cap write a decision even
   if the outcome is unresolved. Missing data produces a blocked-stage receipt,
   not automatic reconstruction of full-session intermediates.
5. End with one nominated intervention or `insufficient_evidence`. Allow one
   bounded candidate-versus-control experiment after its settings and evaluation
   contract are written. No full-session sort, general dashboard, broad sweep,
   new motion estimator or donor model belongs to this checkpoint.

If no actionable case survives, close the checkpoint and return to the existing
two-option programme. Report exactly which observation was unavailable; do not
make solving it a prerequisite for all pipeline work.

## Existing code: reuse and traps

Read these functions before coding:

- `pipeline/qc.py::run_qc` and `truncation_qc`.
- `pipeline/kilosort_results.py::KilosortResults.st`.
- `pipeline/truncation.py::construct_windows`, `analyze_amplitude_truncation`,
  `fit_amp_cdf`, `is_saturated`.
- `testing/luke_c2_v4_truncation_diagnostic.py::load_curated_arrays` and
  `exact_count_windows`, as implementation references, not wholesale imports
  of that experiment's 250-spike protocol.
- `testing/luke_truncation_matched_units.py` and decision 0011, as historical
  matching context. Do not assume a pre-existing matcher proves neuron identity.

Two verified implementation details must be explicit in every output:

1. Production QC amplitudes are `full_st[kept_spikes][:, 2]`, in sorter units.
   `amplitudes.npy` is a different observable. Do not substitute it, convert the
   QC amplitudes to microvolts, normalize each window, or pool amplitudes across
   clusters. Voltage waveform amplitudes in microvolts are separate measurements.
2. Production `construct_windows` stores inclusive endpoints, but
   `analyze_amplitude_truncation` slices `amps[i0:i1]`. A nominal 1,000-spike
   window therefore fits 999 values. This is already documented in the C2 v4
   truncation diagnostic. It is not evidence that it explains biological dropout.

Do not silently fix production QC or overwrite its caches as part of this task.
Reproduce the historical estimate, then compute an explicitly versioned
exact-1,000 estimate for selected cases using the same fitter and window starts.
If the indexing correction changes case eligibility, report that sensitivity
and mark the case unstable; do not replace it after reviewing candidate results.

## Deliverables and CLI

Add only these source files initially:

- `testing/luke_amplitude_dropout_audit.py`: loading, normalization, selection,
  evidence tables, static figures and a short decision report.
- `testing/test_luke_amplitude_dropout_audit.py`: contract and regression tests.

One runner with explicit subcommands is sufficient:

```text
python -m testing.luke_amplitude_dropout_audit inventory --config CONFIG --out-root LOCAL
python -m testing.luke_amplitude_dropout_audit select --config CONFIG --out-root LOCAL
python -m testing.luke_amplitude_dropout_audit inspect --selection SELECTION --out-root LOCAL
```

The JSON configuration supplies, per sort: an immutable sort ID, curated output,
QC directory, source recording, sampling frequency, selected-recording start
sample in the source, duration, channel geometry and available provenance
receipts. Validate supplied frequency/duration against the recording metadata.
Configuration also contains all selection constants below and their units.
`inventory` reads metadata and arrays only; no fitting, voltage extraction or
sorting. `select` uses cached QC, freezes case IDs and writes their input hashes.
`inspect` verifies that selection and computes only the selected evidence.

Write under one local output root, rejecting resolved paths beneath `/mnt` and
any input directory. Output files:

```text
manifest.json          # schema, source identities, code/config hashes, stage status
windows.csv            # normalized cached QC rows and explicit validity flags
selection.json         # frozen case IDs, intervals, ranks, reasons and constants
case_windows.csv       # historical replay and exact-1,000 sensitivity
case_evidence.csv      # evidence classification, limitations and artifact links
figures/<case_id>.png  # one aligned static panel per case
decision.md            # one next action, or insufficient_evidence
```

Manifest schema: `luke-amplitude-dropout-audit-v1`. Hash each consumed array/QC
file, or verify a content receipt that covers that exact file. Record the Git
commit AND hashes of relevant working-tree source files, since the workspace
may be dirty. Write status `running`, `complete`, or `failed` with a reason;
write final files atomically. Existing incompatible outputs must be refused.
Do not implement automatic cache repair or recursive “latest run” discovery.

## Array and time contracts

- Flatten spike times and cluster labels to one dimension. Times must be finite
  integer samples; preserve integer arithmetic until plotting. Reject invalid
  arrays rather than silently rounding. IDs are values, never array positions.
- Validate `kept_spikes` as either a Boolean mask of the full table's length or
  an in-range integer index vector. Validate resulting row count and exact time
  equality with `spike_times.npy`. Apply any stable time sort to all aligned
  arrays together. Preserve duplicate timestamps and original row IDs.
- Reconstruct each cluster's sequence in the same order production QC used.
  Cached `window_blocks` are indices into that cluster's sequence, not global
  spike rows, samples or seconds. If it is not time ordered, fail cached replay
  rather than reinterpret its indices after sorting.
- Validate equal QC row counts for `cid`, `window_blocks`, `popts`, `mpcts`;
  integer-valued IDs and indices; in-range ordered endpoints; and existing IDs.
  Missing QC for a cluster is `no_fit`, never zero missingness.
- For stored `[i0, i1]`, emit historical sample count `i1-i0`, nominal count
  `i1-i0+1`, and time bounds from the two endpoint spikes. Retain both counts.
  Historical replay fits `[i0:i1]`; exact replay fits `[i0:i1+1]`.
- Preserve the production gap rule: split when adjacent spikes are more than
  10 seconds apart; do not construct a fit across that gap. No interpolation of
  missingness across gaps, excluded tails or periods with too few spikes.
- Use selected-recording-relative time internally. Source voltage frames equal
  selected-relative frames plus the verified source offset. Never guess an
  offset from filenames. Clip extraction to bounds and report clipped margins.
- `windows.csv` has one row per stored window: sort ID, cluster ID, source row,
  endpoint indices, first/last sample, start/end seconds, both counts, missing
  percentage, parameters and status. Status precedence: invalid input,
  nonfinite fit, boundary-pinned, finite-interior. Keep all rows and reasons.
- A 50% boundary-pinned fit is not an estimate of exactly 50% or a validated
  biological lower bound. Display it distinctly and exclude it from numerical
  change scores. Finite-interior does not mean the model fits a mixed waveform
  population well; empirical CDF and fitted CDF remain visible.

## Deterministic case selection

These are proposed triage constants, not calibrated biological acceptance gates.
Freeze them in CONFIG before reading rankings; no adaptive relaxation if cases
are scarce. Use cached historical QC for initial selection.

For each cluster, enumerate four consecutive windows inside one gap-free block.
All four must have finite-interior estimates and nominal count 1,000. Treat the
first two as reference and the next two as failing. A qualifying transition has
both reference estimates <=5%, both failing estimates >=15%, and a difference
between the two medians >=10 percentage points. Reject spans longer than 600 s
for this initial audit; report excluded counts so slow units do not disappear
from the inventory. Select the largest difference per cluster, ties by earliest
start. Rank clusters by difference descending, then start and numeric cluster ID.
Take the first two per sort. Select before inspecting waveforms or interventions.

Stable controls require at least four consecutive valid windows within 600 s,
all <=5% with range <=3 percentage points. Select one per sort, excluding failure
IDs, minimizing the absolute log ratio of control-span duration to the first
failure span's duration; tie by start and ID. If there is no failure, choose the
earliest eligible control. Counts below the caps are valid; no backfilling with
new rules. Keep all labels eligible and report KS-good/MUA status descriptively.

This first version detects sustained rises with measurable fits, not every
dropout. Separately inventory boundary-pinned windows and no-fit gaps, without
ranking them as numeric missingness. They remain visible limitations.

## Evidence panel and classification

Each case panel shares a time axis: amplitude/time density, historical and exact
missingness with actual window widths, empirical/fitted amplitude CDFs for the
four windows, waveform/depth summaries, and spike rate as context. Shade no-fit
time separately from boundary-pinned fits. Plot missingness in percent and changes
in percentage points. Keep the frozen four-window interval visible in full.

For the two voltage-review cases, use at most 100 evenly spaced assigned events
per window and at most 16 channels nearest the reference peak channel, frozen
from the reference windows. Show before/during waveform distributions on that
same channel set; do not recenter each waveform to conceal a displacement.
Read bounded chunks, never a full-session voltage array. Include one fixed 50-ms
continuous voltage excerpt centered in each window, with event markers. These
excerpts illustrate evidence; they do not estimate recall or false-positive rate.
Record the voltage view, filter margins, gains and extraction choices. Missing
raw data leaves voltage evidence unavailable and lowers the conclusion strength.

Classify evidence using the following table. Multiple supported categories are
allowed; disagreement is `ambiguous`, not resolved by a majority of metrics.

| Evidence | Permitted conclusion | Next action |
|---|---|---|
| Aligned pre-curation rows exist and are demonstrably removed by curation | Curation contributes to exclusion | Replay one specific curation rule on retained arrays |
| Similar waveform events move into another cluster, supported by spatially restricted exclusive event matching and a fixed shift-null protocol | Identity redistribution is supported | Test one motion-aware identity rule |
| Waveform amplitude/depth changes accompany missingness; no assignment explanation is established | Motion/amplitude change is a candidate explanation | Choose a bounded existing registration or identity experiment; do not claim causality yet |
| Artifacts or voltage processing visibly alter the relevant waveform | Local voltage integrity is suspect | Replay one implicated processing operation on the same voltage |
| Only missingness or rate changes, or required intermediates are absent | Stage is unresolved | Stop with the missing observation identified |

Do not label an event “never detected” solely because it is absent from
`full_st`. Establish the exact semantics of retained arrays against the installed
KS4 source before making stage claims; exported tables need not contain every
universal-template candidate. Full-sort and curated IDs need not be identical.
Prefer explicit retained-row lineage over timestamp matching for curation.

Cross-sort correspondence is optional corroboration, not required to select a
case. If attempted, freeze spatial candidates and the shift-null protocol before
scoring. Use maximum-cardinality one-to-one event pairing within each candidate
cluster pair; preserve ambiguities instead of pooling whole-probe events or
forcing one-to-one neuron labels through a split. No new detection/identity
claim is allowed from time coincidence alone. Do not pool fragments' amplitudes.
Evaluate any proposed merge's refractory burden on the complete output union,
not on the clean anchor or only its matched spikes.

## One experiment and its decision

The audit must nominate one case, one intervention, its expected observable,
and an explicit reason to prefer it over the other available interventions.
Favor a retained-output replay when that directly tests the failure. For a new
sort, use the existing ladder and frozen 12/9 baseline, keeping all unrelated
settings identical. Predeclare evaluation intervals and surrounding context.
If comparable context cannot fit the bounded budget, report that and stop.

Before execution, write an experiment JSON containing both applied-setting maps,
input identities, target and control identities, correspondence criteria, time
intervals, amplitude semantics, contamination/refractory endpoints, maximum
runtime, numerical improvement and regression margins, and the exact decision
rule. These margins require the selected case's baseline evidence and cannot be
honestly specified here. Execution must refuse an incomplete experiment contract.
Do not choose margins or replacement units after viewing candidate results.

Compare both sorts in identical physical-time intervals. Their separate
1,000-spike windows will generally differ; show each window's support, number of
fits and uncovered duration. Do not pair windows by ordinal number or replace
missing fits with zero. Require enough fits within each frozen interval under
both configurations as specified in the experiment contract; otherwise the
completeness comparison is inconclusive. A changed amplitude representation or
ambiguous unit identity also makes the direct fit comparison inconclusive.

An encouraging outcome requires lower estimated missingness in the failing
interval, supported added-event waveform evidence, and no breached contamination,
refractory or healthy-control margin. This is a local candidate-screen result,
not proof of true recall or production superiority. Selected extremes can regress
toward typical behavior; a positive result advances only through the existing
independent-window/context, held-out and session-replication gates. A negative or
ambiguous result closes this checkpoint without automatically launching another
variant. The report must distinguish an operational improvement from a proven
mechanism.

## Implementation order and acceptance tests

Implement sequentially; review each contract against fixtures before adding the
next layer. No parallel agents are required by this prescription.

1. Loader and window-table normalization. Fixtures must catch a wrong amplitude
   source, equal-length but misaligned times, mask/index mistakes, noncontiguous
   cluster IDs, sample/second confusion and nonzero source offsets.
2. Historical/exact replay. A 1,000-event fixture must demonstrate 999 historical
   versus 1,000 exact samples and confirm the final amplitude is included only
   in the exact fit. Test a >10-s gap, exactly 10-s separation, 999/1,000/2,000
   spikes, leftover tails, and empty input. Preserve current production behavior.
3. Selection. Hand-constructed window tables must give exact expected case IDs,
   including threshold equality, ties, duration caps, invalid intervening windows,
   insufficient cases and control exclusion. Candidate outcomes must not be inputs.
4. Evidence/reporting. Test censored/nonfinite/no-fit separation, mismatched
   provenance refusal, selection-hash changes, incompatible output refusal, and
   bounded reads using a recording double that rejects oversized requests.
5. Run one cached real case: reproduce stored historical estimates within
   `rtol=1e-6, atol=1e-6` percentage points under the same runtime, otherwise
   record an input/runtime mismatch and stop interpretation. Inspect the rendered
   panel for time alignment, actual window widths, readable labels and gaps.
6. Run the focused new tests plus `testing/test_truncation_fitter.py`. Broaden
   checks only if shared code is changed. No full sorter run is a unit test.

Completion means the six-case cap is respected, source/selection identities are
auditable, QC semantics are reproduced, and `decision.md` names one concrete next
experiment or explicitly closes with insufficient evidence. A collection of
interesting figures without that decision is not a completed deliverable.

## Related records

- [Current improvement plan](pipeline_improvement_plan.md)
- [Threshold Stage 2 result](luke_c2_train_stability_stage2_result.md)
- [Exclusive matching and detection evidence](decisions/0011-cross-sort-event-matching-and-detection-evidence.md)
- [Limits of corrected cross-sort audits](decisions/0015-corrected-cross-sort-audits-do-not-establish-equivalence.md)
- [C2 v4 truncation diagnostic](luke_20250804_c2_v4_truncation_diagnostic.md)
