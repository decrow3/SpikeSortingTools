# Luke: bounded sorting test with screened MEDiCINe motion

Status: superseded by the user's whole-duration, two-machine request on
2026-09-09 UTC. Use [the full-session rigid/nonrigid plan](luke_full_session_improved_motion_two_machine_plan.md).
The five-minute comparison below is historical planning, not the requested
experimental endpoint. Small checks may support preparation but do not replace
the two full-session sorts. No jobs were launched by this proposal.

## Question and recommended first comparison

Does applying the screened MEDiCINe field improve spike recovery and identity
continuity on Luke 20250804 imec0 relative to the identical uncorrected input?
Start with the cached 930–1230 s interval, not a full-session sort. This is a
development experiment: 930–1030 s informed estimator selection and the later
200 s provide temporal extension evidence, not an untouched final holdout.

Use two paired sorts:

| Arm | Recording | Sorter motion |
| --- | --- | --- |
| Control | Accepted RESCUE voltage, restricted to common supported channels/time | Off |
| Candidate | Same voltage, externally registered once using frozen MEDiCINe | Off |

Nominate 5σ/relaxed screening, 10,000 training steps as the first candidate.
Use the existing 30,000-step and 6σ/full fields for pre-sort sensitivity review;
do not launch three candidate sorts or choose settings from sorting outcomes.
Keep gain at 1 and the existing seed-reference convention. Do not fit a later
offset, sign, gain, or lag to improve agreement.

## Evidence available and limits

The [integration proposal](motion_validation_pipeline_integration.md) reports
better large-excursion waveform agreement for screened MEDiCINe than the best
screened DREDGE configuration in the calibration comparison. This supports a
candidate test, not established downstream improvement.

The extension now has a saved `summary.json` and per-candidate reports under
`testing/outputs/luke_lighthouse_extension_300s_v1`. Its summary records 17
candidates and common field support of approximately 930.004533–1229.754533 s.
These artifacts must be reviewed; artifact presence alone is not a scientific
pass or a verified job exit. Review shared excursions across multiple cells,
strict support, competing identities, gaps, and the known unit161 discrepancy.
Keep lower-score, ambiguous, and unmatched evidence separate. Preserve
whole-probe waveform matching independent of the field.

Historical native KS4 registration failed its operator audit; the older Option A
development contract exercised a different field/domain and calibrated shifts
only through 60 µm. Neither validates application of the new, much larger
nonrigid field. See [native audit](luke_20250804_ks4_native_operator_audit_result.md)
and [Option A readiness](luke_option_a_readiness_assessment.md).

## Cheapest first check: cached events, then a small voltage audit

1. Inspect the existing five-minute waveform/peak overlays and per-cell tables.
   Establish replicated movement and identify unsupported times/depths before
   adding any new estimator or smoothing. Report the two time intervals
   separately and compare training budgets on identical event support.
2. Adapt the existing external-registration path for a versioned experimental
   contract. Starting operator: the earlier Option A SpikeInterface kriging
   configuration, with its exact resolved parameters and implementation pinned.
   Audit it before treating it as usable; do not silently reuse old qualification.
3. On short representative voltage patches, check zero-shift interpolation tax,
   both displacement signs, exact 40 µm translations, fractional shifts, and the
   actual proposed field's displacement range and spatial gradients. Include
   moving waveform support and probe edges. Compare waveform amplitude, cosine,
   residual, clipping, and separation of competing templates. Use existing
   donor/control machinery where applicable and preserve generator dependence.

This cheap check can establish whether the proposed correction is correctly
mapped and plausibly preserves waveforms. It cannot establish spike completeness
or stable sorting identities; that requires the paired sorts.

## Freeze the application contract before sorting

- Verify the accepted recording receipt, sampling frequency, physical geometry,
  channel ordering, field hashes, and the mapping between acquisition,
  recording-relative, and snippet-relative time. Do not copy the old Option A
  acquisition offset into this field without deriving and checking it.
- Restrict both arms to a sample-aligned interval inside actual field support.
  Do not extrapolate the cached field to the nominal 930/1230 s boundaries.
  Read source padding for filtering without scoring padded samples.
- Derive a common interior channel set from supported source coordinates over
  the whole interval and the operator's required spatial support. Apply the
  warp on full geometry before cropping; crop the control identically. Refuse
  unsupported source requests or invalid/noninvertible spatial mappings.
- Keep estimator-only shared-response compensation out of the sorting voltage
  change: both arms begin with the same accepted RESCUE recording. Introducing
  compensation into sorting would require a separate experiment.
- Freeze preprocessing, KS4 version/settings/seed, whitening policy, curation,
  output dtype, and QC. Record any unavoidable arm-specific fitted quantities.
- Version operator/crossover policy and numerical preservation thresholds before
  its new audit. No automatic selective-correction policy or post hoc masking
  of unfavorable cells. If the operator fails, investigate or stop this branch.

The existing `testing/luke_external_warp_pipeline.py` is reuse material, not a
drop-in command: its current contract is digest-bound to another field and
domain. Write a new schema/config and adapter checks rather than editing the
historical frozen contract. Preserve production decision 0002 and its routing
invariant; this proposal concerns an explicit development branch.

## Sorting endpoints and continuation rule

Freeze event matching tolerances, eligible reference support, and numeric
noninferiority margins in the run prespec before viewing candidate sort output.
Use the same reference events for both arms, measured on original voltage.

Primary descriptive endpoint: per-reference recovery of strict waveform events
in the corresponding sorted identity or explicitly reported fragment family,
including recovery during large excursions versus seed/quiet support. Report
both total event assignment and recovery in a single continuous identity so
fragment aggregation cannot conceal identity failure. These are provisional
references, not certified completeness ground truth.

Guardrails: contamination/refractory burden, duplicate assignment, fragmentation,
amplitude loss, and disappearance of previously recovered identities. Show
per-cell results and unmatched/rival assignments, not just pooled medians. Unit
count, total spikes, and KSLabel counts cannot establish success.

Advance to one separately selected interval only if improved recovery/continuity
replicates across multiple eligible identities and frozen guardrails pass.
Sparse support means inconclusive; operator failures or degradation mean stop
and investigate. A severe individual miss remains visible even if the median
improves. Full-session work and production promotion require later evidence.

## Execution and deliverables

Run both sorts and downstream curation/QC through an independent systemd user
service, using the existing launcher only after checking its current lifecycle
contract. Verify any new launch method with a cheap dummy that survives launcher
disconnection. Persist commands, resolved configs, input/code hashes, job IDs,
stdout/stderr, and final exit receipts; inspect live service/process state.

Check actual sorter checkpoint support before launch. Until demonstrated,
budget an interrupted sort as restarting that entire arm; completed input or
field caches are stage reuse, not within-sort resume. Preserve failed evidence
and honor existing holds. Use fresh local output directories outside `/mnt`.
Measure the small audit's resources before estimating sort runtime; the cached
13.6-minute motion benchmark does not estimate sorting runtime.

Deliver a frozen experimental manifest, operator audit, paired sort/QC receipts,
per-reference comparison table, and depth/time plus continuity figures. The
next implementation task is the bounded field/operator adapter and its audit,
followed by the two sorts when the concrete contract is ready.
