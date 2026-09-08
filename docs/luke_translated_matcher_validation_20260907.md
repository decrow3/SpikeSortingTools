# Spatial flexibility restores recovery but exposes identity ambiguity

## Synthetic controls

The exact-geometry translated-template bank searched −80,−40,0,+40,+80µm without DREDGE input. It recovered the injected shift for all20 noiseless template cases. Requiring the existing0.03 margin over same-epoch competitor templates reduced passes to16/20: the late3300µm candidate failed four cases because a shifted3340µm waveform also scored approximately0.985.

A separate shape-specificity stress test placed other same-epoch waveforms at a target's depth where exact same-x geometry permitted it. Six of nine cases passed the target's0.9 cosine/gain threshold. Some stress translations were320–440µm, outside the intended±80µm local search; these show non-unique shape, not direct local false-positive rates. The3300-versus3340µm confusion occurs within the intended search domain and is directly relevant.

The noiseless controls establish geometric recoverability only. They omit real noise, event overlaps, unknown competitors and continuous sub-grid positions. Correct injected shifts do not prove real-data identity.

## Real held-out matching

Applied frozen spatial hypotheses to the same independently detected early970–990 and late9520–9540s peaks. Every accepted event required cosine≥0.9, gain0.4–2.5 and0.03 margin over all other available template/shift hypotheses. Training and detected polarity matched. Same-candidate duplicates within1ms were suppressed. No estimator field was used.

Accepted matches increased to806 and890 for the two early targets,554 for the late2900µm target, and91 diagnostic matches for the late3300µm target. The latter remains disqualified by its synthetic competitor control.

Multiple offsets appear within most five-second bins. Early2920µm observations during977.5s concentrate at−80µm (93/115 matches), whereas the3380µm cohort in that bin is distributed across−80,−40 and0µm (22,24,21 events). Other early bins also contain substantial matches across several offsets. This is not a clean same-cell trajectory, and the observations need not share identity across offsets.

Late2900µm remains predominantly at zero offset with smaller secondary populations. That does not by itself validate zero physical motion: the positive-dominant waveform class has known shape similarities and an incomplete competitor library.

## Decision

Do not use the mean, median or modal matched shift as motion ground truth, and do not tune DREDGE toward it. These candidates currently cannot adjudicate the large excursions. The earlier fixed-template centroids were position-selected; the expanded matcher is now identity-ambiguous. Both limitations must remain visible.

Further corroboration should prioritize more distinctive waveform identities and dense local competitor controls, including large trough-dominant candidates where available. Existing LFP and alternate-input estimates can provide additional independent evidence, but agreement between algorithms alone is not physical validation. The custom continuous lighthouse estimator remains deferred.

## Artifacts and execution

- `testing/luke_shift_matcher_controls.py`; `testing/outputs/luke_shift_matcher_controls_v1/`: all injected recovery cases, shape-confusion cases, settings and reviewed PNG/PDF figure. Completed successfully.
- `testing/luke_translated_holdout_review.py`; `testing/outputs/luke_translated_holdout_review_v1/`: fixed settings, event-level matched shifts/scores/margins, bin counts and reviewed PNG/PDF figure.
- Independent service `luke-translated-holdout-review-v1` was verified live after launcher exit and completed with zero exit status. Launch command, log and receipt are persisted.

No production classifier, correction or spike sort was changed. Full recording-wide motion reliability and downstream benefit remain unverified; goal completion is not established.
