# Motion recovery goal: evidence and outstanding requirements

This ledger preserves the full user objective. Diagnostic progress does not release the full-sort hold or establish goal completion.

| Requirement | Current evidence | Outstanding proof |
|---|---|---|
| Trustworthy recording-wide motion | Central original/compensated DREDGE differences corroborated in direction during 4180–4200 and 4240–4260 s; search-domain discrepancy reproduced and bounded diagnostic implemented | Broad temporal/depth coverage, severe-motion validation, uncertainty handling, cross-algorithm corroboration |
| Neural inputs with demonstrated artifacts excluded | Probe-wide shared-response mechanism documented at acquisition/reference stages; compensation restores central motion sensitivity; frozen model suppresses identified residuals in 12 dispersed voltage samples | Representative preservation and input-quality audit across recording, artifacts beyond shared model, confidence against neural population-rate confounds |
| Independent lighthouse checks | Original nine gentle-epoch tracks; four early/late provisional cohorts persist on subsequent intervals without sort IDs or motion-tuned matching | Stronger competitor/identity specificity, shallow and middle-depth coverage at early/late times, translated footprint qualification through movement |
| Reliability and uncertainty, including difficult motion | Shallow pairwise inconsistency demonstrated; low-count waveform bins remain gaps; broad spatial filtering did not fix shallow failure | Quantified reliability map, validated failure detection, evidence through the worst motion patches |
| Improved sorting and amplitude completeness | Previous full-recording truncation work available; short waveform intervals explicitly not used as amplitude-completeness evidence | Apply a validated correction to a sufficient-duration comparison, independent persistent execution, matched sorting settings and meaningful long-window truncation estimates |

## Current sequence

1. Complete the independent early/late original-versus-compensated DREDGE comparison with strict requested search bounds and waveform preservation.
2. Consolidate interval/depth evidence and unresolved failures; avoid repeatedly tuning on one shallow pair.
3. Expand independent waveform coverage and compare established alternative motion inputs/algorithms using existing server outputs where possible. Inspect existing narrowband/LFP work before launching duplicates.
4. Produce a broader motion/reliability result only after explicit preservation and identity limitations are accounted for.
5. Run downstream sorting/completeness evaluation only when the motion evidence warrants it and the relevant sort authorization/persistence prerequisites are satisfied.

The full-session restart hold in `configs/luke_full_session_rigid.HOLD.json` remains authoritative. The goal asks to justify the next full run, not to silently restart the held full-session job.

## Key evidence reports

- `docs/luke_shared_response_compensation_20260907.md`
- `docs/luke_compensation_validation_20260907.md`
- `docs/luke_dredge_pairwise_audit_20260907.md`
- `docs/luke_dredge_match_profiles_20260907.md`
- `docs/luke_shallow_conditioning_reassessment_20260907.md`
- `docs/luke_profile_scale_diagnostic_20260907.md`
- `docs/luke_shared_model_transfer_survey_20260907.md`
- `docs/luke_neural_transfer_validation_20260907.md`
- `docs/luke_candidate_footprint_audit_20260907.md`
- `docs/luke_transfer_template_holdout_20260907.md`

This is an evidence index, not a claim that all referenced reports prove their broader requirements. Saved arrays, settings, job receipts and process state remain authoritative for each experiment.
