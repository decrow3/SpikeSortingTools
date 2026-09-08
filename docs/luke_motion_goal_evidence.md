# Motion recovery goal: evidence and outstanding requirements

This ledger preserves the full user objective. Diagnostic progress does not release the full-sort hold or establish goal completion.

| Requirement | Current evidence | Outstanding proof |
|---|---|---|
| Trustworthy recording-wide motion | Central original/compensated DREDGE differences corroborated in direction during 4180–4200 and 4240–4260 s; search-domain discrepancy reproduced and bounded diagnostic implemented | Broad temporal/depth coverage, severe-motion validation, uncertainty handling, cross-algorithm corroboration |
| Neural inputs with demonstrated artifacts excluded | Probe-wide shared-response mechanism documented at acquisition/reference stages; compensation restores central motion sensitivity; frozen model suppresses identified residuals in 12 dispersed voltage samples | Representative preservation and input-quality audit across recording, artifacts beyond shared model, confidence against neural population-rate confounds |
| Independent lighthouse checks | Original nine gentle-epoch tracks; four early/late provisional cohorts persist on subsequent intervals without sort IDs or motion-tuned matching | Stronger competitor/identity specificity, shallow and middle-depth coverage at early/late times, translated footprint qualification through movement |
| Reliability and uncertainty, including difficult motion | Shallow pairwise inconsistency demonstrated; low-count waveform bins remain gaps; broad spatial filtering did not fix shallow failure | Quantified reliability map, validated failure detection, evidence through the worst motion patches |
| Improved sorting and amplitude completeness | Previous full-recording truncation work available; short waveform intervals explicitly not used as amplitude-completeness evidence | Apply a validated correction to a sufficient-duration comparison, independent persistent execution, matched sorting settings and meaningful long-window truncation estimates |

## Earlier sequence (superseded by the AP-only implementation update below)

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

## 2026-09-08 resumed implementation: prioritize the existing 100 seconds

User authorized a new version of the paused six-arm diagnostic, retaining DREDGE/AP-only scope and the full-sort hold. The cancelled v1 evidence was preserved. An independent systemd dummy survived launcher disconnection and completed; new v2 comparison, waveform audit, correlation replay, and common-event control each completed with exit 0 under managed services.

- Additional 3 kHz low-pass is **not promoted**. Noise-adjusted filtering changes overall lighthouse disagreement from 1.073 to 1.943 µm and drop/recovery disagreement from 1.270 to 1.826 µm. Gentle broadband screening is mixed (1.044 / 1.331 µm), also not promoted.
- Approximately 98% of the adjusted-filter excess disagreement comes from the shallowest lighthouse region. Exact correlation replay shows new shallow competing alignments and increased cycle inconsistency; the earlier central 55–65 µm pathology is not the main failure in this round.
- A 316,089-exact-event amplitude/localization crossing gives 1.243 µm with original features, 1.481 with filtered amplitudes only, 1.532 with filtered locations only, and 1.934 with both. Event deletion alone does not explain the filter failure. The exact-event intersection is selection-biased and not a population-wide causal decomposition.
- Same-event lighthouse waveform analysis shows heterogeneous attenuation (median peak ratio 0.747, unit510 approximately 0.444 versus unit587 approximately 0.965). Small waveform-energy centroid changes do not establish localization accuracy or uniform waveform preservation.
- Compensated broadband 3σ remains the **development baseline**, exported in `configs/luke_motion_diagnostic_v2.json`; no recording-wide validation is claimed.

The bounded specificity audit is now complete: unit80 fails sensitivity (65.3% injected recovery), while unit154 passes a preliminary quiet/injection gate (92.7%). Its frozen transition matcher supplies 49 and26 events in the first two five-second bins, but only3 and5 after4250s; later bins remain gaps. Fractional controls show that median individual-event centroids compress movement. The centroid of the median waveform gives−5.770µm for the real supported transition rather than−1.225µm, and recovers−9.09µm for an approximate−10µm injection at gain1. This is not calibrated biological truth or a physical-error bound.

A consistent mean-versus-sum amplitude-raster diagnostic also completed. Mean aggregation improves the local unit154 comparison (predicted−7.231 versus−10.312µm), but worsens the full nine-cell comparison to1.532/2.184µm overall/drop-recovery disagreement. It is not promoted. No depth-dependent hybrid was fitted to this development interval. The outstanding work is independently supported shallow motion and transfer beyond the development interval; the220/410µm region and late sparse periods remain unvalidated. No session-wide lighthouse census or full sort was launched.

Reports: `docs/luke_lowpass_evidence_20260908.md`, `docs/luke_lowpass_waveform_preservation_20260908.md`, `docs/luke_lowpass_common_events_20260908.md`, and `docs/luke_ap_conditioning_research_20260908.md`.

Integrated implementation checkpoint: `docs/luke_motion_preconditioning_round_20260908.md`. Nine diagnostic services completed with exit0 and were verified no longer live; terminal audit is saved under `testing/outputs/luke_motion_implementation_round_v3/completed_jobs.json`.
