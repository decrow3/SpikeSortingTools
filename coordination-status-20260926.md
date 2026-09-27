# Luke motion coordination status

Updated: 2026-09-26T21:04:00Z (2026-09-26 14:04:00 PDT)

## Verdict first

**AM.2 frozen verdict: co-moving.** The null passed, 282/339 mapped imec1 episodes (83.19%) were accepted on imec0 with the same sign and |shift| at least 80 µm, and accepted cross-probe shifts had Spearman rho 0.508. This passes the preregistered co-moving thresholds (at least 60% and rho at least 0.4).

The accepted imec0 shifts have median -200 µm, P5 -240 µm and P95 -160 µm. Counts by shift were: -280: 5, -240: 56, -200: 156, -160: 55, -120: 6, -80: 4. The result supports shared displacement across probes rather than an imec1-only event.

## AM.1 provenance and clock validation

| Item | Result |
|---|---|
| Extraction service | `luke-imec0-am-extract-20260925.service` |
| Final state | successful, inactive/dead, exit status 0 |
| Validated caches | 87/87 `population.npz` files and 87/87 completion receipts |
| Extraction wall time | 43,769.46 s (12.16 h) |
| Peak source | 87 independently extracted pilot-frontend blocks |
| Peak count | 55,809,805 |
| Depth span | -25.994 to 3820.392 µm |
| Session-time span | 0.0014 to 10473.5511 s |
| Peak-manifest SHA-256 | `a3f6d8b9101b77d1749033365ab368a9242f7eb3a13cd75dbbf2b0ce5074ba14` |
| Catalogue SHA-256 | `46463cda1348c0e0b0c39ad22ae9c71bc46f7c7347797c1c10588944cfc52df1` |
| Clock mapping | `imec0_s = 0.9999998076003875 * imec1_s - 0.000026388859012494873` |
| Sync edges | 10,473 paired rising edges |
| Clock residual | RMS 22.95 µs; P95 absolute 43.61 µs; maximum absolute 65.44 µs |

## AM.2 null and cross-probe results

| Metric | Result |
|---|---:|
| Resolved matched pseudo-episodes | 60/60 |
| Null mode | 0 µm |
| Null median absolute shift | 0 µm |
| Null gate | PASS |
| Mapped imec1 accepted episodes | 339 |
| Accepted on imec0 | 282 (83.19%) |
| Same-sign accepted with \|shift\| >= 80 µm | 282 (83.19%) |
| Spearman rho among accepted episodes | 0.508 |
| imec0 accepted-shift median | -200 µm |
| imec0 accepted-shift P5–P95 | -240 to -160 µm |
| Frozen classification | **co-moving** |

Per-depth results retain the frozen four-block test. Accepted counts and accepted-shift medians were: block 0, 12/339 and -80 µm; block 1, 145/339 and -200 µm; block 2, 189/339 and -200 µm; block 3, 327/339 and -200 µm. The lowest block is sparsely resolved and heterogeneous; the other three resolved blocks agree on a roughly -200 µm displacement. Full accepted/unresolved evidence remains in the CSV and is not collapsed into the whole-probe verdict.

## Infrastructure retry record

- Initial AM.2 service: `luke-imec0-am2-20260926.service`; failed after the passing null because the per-block call received `depth_range_um` both globally and locally.
- Root-cause fix: the per-block call now removes only the global whole-probe depth range so each frozen block uses its own block +/-150 µm range. No data, threshold, field, metric or scientific rule changed.
- Focused tests after the fix: 9 passed.
- Retry 1 service: `luke-imec0-am2-20260926-r1.service`; completed successfully with exit status 0.
- Cumulative AM.2 infrastructure relaunches: 1 of 3 allowed.
- The first AM.3 fit service completed and preserved all 174 fast-window
  receipts, then failed before the slow fit because the rescue-environment
  controller imported `medicine` directly. MEDiCINe exists only in the frozen
  `MEDPY` environment already used by every fast fit. No scientific gate ran or
  failed. The correction delegates the unchanged slow fit to `MEDPY`, requires
  its receipt, and reuses all 174 completed fast fits. This is AM.3
  infrastructure relaunch 1 of 3 under AO; no field, mask, parameter, metric or
  budget changed.
- The required same-filesystem retry dummy passed. The MEDPY process imported
  MEDiCINe/CUDA, read `full_block_000` (692,200 cached peaks), imported the AM.3
  slow dispatch, and loaded the stitched field. The rescue environment then
  loaded the same field and verified the saved seam gate. AM has no sort stage,
  and sorting remains unauthorized. Receipt:
  `stage5_imec0/am3_infrastructure_dummy_retry1.json`.
- AM.3 fit retry 1 then completed successfully: 174 fast fields plus the one
  frozen slow field, 12,625.68 s (3.51 h) total MEDiCINe runtime against the
  4 h budget. The retry validation follower failed before validation because
  its wait loop still checked the original fit service name, which is expected
  to remain failed as evidence. This is a second, distinct infrastructure
  relaunch for the validation job; the wait helper now takes the service name
  explicitly. Since `am3_fit_complete.json` is sealed, the next durable job
  starts directly at `am3-validate` and does not refit anything.

## AM.3 complete: imec0 two-layer field passes

The AM.2 report-before-start hold was satisfied and the frozen AM.3 recipe was preregistered before fitting. The final fit retry and validation retry both terminated normally with exit status 0. The run reused all 174 completed `amp50_d1` fast windows, stitched them with per-window median centering, chained overlap offsets and triangular blending, then fitted the unchanged 30 s-kernel slow layer after excluding the mapped AE mask plus 3 s. Total MEDiCINe runtime was 12,625.68 s (3.51 GPU-hours), below the 4 h budget.

The frozen validation **passed**:

| Metric | Fast | Two-layer | Gate/result |
|---|---:|---:|---|
| episode error | 14.84 µm | 17.66 µm | degradation 2.82 µm, <=3 µm |
| episode ratio | 0.956 | 0.958 | retained |
| quiet 5 s increment RMS | 5.37 µm | 2.50 µm | quiet is better |
| quiet absolute displacement | 4.43 µm | 2.91 µm | descriptive |
| false-motion fraction | 0.058% | 0.000% | descriptive |

The fast stitch median absolute seam was 1.04 µm against the 10 µm limit. The two-layer boundary distribution had median 2.24 µm, P95 7.70 µm and maximum 13.59 µm; its boundary gate and all top-10 matched nulls passed. The label-free slow-reference null resolved 200/200 samples, mode 0 µm and median absolute shift 0 µm. Both fields were scored on the same 282 accepted episodes, 32,841 quiet samples and 26,193 quiet increment pairs.

The deployable package is `stage5_imec0/luke0804_imec0_two_layer_motion.npz`, SHA-256 `4c769125020aad3263bf6eaa1889cdaa7eccbf19dcf0aa3a068b185221fadb9f`. Its manifest uses AP-frame-zero seconds and sign convention `corrected = observed - displacement`. No sorting or voltage modification occurred.

## Holds, budgets and next action

- AM.3 fitting started at 2026-09-26 09:41:17 PDT after the AM.2 result was reported. Validation and packaging completed at approximately 13:30 PDT.
- No sorting, new sweep or voltage modification occurred.
- AM MEDiCINe fitting budget is 4 GPU-hours. Consumption is recorded from completed receipts between windows; no completed AM.3 receipt existed at this 09:42 checkpoint.
- Data filesystem free: 186,128,961,536 bytes (about 173 GiB), above the 30 GB guard.
- Current AM blocker: none; AM.1--AM.4 are complete.

## AW CPU preparation (parallel, no sort/GPU/new fit)

AW was received and CPU-only preparation has started while AM.3 remains active. The preparation is constrained to at most two CPU threads, one reader, CUDA disabled, low priority, an approximately 8 GB working-set target, no more than 5 GB of new persistent assets, and the existing 30 GB disk guard. It does not touch the active AM.3 source snapshots or services.

The existing C2-v4 result, corrected per-cluster exclusive scorer, operator calibration and lattice-commensurate staircase implementation have been identified and hash-audited for reuse. DARTsort source inspection confirms that matching obtains the template state at the centre sample of each matching chunk (`peel/matching.py`, lines 260--269) and converts the external displacement at that time into integer pitch shifts before selecting static-channel template support (`templates/template_util.py`, lines 62--78). AW will therefore preserve actual chunk-centre sampling rather than treating the deployed field as a continuous per-spike trajectory.

Two AV handoff items are hard dependencies and will not be guessed from nearby local products:

- the accepted static DARTsort S/W2 900--1240 s donor bank, unit metadata, preprocessing identity, and frozen hashes;
- the exact two-layer v1 field used by D2L, with its authoritative hash and time/sign metadata (AI-v2 and the active imec0 AM.3 field are explicitly ineligible substitutes).

The donor-selection rule and exact same-column 40 µm qualification are being frozen before any outcome inspection. CPU fixtures will exercise the corrected exclusive +/-0.4 ms matching boundary, immutable 5 Hz trains, exact channel remapping, per donor/state energy retention >99%, PTP ratio 1.00 +/-0.02, cosine >=0.99, and fail-closed hash/provenance checks. The preparation will be marked worker-ready only after the AV assets arrive and all donors/states qualify; no placeholder HDF5 or synthetic donor bank will be called worker-ready.

### AW dependency update, received 2026-09-26 13:36:29 PDT

The hub independently confirmed the AM.3 receipt/gates/package hash above, so
AM is closed without new fitting. AV found no saved full-probe static W2 bank.
The exact saved S bank is a shallow AP202--AP383 crop with shape 654 x 121 x
182 and SHA-256
`a99b12c3075f038fbad8c05c36c96f63221fd0eac5ba71caee8f18ff97acb75c`;
the final sorting SHA-256 is
`be6106ed0cb0f99759fd23629dd3bcb5f50657dbffa238ad3721c15d1b21fba7`.
The AV manifest exists on huklaban5 with SHA-256
`b3c591f0bde45942c1f5b6e0cf063ffc5685dabf09cf98da31248017d16552cc`,
but neither it nor the h5-local bank path is mounted here. AV/h5 is staging a
compact shared-path bundle and adding the missing authoritative D2L-v1 field
provenance. No nearby local field will be substituted.

The crop supports a necessary interior observed-support screen: >99% of the
energy present in the 182 saved channels must lie farther than maximum state
excursion plus matching interpolation radius from both edges, and every exact
state must pass the remap/PTP/cosine checks. It cannot establish full-probe
energy because AP0--AP201 are unobserved, not observed zeros. Consequently the
strict full-probe AW donor design remains blocked on new full-probe donor
extraction or an equivalent full-probe waveform source. Narrowing the claim to
the shallow deployed crop would make a restricted cohort testable, but requires
explicit scientific authorization and is not assumed here.

DARTsort source inspection also corrects the operator description. The default
`drifty` matcher continuously spatially kernel-interpolates the registered
template basis at the unrounded external displacement evaluated at the matching
chunk centre. Integer pitch selection is a separate path and does not prove
fractional interpolation was absent. The staged matching config must determine
which path S/D2L actually used before qualification.

AW CPU fixture deadline was met at 13:49 PDT: nine focused AW tests and the
existing injected-truth/scorer suite pass; the preparation remains small and
CUDA-free. Next dependency check is 14:00 PDT. If the shared bundle is still
absent, AW remains blocked without a worker launch or deadline guess for AV.

### AW handoff validation, 2026-09-26 14:04 PDT

The AV handoff is now staged at
`/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1/`.
All 21 files pass their manifest sizes and SHA-256 values. The manifest hash is
`3cc7b0a3840e3d48cdcb0fe5b5707dd2eeea06942cab792294b1ab3e459ac843`;
the completion-marker hash is
`71ee43721c7e952d677a32a3f4b7b07b9c54579c11389c382ebe8d766656ff46`.
The D2L-v1 field is authoritatively confirmed at
`85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9`.
The local DARTsort adapter copied it without filtering or resampling and is
identical at copied samples.

Actual configs show that both S and D2L used the `drifty` continuous spatial
interpolation matcher. S used 30,000-sample chunks; D2L used 7,500-sample
(0.25 s) chunks. The authoritative W2 trajectory has 1,360 matching centres and
occupied lattice states 0 through -240 um. Median absolute lattice quantization
error is 7.53 um (P95 17.81, maximum 19.99). The local-adapter and whole-session
source evaluations agree to `4.24e-11` um at those centres.

The handoff does not make the full-probe worker ready. Only 7/654 shallow-crop
templates pass both the >99% observed-interior-energy screen and all occupied
exact-state round trips, versus the frozen target of 30. AP0--AP201 remain
unobserved, and the bundle lacks the frozen `quality` and `isolation_score`
columns. The exact small `qc-phy` table should be staged; no quality definition
will be invented.

The smallest full-probe remedy is a read-only, no-sort W2 template extraction
after donor selection: one approximately 7.83 GB raw read, 15.67 GB temporary
RAM cache, under 0.2 GB persistent output, and an estimated 20--35 minutes CPU
wall time. It has not been launched. The synchronized 5 Hz train is now labelled
fixture-only; the intended population generator uses independent per-unit
seeded renewal trains with a minimum 3 ms refractory interval. Forty combined
CPU regressions pass.

## Outputs and hashes

Sweep directory: `/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage5_imec0/`

| Output | SHA-256 |
|---|---|
| `AM2_REPORT.md` | `c000c321063e9dae37584640d4017bd95fe18c151a6dc2f7521672c5fa8e3816` |
| `am2_null.csv` | `341224151fa91273ab5b754ffc52858ba41f481c03b59344db3b61083fae3370` |
| `am2_null_gate.json` | `d2d70630ca0eea6002db7886f9f5a3a84d26a8d916388b6378ab07241bdbdded` |
| `am2_cross_probe_episodes.csv` | `3a641ca217a16e223a80a9cde628e4e75c5353352634266ed7606c7b0cdf9c88` |
| `am2_per_depth_blocks.csv` | `e4d7ddc2d16d782cf48712f77a4388cb06f7176313c835271fc738e54b5f7325` |
| `am2_summary.json` | `bb6ab38b9fa3c847e13831fe37c534eaf9ef2ae0719cdae750389c9447000702` |
| `peak_cache_manifest.csv` | `a3f6d8b9101b77d1749033365ab368a9242f7eb3a13cd75dbbf2b0ce5074ba14` |
| `luke0804_imec0_two_layer_motion.npz` | `4c769125020aad3263bf6eaa1889cdaa7eccbf19dcf0aa3a068b185221fadb9f` |
| `am3_validation_gate.json` | `6cd41834b3077115e82e0051996e1e065b63e677900bd80785e8c83718596bb2` |
| `am3_scores.csv` | `f4c3eeba59b1af31a3d7f7e411388a28c9e200830ef81d3e3241afb3937940b5` |
| `am3_fit_complete.json` | `93a5971e9ef6c249201a13152e8d8afac9be8819c21a53df82836cd404bf64be` |

The colourblind-safe summary figure is `am2_cross_probe_summary.png`. The failed and successful durable logs are `am2.service.log` and `am2.retry1.service.log`.

AM.3 preregistration is `am3_preregistration.json`. The initial failure is preserved in `am3_fit.service.log`; successful fit retry evidence is in `am3_fit.retry1.service.log`; successful final validation/package evidence is in `am3_validate.retry2.service.log`. Per-window receipts remain below `fields/full/` and the slow receipt is below `fields/slow/medicine_amp50_d1_k30_exclude3s/`.

## AT source confirmation and interpretation qualifications

These are documentation qualifications only. They do not alter any frozen field, score, verdict, mask, threshold or active AM.3 source snapshot.

- The hub independently reproduced the AE mask with zero differing bins, the saved AI-v2 displacement with zero maximum error under the historical endpoint-clamping rule, and the AI adoption scores to below 1.8e-15 error. AI-v2's episode-error margin to its frozen limit is only 0.06493 µm. Outside the mask, v2 minus the selected slow layer is the constant +3.50481 µm caused by final recentering; the two fields are equal there modulo that constant offset.
- Source provenance confirms that `fast_shared_jitter_three_windows.csv` was produced by `testing/luke_imec1_slow_layer_ab_v1.py`'s `fast_jitter` path. The table was written at 2026-09-25 14:21:19 PDT; the v1 manifest written five minutes later names that script and records SHA-256 `b3ac06862e87968a7a4d27fef12427cb229708fab9cf20c678d6d9cf40d2be51`. The script was subsequently modified for AG at 14:45 and its current SHA-256 is `41db6d40866ff3398275321f7456c551f4562f184914b4bb629985931efea0e4`; the historical byte-for-byte snapshot is not retained here beyond the manifest hash.
- The historical three-window 2–10 Hz result is qualified. Its observed spectrum interpolates gaps on the original 50 ms grid, whereas its circular-shift null deletes invalid/episode bins and still uses 50 ms sampling. Retained-bin fractions were 56.7%, 93.1% and 66.0%. Therefore it is not valid to claim a full-session absence of 2–10 Hz shared jitter from that table. The original contiguous-segment T13 result is not affected by this specific defect.
- `motionqc.reference.unit_common_mode` likewise deletes gaps before CSD while retaining the original sample rate and is not a faithful reproduction of T13. AI's `motionqc` report did not supply the optional unit input, so this routine did not contribute to AI's saved field scores or adoption decision. Current `motionqc/reference.py` SHA-256 is `56032d610552e32be0b7909c94c3804b6bc71b683f13fc4d7c60fba0323cf3aa`.
- AM.2's saved baseline is the full `[start-4,start-1)` interval and does not exclude other accepted episodes. Thirteen of 339 mapped baselines overlap another accepted episode; only one of those 13 is an accepted AM.2 row. As a robustness check of the existing rows, omitting all 13 leaves 281 strong shifts among 326 eligible episodes and accepted Spearman rho 0.50806. This is not a rescore and does not replace the frozen co-moving verdict.
- `shift_test` has an identifiability limit: a controlled stationary population rate-swap can pass at -200 µm with gain 0.06687. Ordinary quiet nulls therefore do not rule out every structured activity confound. This is not evidence that Luke's measured episodes are artifacts; positive, negative and zero translations, random thinning, and independent correlation arithmetic all passed review.
- Generic `compose_two_layer` can propagate unsupported endpoint NaNs through its ordinary median. Historical AI explicitly clamped endpoints, and active AM.3 explicitly pre-aligns the slow field with `_edge_sample`, so the observed case is avoided without changing the generic library.
- Frozen fields named `MAE` are median absolute errors, not arithmetic mean absolute errors. Quiet-increment gates require quiet endpoints; a five-second pair can span an intervening episode.

## Instruction and code state

- Latest operative scientific instruction on this host: AW CPU preparation alongside the now-complete AM.3 run; AO remains the AM retry/reporting authority.
- Latest coordination/source-review instruction: AT, following AR.
- AP/AQ are DARTsort diagnostics requiring no work here per AR; no AP.6 operative instruction has reached this task.
- Branch: `codex/motionqc-phase1`.
- Base commit before the infrastructure fix: `22de67700afe39249887700c0006bfbd91460842`.
- Completed AM.2 checkpoint, including the tested infrastructure fix: `0721ab2474f754d1dac102a2bc96da37979e4bf4`.
- Frozen AM.3 fitting orchestration: `20ec186fcdc676b838b39fca2fdd31f99717c5ea`.
- Frozen AM.3 validation/package follower: `c2a6ad81b2b06f5a1fd8650fa3299421f4fa0ac1`.

### AW supplemental-QC correction checkpoint, 2026-09-26 15:17 PDT

The supplemental AV QC bundle is byte-verified. Its manifest is
`483d7233d31d5b5d95efb6d464d82f4b986c5524a0bb9741a37a625d3689ea13`,
its completion marker is
`70b8febef3a3d44c2f8921a2f632154d29264e458f3872f5b7a01017194457c5`,
and the 538-row QC table is
`e0ff5f1a61b2b07a8632dfcaf59dca282fdc213b3983557ee5d67fb1f0aaf7dc`.
The table contains an exact 1 ms adjacent-ISI fraction but contains neither
`quality` nor `isolation_score`. Those two requirements were introduced during
AW preparation and were not part of the user/coordinator criteria. Their
absence is not evidence of poor isolation.

A self-contained prospective correction is saved in
`testing/outputs/luke_au_cpu_preparation/DONOR_RULE_CORRECTION_PROPOSAL.md` and
`donor_rule_v2_proposal.json`. It retires the unavailable fields without
substituting refractory-contamination, presence-ratio or other non-equivalent
metrics. The original frozen blocked result remains unchanged in
`preparation_manifest.json` and `handoff_validation.json`; the proposal has not
been applied and authorizes neither extraction nor a benchmark.

The read-only scalar pre-screen finds 90/538 final labels satisfying at least
800 spikes, raw adjacent-ISI fraction at most 0.005 and rest-spike fraction at
least 0.80. This is not an eligible or selected cohort. Exact one-to-one lineage
from the 654 matching-template units to the 538 final labels remains unresolved,
and full-probe waveform construction, stability, support and placement gates
remain unmeasured. Numeric ID overlap is explicitly not treated as lineage.

If the prospective rule is approved, the remaining no-sort work begins with a
small lineage artifact if available, then one read-only W2 full-probe template
extraction: about 7.83 GB read, 15.67 GB temporary RAM, under 0.2 GB persistent
output and provisionally 20--35 CPU minutes. It must first reproduce the saved
crop overlap before expanding to candidates, and it stops rather than relaxing
gates or reducing the target if fewer than 30 donors qualify.

### AW proposal revision after hub review, 2026-09-26 15:47 PDT

The prior proposal remains preserved at commit `54b6f00`, but its design advice
has been corrected prospectively in `DONOR_RULE_CORRECTION_PROPOSAL_V2.md` and
`donor_rule_v3_proposal.json`. Nothing has been applied: no extraction, donor
selection, GPU work or sort ran.

The revised design constructs a donor directly from a chosen final S label's
saved spike times. A bijection to the 654-unit pre-matching bank is no longer an
automatic eligibility blocker; lineage is a diagnostic/caveat unless the claim
is reproduction of a historical matching template. Biological split-half
stability is diagnostic because the 0.99 cosine/2% PTP tolerances were validated
for deterministic exact-copy/remap fidelity, not two noisy 200-spike estimates.

Actual support at every occupied shift remains mandatory: more than 99% measured
energy support, no padding/extrapolation/many-to-one maps, and deterministic
round-trip operator fidelity. The 440 um interior margin is now explicitly a
conservative diagnostic, not a hard gate absent a separate padding/extrapolation
justification. The design aims for approximately 30 donors; if the count differs,
it reports outcome-blind precision and pauses for an explicit cohort-size
decision rather than enforcing exact 30 or silently reducing the target.

The prescreen used the 9,151-byte stage4 AB mask, SHA-256 `31b39a494ede16e3f5919caf1cf53a6222dd5e737f027d59155e96a9ce9179ed`.
Read-only comparison confirms its 422 intervals and normalized A/U/E sources are
identical to the 8,750-byte hub canonical mask, SHA-256 `86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55`;
the bytes differ only in source serialization such as `A+E` versus `AE`.

The recorded frequency is 29,999.759166666667 Hz. Full 384-channel voltage is
used only to measure final-label templates and their support. Injection and any
future benchmark remain on AP202--AP383, the historical 182-channel deployed
sorter domain. The smallest proposed CPU pass is still one read-only W2 pass
(about 7.83 GB read, 15.67 GB RAM, under 0.2 GB persistent, provisionally
20--35 minutes) and can compute support plus lineage/stability diagnostics
together. It remains unapproved and unlaunched.

### AW support-denominator clarification, 2026-09-26 16:20 PDT

The prospective design now defines mandatory retention exactly as translated
energy on actual AP202--AP383 target channels divided by the donor's total
measured 384-channel energy before translation. The denominator is never
renormalized after crop, and retention must be at least 99% for every placement
and occupied state. Actual remap and deterministic operator-fidelity gates are
unchanged.

The earlier 20--35 minute estimate applied to approximately 30 templates, not
all 90 scalar-prescreen candidates. The unapproved prospective extraction is
therefore capped at 90 candidates, two CPU threads, 20 GB RAM, 60 minutes wall
time and 0.2 GB persistent output. Reaching a cap stops with a report rather
than silently dropping candidates. The corrected design is ready for a user
decision and awaits explicit extraction authorization; no job was launched.

### AZ approved batch received, 2026-09-26 17:25 PDT

The complete approved AZ instruction is saved as `AZ_APPROVED_BATCH.md`.
Receipt is acknowledged before its first checkpoint deadline of 2026-09-27
12:23 UTC (05:23 PDT). AZ removes the separate extraction and 20--30-donor
cohort approval pauses within its stated bounds. This host owns CPU donor
measurement/qualification, generator/scorer qualification and CPU analysis;
huklaban5 owns the hybrid worker and GPU arms. The accepted D2L-v1 hash remains
`85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9`.

The first authorized extraction attempt ran from 17:20:11 to 17:23:08 PDT and
failed closed before caching/templates because exact `ibllikecmr` preprocessing
changed the expected 384-channel contract. The failed service, log,
`RUNNING.json`, prescreen and `FAILED.json` are preserved. This is currently an
infrastructure/preflight mismatch, not a donor scientific verdict. It used
about 2.95 process-wall minutes of the cumulative 120-minute extraction budget.
The next action is a bounded read-only audit of the removed channel IDs, then an
equivalent repair or a concrete scientific blocker; no GPU, sort or voltage
write is authorized on this host.

### AZ extraction attempt 2 resource stop, 2026-09-26 17:37 PDT

The retained-channel repair independently reproduced the ordinary 383-good-
channel operator while keeping the measured AP191 trace, but materializing the
complete 340 s preprocessed window in `/dev/shm` raised the service cgroup above
AZ's 20 GB RAM ceiling (about 35 GB observed). The service was stopped before
template output, exited failed at 17:37:12 PDT, and its failed-run evidence was
preserved. Its private RAM scratch was removed; scientific disk free space was
174 GB. This is the first of AZ's two allowed additional infrastructure
relaunches.

The final allowed relaunch uses the identical lazy SpikeInterface preprocessing
graph directly, so DARTsort reads only requested waveform neighborhoods. It
serializes voltage reads, allows at most two numerical threads, writes no
preprocessed voltage, and will run under a hard 20 GB service memory limit. This
is a memory-bounded execution repair, not a scientific-operator change.
