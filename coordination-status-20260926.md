# Luke motion coordination status

Updated: 2026-09-26T20:05:00Z (2026-09-26 13:05:00 PDT)

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

## AM.3 active background compute

The AM.2 report-before-start hold was satisfied and the frozen AM.3 recipe was preregistered before fitting. `luke-imec0-am3-fit-20260926.service` is active under main PID `1820851`. At 10:26 it had completed 35/174 `amp50_d1` fast windows in 2,652.7 s wall time, using 2,528.1 s of MEDiCINe runtime. The GPU is healthy and no pre-existing AM.3 fit was duplicated. Durable follower `luke-imec0-am3-validate-20260926.service` (PID `1823164`) is waiting for fit completion, then runs the frozen validation and packages only on a pass.

The service will fit the 120 s / 60 s-step fast windows, stitch them with per-window median centering, chained overlap offsets and triangular blending, enforce the 10 µm median seam gate, and then fit the 30 s-kernel slow layer after excluding the mapped AE mask plus 3 s. The 35-fit checkpoint projects 2.93 hours remaining; expected fit completion is approximately 13:22 PDT, followed by validation and packaging. Expected user-facing completion is approximately 14:00–14:30 PDT if all gates pass.

## Holds, budgets and next action

- AM.3 fitting started at 2026-09-26 09:41:17 PDT after the AM.2 result was reported. Validation and packaging remain pending.
- No sorting, new sweep or voltage modification occurred.
- AM MEDiCINe fitting budget is 4 GPU-hours. Consumption is recorded from completed receipts between windows; no completed AM.3 receipt existed at this 09:42 checkpoint.
- Data filesystem free: 186,128,961,536 bytes (about 173 GiB), above the 30 GB guard.
- Current blocker: none. Stop immediately if a frozen seam, validation, disk or budget gate fails; do not tune.

## AW CPU preparation (parallel, no sort/GPU/new fit)

AW was received and CPU-only preparation has started while AM.3 remains active. The preparation is constrained to at most two CPU threads, one reader, CUDA disabled, low priority, an approximately 8 GB working-set target, no more than 5 GB of new persistent assets, and the existing 30 GB disk guard. It does not touch the active AM.3 source snapshots or services.

The existing C2-v4 result, corrected per-cluster exclusive scorer, operator calibration and lattice-commensurate staircase implementation have been identified and hash-audited for reuse. DARTsort source inspection confirms that matching obtains the template state at the centre sample of each matching chunk (`peel/matching.py`, lines 260--269) and converts the external displacement at that time into integer pitch shifts before selecting static-channel template support (`templates/template_util.py`, lines 62--78). AW will therefore preserve actual chunk-centre sampling rather than treating the deployed field as a continuous per-spike trajectory.

Two AV handoff items are hard dependencies and will not be guessed from nearby local products:

- the accepted static DARTsort S/W2 900--1240 s donor bank, unit metadata, preprocessing identity, and frozen hashes;
- the exact two-layer v1 field used by D2L, with its authoritative hash and time/sign metadata (AI-v2 and the active imec0 AM.3 field are explicitly ineligible substitutes).

The donor-selection rule and exact same-column 40 µm qualification are being frozen before any outcome inspection. CPU fixtures will exercise the corrected exclusive +/-0.4 ms matching boundary, immutable 5 Hz trains, exact channel remapping, per donor/state energy retention >99%, PTP ratio 1.00 +/-0.02, cosine >=0.99, and fail-closed hash/provenance checks. The preparation will be marked worker-ready only after the AV assets arrive and all donors/states qualify; no placeholder HDF5 or synthetic donor bank will be called worker-ready.

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

The colourblind-safe summary figure is `am2_cross_probe_summary.png`. The failed and successful durable logs are `am2.service.log` and `am2.retry1.service.log`.

AM.3 preregistration is `am3_preregistration.json`; the active durable log is `am3_fit.service.log`; per-window receipts are written below `fields/full/`, and progress is checkpointed in `am3_fit_progress.json`.

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

- Latest operative scientific instruction on this host: AO.
- Latest coordination/source-review instruction: AT, following AR.
- AP/AQ are DARTsort diagnostics requiring no work here per AR; no AP.6 operative instruction has reached this task.
- Branch: `codex/motionqc-phase1`.
- Base commit before the infrastructure fix: `22de67700afe39249887700c0006bfbd91460842`.
- Completed AM.2 checkpoint, including the tested infrastructure fix: `0721ab2474f754d1dac102a2bc96da37979e4bf4`.
- Frozen AM.3 fitting orchestration: `20ec186fcdc676b838b39fca2fdd31f99717c5ea`.
- Frozen AM.3 validation/package follower: `c2a6ad81b2b06f5a1fd8650fa3299421f4fa0ac1`.
