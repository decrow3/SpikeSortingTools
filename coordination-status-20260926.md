# Luke motion coordination status

Updated: 2026-09-26T16:42:00Z (2026-09-26 09:42:00 PDT)

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

## AM.3 active background compute

The AM.2 report-before-start hold was satisfied and the frozen AM.3 recipe was preregistered before fitting. `luke-imec0-am3-fit-20260926.service` is active under main PID `1820851`. At 09:45 it had completed 3/174 `amp50_d1` fast windows in 225.5 s wall time, using 213.3 s of MEDiCINe runtime. The GPU is healthy and no pre-existing AM.3 fit was duplicated. Durable follower `luke-imec0-am3-validate-20260926.service` (PID `1823164`) is waiting for fit completion, then runs the frozen validation and packages only on a pass.

The service will fit the 120 s / 60 s-step fast windows, stitch them with per-window median centering, chained overlap offsets and triangular blending, enforce the 10 µm median seam gate, and then fit the 30 s-kernel slow layer after excluding the mapped AE mask plus 3 s. The first-three measured projection is 3.57 hours remaining; expected fit completion is approximately 13:20 PDT, followed by validation and packaging. Expected user-facing completion is approximately 14:00–14:30 PDT if all gates pass.

## Holds, budgets and next action

- AM.3 fitting started at 2026-09-26 09:41:17 PDT after the AM.2 result was reported. Validation and packaging remain pending.
- No sorting, new sweep or voltage modification occurred.
- AM MEDiCINe fitting budget is 4 GPU-hours. Consumption is recorded from completed receipts between windows; no completed AM.3 receipt existed at this 09:42 checkpoint.
- Data filesystem free: 186,128,961,536 bytes (about 173 GiB), above the 30 GB guard.
- Current blocker: none. Stop immediately if a frozen seam, validation, disk or budget gate fails; do not tune.

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

## Instruction and code state

- Latest operative scientific instruction on this host: AO.
- Latest coordination instruction: AR.
- AP/AQ are DARTsort diagnostics requiring no work here per AR; no AP.6 operative instruction has reached this task.
- Branch: `codex/motionqc-phase1`.
- Base commit before the infrastructure fix: `22de67700afe39249887700c0006bfbd91460842`.
- Completed AM.2 checkpoint, including the tested infrastructure fix: `0721ab2474f754d1dac102a2bc96da37979e4bf4`.
- Frozen AM.3 fitting orchestration: `20ec186fcdc676b838b39fca2fdd31f99717c5ea`.
- Frozen AM.3 validation/package follower: `c2a6ad81b2b06f5a1fd8650fa3299421f4fa0ac1`.
