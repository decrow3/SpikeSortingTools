# Luke motion coordination status

Updated: 2026-09-26T16:28:36Z (2026-09-26 09:28:36 PDT)

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

## Holds, budgets and next action

- AM.3 has **not** started. AO/AR requires this AM.2 result to be reported before AM.3.
- No sorting, new sweep or voltage modification occurred.
- AM MEDiCINe fitting budget consumed: 0 of 4 GPU-hours; extraction and AM.2 do not consume that fitting budget.
- Data filesystem free: 186,128,961,536 bytes (about 173 GiB), above the 30 GB guard.
- Current blocker/hold: coordination handoff of this AM.2 result. After that acknowledgement, AM.3 may use the already-authorized frozen recipe unless a more restrictive instruction arrives.

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

## Instruction and code state

- Latest operative scientific instruction on this host: AO.
- Latest coordination instruction: AR.
- AP/AQ are DARTsort diagnostics requiring no work here per AR; no AP.6 operative instruction has reached this task.
- Branch: `codex/motionqc-phase1`.
- Base commit before the infrastructure fix: `22de67700afe39249887700c0006bfbd91460842`.
- The tested one-line cross-probe infrastructure fix is currently in the working tree and should be committed separately from unrelated user changes.
