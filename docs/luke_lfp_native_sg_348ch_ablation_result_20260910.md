# Native-250-Hz LFP filter ablation result

Date: 2026-09-10

## Answer

Savitzky–Golay smoothing itself improves the movement-focused lighthouse
endpoint relative to the unfiltered native approximately 250 Hz LFP field.
On identical voltage, time range, interpolation, reference, 348-channel
support, and sorter settings, SG25/order 2 increased family-balanced
single-cluster concentration during movement from 0.480 to 0.567
(+0.087). The five eligible movement families gave three SG wins, one tie,
and one loss.

This benefit is regime-specific rather than universal. The quiet score was
effectively unchanged (0.352 versus 0.355), while quiet duplicate-event
fraction worsened. Global sorter guardrails also mildly favored the unfiltered
arm: 44 versus 43 KS-good units and lower median contamination. Thus this
control supports SG25 for the stated priority of handling fast dramatic
offsets, but does not establish it as the best full-session correction.

## Matched result

The primary endpoint is the family-macro single-cluster fraction among frozen,
held-out strict lighthouse events. Lighthouse identity matching and family
votes were fixed independently of either motion field.

| Endpoint | Native 250 Hz | SG25/order 2 | SG minus native |
|---|---:|---:|---:|
| All-event single-cluster fraction | 0.370 | 0.429 | +0.059 |
| Movement single-cluster fraction | 0.480 | 0.567 | +0.087 |
| Quiet single-cluster fraction | 0.352 | 0.355 | +0.003 |
| Movement duplicate-event fraction | 0.235 | 0.220 | -0.015 |
| Quiet duplicate-event fraction | 0.316 | 0.384 | +0.068 |
| All-event median fragments | 4 | 5 | +1 |
| Movement median fragments | 3 | 3 | 0 |
| KS-good units | 44 | 43 | -1 |
| Median contamination, percent | 73.1 | 78.6 | +5.5 |

Event recovery was identical between arms: 0.993 overall and 1.000 in both
movement and quiet regimes. The movement result used five eligible families
and 201 events; quiet used three families and 240 events. Movement-family
deltas for SG minus native were 0.000, -0.045, +0.056, +0.280, and +0.143.
The improvement therefore is not a single-family-only result, although the
small family count remains an important uncertainty.

## Premise failure and repaired comparison

The first native-only attempt deliberately required the previous matrix's 350
common channels. It stopped before materialization because the unfiltered
native field retained only 348 of them, excluding AP24 and AP25. That failed
run was preserved with its receipt, traceback, and source checksum evidence;
it was not silently restarted.

The decisive repair reran **both** native and SG arms on the exact same 348
channels (AP26–AP373). This makes filtering the only arm difference. It also
means these scores should not be numerically ranked against the earlier
350-channel AP-rigid/unwarped matrix without rerunning those candidates on the
same 348-channel contract.

Follow-up: the matched 348-channel unwarped and AP-rigid reruns, plus a native
DARTsort nonrigid arm, are now complete. SG25 remains best on the movement
concentration endpoint; native DARTsort is second during movement and close to
AP rigid overall. See
`docs/luke_motion_348ch_candidate_matrix_result_20260910.md`.

## Operational verification

The paired run was launched as an independent systemd user service after a
dummy disconnection-survival test. The real service completed successfully
with exit status 0, and the persistent receipt records `complete`. Both saved
operators have effective `nblocks=0`, no internal displacement, CAR enabled,
and artifact threshold set to infinity. The accepted source-binary checksum
proof from the failed attempt was reused only after verifying its receipt,
failure location, worker/config/stderr hashes, manifest hash, binary size and
mtime, and recency.

## Artifacts

- Output: `/media/huklab/Data/luke_lfp_native_sg_348ch_v1`
- Machine-readable result: `analysis/summary.json`
- Regime table: `analysis/regime_summary.csv`
- Figure: `analysis/01_native_lfp_filter_ablation.png`
- Persistent job receipt: `/media/huklab/Data/luke_lfp_native_sg_348ch_v1_job/receipt.json`
- Preserved failed premise check: `/media/huklab/Data/luke_lfp_native_250hz_100s_v1_job`
- Worker: `testing/luke_lfp_native_sg_348ch_ablation.py`
- Launcher: `testing/launch_luke_lfp_native_sg_348ch_ablation.py`
