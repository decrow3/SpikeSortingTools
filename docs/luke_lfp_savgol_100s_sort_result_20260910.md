# Native-grid LFP Savitzky–Golay correction result

Date: 2026-09-10

## Answer

A conservative native-grid Savitzky–Golay candidate improves the previous LFP
candidate, but it still does not beat unwarped voltage or AP rigid correction.
The filtered LFP candidate therefore does not change the current decision: AP
rigid remains the best candidate for fast dramatic offsets in this interval.

The filter was applied to the underlying approximately 250 Hz estimate, not to
the 25 Hz display or 4 Hz comparison CSV. Parameters were frozen before the new
sort result was inspected: 25 samples (100.0 ms), polynomial order 2, and
`mode="interp"`. It reduced the largest 4 ms displacement step from 52.8 to
13.5 micrometers while retaining 320.9 of the raw 334.1 micrometer range.

## Sort result

| Arm | All-family single cluster | Movement | Quiet | KS-good units |
|---|---:|---:|---:|---:|
| Unwarped | **0.597** | 0.470 | 0.751 | 46 |
| Previous LFP rigid | 0.373 | 0.403 | 0.346 | 46 |
| Native-grid LFP Savitzky–Golay | 0.394 | 0.444 | 0.438 | 41 |
| AP rigid | 0.589 | **0.504** | **0.794** | 39 |
| AP nonrigid | 0.572 | 0.385 | 0.786 | **48** |

Relative to the previous LFP arm, the Savitzky–Golay candidate increased
family-balanced single-cluster concentration by 0.022 overall, 0.041 during
movement, and 0.092 during quiet bins. It remained below unwarped by 0.203
overall, 0.026 during movement, and 0.313 during quiet bins. Movement duplicate
fraction improved slightly relative to the previous LFP arm (0.308 versus
0.316), but median movement fragmentation remained four clusters.

The filtered arm produced 201,692 spikes, 187 units, and 41 Kilosort-good units.
The independently managed service completed successfully with return code 0;
saved Kilosort state has effective `nblocks=0` and no internal displacement.

## Important ablation limit

This is a candidate comparison, not a pure filter ablation. The previous
`lfp_rigid` matrix arm was constructed from the 4 Hz comparison CSV, whereas
the new Savitzky–Golay arm was constructed from the native approximately 250 Hz
motion package. Its improvement cannot therefore be attributed uniquely to
Savitzky–Golay smoothing. A matched unfiltered-native-250-Hz arm is required to
separate the effect of native temporal sampling from the effect of filtering.

That unfiltered native arm is the cheapest decisive next check. It can reuse
all frozen controls and add only one new sort. A wider filter sweep should wait
until that control establishes whether smoothing itself helps.

Follow-up: the matched native-grid control is now complete. Because native
support excluded AP24 and AP25, both native and SG arms were rerun on the same
348-channel contract. SG25 improved the movement endpoint by 0.087 while the
quiet endpoint was effectively unchanged. See
`docs/luke_lfp_native_sg_348ch_ablation_result_20260910.md`.

## Native-grid support and artifacts

Within 930–1030 s, 24,516 of 25,000 native samples were directly supported.
The 484 unsupported samples occurred in 104 short gaps, each no longer than
9 samples (36 ms), and were linearly filled before filtering. The field was
referenced to 970–975 s and applied to the same 350 common channels as the prior
matrix.

- Output: `/media/huklab/Data/luke_lfp_savgol_sg25_100s_v1`
- Result summary: `analysis/summary.json`
- Combined regime table: `analysis/regime_summary_with_savgol.csv`
- Filter audit: `field_resolution_audit.json`
- Figure: `analysis/01_lfp_savgol_sort_comparison.png`
- Persistent job receipt: `/media/huklab/Data/luke_lfp_savgol_sg25_100s_v1_job/receipt.json`
- Worker: `testing/luke_lfp_savgol_100s_sort.py`
- Launcher: `testing/launch_luke_lfp_savgol_100s_sort.py`
