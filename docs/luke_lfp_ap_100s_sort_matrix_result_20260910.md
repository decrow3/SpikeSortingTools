# Luke 930–1030 s LFP/AP correction-sort head-to-head

Date: 2026-09-10

## Answer

For follow-up sorting that prioritizes fast, dramatic offsets, the AP rigid
field is the best-supported candidate in this bounded comparison. It produced
the highest family-balanced single-cluster concentration during lighthouse
movement bins (0.504 versus 0.470 unwarped), and the improvement was distributed
as three lighthouse families better, one tied, and one worse. The gain is
modest and comes with seven fewer Kilosort-good units than unwarped, so this is
not yet evidence for replacing the full-session correction policy.

The LFP rigid field should not advance in its present form. It reduced
single-cluster concentration in movement bins (0.403) and especially quiet bins
(0.346 versus 0.751 unwarped), while increasing fragmentation. AP nonrigid had
the largest Kilosort-good count (48) but the weakest movement concentration
(0.385), so raw sorter yield does not support it for the stated fast-offset
priority.

## Matched experiment

The frozen LFP candidate only spans 930–1030 s, so this is a 100 s matrix rather
than a literal reproduction of the earlier 930–1230 s MEDiCINe matrix. Four
arms used the same accepted voltage, 350 common channels, 970–975 s reference,
linear temporal interpolation, kriging spatial interpolation, int16
materialization, and Kilosort 4.0.27 with internal motion correction disabled:

- unwarped
- LFP rigid
- AP rigid
- AP nonrigid

Only seven isolated unsupported 250 ms LFP samples were linearly interpolated.
The frozen held-out strict lighthouse events excluded 930–940 s training data.
Movement and quiet labels were inherited from the prior depth-aware lighthouse
audit: at least 20 micrometers versus less than 20 micrometers of 5 s
lighthouse increment. Matching used the correction actually applied to each
arm; no candidate motion field entered lighthouse identity discovery.

## Primary and guardrail results

| Arm | All families: single cluster | Movement | Quiet | Movement duplicate fraction | KS-good units |
|---|---:|---:|---:|---:|---:|
| Unwarped | 0.597 | 0.470 | 0.751 | 0.296 | 46 |
| LFP rigid | 0.373 | 0.403 | 0.346 | 0.316 | 46 |
| AP rigid | 0.589 | **0.504** | **0.794** | 0.276 | 39 |
| AP nonrigid | 0.572 | 0.385 | 0.786 | **0.273** | **48** |

The movement subset contains five eligible families (six lighthouse units,
201 events); the quiet subset contains three families (three units, 240
events). AP rigid beat/tied/lost against unwarped in 3/1/1 movement families
and 2/1/0 quiet families. Across the broader nine-family all-event endpoint it
was essentially tied with unwarped (-0.008), with a 3/3/3 win/tie/loss split.
This small sample and mixed all-event result are the main limits on the claim.

Global sorter counts were: unwarped 204,345 spikes / 178 units; LFP rigid
193,724 / 181; AP rigid 216,332 / 181; AP nonrigid 211,329 / 191. Median
contamination was lower for AP rigid (62.9%) than unwarped (71.15%), but AP
rigid had fewer Kilosort-good units (39 versus 46). These global outputs are
guardrails, not identity efficacy endpoints.

## Interpretation and next decision

This head-to-head does support the asymmetric objective: if fast-offset
preservation matters more than quiet-period perfection, AP rigid is the arm to
replicate. It does not support the present LFP rigid trace, despite the visually
clean registered LFP, and it does not support choosing AP nonrigid from its
larger unit count alone.

The cheapest next check is a targeted AP-rigid-versus-unwarped replication on
additional independently selected dramatic-offset intervals. That can test
whether the +0.034 movement advantage replicates before paying for a
full-session matrix. A longer LFP comparison would first require extending the
LFP estimator beyond its frozen 100 s support and then freezing that estimate
before lighthouse review.

## Artifacts

- Matrix output: `/media/huklab/Data/luke_lfp_ap_100s_sort_matrix_v1`
- Primary summary: `analysis/summary.json`
- Arm table: `analysis/arm_summary.csv`
- Regime table: `analysis/regime_summary.csv`
- Per-lighthouse table: `analysis/lighthouse_unit_metrics.csv`
- Event matches: `analysis/matched_events.csv`
- Figure: `analysis/01_sort_matrix_summary.png`
- Persistent job receipt: `/media/huklab/Data/luke_lfp_ap_100s_sort_matrix_v1_job/receipt.json`
- Worker: `testing/luke_lfp_ap_100s_sort_matrix.py`
- Launcher: `testing/launch_luke_lfp_ap_100s_sort_matrix.py`
- Analysis: `testing/luke_lfp_ap_100s_matrix_analysis.py`

The systemd service exited successfully with status 0. Every saved Kilosort
`ops.npy` has effective top-level `nblocks=0`, no displacement field, and no
internal drift correction; requested-settings metadata retains Kilosort's
default `nblocks=1`, but execution logs and effective state both record the
disabled correction.
