# Matched 348-channel motion-candidate comparison

Date: 2026-09-10

## Answer

The matched comparison does not produce one universal winner. It separates the
candidates by regime:

- **Fast dramatic offsets:** SG25 LFP has the highest movement single-cluster
  concentration (0.567). Native DARTsort is second (0.498) and has the lowest
  movement duplicate-event fraction (0.159).
- **All held-out events:** AP rigid (0.584) and native DARTsort (0.578) are
  effectively close at this resolution and both exceed unwarped (0.550).
- **Quiet events:** unwarped is best (0.684). AP rigid (0.649) and native
  DARTsort (0.648) are nearly identical; both LFP candidates are substantially
  worse.

For the stated priority of solving fast offsets, SG25 LFP remains the leading
candidate on the primary concentration endpoint. Native DARTsort is the most
credible balanced challenger because it improves movement concentration and
duplicates while remaining close to AP rigid overall and during quiet periods.

## Matched lighthouse results

The primary endpoint is the family-macro single-cluster fraction among frozen,
held-out strict lighthouse events. All arms use the same 930–1030 s source
voltage, AP26–AP373 channel support, 970–975 s reference, interpolation policy,
KS4 settings, and independently selected lighthouse identities.

| Arm | All | Movement | Quiet | Movement duplicate fraction | Movement fragments | KS-good units |
|---|---:|---:|---:|---:|---:|---:|
| Unwarped | 0.550 | 0.409 | **0.684** | 0.256 | 4.0 | 37 |
| AP rigid | **0.584** | 0.439 | 0.649 | 0.220 | 3.5 | 36 |
| DARTsort native nonrigid | 0.578 | 0.498 | 0.648 | **0.159** | **3.0** | 40 |
| LFP native 250 Hz | 0.370 | 0.480 | 0.352 | 0.235 | **3.0** | **44** |
| LFP SG25/order 2 | 0.429 | **0.567** | 0.355 | 0.220 | **3.0** | 43 |

All movement and quiet arms recovered 1.000 of eligible events. The all-event
recovery values were 0.971 unwarped, 0.985 AP rigid, 0.983 DARTsort, and 0.993
for both LFP arms.

Across the five eligible movement families, SG25 had the best or tied-best
concentration in two families, DARTsort in two, AP rigid in one, and native LFP
in one; ties make these counts overlap. Mean within-family ranks were 2.2 for
SG25, 2.5 for DARTsort, 3.0 for native LFP, 3.1 for AP rigid, and 4.2 for
unwarped. The small family count limits precision.

## Global guardrails

| Arm | Spikes | Units | KS-good | Median contamination, percent | Median refractory fraction |
|---|---:|---:|---:|---:|---:|
| Unwarped | 197,954 | 171 | 37 | 71.5 | 0.0358 |
| AP rigid | 205,476 | 172 | 36 | **65.1** | **0.0344** |
| DARTsort native nonrigid | 209,795 | 167 | 40 | 74.1 | 0.0407 |
| LFP native 250 Hz | 203,058 | 183 | **44** | 73.1 | 0.0462 |
| LFP SG25/order 2 | 199,817 | 170 | 43 | 78.6 | 0.0476 |

Raw good-unit counts favor the LFP arms, but their poor all/quiet lighthouse
concentration and higher duplicate burden argue against treating that count as
the efficacy endpoint. AP rigid has the strongest contamination and refractory
guardrails. DARTsort has four more KS-good units than AP rigid, with weaker
contamination and refractory metrics.

## DARTsort motion provenance

The supplied `motion.pkl` was produced by DARTsort 0.5.23+d20260910 and contains
a native DREDGE nonrigid estimate: 300 one-second time bins and nine spatial
bins centered from 312 to 3512 micrometers. It covers the original 930–1230 s
DARTsort window. The comparison preserved DREDGE's sign convention
(`corrected depth = observed depth - displacement`), shifted the time origin to
absolute recording seconds, and referenced each spatial bin independently at
970–975 s. The centered field spans -161.1 to +41.9 micrometers.

The pickle referenced a SciPy implementation detail unavailable in the KS4
production environment. It was therefore deserialized only in its recorded
DARTsort environment and converted to a hashed, array-only NPZ. The converted
geometry matched the accepted 384-channel source exactly, and the field retained
all 348 frozen comparison channels.

## Operational verification

The comparison ran in an independent systemd user service after a successful
launcher-disconnection dummy. The real service completed with exit status 0;
the persistent receipt records `complete`. All three new saved KS4 operators
have effective `nblocks=0`, `dshift=None`, CAR enabled, and infinite artifact
threshold. There is no within-sort checkpoint; no restart was needed.

An initial dummy launcher attempt used the shell Python and stopped at the
production-environment gate before starting a service or sort. Its partial
launcher evidence was preserved; the successful v2 dummy used the pinned
production interpreter and fresh paths.

## Artifacts

- Output: `/media/huklab/Data/luke_motion_348ch_matrix_v1`
- Machine-readable result: `analysis/summary.json`
- Regime table: `analysis/regime_summary.csv`
- Lighthouse unit table: `analysis/lighthouse_unit_metrics.csv`
- Matched events: `analysis/matched_events.csv`
- Figure: `analysis/01_motion_348ch_candidate_matrix.png`
- Persistent job receipt: `/media/huklab/Data/luke_motion_348ch_matrix_v1_job/receipt.json`
- Frozen DARTsort conversion: `/media/huklab/Data/luke_motion_348ch_matrix_v1_job/dartsort_native_motion_export.npz`
- Worker: `testing/luke_motion_348ch_candidate_matrix.py`
- Launcher: `testing/launch_luke_motion_348ch_candidate_matrix.py`
- Exporter: `testing/export_luke_dartsort_motion.py`

Follow-up unit reconciliation is complete. AP-rigid, DARTsort-motion, and
unwarped Kilosort match 61–65% of eligible Kilosort units and 73–81% of
KS-good units to native DARTsort, whereas the LFP arms match 21–23% and
32–44%, respectively. See
`docs/luke_kilosort_dartsort_unit_comparison_20260910.md`.
