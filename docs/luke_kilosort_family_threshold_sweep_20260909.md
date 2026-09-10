# Depth-blind Kilosort family threshold sweep

## Result

Rebuilding the waveform graph at **0.97** is a useful tightening. Raising it to
**0.98** is counterproductive: it removes every depth/time-coherent family while
retaining several high-cosine implausible families.

| Cosine edge threshold | Multi-template families | Templates retained | Largest component | Depth/time coherent | Implausible | Underpowered |
|---:|---:|---:|---:|---:|---:|---:|
| 0.95 | 24 | 119 | 25 | 5 | 17 | 2 |
| **0.97** | **17** | **83** | **20** | **3** | **13** | **1** |
| 0.98 | 9 | 53 | 18 | 0 | 9 | 0 |

The graphs were independently rebuilt at each threshold. These are not the old
0.95 labels with rows filtered afterward. Family identity matching still uses
only relative waveform geometry; depth and time are revealed only for the
frozen post-hoc plausibility screen.

## The three 0.97 families

| 0.97 family | CIDs | Origin at 0.95 | Min cosine | Internal-minus-external margin | Prior median r | Shift p | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| **c097_F001** | 24, 65 | F001 | 0.9749 | **0.2075** | 0.39 | 0.20 | strongest waveform isolation; coherent depth/time |
| **c097_F012** | 469, 475 | F016 | 0.9700 | **0.0392** | **0.48** | **0.05** | strongest motion replication; boundary exact null |
| c097_F017 | 664, 667 | two-CID child of giant F018 | 0.9773 | 0.0083 | 0.29 | 0.20 | plausible child, but thin isolation and no motion replication |

Only two of the five original 0.95 coherent families remain intact at 0.97:
F001 and F016. F010, F015, and F022 dissolve because their pair similarities
are 0.9594, 0.9558, and 0.9666 respectively. The new c097_F017 is useful: it
shows that tightening can carve a locally coherent pair out of a large chained
component. Its small external margin warns that it is still embedded in a
crowded waveform neighborhood.

## Why 0.98 fails

At 0.98, none of the surviving components passes the same depth/time screen.
The surviving graph still contains components as large as 18 templates, and
several retained pairs show simultaneous separated depth bands or rapid large
jumps. Thus very high pairwise cosine is not equivalent to identity.

This also explains the apparent paradox: increasing the threshold removes some
moderately similar genuine tracklet candidates before it removes pathological
templates in highly degenerate regions of the bank.

## Recommended cheap rule

Use **cosine >=0.97** for the primary screen, then:

1. treat complete-link pairs/small cliques as primary and chained components as
   containers requiring subdivision;
2. rank by weakest-internal minus closest-external cosine, rather than internal
   cosine alone;
3. keep 0.95–0.97 families as a secondary recovery tier, preserving F010 and
   F015 rather than silently discarding them;
4. reveal depth/time only after those waveform-only ranks are frozen.

On this snippet, c097_F001 and c097_F012 are the strongest primary candidates.
c097_F017 is a secondary candidate. A hard isolation-margin cutoff should not
yet be learned from these 24 families.

## Files

- `testing/luke_kilosort_family_threshold_sweep_v4.py`
- `testing/test_luke_kilosort_family_threshold_sweep.py`
- `testing/outputs/luke_kilosort_family_threshold_sweep_v4/threshold_summary.csv`
- `testing/outputs/luke_kilosort_family_threshold_sweep_v4/component_audit_all_thresholds.csv`
- `testing/outputs/luke_kilosort_family_threshold_sweep_v4/baseline_0p95_family_survival.csv`
- `testing/outputs/luke_kilosort_family_threshold_sweep_v4/coherent_family_motion_all_thresholds.csv`
- joined-family atlases for 0.97 and 0.98, summary figures in PDF and PNG, and
  hashed `summary.json`

Ten tests across the complete cached tracklet-family analysis pass. No voltage
was read, no sorter was run, and no family used motion or depth for identity.
