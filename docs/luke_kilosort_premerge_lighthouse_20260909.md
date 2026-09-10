# Kilosort premerge lighthouse screen

## Headline

Running the depth-blind family screen on the preserved Kilosort state before
`template_matching.merging_function` exposes **one compelling moving candidate**:
**c097_F009**, an 11-tracklet family spanning 240 um of template depth and about
196 um of 5-s binned displacement.

This family looks much better in depth/time than the large chains from the
final-CID analysis. Its chronological plot repeatedly moves between the same
depth bands, its merged train is extremely refractory-clean, and it permits a
small number of >100 um transitions without admitting much larger jumps.
However, agreement with the previous lighthouse cells remains suggestive rather
than significant after the circular-shift controls.

## What was run

The input is the preserved 930--1230 s Kilosort checkpoint produced by
`clustering_qr.run(mode="template")`, explicitly before
`template_matching.merging_function`. The lighthouse review is restricted to
930--1030 s.

- 267,004 detections in the review interval
- 342 active premerge clusters
- 323 waveform-eligible clusters with at least 20 snippet spikes
- no AP-voltage read
- no sorter launch
- cluster waveforms reconstructed exactly as `Wall @ wPCA`
- absolute depth and event time excluded from family construction

The checkpoint templates were learned from the full bounded 930--1230 s
clustering interval, so this is candidate discovery rather than a seed-only
identity validation.

## Threshold result

| Cosine threshold | Families | Templates in families | Largest component | Depth/time coherent |
|---:|---:|---:|---:|---:|
| 0.95 | 22 | 108 | 23 | 1 |
| **0.97** | **11** | **65** | **20** | **2** |
| 0.98 | 9 | 48 | 16 | 2 |

At 0.97, the two coherent families are c097_F009 and c097_F010. At 0.98,
c097_F009 breaks apart; the two survivors are stationary pairs rather than
large-excursion lighthouse candidates. This again favors 0.97 for discovery.

## Moving candidate c097_F009

Member CIDs are 241, 251, 255, 270, 278, 279, 283, 287, 290, 291, and 294.

| Diagnostic | Result |
|---|---:|
| Template-depth span | 240 um |
| 5-s displacement range | -97 to +99 um |
| Cross-CID joins within 2 s | 598 |
| Jumps >=100 um | 13 (2.17%) |
| Jumps >=160 um | 1 (0.17%) |
| Maximum jump | 164.8 um |
| Jumps >=180 um | 0 |
| Coactive 1-s bins | 53 |
| Coactive bins separated by >=80 um | 5 (9.43%) |
| Merged/Poisson short-ISI ratio | 0.038 |
| Maximum-spanning-tree bottleneck cosine | 0.9773 |
| Closest external cosine | 0.9592 |
| Tree-bottleneck isolation margin | 0.0181 |
| All-pairs minimum cosine | 0.8586 |

The low all-pairs minimum is expected for a long chain crossing several depth
states and should not be confused with a weak connection path. Every member is
connected by a maximum-spanning-tree edge of at least 0.9773, and the weakest
required edge still exceeds the closest external match by 0.0181.

The family therefore fits the intended >100 um policy: total motion may be much
larger than 100 um, occasional rapid transitions slightly above 100 um are
retained, and essentially no teleportation much beyond 160 um is present.

## Comparison with previous lighthouse cells

For c097_F009:

- previous-cell median: Pearson r=0.443, circular-shift p=0.10;
- best previous cell: unit 657, r=0.825 over 19 matched bins;
- best-of-library circular-shift p=0.20.

No lag, sign, or gain was fit. The comparison is directionally encouraging but
does not pass the multiple-comparison-aware null.

The other coherent 0.97 family, c097_F010 (CIDs 267/272), is a tight and isolated
pair but moves only about 4 um peak-to-peak. It is a plausible duplicate/split
identity, not a useful motion lighthouse.

## Did disabling the final merge create the moving family?

No. Ordinary static Kilosort merging leaves all 11 c097_F009 members as 11
separate output clusters. It also leaves the two c097_F010 members separate.

Therefore the useful difference is not that the final merge previously
collapsed the lighthouse fragments. It is that the bounded premerge clustering
provides a different, finer candidate partition than the final full-session
curated CIDs used in the earlier screen. The premerge checkpoint is useful for
discovery, but this experiment does not show that Kilosort's merge stage caused
the original loss.

## Outputs

- `testing/luke_kilosort_premerge_lighthouse_v5.py`
- `testing/test_luke_kilosort_premerge_lighthouse.py`
- `testing/outputs/luke_kilosort_premerge_lighthouse_v5/component_audit_all_thresholds.csv`
- `testing/outputs/luke_kilosort_premerge_lighthouse_v5/members_all_thresholds.csv`
- `testing/outputs/luke_kilosort_premerge_lighthouse_v5/edges_all_thresholds.csv`
- `testing/outputs/luke_kilosort_premerge_lighthouse_v5/coherent_family_motion_all_thresholds.csv`
- `testing/outputs/luke_kilosort_premerge_lighthouse_v5/coherent_family_static_merge_fate.csv`
- joined-family atlases at all three thresholds, summary figures in PDF and PNG,
  and hashed `summary.json`

Thirteen tests across the cached depth-blind family analysis pass.
