# Cluster 452 cached identity and curation replay result

Date: 2026-09-28. Status: **complete; no retained-output intervention nominated.**

## Decision

The frozen cached-sort checks do **not** support either identity redistribution
or final-curation exclusion as the cause of cluster 452's amplitude-completeness
deterioration. Do not merge cluster 452 with a neighboring rescue label and do
not replay the final curation as a candidate intervention for this case.

This closes the bounded follow-up proposed in
[`luke_cluster452_exact_replay_proposal_v1.md`](luke_cluster452_exact_replay_proposal_v1.md).
It does not establish normal detection or voltage integrity. The remaining
supported classification is an unresolved detection/waveform-amplitude change.

## Independent event-anchor result

The v2 contract selected candidates by template peak depth within 100 um before
examining the endpoint windows. It selected the legacy anchor using only the
two reference windows, maximum-cardinality one-to-one event pairing within
0.5 ms, and fixed 137/431/997 s circular-shift nulls.

Legacy cluster 356 is a strong reference anchor:

- 1,909 reference events;
- 97.17% match some spatially eligible rescue event;
- 73.91% match rescue cluster 452 specifically;
- null median 16.61%, giving 80.57 percentage points of excess matching;
- rescue 452 is the dominant reference label.

The anchor remains mostly assigned to rescue 452 in both failing windows:

| window | anchor events | any rescue match | rescue 452 fraction | largest alternate |
|---|---:|---:|---:|---:|
| reference 1 | 961 | 97.40% | 73.47% | 459: 3.85% |
| reference 2 | 947 | 96.94% | 74.45% | 459: 3.38% |
| failing 1 | 911 | 94.84% | 70.25% | 459: 4.61% |
| failing 2 | 756 | 95.11% | 71.96% | 459: 4.23% |

The target fraction falls by only 2.85 percentage points from the reference
mean. The largest alternate, cluster 459, rises by 0.81 points, remains below
the frozen 10% per-window and 10-point increase gates, and has template cosine
0.071 to cluster 452. No candidate passes the waveform, redistribution, and
union-refractory rules together.

The first v1 execution is preserved as a setup failure. Its selection rule
chose a three-event, 100%-matched anchor before applying the already declared
target-dominance gate, making the endpoint unevaluable. V2 corrected selection
to operate only among anchors passing the declared reference gates and the
repository's existing 20-event minimum. Five known-answer tests cover exclusive
matching, denominator accounting, template lag, duplicate timestamp handling,
and eligible-anchor selection.

## Exact curation lineage

The final cluster maps exactly to raw Kilosort cluster 470 and was not created
by a merge. Reconstructing the production curation order—duplicate-row removal,
merge relabeling, redundant-unit removal, and renumbering—reproduces all 57,664
curated target events.

| window | pre-curation events | duplicate rows removed | removed fraction |
|---|---:|---:|---:|
| reference 1 | 1,029 | 30 | 2.92% |
| reference 2 | 1,037 | 38 | 3.66% |
| failing 1 | 1,038 | 39 | 3.76% |
| failing 2 | 1,021 | 22 | 2.15% |

The minimum failing-window removed fraction minus the maximum reference-window
fraction is -1.51 percentage points, far below the frozen +10-point support
threshold. Final curation therefore does not explain the 19-point completeness
deterioration. The similar pre-curation event counts also make a final-curation
replay an unmotivated intervention, while remaining agnostic about events KS4
never placed in this raw cluster.

## Reproducibility and scope

- Identity configs:
  [`v1`](../configs/luke_cluster452_identity_replay.v1.json) and
  [`v2`](../configs/luke_cluster452_identity_replay.v2.json); v2 SHA-256
  `cae6cb75c662ff4afe5349ccc6fcda009d48fb447d3e737ae948e61932a1ee4c`.
- Identity implementation:
  [`testing/luke_cluster452_identity_replay.py`](../testing/luke_cluster452_identity_replay.py).
- Curation config:
  [`configs/luke_cluster452_curation_lineage.v1.json`](../configs/luke_cluster452_curation_lineage.v1.json),
  SHA-256 `d1e5c6173e84a57ffd8f34e10f97b5d6037e16a67ab47c2d38cb673f435af1a6`.
- Curation implementation:
  [`testing/luke_cluster452_curation_lineage.py`](../testing/luke_cluster452_curation_lineage.py).
- Local generated evidence:
  `testing/outputs/luke_cluster452_identity_replay_v1/`,
  `testing/outputs/luke_cluster452_identity_replay_v2/`, and
  `testing/outputs/luke_cluster452_curation_lineage_v1/`.

No raw voltage was read, no sort was launched, and no production output or
label was changed.
