# EN Arm B full-session sort result — 2026-09-30

**Technical verdict: pass. Scientific verdict: pending the frozen four-arm
Tier-1 panel.** The rounded XR-field control completed its full 10,473.6 s
imec0 Kilosort 4.0.27 run under the accepted EN contract. All saved counts,
native settings, request bindings, and raw sorter arrays passed independent
checks. Yield alone does not establish that motion correction improved the
pipeline.

## Accepted result

- Output: `/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_b_v2`
- Persistent service: `en-rounded-ks129-arm-b-20260929-v2.service`, exit 0
- Full recording: 314,204,894 samples, 384 channels, 241,309,358,592 bytes
- Recording binary SHA-256:
  `a63d61d57d214d7a90a9698143c0fd1ea5c872a73bcac498b683a6d99efed8a2`
- Recording-content digest:
  `867218a76666c0a35a4cfaaab8a3bdedcb5938217baf4cf61df407393182b289`
- Sort request digest:
  `026e51365fc38440be43c2f7e840c96064f3e30cd8f383ffb386e5bad5f3fe4a`
- Units: 812 raw, 332 Kilosort-good
- Assigned spikes: 31,052,960

The sort manifest and top-level completion receipt agree on both the recording
request and sort request. Independent array inspection found 31,052,960 rows
in both `spike_times.npy` and `spike_clusters.npy`, 812 unique cluster IDs, and
332 good labels.

## Executed settings

Native `ops.npy`, rather than only the SpikeInterface request, confirms:

- `Th_universal=12`, `Th_learned=9`;
- `do_CAR=true`;
- effective `nblocks=0` and `dshift=None`;
- infinite artifact threshold.

The request records `do_correction=false`. The Kilosort 4.0.27 empty-clustering
center compatibility patch was active with source SHA-256
`19ba98a8cc752889a4eb2833410f8d69da6a275688fb9580b840075092fde259`.

## Tier-1 handoff hashes

The frozen Tier-1 validator accepted Arm B and independently hashed its four
required raw sorter products:

| Product | Bytes | SHA-256 |
|---|---:|---|
| `spike_times.npy` | 248,423,808 | `42dd56e8cfec65f233f40ca88777c56998baca528b0775e8b3019feb4d4d0ac5` |
| `spike_clusters.npy` | 124,211,968 | `87cde5de444716e9914fed242f674bd08a3c10992a64a8ab3718d578acdc812a` |
| `spike_positions.npy` | 248,423,808 | `f855d2fb9930adbffaa44b7c7912a9e76c60638baf43270cceb37cfc5f377256` |
| `cluster_KSLabel.tsv` | 6,737 | `db289d66af213d54c078b71f32688ff9143b58593ac80db57679864f024e5753` |

Tier 1 now reports REF and B as ready. Arm A still awaits the exact AM.3 field,
and XR still awaits its raw-sort handoff; therefore the frozen panel has not
run and no profile-based production decision is available.

## Resources

- Full materialization including acceptance hash: 4,268.8 s
- Full sort wrapper including independent input validation: 12,156.1 s
- Kilosort runtime: 10,341.2 s
- End-to-end service wall time: 16,555.3 s (4 h 35 m 55 s)
- Systemd CPU time: 5 h 48 m 34.7 s
- Service guards: `MemoryHigh=150G`, `MemoryMax=180G`, `CPUQuota=800%`

The service survived launcher disconnection, retained its launch command and
logs, atomically accepted the sort, wrote its completion receipt, and exited
cleanly. Kilosort had no within-sort checkpoint; an interruption would have
required preserving the partial evidence and restarting the sort.

## Published handoff

The compact verified result packet is published at
`/mnt/NPX/Luke/DARTsort_motion_experiments/en_arm_b_full_sort_result_20260930_v1`.
Its manifest SHA-256 is
`9faeba59801dd5c7bf44b7ed85c65ed6d99f9a8ecb80b73de6c15ea501afee54`.
The packet contains receipts, settings, this result, and hashes for the Tier-1
inputs. It contains no voltage or sorter arrays. `COMPLETE.json` was written
last, and its manifest binding and every product hash were independently
verified after publication.
