# EN XR raw-sort Tier-1 handoff request

The existing XR comparison handoff contains curated arrays. EN's frozen Tier-1
panel must instead start from like-for-like saved raw Kilosort outputs for REF,
XR, A, and B. This request transfers only the four compact XR arrays and two
identity receipts needed for yield, short-interval, presence, spatial
coincidence/edge, state-rate, and time-only correspondence measurements.

## Frozen identities and paths

- Producer source on huklaban1:
  `/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1`
- Expected XR sort identity:
  `06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555`
- Destination:
  `/mnt/NPX/Luke/DARTsort_motion_experiments/en_xr_tier1_raw_handoff_20260929_v1`
- Publisher/verifier:
  `testing/en_xr_tier1_handoff.py`
- Publisher SHA-256:
  `6e9a47999c0ae8393027ce40bb647aa231459f3e282711c29430fe9ed164cf5f`

## Producer command

Launch this command through a persistent systemd user service and retain its
unit, logs, command, and terminal status outside the task:

```bash
python -m testing.en_xr_tier1_handoff publish \
  --source /media/huklab/Data/luke_medicine_rigid_sort_20260909_v1 \
  --destination /mnt/NPX/Luke/DARTsort_motion_experiments/en_xr_tier1_raw_handoff_20260929_v1 \
  --expected-sort-identity 06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555
```

The publisher verifies the three files already bound by `sort_identity.json`,
hashes every copied file, writes `COMPLETE.json` last, verifies the staged
packet, and exposes it with an atomic rename. It refuses to replace either a
completed destination or a preserved partial transfer.

The packet contains `spike_times.npy`, `spike_clusters.npy`,
`spike_positions.npy`, `cluster_KSLabel.tsv`, `sort_identity.json`, and
`summary.json`. It excludes voltage, templates, features, waveforms, and sorter
scratch.

## Why the existing curated handoff cannot substitute

The existing rigid-comparison handoff was checked directly.  Its
`spike_times.npy`, `spike_clusters.npy`, and `spike_positions.npy` each contain
26,659,257 events.  Its `full_st.npy` also has 26,659,257 rows, and
`kept_spikes.npy` is an all-true vector of that same length.  The pinned raw XR
sort identity records 28,176,859 events.  Thus the curated packet omits
1,517,602 original events together with their raw labels and positions; those
values cannot be reconstructed from its retained arrays.  EN Tier 1 must wait
for the identity-bound raw files requested above.
