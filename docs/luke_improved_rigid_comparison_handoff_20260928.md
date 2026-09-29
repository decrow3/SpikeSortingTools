# Compact rigid-arm comparison handoff

The full-session external-rigid MEDiCINe sort on huklaban1 is complete, but its
shared summary is insufficient for the same paired comparison already applied
to the external-nonrigid arm. Raw counts cannot establish improvement. This
handoff transfers only the completed sort evidence needed for spike-train
correspondence, amplitude-completeness coverage, refractory burden,
chance-aware spatial coincidence, and edge guardrails.

## Frozen identities and paths

- Producer source on huklaban1:
  `/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1`
- Candidate sort identity:
  `06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555`
- Shared destination:
  `/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/huklaban1_rigid_comparison_handoff_v1`
- Consumer config:
  `configs/luke_improved_rigid_comparison.v1.json`

The equivalent files from the completed nonrigid arm total 1,196,114,046 bytes
(1.114 GiB), plus a small application contract and manifest. This is small
relative to the corrected recording and sorter workspace, but the shared mount
has only about 232 GiB free and must not receive broader output trees.

## Producer command

Run the publisher on huklaban1 from a checkout containing
`testing/rigid_comparison_handoff.py`. Because the transfer is long enough to be
interrupted, launch this exact command through a run-specific persistent user
service and preserve the service definition, stdout/stderr, and terminal exit
status beside the producer job record:

```bash
python -m testing.rigid_comparison_handoff publish \
  --source /media/huklab/Data/luke_medicine_rigid_sort_20260909_v1 \
  --destination /mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/huklaban1_rigid_comparison_handoff_v1 \
  --expected-sort-identity 06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555
```

The publisher refuses missing or incomplete stage receipts, checks their sort
identity, hashes every copied file, verifies the staged package, and exposes it
with an atomic directory rename. An interrupted
`huklaban1_rigid_comparison_handoff_v1.partial` is deliberately preserved and
must be investigated rather than treated as success or overwritten.

The package includes six curated comparison inputs, cached truncation QC, the
application contract, pinned identity, final summary, and four downstream
receipts. It excludes corrected voltage, sorter scratch, templates, `tF.npy`,
`pc_features.npy`, waveforms, PDFs, and MATLAB exports.

## Consumer check and comparison

On huklaban5, first verify without copying the package:

```bash
python -m testing.rigid_comparison_handoff verify \
  --package /mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/huklaban1_rigid_comparison_handoff_v1 \
  --expected-sort-identity 06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555
```

The persistent service declared in
`configs/luke_improved_rigid_comparison.v1.json` may be started before the
handoff arrives. It waits only for the atomically published `MANIFEST.json`,
then independently verifies every artifact before loading any candidate data.
Its scientific settings and spatial domain are identical to the completed
nonrigid-vs-reference comparison. `ManagedOOMPreference=avoid` retains the
resource protection required after the earlier user-manager oomd termination.

Do not infer a rigid-arm decision from the 480 raw units, 108 raw KS-good units,
or 28,176,859 raw spikes. The arm remains `comparison_pending` until the paired
result is complete. No new sort is authorized or needed for this closure.
