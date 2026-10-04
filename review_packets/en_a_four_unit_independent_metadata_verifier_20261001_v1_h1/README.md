# Independent arm-A four-unit metadata verifier

Status: **authoring complete; this is not the final v3 review or approval.**

`en_a_four_unit_metadata_verifier.py` is an H1-authored, read-only verifier for
the frozen 20-slot selection contract. It accepts only saved sorter/model arrays,
channel geometry, the A motion field, ops, and the frozen CSV contract. It has no
recording-binary argument and rejects `.raw`, `.bin`, `.dat`, and `.cbin` inputs.

The verifier hashes every input before loading it, memory-maps the event/model
arrays, confirms alignment and global ordering, and independently reconstructs:

- original adjacent same-unit event sequences before eligibility filters;
- half-away 40-um field states and numeric contiguous-state segments;
- exact-lattice saved-y support membership;
- the earliest qualifying eight short pairs, eight ordinary controls, and four
  q0 controls under the unchanged H1 selection rules;
- complete final-unit/template distributions for the four units; and
- scalar post-sort exported-template footprint summaries for every associated
  template.

Outputs are compact `SELECTION_EVIDENCE.csv`, `EVIDENCE.json`, `MANIFEST.json`,
and `COMPLETE.json`. No template arrays, waveform samples, or voltage values are
written. Existing output directories are never overwritten.

## Frozen H5 invocation

Run from an environment with NumPy, substituting only `OUTPUT_DIR`:

```bash
python en_a_four_unit_metadata_verifier.py \
  --spike-times /media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/sort/sorter_output/spike_times.npy \
  --spike-times-sha256 1c7643385dc3f128451880871a643b93fd343dfaa90c4cd7c65a229238ee3bcc \
  --spike-clusters /media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/sort/sorter_output/spike_clusters.npy \
  --spike-clusters-sha256 84409cbbf3548939fc88d44b191824acceb5ebdcb5432a2faf0619960b50454a \
  --spike-templates /media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/sort/sorter_output/spike_templates.npy \
  --spike-templates-sha256 84409cbbf3548939fc88d44b191824acceb5ebdcb5432a2faf0619960b50454a \
  --spike-positions /media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/sort/sorter_output/spike_positions.npy \
  --spike-positions-sha256 cb49eda92a72aaf5cfe48cc09b2c4a7a75f07682877cef26f86089892f70c762 \
  --templates /media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/sort/sorter_output/templates.npy \
  --templates-sha256 19133339863c9c8f58e07128e035fdf7099c45cba30f1342fa0acfac57599ba7 \
  --ops /media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/sort/sorter_output/ops.npy \
  --ops-sha256 07d61baf2c0da759db2ee2a21ef1d2d7a561bcb90293f568bccfe0a9e0906906 \
  --channel-positions /media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/sort/sorter_output/channel_positions.npy \
  --channel-positions-sha256 0469ca92fb739a0cfd2f1613262d3a2d75af1098385385d7462ee6e3fd038d75 \
  --field /mnt/NPX/Luke/DARTsort_motion_experiments/en_am3_imec0_field_request_20260929_v1/payload/luke0804_imec0_two_layer_motion.npz \
  --field-sha256 4c769125020aad3263bf6eaa1889cdaa7eccbf19dcf0aa3a068b185221fadb9f \
  --selection-contract /mnt/NPX/Luke/DARTsort_motion_experiments/en_a_four_unit_physical_signal_assessment_20261001_v1_h1/SELECTION_SLOTS.csv \
  --selection-contract-sha256 d34c62c502b6ef5d37c69037c4d0bc3c2d390ed1ce9c726f330d5d0ca69af30c \
  --expected-source-sha256 ea292d8f6ecbaa9f4306bb9e59754ecab3447c0a2b04f539cc343110111a8a2e \
  --sampling-frequency-hz 29999.835983263598 \
  --output OUTPUT_DIR
```

The script must be run unchanged at the manifest-bound source hash. The compact
output may then be included in H5's v3 packet for the single final H1 review.

## Local validation

`test_en_a_four_unit_metadata_verifier.py` builds only synthetic metadata and
exercises the complete CLI/output path. It passes without opening recording data.

Implementation checks

- Done: known-answer end-to-end synthetic fixture -> 20 selections, four exact
  unit/template distributions, scalar footprints, manifest, and completion.
- Done: source self-hash, every input hash, selection-rule text, array alignment,
  global ordering, field uniformity, ops clock/template length/identity chanmap,
  and output non-overwrite gates fail closed.
- Not done: H5 saved-array execution -> delegated to H5 for the v3 packet.
- Can establish after H5 execution: independently generated compact metadata
  evidence from the hash-bound saved arrays.
- Cannot establish: waveform behavior, biological identity, purity, or any
  voltage-domain claim.
