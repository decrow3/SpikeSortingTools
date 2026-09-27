# BY: BQ packet input-availability review

## Verdict

The approved BQ packet is a strong **prospective reference/output packet**, but
it is not self-contained input for the proposed 30,000-versus-3,000-sample
template-construction control. It supports exact lineage checks against the BQ
state and supplies the completed old template bank, temporal basis, distance/
shift matrices, route masks and maps. New construction still needs explicitly
resolved external inputs: the accepted 182-channel float32 recording/cache,
event channels, physical geometry, motion field/time origin and frozen configs.

The control does not need pcmerge or TMM rerun once those missing references are
resolved. A new deterministic construction membership shared by both arms is a
valid prospective control, but must not be described as BQ's or the historical
run's original sampled membership.

Review was read-only in:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-bq-w2-agglomeration-capture-v1/`.
No arrays, templates, distances or QDA statistics were recomputed.

## Exact packet inventory

### Pre-stage/event state

`capture/arrays.npz` contains:

- `event_row_ids [641588] int64`, exactly `0..641587`;
- `event_labels_pre [641588] int64`, one noise row and dense nonnoise IDs
  `0..747`;
- `event_times_samples_pre [641588] int64`, range `98..10199836`;
- `stage_unit_ids [748] int64`, exactly `0..747`;
- `event_gmm_candidates [641588,5] int32` and
  `event_gmm_log_liks [641588,6] float32`.

It does **not** contain event channels, GMM responsibilities, merged candidates/
likelihoods/responsibilities, `gmm_train`, unit log proportions, alignment signs,
the parent HDF5 path/hash, or a serialized `DARTsortSorting`. Gate P reports that
several of those arrays existed and were equal between its paired arms, but its
summary is not their payload.

### Constructed template state

`constructed_template_data.npz` is a loadable `TemplateData`-shaped packet:

- templates `[748,121,206] float32` and dense unit IDs `0..747`;
- spike counts `[748]` (range 7–500; sum 299,635) and per-channel counts
  `[748,206]`;
- registered geometry `[206,2]`, depth 2020–4060 µm;
- trough offset 42, sampling frequency 29,999.7591667 Hz;
- whitening strategy `none`;
- TSVD components `[5,121]`, mean `[121]`, whiten flag false, and five
  explained-variance entries. Components are finite; the explained-variance
  entries are all nonfinite and cannot be used as fit-quality evidence.

This bank can serve as the exact old-output comparator and supplies a temporal
basis candidate. It does not reveal which event row IDs produced each template,
the physical 182-channel geometry, motion field, interpolation configuration,
full `TemplateConfig`/`WaveformConfig`, or source recording identity. The
206-channel registered geometry is not a substitute for the physical input
geometry. `PREFLIGHT.json`'s `[30000,2]` recording shape is the dummy preflight,
not the accepted scientific recording.

### Distances, QDA and maps

`capture/arrays.npz` also contains the complete old `[748,748]` raw template
distances and directed shifts, distance/force linkage inputs, condensed strict
upper triangles, direct masks, linkage-expanded masks, requested/QDA/union
masks, firing correlation and reported QDA score/IoU/coverage/min-ratio arrays,
plus pre-depth map `[748]`, depth reorder `[551]` and final map `[748]`.
`contrast_arrays.npz` adds current/no-force fixed labels `[641588]`, no-force
stage-aware times/labels, union masks and component maps.

These are exact old-result controls. They do not provide all inputs for a new
full QDA: source `get_gmm_scores` requires candidates, log likelihoods **and
responsibilities** (`util/data_util.py:1501-1524`), while the public packet lacks
responsibilities and merged score arrays. Moreover, the manifest explicitly
marks QDA completion/status and IoU/coverage availability masks as not passively
observed. Zero score entries therefore cannot be treated as completed rejects.
If short-chunk distances nominate pairs outside the old requested mask, the
packet has no demonstrated QDA result for them. Full QDA/agglomeration is not
reconstructible from this packet alone.

## Minimum construction-only API sequence

After resolving and hashing the missing external references, the smallest
prospective paired sequence is:

1. Build one `DARTsortSorting` from the saved BQ row IDs/times/labels plus the
   exact corresponding event channels (and alignment signs if present) from the
   accepted source. Assert row-wise equality before use.
2. Define one deterministic eligible construction-row set per unit from this
   fixed state. Save the selected row IDs. Use that identical membership/order
   in both arms; avoid a second internal subsampling step by masking labels to
   the frozen membership and setting the cap consistently. This is new
   prospective membership, not historical membership.
3. Resolve the exact 182-channel accepted float32 recording/cache, physical
   geometry, Q field/MotionInfo with time origin and registered geometry, full
   template/waveform/interpolation configs, computation dtype/device/workers,
   and executed source object. Supply the saved finite TSVD basis explicitly;
   whitening is declared `none`, so no absent whitener should be invented.
4. Call `TemplateData.from_config(...)` twice on the same sorting/recording/
   motion/basis/config, changing only
   `grab_chunk_length_samples=30000` versus `3000`. The active peel-reduce route
   subsamples before engine construction and passes the grab length only to
   `TemplateReduction` (`peel/reduction_template.py:60-81, 181-358`).
5. Verify identical selected row IDs, unit IDs and expected event counts. Compare
   each arm to the old bank as a diagnostic, without asserting historical
   membership equivalence.
6. For force-distance comparison only, call `template_distances(...)` on each
   new `TemplateData` with the same fixed sorting, motion and exact merge config,
   then apply the same direct threshold and linkage. Weighted distance requires
   sorting channels/times/labels plus physical/registered geometry and motion to
   reconstruct empirical support weights (`clustering/agglomerate.py:281-405,
   742-777`).

No prefix rerun is required by this sequence. Do not call full `agglomerate` or
QDA unless the missing full score state and QDA coverage semantics are separately
provided and authorized.

## Exact blockers to resolve before a proposal can launch

- accepted 182-channel float32 recording/cache path, content hash, wrapper and
  sample/time origin;
- event channel vector aligned exactly to `matching1_event_rows`; optional
  alignment-sign vector and authoritative parent source hash;
- physical `[182,2]` geometry and the mapping/derivation to saved `[206,2]`
  registered geometry;
- exact external motion field/hash, sign convention, interpolation/extrapolation
  implementation and time origin;
- full template, waveform, interpolation, merge and computation configs,
  including reduction method, padding/alignment, support thresholds, distance
  parameters and linkage;
- new shared membership algorithm/seed and saved selected row IDs;
- confirmation that the saved TSVD basis is accepted despite nonfinite reported
  explained variance, and that no other fitted model is required by the frozen
  route.

The packet supplies no evidence for inventing any missing model or config.
If an exact item cannot be recovered, the corresponding comparison must be
labelled a new prospective implementation rather than a reconstruction.

## Corrected BV/BX footprint statement

The proposed waveform review targets the accepted **182-channel float32 cache**,
not 384-channel int16 raw voltage. Its logical full-row payload is exactly
`7200 × 121 × 182 × 4 = 634,233,600` bytes (about 604.9 MiB). This does
not establish physical I/O: cache layout, filesystem blocks, duplicate/coalesced
reads, filter buffers and temporary arrays remain unmeasured. The prior 384-
channel int16 wording is superseded.

## Next dependency

Before BW.2 is executable, publish a compact resolver manifest for the blockers
above and demonstrate exact row alignment for channels against the 641,588 saved
row IDs. Then freeze and hash the new construction membership before either arm.
This review does not authorize voltage access, distance recomputation or launch.
Next unused queue label: BZ.

