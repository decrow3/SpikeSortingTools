# CF: same-state actual-field versus zero-field construction review

## Verdict

The proposed 3,000-sample actual-field versus zero-field control is executable
from one frozen BZ state, but it has **two different legitimate operators** that
must not be conflated. DARTsort's native operator constructs each arm on its own
motion-dependent registered geometry and recomputes its own empirical spatial
weights. A fixed-common-physical-support operator can compare waveform content
on identical physical sites, but is a diagnostic projection rather than the
native force route. Raw registered-channel indices and zero-padded full-array
energy are invalid cross-arm comparisons because the actual field expands the
182 physical contacts to 206 registered sites while a zero field need not.

This control is not a truth experiment. Zero motion deliberately omits the real
Luke episodes. Therefore either a large or a small graph gap can reflect field
removal, changed template composition/support, or genuinely distinct units; it
does not by itself establish motion-estimator error. The failed BQ/BW numerical
criterion remains failed and no historical QDA/final-label claim is restored.

Review basis: local DARTsort commit `edcfe1b51d672b4136eb13cc78c0875da804b851`.
No construction, distance, voltage read, GPU job, QDA, or sort was run.

## What the native route actually does

`MotionInfo` derives a motion-dependent registered geometry from the supplied
displacement (`util/motion.py:186-216`); `registered_geometry` explicitly pads
the physical layout to the displacement extent (`util/drift_util.py:44-52,
175-187`). Template construction subsamples before building the reduction
engine (`peel/reduction_template.py:60-81`), passes labels/channels and the grab
chunk setting into that engine (`peel/reduction_template.py:339-357`), and
returns templates plus per-registered-channel counts and registered geometry
(`peel/reduction_template.py:147-175`). In the alternate direct-template route,
raw waveforms are mapped to registered sites using the arm's pitch shifts and
registered geometry, and support counts are mapped through the same operation
(`templates/get_templates.py:570-672`; `util/drift_util.py:253-312`).

For weighted force distance, `template_distances` compresses each arm's template
bank in a shared temporal basis (`clustering/agglomerate.py:337-350`) and then
calls `count_radial_weights` (`clustering/agglomerate.py:377-394`). That function
corrects every retained event's physical channel coordinate at its event time,
snaps it to the nearest registered site, counts by label, expands counts over a
registered-geometry radius, and normalizes per unit
(`clustering/agglomerate.py:742-777`; `clustering/cluster_util.py:385-397`).
Thus motion changes both the template coordinate system/content and the native
distance weights. The result is consumed through the configured threshold and
linkage; independent threshold crossings are not the final force components.

## Frozen paired construction contract

Both arms must begin from the same BZ/D2L post-TMM state and differ only in the
motion object:

- actual arm: the exact saved field, SHA-256
  `ee34e8a6a036b10d77a0bc8563b65d171368e8869be5e96f654ee072c5bb24cf`;
- zero arm: an explicit all-zero displacement on the same time grid, clock
  origin, depth convention, interpolation/extrapolation implementation and sign;
- both: `grab_chunk_length_samples=3000`, identical ordered construction row
  IDs (`b3603e70184dbaa8cf050d54fc77cd9f8723fa70a43198cc3f53e1a1755e8c34`),
  748 labels/units and 299,635 selected events, physical `[182,2]` geometry,
  recording wrapper/dtype, waveform window and padding, median reduction,
  supplied rank-5 TSVD basis/mean, whitening=`none`, RNG state, worker/rank
  assignment, device/dtype, distance settings, threshold and linkage.

The QDA source uses candidates, log likelihoods and responsibilities only
(`util/data_util.py:1501-1524`; `clustering/agglomerate.py:537-571`). The silently
empty `source_proportions` item is therefore a provenance defect to record, but
it is not an input to this construction/distance comparison and is not used by
the shown QDA call. QDA still must not be run from the public BQ packet because
its complete responsibility state and coverage semantics are absent.

### Required equality sentinels before interpretation

1. Hash and compare ordered row IDs, event times/channels/labels, per-unit row
   counts, unit ordering, valid-time exclusions, TSVD arrays, configuration,
   physical geometry, recording identity and computation state.
2. Assert each retained row appears exactly once in both arms and that finite
   template masks/count totals are reported before any distance result.
3. Save each arm's registered geometry and the physical-to-registered mapping;
   do not require their shapes or channel indices to match.
4. Save native templates, per-channel counts, radial weights, spatial IoU,
   symmetric distances, directed shifts, direct threshold mask, linkage-expanded
   mask and component map. Verify symmetry/antisymmetry and strict-upper-triangle
   accounting.
5. Run a zero-versus-zero repeat sentinel with the same frozen membership. Any
   difference beyond the declared numeric tolerance is implementation noise and
   blocks scientific interpretation.

## Two operator outputs to keep separate

### 1. Native geometry/channel-weight operator — primary implementation result

Run the unmodified DARTsort construction and weighted-distance path separately
for each arm. Each arm owns its registered geometry, templates, channel counts,
event-derived radial weights and IoU. Compare final scalar pair distances and
graph decisions by stable unit ID, while retaining each arm's support coverage.
This answers how the actual DARTsort force route changes when motion is removed.

### 2. Fixed common physical support — diagnostic waveform result

Freeze the support rule before viewing arm differences. For each unit/pair,
select physical contact coordinates from the 182-contact input geometry using a
common rule (prefer the intersection of sites with finite, nonzero empirical
coverage in both arms inside the frozen radius; mark insufficient coverage
rather than padding). Map/interpolate both arm templates back to those same
physical coordinates with the same kernel and transform convention. Compute
cosine, norm/amplitude ratio, lagged error and coverage on that support. Preserve
the site coordinates and mappings in the output.

This projection is useful for separating waveform-content changes from native
support changes. It must not replace native force distances, and neither raw
registered indices nor padded-array norms are allowed: both confound the arms'
different registered grids with artificial zeros.

## Interpretation table

| Native force gap | Common-support waveform gap | Bounded reading |
|---|---|---|
| large | small | support/weight/geometry path is influential |
| small | large | waveform changes are down-weighted or outside force support |
| large | large | motion removal changes both construction and force decisions |
| small | small | little sensitivity under this frozen state |

None of these cells establishes that the actual field is wrong, that the zero
field is correct, or that linked units are biologically the same. Report direct
edges and linkage-expanded relations separately and preserve the failed BW/BQ
criterion unchanged.

## Smallest same-state raw-snippet anchor pilot — proposal only

The smallest useful replacement for the withdrawn cross-sort 48 anchors is an
eight-unit, depth-stratified pilot entirely inside the D2L/BQ namespace:

1. Before reading waveform outcomes, select eight stage units across probe depth
   that meet the existing native count/reliability preconditions and are not in
   any disputed force-connected group under either the long or corrected-short
   native graph. Do not join to AW/static IDs.
2. Freeze 100 events from each of two disjoint time halves per unit, balanced by
   5-s block where available. Preserve competing identities and unmatched rows;
   shortfalls are unresolved.
3. Read accepted float32 snippets on at most 48 **physical** contacts covering a
   preregistered radius around the original stage-unit peak site. Move the
   measurement support with each tested depth hypothesis, but score on common
   physical coordinates. Matching localizations may be recorded only as a
   diagnostic.
4. Qualify with the existing waveform-reliability threshold on disjoint halves,
   plus finite energy, amplitude/gain, support-coverage, block-count and nearest-
   competitor margins. Bootstrap 5-s blocks, not events. No reliable winner or
   inadequate coverage means unresolved.
5. Stop after anchor qualification. Do not estimate or fit a residual field.

Maximum logical waveform payload is
`8 units × 2 halves × 100 events × 121 samples × 48 channels × 4 bytes =
37,171,200 bytes` (35.4 MiB), leaving room for tables/figures under the 100 MB
output cap. Expected resources are one reader, two CPU threads, under 2 GB RAM,
no GPU, under 100 MB output and approximately 4–6 minutes CPU after a small
source-alignment preflight. Raw access and this pilot require separate launch
authorization; this review performed neither.

## CD sidecar dependency

At the single allowed availability check, the CD directory still contained its
original packet only; no frozen support/noise/state/replicate sidecar was
present. Therefore pair 1 (parent 91, units 6/9) and pair 5 (parent 12, units
535/538) still satisfy worker-reported inequalities, but their intervals and
finite replicate accounting are not independently reproduced. H5's healthy
sidecar work was not duplicated.

