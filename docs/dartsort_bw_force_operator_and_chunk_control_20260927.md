# BW: force-distance inputs and prospective chunk control

## Verdict

The BQ force route is not a function of waveform templates alone. Its exact
input state includes template construction membership and values, temporal
basis, whitening, channel support, event counts, physical/registered geometry,
motion queries, and linkage configuration. Therefore nearest-neighbour
distributions from different matching banks are descriptive but not a clean
registration contrast.

A prospective `30_000` versus `3_000` grab-chunk experiment **can isolate the
template-registration chunk-centre approximation without rerunning pcmerge or
TMM**, if it begins from one fixed post-TMM sorting and freezes every input
listed below. It remains a proposal; no real run was performed.

References are to executed DARTsort source object
`fdaad62118d778eec63f8e755a0b0f1422574aff`, unchanged in the relevant local
`edcfe1b...` files.

## BW.1: exact force-distance operator

`template_distances` consumes `TemplateData`, `TemplateMergeConfig`, and—for
weighted distance—the sorting and motion object
(`clustering/agglomerate.py:281-334`). Its effective inputs are:

- template tensor, unit IDs, per-unit counts, per-channel counts, registered
  geometry, sampling rate, trough offset, temporal SVD, whitener/covariance/
  temporal kernel, and whitening declaration (`templates/templates.py:29-57,
  146-180`);
- the shared temporal basis. If `TemplateData.tsvd` exists its components are
  reused; otherwise the basis is fitted from the full template bank. Templates
  are projected into that basis before pair distances
  (`clustering/agglomerate.py:337-350`; `templates/template_util.py:272-337`);
- the exact whitening strategy and operator. Missing whitening may force
  template reconstruction (`clustering/agglomerate.py:296-334`);
- empirical spatial-support weights. For each retained event, its physical
  channel coordinate and event time are registered through the motion object;
  counts are accumulated by label and registered channel, expanded over a
  geometry-radius neighbourhood, normalized per unit, and then used pairwise
  (`clustering/agglomerate.py:377-394, 742-777`);
- amplitude-scaling prior/bounds, maximum lag, minimum spatial IoU, support
  radius, compression rank, distance threshold and linkage method
  (`util/internal_config.py:439-471`). Per-channel counts also affect template
  masking during construction and SNR-based reference selection after merging
  (`peel/reduction_template.py:147-175`; `templates/templates.py:89-96`;
  `clustering/cluster_util.py:84-101`).

For weighted scaled normalized Euclidean distance, unit-specific weights are
combined by a symmetric minimum-support intersection, normalized, contracted
with lagged template components, and tested for soft IoU
(`util/spiketorch.py:337-447`). The resulting distances are symmetric; shifts
are directed/antisymmetric. Hierarchical clustering consumes only the strict
upper triangle (`clustering/cluster_util.py:104-164`). `linkage_mask` returns
co-membership after the configured hierarchical linkage, not merely a matrix of
independent direct threshold crossings (`cluster_util.py:167-178`). Force is
that linkage-expanded mask at the force threshold, later unioned with QDA
(`clustering/agglomerate.py:222-244`). Exact controls must preserve both the raw
upper-triangle values and the configured linkage semantics.

### Minimum exact-control state

An exact force-route comparison must freeze and hash:

1. event-row namespace, post-TMM times/channels/labels and all GMM properties;
2. exact selected template-construction row IDs per unit, including valid-time
   exclusions and ordering;
3. preprocessed recording identity/dtype, waveform window/padding/alignment,
   physical geometry, registered geometry, motion field and time origin;
4. template reduction/interpolation configuration, template values, counts by
   channel, TSVD basis, whitening matrix/covariance/kernel and their provenance;
5. computation device/dtype, worker/rank assignment and source object;
6. distance parameters, raw symmetric distance matrix, directed shift matrix,
   spatial weights/IoU, force threshold, linkage method, upper-triangle vector,
   expanded force mask and final component map.

BQ's constructed pre-agglomeration template bank differs from the historical
matching banks, while Gate H shows that its post-TMM state is not historical.
Accordingly BQ can support its prospective same-state contrast but not recover
historical force decisions.

Cross-bank nearest-neighbour distance distributions cannot isolate registration:
bank size changes the number of nearest-neighbour opportunities; bank density,
unit splitting/merging, waveform membership, support geometry, basis,
whitening, counts and motion all change the metric itself. A smaller nearest
neighbour distance in a larger/denser bank is partly mechanical. Matching-bank
NN summaries should therefore be labelled descriptive and adjusted or matched
for bank composition before any causal interpretation.

## BW.2: minimal prospective paired control

### Frozen arms

Start once from the exact same in-memory post-TMM sorting. Freeze its labels,
times, channels, GMM arrays, actual template-construction spike row IDs and
their order. Freeze the preprocessed recording, motion field, clock origin,
geometry, waveform/padding/alignment parameters, interpolation parameters,
TSVD basis, whitening state, RNG seeds, computation device/dtype, worker/rank
assignment, and every template/distance/linkage parameter. Disable saved-template
reuse. Construct two template banks differing only in:

- control: `grab_chunk_length_samples = 30_000`;
- intervention: `grab_chunk_length_samples = 3_000`.

Then run the same distance and force-mask calculation on both. Preserve exact
row lineage and compare per-event registered waveforms, template/count arrays,
support weights, distances, shifts, direct threshold crossings, expanded force
mask and final component mapping. This does not require rerunning the prefix;
rerunning it would introduce the very state variation the control is meant to
exclude.

### What the source says changes

Template construction subsamples units before creating the reduction engine
(`peel/reduction_template.py:60-81`). Membership uses an explicit seed and is
independent of grab chunk length (`util/data_util.py:1760-1788`). The reduction
engine receives chunk length only at construction
(`peel/reduction_template.py:339-358`).

Events are assigned to half-open chunk interiors by clipping against
`[start,end-1]`, while waveform extraction reads a margin equal to the larger
side of the waveform window (`peel/grab.py:84-141`; `peel/peel_base.py:530-554`).
Absolute event times and channels are then used to grab snippets
(`peel/grab.py:184-235`). With identical preprocessing and valid event rows,
changing chunk length should not change which raw samples form a waveform.

It **does** change registration: the interpolation transformer receives one
`chunk_center_s` for every event in a chunk, not each event's time
(`peel/peel_base.py:556-580`; `transform/interp.py:49-57`). At approximately
30 kHz, the proposed arms reduce that shared query interval from about 1 s to
about 0.1 s. Chunks start on `np.arange(t_start, t_end, chunk_length)`
(`peel/peel_base.py:817-845`), so the test must fix `t_start=0`, recording time
conversion, last-chunk handling and query origin. Boundary margins must be
verified, and every retained row must appear exactly once in both arms.

Chunking can also alter numerical grouping. Median reduction saves individual
waveforms and takes a per-unit `nanmedian`, making it largely order-insensitive;
mean reduction uses batchwise Welford updates, whose floating-point grouping can
change (`transform/reduction.py:19-24, 63-116, 118-177`). Any TSVD or whitening
refit could add larger sampling/arithmetic differences, which is why both must
be supplied frozen. A tiny no-motion/interpolation-disabled equality control is
useful as an arithmetic sentinel, but it is not itself evidence about field
accuracy.

### Gates and cost risks

Before interpreting force changes, require equal construction row IDs/order,
counts and finite masks; exact raw snippets when interpolation is disabled; and
no unexpected output-row loss or duplication. Report template deltas before
distance/graph deltas. Tenfold shorter chunks increase scheduler, recording-read,
HDF5 and GPU-kernel overhead and may amplify worker-order nondeterminism; total
waveform payload is similar, but wall time and temporary-file overhead need a
dry-run estimate. BW.2 GPU work and any residual loop remain unlaunched
proposals.

The established fact is only that chunk-centre approximation exists and is
experimentally isolatable. It is **not yet causal degradation**. A short-chunk
null cannot by itself show that the field is wrong, BJ's `.989` is not a causal
percentage of any cross-bank gap, and cross-probe co-motion does not establish
that a rigid residual is conservative.

### Timing clarification: 0.1 s diagnostic versus 0.25 s candidate

The deployed field's 0.25 s knot spacing and a matcher's nominal 0.25 s chunks
do not make 0.25 s template-construction chunks automatically equivalent.
`MotionInfo` exposes the original estimator's time bins but delegates every
query to that estimator's `disp_at_s` implementation
(`util/motion.py:254-296`). For the deployed linear interpolant, querying at
0.1 s centres introduces no motion samples or bandwidth beyond those 0.25 s
knots; it evaluates the same piecewise-linear representation closer to each
event time. Thus 0.1 s is the cleaner mechanism diagnostic, while 0.25 s is a
cheaper deployment candidate. The latter is not an additional treatment arm
in this review.

Equal nominal durations are insufficient. Equivalence also requires identical
sample-zero/time origin, chunk starts and centres, final partial-chunk rule,
left/right waveform buffers, motion-query interpolation and edge
extrapolation/quantization, field representation, registered geometry, and
transform direction. Template grabbing computes a centre from the actual
`[chunk_start, chunk_end)` interval (`peel/peel_base.py:556-580`), whereas
matching uses `chunk_start + configured_chunk_length // 2`
(`peel/matching.py:247-276`); their last partial chunks can therefore query
different times even with equal configured durations. Registration to the full
probe shifts target registered geometry by `+disp`, while the inverse full-probe
transform shifts source geometry by `-disp`
(`util/interpolation_util.py:1201-1232, 1264-1292`). A same-duration comparison
must verify that it is exercising the same representation and direction rather
than assuming this from names.

The reviewer supplied, but this audit did not recompute, a corrected summary
for the 0.1 s centre-hold comparison: fraction above 20 µm W2/W3 =
`0.2%/0.0%`, and P95 = `11.2/9.0 µm`, versus a reported `7–11%` above
20 µm for 0.25 s. These values are external context only until their sample
mask, origins, interpolation direction and calculation artifact are reviewed.
They do not establish a lower bound on true motion error: field-relative error
measures disagreement with the field representation, while the unknown true
displacement may differ from both.

For a later residual-loop evaluation, prediction must be frozen on held-out
time: fit/calibrate only on development intervals, predict signed residual
shift and uncertainty for untouched intervals, and score error/calibration
against label-free held-out measurements plus matched nulls. Registration,
identity ambiguity and reference quantization uncertainty must be propagated;
claims should be restricted to intervals where the reference resolves.

## Next dependency

Before any BW.2 launch, the executor must publish a compact preregistration
containing the exact frozen row-ID hash, basis/whitener/template-config hashes,
recording/motion/geometry hashes, field-knot/interpolation identity, chunk
origin/centres/buffers and transform direction, worker assignment, equality
sentinels, output footprint and projected GPU/wall time. The 0.1 s arm is the
preferred diagnostic; a 0.25 s version is a cheaper candidate only after those
equivalence details are explicit. No new prefix run is scientifically required
if that state is available.
