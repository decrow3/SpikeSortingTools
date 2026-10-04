# Independent F fixed-event local-refinement feasibility review

## Verdict

**NOT FEASIBLE FROM SAVED ARTIFACTS.**

The saved `matching1.h5` state is sufficient to define a fixed W3 event view,
but it is not a recovery checkpoint for changing local clustering/refinement
context while replaying the executed downstream chain to comparable final
labels.  The exact missing state is:

1. post-PCMerge membership keyed to canonical matching-HDF5 parent rows;
2. post-TMM membership plus the fitted TMM means, bases, neighbourhood
   covariance/EmbeddedNoise parameters, and train/validation membership;
3. post-TMM templates and the agglomeration mapping.

Substituting `dartsort_sorting.npz` final labels for any of these stages is
invalid: those labels are the output after TMM and agglomeration, not the
matching-stage initializer.  Silent regeneration is also invalid because it
reruns PCMerge, refits TMM/noise, and estimates new post-TMM templates.

## Executed stage boundary

At recorded base commit `fdaad62118d778eec63f8e755a0b0f1422574aff`,
`src/dartsort/main.py:217-220` calls `ds_fast_forward`.  The recovery helper
only advances beyond a completed clustering step when intermediate labels were
saved; otherwise a present `matching1.h5` returns the previous sorting at step
1 (`src/dartsort/util/main_util.py:385-434`).  Matching is then called for that
step and the cluster/refinement chain runs before only the final sorting is
saved (`src/dartsort/main.py:333-420`).  There is no later rematching when
`matching_iterations=1`.

The base source constructs refinement wrappers in listed order
(`src/dartsort/clustering/clustering.py:62-107`) and the step configuration
provides pre-refinement, refinement, then agglomeration
(`src/dartsort/util/main_util.py:437-495`).  The producer reports the effective
strategies as PCMerge, TMM, and agglomeration with
`recluster_after_matching=false`; therefore matching labels are the reported
initializer and no DPC reclustering occurs.  The packet binds but does not
embed the effective config or the tracked diff, so H1 could verify this control
flow at the recorded base commit but could not independently inspect the exact
producer-side diff/config values.  That provenance gap prevents a stronger
feasibility claim; it does not rescue missing checkpoint state.

## Hidden fit and voltage dependencies

- TMM builds/bootstraps a fresh mixture and deletes the model/datasets after
  adding final scores (`src/dartsort/clustering/mixture.py:118-260`).  Saved
  candidates, log likelihoods, responsibilities, neighbourhood IDs, and unit
  proportions are outputs, not a serialized TMM checkpoint.  The score writer
  at `mixture.py:5744-5772` stores labels and scores, not means, bases,
  covariance/noise, or split membership.
- When no saved noise model is supplied, `get_truncated_datasets` calls
  `EmbeddedNoise.estimate_from_hdf5` (`mixture.py:4196-4265`).  That method
  reads/interpolates HDF5 residual snippets and estimates covariance
  (`src/dartsort/util/noise_util.py:1054-1097`).  This is waveform-backed
  fitting, not metadata-only continuation.
- `AgglomerateRefinement` requires a recording
  (`src/dartsort/clustering/clustering.py:864-885`).  Agglomeration passes no
  post-TMM `TemplateData`, and `template_distances` therefore calls
  `TemplateData.from_config(recording=...)`
  (`src/dartsort/clustering/agglomerate.py:59-92,281-335`).  The saved matching
  template bank predates the new TMM units and cannot be silently substituted.

## Fixed-event identity and fixtures

The frozen global half-open W3 interval `[239998073,250197991)` maps to
matching-parent rows `[18112500,18603444)`, exactly 490,944 rows.  Independent
checks of the published fixture confirm 62 unique sorted probe rows, first/last
parent rows 18,112,500/18,603,443, every published time inside W3, and all 62
matching labels numerically different from final labels.  The audit source uses
scalar lower bounds without timestamp conversion and direct stored-NPZ row
reads (`audit_fixed_event_view.py:24-91,148-203`).

The 62-row producer checks establish sampled parent/final time and channel
agreement and event-dataset length/shape consistency.  Direct row indexing also
keeps matching labels, template indices, and TPCA rows on the selected parent
indices.  It does not justify reordering events by time or final labels.

The synthetic wrapper correctly rejects whole-view add, drop, and reorder, plus
feature-only reorder (`audit_fixed_event_view.py:94-132`).  It includes time
inversion, duplicate times, crop edges, and a negative label.  However its
validator never checks label values and has no neighbourhood field, and the
real checks only read final neighbourhood IDs without asserting an independent
mapping.  Therefore the published fixture is not fail-closed against
label-only or neighbourhood-only mutation.  This is a correction to the
producer's broad wrapper claim, not evidence that scientific replay is
feasible.  Any future loader must explicitly bind and validate parent row,
time, channel, matching label, feature row, and neighbourhood row together.

## Saved inventory and exact cost to close

Available saved state includes the 21,822,563-row matching HDF5, row-aligned
features and matching labels, matching featurization/template state, motion and
configs, 8,192 residual snippets, and the final sorting.  The packet inventory
reports no post-PCMerge/post-TMM membership files and no serialized TMM/noise
model.  H1 could not directly list producer-local `/home/huklaban5/...` files;
this absence is supported by the producer's metadata inventory and by the
executed save paths, not independently regenerated on H1.

The minimum exact-semantics regeneration is not a thin saved-state loader.  It
must read the 490,944 W3 feature rows, rerun PCMerge, refit TMM and EmbeddedNoise
from residual snippets, estimate post-TMM templates from bounded recording
voltage, and run agglomeration.  It must save parent-row-keyed post-PCMerge and
post-TMM memberships, fitted TMM/noise state, templates, and agglomeration
mapping before interpretation.

The exact lower-bound payloads are:

- W3 TPCA features: 282,783,744 bytes = 269.68359375 MiB;
- residual snippets: 721,616,896 bytes = 688.1875 MiB;
- combined lower bound: 1,004,400,640 bytes = 957.87109375 MiB,
  before other event arrays, fitted models, working tensors, templates, or
  recording voltage.

Historical full-session `cluster1` time was 1,133.1946879187599 seconds on its
configured CUDA path.  That is provenance, not a W3 runtime estimate.  The
required voltage volume, peak memory, GPU condition, and W3 runtime remain
unmeasured and require a separate preflight and frozen regeneration contract.

## Packet integrity correction

All nine manifest members match
`883b35835e30fbce346b337ba74526f95b8c45ef152f002b01f2f578a4e83fc2`,
and the supplied COMPLETE hash matches.  On shared storage, however,
`COMPLETE.json` is older than the other packet members, so v2 is hash-consistent
but does not satisfy the project's strict COMPLETE-last publication rule.  H1
did not mutate the producer packet.  This seal-order defect should remain
visible; it does not change the conservative feasibility verdict.

## Implementation checks

- Done: packet contents -> all manifest member hashes, v1-to-v2 units-only
  correction, decision, inventory, source audit, fixture output, and
  coordination receipt inspected.
- Done: executed boundary -> recorded base-commit `ds_fast_forward`, matching,
  clustering/refinement, final save, PCMerge, TMM/noise, and agglomeration paths
  read directly at cited lines; no CLI-name inference used.
- Done: identity -> W3 parent span, event count, 62-row selection, half-open
  timing, sampled time/channel bindings, and final-label non-substitution checked.
- Done: dependencies -> missing intermediate memberships/model state and hidden
  residual-waveform/noise-fit and recording-voltage/template paths identified.
- Done: fixture scope -> add/drop/whole-row/feature-reorder checks verified;
  label-only and neighbourhood-only fail-closed coverage found absent.
- Done: cost -> feature/residual byte and MiB arithmetic independently checked;
  unmeasured voltage/runtime/resource costs kept explicit.
- Not done: exact producer tracked diff and effective-config content -> hashes
  are published, but the source/config snapshots are not in the packet.
- Not done: direct producer artifact-directory inspection -> paths are local to
  H5; no scientific reads or regeneration were authorized.
- Can establish: saved artifacts do not support exact fixed-event W3 replay of
  the executed downstream chain to comparable final labels.
- Cannot establish: scientific value of local context, final label quality,
  biological identity/purity, or resource sufficiency for regeneration.
