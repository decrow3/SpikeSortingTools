# BU: independent review of the completed BQ W2 capture

## Verdict

**Share with caveats.** The saved arrays support a valid *prospective,
current-state* force-route counterfactual: Gate P passed, row correspondence is
fixed, route masks reconstruct, graph closure matches the saved component maps,
and permutation-invariant accounting reproduces 85 current groups split across
284,832 event rows when force is removed. They do not support attribution to the
accepted historical output, motion specificity, neuronal yield, or biological
continuity.

Gate H is a substantive exact-reproduction failure, not merely arbitrary final
label numbering. It reports identical row count, channels and training mask,
but different times, labels, GMM/merged candidates, likelihoods,
responsibilities, source-to-merged map and rest 9–29-sample ISI result;
likelihood finite masks differ and unit-log-proportion shapes are 748 versus
760. However, only equality/shape/finite-support summaries were published for
H. The compared arrays and mismatch magnitudes were not saved, so the review
cannot say whether most differences are tiny or widespread, compute a
permutation-invariant H partition similarity, or localize the first divergence.

The correct strength of the caveat is therefore:

- **strong rejection of exact historical reproduction and historical route
  attribution;**
- **no quantitative grade of H mismatch magnitude from this package;** and
- **no invalidation of the internally paired P contrast**, which freezes one
  newly constructed current state and shows what force removal does within that
  state.

H used its own template construction. P's separately frozen current state must
not be substituted for the accepted historical state.

## Integrity and source identity

The requested hashes match exactly:

| Artifact | SHA-256 |
|---|---|
| `RESULT.json` | `d268fcb9607f0bce10ac61e12603b187242eddf54dd30f6d4423d8cd8de84f2c` |
| `capture/arrays.npz` | `234b6ebb70fe6b913e1378723245f78c0893353cf2e524f2e7f9a511d9f96ae8` |
| `contrast_arrays.npz` | `636cada9b7fd6d1da82bd3a25976ae760b9870dcbd5b18f3226bfbbc8feab9b4` |

Every file listed in `COMPLETE.json` matches its recorded individual hash and
size. Its aggregate `bytes` field is low by 1,891 bytes: exactly the 129-byte
preflight plus the 1,762-byte publication manifest, even though both appear in
the file list. This is a bookkeeping defect, not evidence of array corruption.

The capture records executed DARTsort Git object
`fdaad62118d778eec63f8e755a0b0f1422574aff`. The locally reviewed later commit
`edcfe1b51d672b4136eb13cc78c0875da804b851` has no differences from that commit
in `agglomerate.py`, `cluster_util.py`, `motion.py`, or `internal_config.py`.
Installed DARTsort 0.5.16 remains a distinct implementation and must not be
treated as the executed source.

## Independent saved-array checks

### Rows, IDs, maps and noise

- 641,588 immutable event-row IDs are exactly `0..641587`; the event arrays all
  share that length.
- Stage unit IDs are dense `0..747`; pre-stage labels contain 748 positive units
  and one `-1` row.
- The actual pre-depth merge map has 551 components; the 551-entry depth reorder
  is a permutation; and composing the actual map with that reorder exactly
  reproduces `merge_mapping_final`.
- `fixed_current_labels` exactly equals the actual **pre-reorder** merge map
  applied to the pre labels. It is partition-equivalent, but numerically unequal,
  to the final depth-reordered labels. The contrast file does not declare this
  map namespace, so consumers must not infer it from label values.
- `fixed_no_force_labels` exactly equals the saved no-force component map applied
  to the same pre rows. Current and no-force fixed arrays retain the identical
  single noise row. This is event conservation on the fixed comparison clock.

### Routes and closure

All masks have the expected diagonal/symmetry convention. Both saved condensed
vectors exactly equal the upper triangles supplied to linkage. Reapplying the
source float32 preprocessing to the raw distance matrix exactly reproduces the
distance (`0.6`) and force (`0.3`) linkage inputs. The raw matrix contains
535,338 positive infinities and no NaNs or negative infinities; these are source
sentinels, not missing data.

| Quantity | Independent count |
|---|---:|
| QDA scheduled upper pairs | 2,494 |
| QDA accepted pairs | 15 |
| Direct force edges | 236 |
| Direct force edges not QDA-accepted | 232 |
| Force-linkage expanded connections | 611 |
| Expanded force connections not QDA-accepted | 606 |
| Current union edges | 621 |
| Current final-closure pairs / components | 631 / 551 |
| No-force union edges | 15 |
| No-force final-closure pairs / components | 18 / 733 |

The counts preserve overlaps: four direct force edges and five expanded force
connections are also QDA accepted. Thus `236 + 15` is not a valid union count.
Current union is exactly QDA OR expanded force; no-force union is exactly QDA
plus the diagonal. Connected components of those masks reproduce the saved maps.

The QDA schedule is exactly the upper triangle of the requested mask, and every
accepted edge was requested. Completion/status and IoU/coverage availability
are explicitly not passively observed, so zero-valued score arrays cannot be
used to count rejection or completion.

### Partition and adjacency

An O(N) contingency calculation, insensitive to arbitrary component numbers,
finds 85 current groups split, zero no-force groups merged, 733 non-noise
contingency cells and 284,832 affected rows—matching the report. A minimal
counterexample explains the inference limit: if reconstructed templates create
a force edge A–B that the historical templates did not, removing force can
cleanly split A/B in P even though it says nothing about how the historical
accepted output connected them.

On the single fixed pre-stage clock, without crossing unit boundaries, the
saved rows give:

| Partition | Adjacent pairs | ISIs in inclusive 9–29 samples |
|---|---:|---:|
| Pre-stage | 640,839 | 2,882 |
| Current force-on | 641,036 | 7,406 |
| No-force | 640,854 | 2,900 |

These are whole-window fixed-clock checks, not the reported state-stratified
metrics. The published stage-aware rest/episode values (current 6,894/621;
no-force 2,707/193) cannot be independently recomputed: the package lacks the
canonical catalogue/AS-padded state masks and lacks current post-stage
times/labels. It saves only the no-force stage-aware arrays. The one-event
difference in published current versus no-force state totals therefore also
cannot be traced here. No clock origin or bounds are declared.

## What stronger validation is and is not possible

The package does save finite templates `[748,121,206]`, spike/channel counts,
registered geometry `[206,2]`, T-SVD components and sampling frequency
29,999.7591667 Hz. That is enough for template-level geometry, distance and
changed-group visualization on the newly constructed P state. It is not enough
for biological identity or continuity: there are no competing-identity event
assignments, per-event waveforms, accepted historical template arrays, original
geometry/motion field, or matched historical clock lineage. The saved T-SVD
explained-variance vector is entirely NaN and cannot strengthen that evidence.

Consequently, the rise in units active in both reported states from 544 to 724
is a partition count only. It is not evidence of 180 recovered neurons or
improved episode continuity. D2L-only BQ also cannot establish motion
specificity; an S/W2 arm remains a separate unapproved experiment.

## Resources

BU read only the new BQ package in place and local DARTsort source. It did not
access BH, copy h5 worker source, use raw voltage/GPU, replay, sort, or integrate
an engine. Known active process wall before receipt finalization was 13.4 s; a
conservative **30/900 s** is charged, taking cumulative h1 from 6833.22 to
**6863.22/14400 s**. Outputs remain under 100 MB. BU stops here pending h5's
separate fixed-versus-stage-clock and fragmentation audit.
