# DB rematching stage attribution

## Frozen scope

Read-only independent audit of the completed W2 accepted D2L, REMATCH0, and
CD1_FULL artifacts. Attribute what is visible at matching versus final
refinement without rerunning a sort or functional analysis.

The audit must:

- prove whether matching HDF5 rows and final NPZ rows are aligned within each
  arm using row count, time-delta, channel, geometry, frequency, and parent-HDF5
  checks; preserve any bounded final timestamp realignment rather than assuming
  byte-identical times;
- report matching-to-final assignment/noise and label-routing summaries in
  accepted, unresolved, and rest states;
- independently reproduce the ±7-sample, maximum-cardinality time-only
  unmatched-state table from source arrays;
- independently verify the compatible bank's unchanged CE arrays and its
  accepted W2 `prewhiten_postapply` nuisance state;
- keep new matching label namespaces separate and make no biological-identity
  claims.

Limits: 300 additional all-work CPU seconds, two numerical threads, one reader,
20 GB RAM, 250 MB final output, at least 30 GB free, no raw voltage reads, no
GPU, and no sorting. H1 cumulative accounting begins at 17,453.22 seconds.

This finite audit can localize row rejection and label repartition between
matching and final refinement. It cannot isolate a pure coordinate-descent
effect at final output because adaptive localization models differ between the
two rematching arms.
