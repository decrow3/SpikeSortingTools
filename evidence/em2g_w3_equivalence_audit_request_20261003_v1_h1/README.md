# EM.2g exact full-vs-W3 rounded-kriging equivalence support request

This is a bounded metadata-only request for huklaban5. It binds the exact full
sorting SHA-256 `3b324bbc...de660` and exact W3 window sorting SHA-256
`e82b0402...ec1b`. The corresponding exact-window execution evidence was not
found in the shared project packets on huklaban1.

The completed `em2g_full_validation_20260929` packet is useful but not an
equivalent audit: its W3 comparator is exact-lattice SHA `901de0...`, with 485
active and 394 eligible units, rather than the requested rounded-kriging W3
window sort. Its report already warns that independently trained models differ
in population and that event overlap cannot isolate the correction operator.

The only proposed candidate test is saved-artifact global-frame equivalence.
No voltage, sorting, scoring, bootstrap, plotting, RF/holdout access, target-A
arrays, or large-array transfer is requested.

## Implementation checks

- Done: exact-hash reuse search -> no shared packet containing W3 SHA
  `e82b0402...ec1b` was found under the authorized shared root.
- Done: closest completed packet inspection -> full SHA matches, but its W3
  comparator SHA and operator differ; it cannot answer this audit.
- Done: recovery/report inspection -> full execution used SI 0.104.8 and
  DARTsort `0.5.23.post4+g6c0566b2`; a complete matching checkpoint was reused
  after systemd-oomd, then clustering/refinement restarted.
- Not done: exact two-run config/source/field/geometry/preprocessing/cache diff
  -> the exact W3 window execution artifacts are only available on huklaban5.
- Can establish: the precise minimal evidence needed to decide whether an
  implementation/input mismatch exists without another sort or voltage read.
- Cannot establish: that the rho difference is caused by the correction
  pipeline, or that either population is biologically more valid.
