# BV: BQ one-call versus two-call boundary

## Verdict

**Equivalent under explicit state-preservation assumptions; the Python call
boundary itself is not a source of scientific divergence.** At executed source
object `fdaad62118d778eec63f8e755a0b0f1422574aff`, the one-call
pcmerge → TMM → agglomerate route is a nested composition. Splitting it after
TMM gives the same agglomeration operation provided the exact returned sorting
object (including ephemeral GMM arrays), recording, motion, configs, waveform
config, computation config, and template-construction inputs are retained.

This does **not** repair historical Gate H. BQ's prospective state differs from
the historical accepted state before agglomeration, and the historical
pre-agglomeration arrays/templates needed to identify the first divergence are
absent.

## Concrete source trace

References below are to DARTsort Git object `fdaad621...`; the relevant files
are unchanged in the locally reviewed `edcfe1b...` checkout.

1. `clustering/clustering.py:77-105` builds the requested refinements by wrapping
   one `Clusterer` object in the next. There is no combined special-case path.
2. `clustering/clustering.py:719-731` implements every refinement wrapper as
   `inner.cluster(...)` followed immediately by `self.refine(...)` on the
   returned sorting. It passes the same feature, stable-feature, recording and
   motion arguments onward.
3. `clustering/clustering.py:864-882` shows that the agglomeration wrapper does
   nothing scientific besides call the public keyword-only
   `agglomerate.agglomerate(...)` and return `.agglomerated_sorting`.
4. TMM's final assignment attaches labels, `gmm_train`, and
   `unit_log_proportions` to the returned sorting at
   `clustering/mixture.py:236-247`. `DARTsortSorting.ephemeral_replace` makes a
   shallow object copy while retaining the ephemeral-feature dictionary
   (`util/data_util.py:419-439`). Thus separating the calls in the same process
   does not discard or rebuild model-derived features.
5. Agglomeration performs the same QDA/force union, reclustering, score
   combination, deduplication and depth reorder in either invocation
   (`clustering/agglomerate.py:229-258`). The wrapper adds no alternative
   postprocessing. Its only extra behavior is optional intermediate-label
   saving after refinement (`clustering/clustering.py:744-758`), which does not
   alter the returned scientific state.

## RNG and template conditions

The boundary does not save, reset or advance an RNG. TMM sampling creates its
own seeded generator (`clustering/clustering.py:156-187`), and TMM receives an
explicit default seed of zero (`clustering/mixture.py:118-152`). Agglomeration's
QDA traverses the fixed upper-triangle candidate list and contains no random
draw (`clustering/agglomerate.py:537-602`). When templates are constructed,
their low-level API also uses explicit `random_seed=0`, with worker-local seed
`random_seed + rank` (`templates/get_templates.py:127-143, 211-247, 536-548`).
Consequently equivalence requires the same computation configuration and worker
rank assignment, but not continuity of ambient NumPy/Python/Torch RNG state.

There is one API-sensitive caveat:

- The production wrapper passes `template_data=None`
  (`clustering/clustering.py:874-881`). `agglomerate` then flattens the returned
  TMM sorting, including GMM properties, in place
  (`clustering/agglomerate.py:73-81`; `util/data_util.py:932-986`) and constructs
  templates from that state (`clustering/agglomerate.py:83-92, 281-334`).
- A separate call that supplies `template_data=` skips that flattening. It is
  equivalent only if the sorting was already flattened with the same GMM
  remap, the supplied template data was built from precisely that flattened
  sorting with the same configs/worker assignment, and neither object was
  serialized/reloaded or mutated between steps.

For BQ's *prospective* paired arms, Gate P records an identity flatten map
`0..747`, exact sorting equality, equal template hashes, equal sampling lineage
and equal RNG lineage. Those checks make the supplied-template versus internal
template distinction inert **between the two BQ arms**. They do not make the BQ
state historical: Gate H reports different times, labels, GMM/merged arrays,
finite likelihood support, source-to-final map, and 748 versus 760
unit-log-proportion entries.

## Exact remaining missing state

To claim equivalence to the accepted historical result, the missing evidence is
the historical sorting at the TMM→agglomerate boundary: event-row times and
labels; all GMM candidate/likelihood/responsibility arrays and training mask;
unit log proportions; the exact flattened remap; the templates/whitener/TSVD
produced from that state; computation/worker assignment; and the resulting
pre-depth merge map, dedup survivors and depth reorder. BQ does not preserve
that historical boundary state, so source equivalence cannot localize Gate H's
first divergence.

Finally, the h5 BU pair-overlap values `.697/.693` compare the fixed
pre-label-map partition with the historical final partition. They do **not**
compare against a saved prospective-final partition; that prospective-final
array is absent. Those values therefore cannot close the one-call/two-call or
historical-reproduction question.

No synthetic fixture was needed. No BU graph tests, raw/GPU work, real stage,
replay, sort, voltage access, BH/h5 source copying, or service launch occurred.

