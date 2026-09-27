# CI: secondary gating of DARTsort force merges

## Verdict

A secondary gate is algebraically feasible, but it must act on **direct force
edges before connectivity expansion**. QDA and optional SpikeInterface routes
remain unchanged and are unioned only afterward. If the secondary rule accepts
no force edges, the result is exactly the existing QDA/SI no-force control—not a
third treatment. A scientifically distinct arm requires a nonempty strict subset
of direct force edges to pass.

The one proposed secondary rule is a precommitted rest-time cross-refractory-dip
test. It is independent of template distance and later ISI benefit. It is a
candidate gating signal, not biological identity truth. No threshold search,
raw read, GPU work, W3 launch, sort or new field was performed.

## Exact source order and graph algebra

Executed DARTsort builds `qda_mask`, a linkage-expanded `force_mask`, and an
optional SpikeInterface mask, then unions them before reclustering
(`clustering/agglomerate.py:169-244`). The saved `force_mask` is therefore too
late for edge-level gating.

The correct order is:

1. define strict-upper-triangle direct force edges as finite template distances
   below the frozen force threshold;
2. evaluate the frozen secondary rule only on those direct edges;
3. symmetrize accepted edges and recompute connected components;
4. expand each component to a force co-membership mask;
5. union that mask with unchanged accepted QDA and optional SI masks;
6. run the existing reclustering/alignment/deduplication path.

`testing/ci_force_gate.py` implements only this graph operation. Its minimal
tests verify route order, exact no-force collapse and indirect reconnection.

On both saved CE arms, the all-pass result exactly reproduces the published
expanded force mask:

| CE arm | Direct edges | Expanded relations | Indirect relations |
|---|---:|---:|---:|
| 30,000 samples | 229 | 476 | 247 |
| 3,000 samples | 222 | 437 | 215 |

The zero-pass result exactly equals accepted QDA plus the identity diagonal in
both arms. Optional SI was absent in this check; if present it is carried
unchanged. Two focused tests pass.

Connectivity can merge endpoints of a rejected direct edge through a path of
accepted edges. Every run must therefore report: accepted direct edges,
force-expanded relations, transitive-only relations, and original rejected
direct edges whose endpoints reconnect indirectly. Net component or edge counts
alone are insufficient.

## One frozen secondary rule: rest-time cross-refractory dip

### Population and clock

- Evaluate only direct force edges from one run's own stage-unit namespace.
- Use unique pre-agglomeration event rows on the fixed pre-recluster sample
  clock, before force alignment and post-merge deduplication.
- Use verified rest time outside the canonical episode/censor mask and its
  padding. Apply only the already-frozen recording-bound eligibility. Do not
  remove evaluated-pair collision or coincident events: that would manufacture
  the dip being tested.
- Require at least 100 eligible events per unit and 20 common nonoverlapping
  5-s blocks. Missing support is `unresolved`, never evidence of biological
  difference.

The rows already inherit matching competition, collision cleaning and prior
deduplication from the saved post-TMM state. Those upstream operations can
suppress zero-lag coincidences and limit biological interpretation; they are
documented censoring, not reapplied or corrected after seeing outcomes.

QDA-request status is not an input. An unrequested or otherwise unknown QDA test
is not a failed QDA test. Accepted QDA edges remain active regardless of the
force gate.

### Statistic and frozen pass rule

For each eligible pair, compute the cross-correlogram on the common rest domain.
At the saved 29,999.759-Hz clock, the exact lag sets are:

- central: integer sample lags `-29..+29`, including zero (59 lag values);
- shoulder: `-89..-45` and `+45..+89` (90 lag values);
- dip ratio
  `R = (central_count + 0.5) / (shoulder_count × 59/90 + 0.5)`.

No evaluated-pair event or lag is removed for collision/coincidence. Counts use
only pairs wholly inside the same eligible 5-s block, with the first and last
89 samples excluded from both trains so central and shoulder have identical
boundary exposure.

Require an expected central count of at least 20. Draw 1,000 common-5-s-block
bootstrap replicates and require all finite. Independently create 1,000 seeded
nulls by randomly deranging complete 5-s blocks of one train **within each
contiguous rest segment**. A segment contributes only when it contains at least
two complete blocks; incomplete blocks are excluded from observed and null
domains alike. Within-block sample offsets are preserved, pairs are counted only
inside the target block, and the same 89-sample edge trim applies. This is
nonwrapping and gives observed and null statistics identical temporal support;
it cannot introduce last-to-first boundary pairs.

The direct force edge passes only when both are true:

1. the 95% block-bootstrap upper bound for `R` is at most `0.50`;
2. observed `R` is at or below the null's fifth percentile.

These thresholds are frozen before any downstream duplicate/ISI/coherence
outcome. They may not be tuned on the same ISI benefit used to evaluate the
arm. Failed numeric criteria mean `secondary_fail`; insufficient events, blocks,
shoulder exposure or finite replicates mean `unresolved`. Both withhold the
force route, but only the former is evidence against this specific dip rule.
QDA/SI may still connect the units.

### Interpretation safeguards

A refractory dip is compatible with fragments of one spike train, but can also
reflect firing suppression, censoring or sorting competition. Passing does not
prove identity; failing does not prove distinct neurons. The two CD compatible
pairs may serve as BQ-lineage implementation safeguards only. They are not
truth labels and must not set thresholds. CE or W3 integer IDs cannot be joined
to those pairs.

## Held-out W3 capture plan

Apply the exact frozen rule once to W3 only after these same-run inputs exist:

- W3 post-TMM row IDs, fixed times, channels and dense stage labels;
- W3 direct template-distance matrix, force threshold and stage unit IDs;
- accepted QDA mask and its requested/completed-status receipt;
- optional SI accepted mask or an explicit absence receipt;
- canonical rest/censor mask on the same fixed clock, recording bounds,
  sampling frequency and 5-s block origin;
- source/config/basis/template/geometry hashes and final actual-clock output.

Before evaluation, write a frozen edge table containing every direct force edge,
eligibility/missingness reason and the rule/version hash. Then publish counts and
IDs for accepted direct edges, unresolved edges, gated components, indirect
relations, rejected-edge reconnections, QDA/SI overlaps and final union. Preserve
row-level clock/dedup accounting as in CE. Do not join W2/CE IDs to W3.

If those saved W3 pre-agglomeration inputs already exist, this is a sub-100-MB,
two-thread, one-reader calculation expected to take under five CPU minutes with
no voltage or GPU access. If they do not, the exact blocker is a same-run W3
post-TMM and route capture; generating it requires a separately managed pipeline
stage and resource preflight. No W3 job is launched by this preparation.

## Recommendation

This is finite and testable and should precede a speculative large-field
feedback loop. Its value is narrower: it asks whether an independently frozen
refractory signature can retain a nonempty subset of force edges without the
known rest-time harm. A zero-edge result is simply the established no-force
control; an all-edge result is the original force arm. Any intermediate result
must be evaluated on held-out W3 with cheap M3/M4/M5 outcomes before RF work.
