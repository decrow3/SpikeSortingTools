# BR: independent BQ worker orchestration review

## Verdict

The prepared BQ concept is not launch-ready on API repair alone. A future worker
must bind every executed call against the exact source, run the accepted
pcmerge → TMM → agglomerate prefix, flatten exactly once, and carry the actual
recluster mapping through the actual depth permutation. Route and outcome
accounting must be permutation invariant and state stratified. This review did
not launch a service or access the h5 payload.

The smallest safe retry recommendation is **zero now**. After h5 reconciles the
durable BQ ledger and its end-to-end mocked worker passes all assertions below,
the coordinator may request exactly one additional infrastructure launch within
the unchanged BQ budgets. A scientific gate failure is not retryable.

## Executed-source contract

Reviewed checkout: `/home/huklab/Documents/DARTsort`, Git
`edcfe1b51d672b4136eb13cc78c0875da804b851`.

| Contract | Source evidence | Consequence |
|---|---|---|
| Motion construction is keyword-only | `util/motion.py:163-171` | `geom=` and the selected motion argument must be named. |
| Agglomeration is keyword-only | `clustering/agglomerate.py:59-70` | Signature binding must precede any service. |
| Refinement wrappers are ordered | `clustering/clustering.py:77-105` | The effective chain, not only nested config values, must be recorded. |
| Accepted defaults include pcmerge, TMM, then agglomerate | `util/internal_config.py:915-926, 944-948` | Omitting the pcmerge/TMM prefix changes the starting state. |
| Matching-path behavior depends on `recluster_after_matching` and final/subsample state | `util/main_util.py:437-495` | Record the actual branch and all effective waveform/refinement configs; do not infer it from defaults. |
| No supplied templates causes flattening with GMM-property remapping; supplied templates skip it | `clustering/agglomerate.py:77-81`; `util/data_util.py:932-986` | Templates and sorting must share one dense ID domain with explicit sparse→flat lineage. |
| `recluster` requires dense template IDs | `clustering/cluster_util.py:25-59` | `template_data.unit_ids` must equal `arange(K)` in the flattened state. |
| Final union is QDA OR force OR optional SI, followed by binary reclustering | `clustering/agglomerate.py:229-244` | Direct route edges and linkage-expanded co-membership must stay separate. |
| Actual merge mapping is composed with actual depth reorder after dedup | `clustering/agglomerate.py:246-255` | A component map re-derived from the union is diagnostic only, never persisted as the actual map. |
| Dedup uses score priority, marks discarded events `-1`, stably sorts only when needed, and restores input order | `clustering/agglomerate.py:947-993` | Gate P must preserve event-row lineage, times, labels, dedup survivors, and ordering. |

The executed agglomeration uses merge cutoff `0.6` and force cutoff `0.3` in the
accepted configuration. `TemplateMergeConfig.cross_merge_distance_threshold`
defaults to `0.5` (`internal_config.py:447-448`) but is not read by this
agglomeration path. The `0.5` at `agglomerate.py:242` is instead the threshold
for reclustering a binary final mask. A receipt must not conflate them.

## Version differences

The checkout and installed DARTsort 0.5.16 are not interchangeable:

- checkout `agglomerate.py` SHA-256:
  `3e3b5533f0c9d1508092672407f0d7972de43584bc4842d6f3cd4e3232dfd345`;
- installed 0.5.16 file SHA-256:
  `55b7285c62a708155b82bdac252c9ac0c2e87547da4fb1bae070c01c662ab34b`;
- the checkout flattens and depth-reorders in place, while 0.5.16 uses copied
  state;
- the checkout defines `reference_peak_minus_source_peak_v1` and stores
  `-best_lag`; installed 0.5.16 stores `best_lag` without that convention.

Therefore the worker must record the imported file identity and shift
convention, not merely a package version. Direct imports were deliberately not
used for validation: the default interpreter lacked KDEpy, and the challenger
environment import hit a SpikeInterface/Numba cache-locator error. Both failures
were preserved; AST/source checks then established signatures without executing
DARTsort.

## Accounting semantics

- Sparse pre-IDs are flattened once and retained in a flat→original table.
- The actual pre-reorder map returned by reclustering is composed with the
  actual reorder. A separately derived union partition must carry its own label
  crosswalk.
- Changed-event/group counts use an O(N) contingency table. Pure label renaming
  changes zero events and zero groups.
- QDA source zeros are values, not completion evidence. Without passive status,
  scheduled pairs remain unknown; only explicit completed status can support a
  completed-rejection count.
- Direct threshold edges, linkage-expanded co-members and final component
  changes are separate quantities. In the three-node chain fixture there are
  two direct force edges but three expanded co-member pairs.
- Rest and episode ISIs use frozen catalogue/AS-padded half-open segments on the
  original sample clock. Adjacent pairs never cross a segment/state boundary;
  each state reports spikes, adjacent-pair denominator and violations. The
  broad censor mask is not a substitute.
- Missing required arrays in both arms fail Gate H. Recorder-off/on equivalence
  also requires shared serialized RNG, template/model construction and waveform
  state. It is prospective conditional evidence only, not historical
  reproduction.
- D2L-only BQ cannot establish motion specificity. S/W2 remains an unapproved
  later arm and was not added.

## Targeted qualification

`pytest -q testing/test_bq_worker_contract_br.py`: **9 passed in 0.23 s**.
No old suite was rerun. The tests cover the two observed API classes, exact
source-version shift semantics, prefix omission, sparse flatten lineage, actual
map/reorder composition, permutation invariance, state-boundary ISIs,
missing-both equivalence failure, QDA unknowns and direct-versus-expanded force
accounting.

BR used no GPU, raw/real data, sorting, waveform execution, QDA execution,
engine integration, h5/BH payload or service launch. Known process wall was
13.9 s before receipt finalization; a conservative **30/600 s** is charged,
taking cumulative h1 from 6803.22 to **6833.22/14400 s**. BI remains separate.

## Required end-to-end dummy assertions

The next mocked worker must assert, in one traversal:

1. imported source path/hash and keyword binding match the recorded checkout;
2. effective stages are exactly pcmerge → TMM → agglomerate, with
   `recluster_after_matching=false` and resolved waveform/config values;
3. disabled/enabled arms receive identical serialized RNG, template/model,
   waveform, sparse event-row and motion states;
4. sparse IDs flatten once; templates use that same dense domain and no hidden
   template regeneration or whitening occurs;
5. missing required Gate H fields fail even when absent from both arms, while a
   deliberate mismatch stops before force contrast;
6. actual recluster map, actual reorder, input immutability, dedup survivor/drop
   lineage, times and row order pass Gate P;
7. a pure label permutation yields zero changed events/groups, while a true
   split/merge is detected by contingency accounting;
8. zero-valued QDA arrays with unavailable completion status remain unknown;
9. direct `.3` force edges, expanded connections and final component changes
   are reported separately; `.6` merge and inactive cross-merge `.5` are named;
10. rest/episode counts reproduce the frozen catalogue/AS domains and ISI pairs
    do not cross state or contiguous-segment boundaries;
11. episode continuity/recovery, competing-identity availability and all
    prespecified cheap metrics are present or explicitly unavailable—lower ISI
    alone cannot support a biological claim; and
12. an early exception writes durable timing/failure accounting, while guarded
    publication writes validated schema, hashes and `COMPLETE` last and stays
    within cumulative time/RAM/disk/output bounds.
