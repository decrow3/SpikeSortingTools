# CJ: saved-W2 secondary force-gate result

## Verdict

The preregistered fixed-clock rest-time diagnostic retained **6 of 222** direct
force edges. It rejected 28 resolved edges and left 188 unresolved under the
frozen support requirements. Those unresolved edges were not treated as
failures and no threshold was relaxed. Recomputing connectivity from the six
accepted direct edges produced six force relations and no transitive-only
relations. After union with the unchanged accepted-QDA route, the saved-state
graph has 21 upper-triangle relations: 16 QDA relations plus five new force-only
relations (one accepted force edge was already accepted by QDA).

This is a finite **saved hard-partition/graph diagnostic**, not an observed
post-aggregation benefit. Actual aggregation, reassignment and deduplication
were deliberately not run, and W3 was untouched.

## Frozen method and result

The methodological correction was committed as `07811bc` before examining CJ
outcomes. Events use the saved fixed pre-recluster sample clock and hard
partition. Evaluated pairs receive no additional collision/coincidence event
filter; the inherited matching competition, collision cleaning and
deduplication remain part of the saved upstream state. Only complete 5-s rest
blocks are used, trimmed by 89 samples on each side.

| Quantity | Frozen rule | W2 result |
| --- | --- | ---: |
| Direct force edges | strict upper triangle | 222 |
| Complete rest blocks | 5 s, identical trimmed support | 48 |
| Resolved and passed | bootstrap upper <= 0.50 and observed <= null q05 | 6 |
| Resolved and failed | same fixed rule | 28 |
| Unresolved: common blocks | fewer than 20 | 39 |
| Unresolved: events | fewer than 100 for either unit | 1 |
| Unresolved: expected central | fewer than 20 | 148 |
| Accepted force relations | connectivity recomputed after gating | 6 |
| Indirect-only force relations | accepted connectivity minus direct | 0 |
| Rejected direct edges reconnected | accepted paths | 0 |
| Accepted QDA relations | unchanged | 16 |
| Final union relations | gated force union unchanged QDA | 21 |

The central window is exactly sample lags -29 through +29, including zero (59
lag values). Shoulders are absolute lags 45 through 89 (90 lag values), so the
expected central exposure is shoulder count multiplied by 59/90. Each resolved
edge uses 1,000 block-bootstrap draws and 1,000 within-segment, nonwrapping
5-s-block derangements with identical support. Every segment included in the
null has at least two complete blocks.

The six accepted direct edges are:

| Units | Blocks | Events i/j | Observed ratio | Bootstrap upper | Null q05 | QDA accepted? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 / 4 | 48 | 1696 / 1152 | 0.0230 | 0.0564 | 0.7707 | no |
| 151 / 158 | 48 | 3186 / 1156 | 0.2064 | 0.3161 | 0.8176 | no |
| 212 / 216 | 48 | 1285 / 3311 | 0.0428 | 0.0845 | 0.8141 | yes |
| 270 / 271 | 34 | 1171 / 1307 | 0.0066 | 0.0098 | 0.6405 | no |
| 277 / 653 | 48 | 1890 / 704 | 0.1811 | 0.3301 | 0.6364 | no |
| 324 / 337 | 37 | 759 / 334 | 0.0434 | 0.0936 | 0.2027 | no |

## Controls and validation

The all-pass control exactly reconstructs CE's saved 3,000-sample expanded
force mask. The zero-pass control exactly equals QDA-only/no-force. The observed
gate is not the zero-pass control. An independent post-run check verifies all
five product hashes and sizes, all four saved graph arrays, all 222 edge rows,
and all 48 exact-duration rest blocks. Every statistic on a resolved edge is
finite.

The focused fixture verifies zero and +/-29 as central, +/-30 as neither,
+/-45 and +/-89 as shoulders, and +/-90 as outside; it also verifies a
nonwrapping derangement has no fixed points. The helper plus CJ fixture report
`3 passed`.

The first command-form launch failed before output creation because executing
the file directly did not expose the repository package root. The unchanged
frozen code was then launched with `python -m testing.cj_force_gate_w2`. No
scientific parameter changed and no partial shared output existed to reuse.

## Outputs and scope

Shared output:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cj_force_gate_w2_20260927/`

- `EDGE_RESULTS.csv`: all 222 direct edges, including unresolved reasons.
- `REST_BLOCKS.csv`: the 48 complete fixed-clock rest blocks.
- `GATED_ARRAYS.npz`: pass mask, accepted-direct mask, recomputed force mask,
  unchanged-QDA union and component IDs.
- `SUMMARY.json`, `RECEIPT.json`, and `COMPLETE.json` (written last).

Execution took 22.17 s with two-thread limits. CJ conservatively charges
300/1,800 CPU seconds for implementation, validation, execution and review,
taking cumulative h1 usage from 12,233.22 to **12,533.22/14,400 s**. Output is
about 41 KB. There were zero raw/voltage reads, GPU seconds, W3 launches, sorts,
aggregation/dedup executions or threshold searches.

## Interpretation boundary

The result supports retaining six direct W2 force relations for a future
precommitted secondary treatment. It does not demonstrate that those relations
improve ISI, duplicates, coherence, identity continuity, or final unit yield.
Those claims require the separately authorized actual aggregation/dedup outcome
and, for generalization, the held-out W3 capture described in CI.
