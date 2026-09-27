# CJ v2: corrected saved-W2 secondary force gate

## Verdict

The corrected, precommitted v2 gate retains **6 of 222** direct force edges.
Only 15 edges resolve under the frozen support rules: six pass, nine fail and
207 remain unresolved. No threshold was changed. Recomputed connectivity gives
six force relations with no indirect-only relations; union with unchanged QDA
gives 21 final relations because one passing force edge is already accepted by
QDA.

The intended W2 graph happens to equal v1's graph, but the evidence accounting
does not: excluding inherited dedup-censored lags 0--8 reduces resolved failures
from 28 to 9 and raises unresolved edges from 188 to 207. V1 remains preserved
as exploratory and must not be used as the intended gate.

## Corrected method and result

Method commit `48868fc` was created before the v2 outcome. The central annulus
is exactly `9 <= abs(lag) <= 29` samples (42 lag values), shoulders are exactly
`45 <= abs(lag) <= 89` (90 values), and expected central exposure is shoulder
count times 42/90. There is no evaluated-pair collision filter.

Complete 5-s rest blocks come from the exact complement of the canonical AE
mask within W2's exact `[0,10199918)` local-frame interval and 900-s session
origin. The mask SHA-256 is
`86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55`.
No boundary is inferred from spike extrema.

| Quantity | Frozen rule | V2 result |
| --- | --- | ---: |
| Direct force edges | strict upper triangle | 222 |
| Exact rest blocks | complete 5-s blocks in mask complement | 48 |
| Retained source rows before edge trim | saved fixed clock | 492,117 |
| Retained censored rows | must be zero | 0 |
| Resolved and passed | bootstrap upper <= 0.50 and observed <= null q05 | 6 |
| Resolved and failed | same fixed rule | 9 |
| Unresolved: common blocks | fewer than 20 | 39 |
| Unresolved: events | fewer than 100 for either unit | 1 |
| Unresolved: expected central | fewer than 20 | 167 |
| Accepted force relations | connectivity after direct-edge gating | 6 |
| Indirect-only/reconnected relations | report both | 0 / 0 |
| Accepted QDA/final union relations | unchanged QDA then union | 16 / 21 |

The passing edges remain units 2/4, 151/158, 212/216, 270/271, 277/653 and
324/337. Unit pair 212/216 is also QDA-accepted; the other five are force-only.

## Source and calculation validation

The CK interval packet's manifest, sizes and hashes pass before execution. Its
canonical mask labels exactly the same 81,156 saved source rows as CE's fixed
episode state. Independently, all 492,117 rows retained by complete blocks are
rest rows, and every block lies wholly within the exact mask complement and W2
frame bounds.

All five v2 product hashes and sizes pass. The four saved graph arrays reproduce
from the direct-force, secondary-pass and unchanged-QDA masks. All-pass exactly
reconstructs CE's saved expanded-force mask; zero-pass exactly equals QDA-only.
The corrected boundary fixture checks that ±0--8 are excluded, ±9 and ±29 are
included, ±30 is excluded, ±45 and ±89 are shoulders, and ±90 is excluded. A
separate off-grid fixture verifies exact half-open mask conversion and complement
tiling. Four focused tests pass.

The block-derangement reference is a same-genuine-segment, nonwrapping,
shared-support control. It is not a biological null: nonstationary firing rates,
matching competition and inherited missingness remain possible explanations.
Passing is compatible with fragments of one spike train but does not prove
identity. The 207 unresolved edges also show that W2 alone has low coverage for
this gate.

## Frozen W3 validation scope

The next validation is held-out W3, with no W2/W3 integer-ID join and no
threshold search. Before viewing W3 outcomes, capture from one W3 run:

1. unique post-TMM rows, fixed sample times, dense stage labels and the exact
   recording frame bounds/session-time origin;
2. W3 direct force edges plus force threshold and stage-unit namespace;
3. accepted and requested QDA masks, optional SI mask or an explicit absence;
4. the canonical censor-mask hash and a row-level assertion that retained block
   rows are rest;
5. source/config/template/basis/geometry hashes and final actual-clock output.

Then apply commit `48868fc` unchanged: the same annular lag windows, exact block
construction, event/block/exposure minima, 1,000 bootstrap draws, 1,000
within-segment derangements, and two fixed pass inequalities. Publish every edge
including unresolved reasons; accepted direct, indirect-only and reconnected
relations; QDA/SI overlaps; and all-pass/zero-pass controls. A zero-pass result
is the established no-force control, not a new treatment. W3's required same-run
route capture is not currently present on the shared experiment filesystem, so
execution is blocked without launching or reconstructing a sort stage.

Actual aggregation/deduplication and cheap held-out M3/M4/M5 outcomes require
separate authorization and are the only basis for a downstream benefit claim.
The graph-only result cannot establish ISI, duplicate, coherence or identity
improvement.

## Outputs and accounting

Corrected shared output:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cj_force_gate_w2_20260927/v2/`

The calculation took 24.33 s with two-thread limits. This repair conservatively
charges 300/900 CPU seconds, taking cumulative h1 usage from 12,533.22 to
**12,833.22/14,400 s**. The prior CJ 300-s charge remains counted once. There
were no raw/voltage reads, GPU seconds, W3 launches, sorts,
aggregation/deduplication runs or threshold searches.
