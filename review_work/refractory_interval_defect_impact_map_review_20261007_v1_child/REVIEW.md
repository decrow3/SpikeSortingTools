# Independent review: H1 refractory-interval defect impact map

## Verdict

**REPAIR_REQUIRED** for the exact review scope.

The central interval-defect diagnosis, the R1/R1-T impact, the saved-statistic gap, and the choice of a spike-time-only correction are supported, with the exact R1c-source provenance qualification retained. The packet is not ready to route unchanged because it classifies D0-D2 too coarsely and the correction plan does not freeze the historical ACG pair-inclusion semantics tightly enough.

No corrected scientific outcome was computed. Existing D0 inventory/comparison choice, D1 adapter mechanics, D2 managed synthetic proof, direct 9-29-sample guardrail, Allen-like metrics, and panel measurements conditional on the historical panel are not invalidated by this finding.

## Reviewed snapshot binding

- Snapshot: `/mnt/NPX/Luke/DARTsort_motion_experiments/refractory_interval_defect_impact_map_h1_20261007_v1`
- MANIFEST SHA-256: `337d9da7419843617482c10434927b50c335ede6787803e8b0c2f8117fbf8059`
- COMPLETE SHA-256: `60b27ec71f4f09fc3006253e9246fa2b74f0c62704ba803dc33dca5f8dc10c6e`
- Author member hashes all reproduce the manifest.
- Reviewer: separate child `/root/review_refractory_impact`, host `huklaban1-Precision-5820-Tower`, cwd `/home/huklab/Documents/RyanSorting/SpikeSortingTools`.
- Tool-observed start: `2026-10-07T21:07:39-07:00`.

## Findings

### 1. The observed/model clock mismatch is real in the available frozen source

The available earlier evaluator cumulatively consumes forty ACG bins, models their upper edges as `(1..40)*0.00025` s, and admits modeled centers strictly above 0.5 ms (`full_session_r1_evaluation.py:98-111`). The counter assigns integer lags by `d // 7`, stops at `d >= 280`, and therefore supplies forty half-open 7-sample bins (`:114-133`). At the census-bound clocks, this is:

- imec0: `7/fs = 0.0002333346090260354` s and `280/fs = 0.009333384361041416` s;
- imec1: `7/fs = 0.00023333520649651882` s and `280/fs = 0.009333408259860752` s.

The nominal modeled support is 0.010 s, about 7.142% longer relative to the observed support. These independently calculated values reproduce `SRP_AUDIT.csv:2-13` and the census reconstruction at `SOURCE.py:131-166,366-405`.

The exact R1c executed file with SHA-256 `415f29d7f26102606143d0bd4fbb8ed9a136199a6642e664d50c2a6716b7c423` remains unavailable. The census contract binds that hash and declares the same timing semantics (`CONTRACT.json:32-38,84-99`), but a contract/audit is not a substitute for executed-source inspection. The supplemental recovery ledger (SHA-256 `fdccf32f1a7f4f6b94f40342474578652a63bff6557e15314a709ef77ad3d9b8`) confirms the object is absent from the H1 repository, shared tree, and advertised remote history. Thus the R1c impact is strongly supported but remains explicitly provenance-qualified.

### 2. R1 point values, bootstrap P/K, and verdicts are affected; the separate guardrail is not

The available source feeds the all/q0/displaced ACGs into the inconsistent model (`full_session_r1_evaluation.py:288-318`), reuses it per bootstrap draw (`:323-359`), and constructs P, K, intervals, and the final verdict from those flags (`:473-490`). The census's historical point summaries show the resulting P/K and R1-T readings (`HISTORICAL_DECISIONS.json:115-163,278+`). Direction alone cannot predict a flag or verdict transition.

The 9-29 guardrail is a separate direct inclusive sample counter and uses no sliding-RP edge (`full_session_r1_evaluation.py:136-146`). Allen-like fields are separately computed (`:161-175,304-308`). Their historical values remain valid for this defect, subject to their existing proxy limitations. The rejected earlier R1 packet remains rejected for its separate frozen-amplitude-histogram bootstrap defect (`en_full_session_r1_imec0_rescore_final_review_20261004_v1_h1/REVIEW.md:3-13`); this review does not rehabilitate it.

### 3. Census, selection, and panel classifications are mostly correct

The census derives `srp_status`, failure flags, strict joint patterns, strict prevalence, and through-displacement prevalence from the historical SRP-derived columns (`post_sort_structure_census_h5_20261007_v1/SOURCE.py:234-270,287-341`). Those products are affected and must remain historical.

Selection directly uses `well_isolated_all` for `strict_pass` and the historical-time zero-count/SRP-undefined category (`post_sort_structure_panel_selection_h5_20261007_v1/SOURCE.py:76-124`). Anchor and candidate membership are therefore affected upstream. Panel interval/matching/CCG arithmetic does not implement sliding-RP, but it consumes those selected anchors/candidates and copies historical strict labels (`post_sort_structure_panel_analysis_h5_20261007_v1/SOURCE.py:345-374,400-479`). It remains valid only conditional on the historical enriched panel.

### 4. Blocking defect: D0-D2 must be split into mechanics versus frozen anchor membership

`IMPACT_MAP.csv:11` labels the entire “D0-D2 divergence localization workflow” unaffected. That is too broad.

- D0's choice of comparison, inventory, and saved-output correspondence design is unaffected (`dartsort_divergence_D0_h5_20261007_v1/DECISION.md:3-21`).
- D1 adapter/matcher semantics are unaffected, but the D1 contract explicitly binds the panel-selection packet and freezes four panel-derived imec1 groups (`dartsort_divergence_D1_adapter_h5_20261007_v1/CONTRACT.json:8-15,36-42`). Two groups are directly SRP-derived: `srp_undefined_but_zero_pass_possible` and `strict_pass_control`.
- D2's synthetic managed-runner proof is unaffected, but its production draft freezes the same exact anchors (`dartsort_divergence_D2_runner_preparation_h5_20261007_v1/RUN_CONTRACT.json:46-57`). That selection is affected upstream.

Concrete failure enabled: a scientifically correct correspondence runner could execute on obsolete category anchors and be reported as answering the corrected category question. Repair the map by retaining the engineering mechanics/synthetic receipts while marking D1/D2 scientific neighborhood membership `affected_upstream`; prohibit carrying those exact anchors into production unless the corrected selection is unchanged or the run is explicitly labeled historical-panel-only.

A D3 frozen nonexecuting production contract appeared after the reviewed snapshot was sealed and carries the same anchors (`dartsort_divergence_D3_production_contract_h5_20261007_v1/RUN_CONTRACT.json:1-18,56-60`; completed 21:07:15 PDT, production not started). This is not evidence that the earlier snapshot should have predicted a future artifact, but current coordination must hold/revise its anchor selection while preserving its runner review.

### 5. Saved summaries lack the exact sufficient statistic

The availability ledger marks observed lag vectors and confidence matrices absent for all four probe/arm tables (`AVAILABILITY_LEDGER.csv:33-34,67-68,101-102,135-136`). A confidence matrix is derived and need not be saved. The exact minimum missing data are:

- point correction: per-unit/per-scope (`all`, `q0`, `displaced`) forty-bin integer observed ACG vectors. Spike counts, exposures, and clocks are already published, but must be hash-bound to the same run;
- frozen bootstrap correction: per-unit × 300-s-block × scope forty-bin integer ACG tensor and the matching per-unit × block × scope spike counts, plus the block × scope exposure matrix and original deterministic resample indices (or exact seed/generator/draw contract).

Published minimum-contamination values, flags, P/K, confidence intervals, and `per_draw_deltas` are downstream aggregations and cannot reconstruct unit/draw transitions. `SRP_AUDIT.csv` zero-count calculations diagnose feasibility only; they do not supply unknown observed counts.

### 6. Spike-time-only recalculation is the cheapest adequate action, after two contract repairs

No voltage, waveforms, sorter, RF, or holdout access is needed. One bounded traversal of the frozen spike-time/cluster arrays can emit reusable compact sufficient statistics, after which point and deterministic-bootstrap corrections are cheap.

The correction contract must freeze:

1. sample-derived half-open edges `k*7/fs` through `(k+1)*7/fs`, exact per-probe clocks, and the center eligibility rule (the natural centers `(k+0.5)*7/fs > 0.5 ms` retain bins 2-39 at both clocks);
2. the historical pair-inclusion semantics: within unit, `j>i`, integer `d = t[j]-t[i]`, `0 <= d < 280`, bin `d//7`, same 300-s block; q0/displaced require equal Boolean nonzero-state class; **no exact field-segment gate is present in the historical ACG counter**.

The author's `RECALCULATION_PLAN.md:7` says to preserve “block and field-segment restrictions,” but the actual historical `_lag_counts` does not receive segments (`full_session_r1_evaluation.py:114-133`). Adding a segment gate would change a second factor and defeat a one-defect correction. The contract must say “no segment gate” unless a separately labeled redesign is also run. It must also explicitly decide whether same-sample (`d=0`) pairs remain included: the historical code includes them despite describing the tensor as positive ACG.

Concrete failure enabled: an added segment filter or implicit removal of zero-lag pairs could alter observed counts independently of the clock repair, making old-versus-corrected attribution invalid.

The exact R1c source snapshot is also a prerequisite for claiming a one-factor historical reproduction. If it cannot be recovered, the correction contract must label the implementation as reconstructed and require independent reproduction of every saved unchanged field/point result that can be reproduced before applying the edge change.

## Optional hardening (nonblocking after the above repair)

- Split “unaffected” into `unaffected_implementation` and `valid_only_conditional_on_historical_membership` so engineering receipts are not confused with scientific population validity.
- Record the D1/D2/D3 contract hashes in impact-map provenance rather than citing an unbounded packet search.
- Save the compact ACG/count/exposure sufficient statistics in future R1-like runs; confidence matrices can remain derived.

## Implementation checks

- Done: immutable snapshot integrity -> MANIFEST/COMPLETE and all five member hashes reproduce the dispatched binding.
- Done: parameter-to-consumer trace -> available evaluator shows 7-sample observed bins, nominal 0.25-ms model edges, all/q0/displaced consumers, bootstrap reuse, P/K, and verdict mapping (`full_session_r1_evaluation.py:98-133,288-359,473-490`).
- Done: independent clock fixture -> `7/fs`, `280/fs`, relative mismatch, and eligible sample-derived centers recomputed for both exact clocks; values reproduce `SRP_AUDIT.csv`.
- Done: affected/unaffected downstream trace -> census, selection, panel, D0, D1, D2, and post-snapshot D3 contracts/sources inspected; D1/D2 membership omission found.
- Done: sufficient-statistic audit -> observed vectors/confidence matrices are absent in all four ledgers; derived summaries cannot recover per-unit/per-draw transitions.
- Done: correction-boundary trace -> historical ACG has block/state-class gates but no segment gate and includes possible `d=0`; proposed wording could change a second factor.
- Not done: exact R1c executed-source inspection -> required SHA-256 `415f29d7...` remains unavailable; census contract/audit and earlier source are supporting, not exact provenance.
- Not done: corrected outcomes, raw voltage, waveforms, RF/holdout, sorting, or production launch -> outside scope and not authorized/requested.
- Can establish: the temporal inconsistency in the available implementation; which saved fields and downstream decisions consume it; the missing sufficient statistic; and the cheapest bounded correction with required freeze repairs.
- Cannot establish: exact R1c source equality beyond the frozen declaration, which units/anchors/verdicts change after correction, biological identity/purity, causal arm advantage, or validity of a future production run.

