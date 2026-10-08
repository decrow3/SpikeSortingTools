# DARTsort divergence production-readiness review checklist

Preparation started: 2026-10-07 20:57:11 PDT (tool-observed)
Direct review: blocked pending immutable production-contract snapshot

Preserved accepted dependencies

- D1 repaired adapter MANIFEST `c32b16fc...4091` and H1 review MANIFEST
  `9e705504...b919`.
- D2 runner preparation MANIFEST `ed866292...b000`, contract
  `6cdb5482...e696`, and H1 review MANIFEST `4f12ca25...bfcc`.
- Accepted matcher `98919857...6c2b` and its prior 176,400-case audit.
- Re-review only changed contract, production input bindings, resource policy,
  production launcher, evaluator, and report consumer hashes.

Pre-outcome scientific interpretation freeze

- Unit of analysis is one of exactly four frozen imec1 A/REF anchor groups.
- Pair relations remain REF--A, REF--DARTsort, and A--DARTsort separately; no
  transitive identity or forced three-way correspondence.
- Event match is chronological maximum-cardinality one-to-one with inclusive
  +/-15 acquisition samples; all-neighbor degree/ambiguity is separate.
- Common support is the declared intersection on AP-frame-zero int64 samples,
  start inclusive and stop exclusive, at 29999.759166666667 Hz.
- Candidate universe is every DARTsort final label >=0 with at least 100 events
  in common support. Passing support is >=25 exclusive matches and
  smaller-train retention >=0.01. Ranking is matches, retention, F1 descending;
  exact ties share dense rank; label is serialization only; no cap.
- Temporal correspondence is descriptive. It does not establish biological
  identity, purity, merge safety, sorter-only causality, or arm superiority.
- Earliest observable divergence may be labeled only from preserved checkpoints:
  Kilosort exported/original versus curated accounting, DARTsort matching-stage
  versus final accounting, or cross-pipeline final correspondence. Missing
  checkpoints remain `unresolved`, never inferred.
- REF historical executable semantics remain unavailable. Empirical internal
  ancestry can validate consumed REF arrays but cannot reconstruct producer
  code semantics.

Exact production inputs and gates

- Bind exact four anchors and their source selection manifest.
- Bind original and curated Kilosort REF/A roots, sort identities, consumed-file
  sizes/hashes, clock/origin/coverage, detection-template semantics, and
  `full_st/full_clu/kept` ancestry.
- Bind DARTsort final NPZ and matching H5 sizes/hashes, config/source/lock, row
  counts, and full-array `times_samples` equality before feature use.
- Reject drift, missing files, out-of-order times, nonintegral times, origin or
  half-open support mismatch, wrong template field, unknown pair, or ambiguous
  parent promoted to unique.
- No amplitude/depth cross-pipeline identity gate; no voltage, GPU, fit, replay,
  sort, RF/holdout, or large transfer.

Evaluator and report consumer

- Consumer reads only sealed runner outputs whose manifest and COMPLETE verify;
  refuses FAILURE/COMPLETE coexistence, missing members, unmanifested scientific
  tables, schema drift, duplicate primary keys, or nonfinite required metrics.
- Recompute stage and pair summaries from event-level tables; verify candidate
  ranks, dense ties, support flags, threshold metadata, and complete universe.
- For every anchor/pair report denominators: input rows, in-support rows,
  exclusive matches, unmatched each side, any-neighbor counts, degree>1 counts,
  unique/ambiguous/unmatched parent counts, and eligible/passing candidates.
- Report every passing candidate and the complete eligible ranking, including
  zero-pass cases; never select by inspected outcome or suppress ambiguity.
- Earliest-divergence classification must be deterministic from frozen fields,
  preserve ties/unresolved states, and cite the exact checkpoint evidence.
- Scientific report contains strongest counterexample/negative control, four
  group results, data-quality failures, resource receipt, provenance, and
  interpretation limits. No automatic intervention, merge, deletion, or
  production-sorting recommendation.

Output closure

- DARTSORT_STAGE_EVENTS: unique parent_row_id; exact row count and times;
  matching_label and final_label distinct fields.
- KILOSORT_PARENT_RELATION: one row per curated event; status closure; every
  candidate parent retained; deleted-key summary reconciles originals.
- PAIR_EVENT_RELATION: unique pair/side/local index; symmetric exclusive partner
  links; complete all-neighbor lists/degrees; unmatched consistency.
- PAIR_SUMMARY recomputes exactly from event rows and accepted matcher.
- DARTSORT_CANDIDATE_RANKING recomputes from stage events and anchors; no hidden
  candidate cap or threshold override.
- SUMMARY/REPORT claims must be machine-derivable from sealed tables; report
  cannot silently reinterpret missing/ambiguous fields.

Production launch and resources

- New immutable production contract and request; literal pre-parse contract
  hash; exact runner/evaluator/adapter/matcher hashes; fresh namespace.
- H5 final dedup/process/input/resource reconciliation must show no competing
  run, all inputs local/readable, enough disk/RAM, and aggregate read/output
  bounds adequate for actual selected datasets and columns.
- Managed independent job with exact unit properties, environment, argv,
  working directory, job ID, journals, terminal state, resource accounting,
  and COMPLETE last. Each retry uses a new namespace; interruption behavior is
  explicit and failures preserved.
- H1 GO can authorize only the preassigned one-shot saved-output launch after
  H5 reconciliation; H1 never launches it.

