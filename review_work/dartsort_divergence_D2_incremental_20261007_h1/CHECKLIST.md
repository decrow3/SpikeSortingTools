# D2 incremental review checklist

Preparation started: before 2026-10-07 20:29:23 PDT; exact time was not
tool-observed and the earlier 21:13 prose estimate is withdrawn
Direct review started: 2026-10-07 20:29:23 PDT
Snapshot reviewable at: 2026-10-07 20:27:03 PDT
Final incremental review started: 2026-10-07 20:48:26 PDT
Direct review status: repaired final packet under review

Preserved accepted D1 dependencies

- Final D1 repair review verdict:
  `GO_TO_FREEZE_BOUNDED_EXECUTION_CONTRACT_WITH_GATES`.
- Accepted exclusive matcher and prior 176,400-case independent audit.
- D1 parent-map, event-relation, half-open-support, per-arm-provenance, and
  candidate-ranking source checks, unless their hashes change.
- Historical REF sorter/package/wrapper bytes remain unavailable; require
  empirical array ancestry and do not demand impossible byte recovery.

Frozen production conventions

- Candidate universe: final DARTsort labels >=0 with at least 100 events in
  common half-open support.
- Passing support: at least 25 exclusive matches and smaller-train retention
  at least 0.01.
- Ranking: exclusive matches descending, retention descending, F1 descending;
  exact ties share dense rank; label order is serialization only.
- Report complete eligible table and every passing candidate; no top-N cap,
  post-outcome relaxation, depth/waveform/state/identity gate, or transitive
  cross-pipeline identifier.
- Runner must hard-bind 100/25/0.01 or derive every reported threshold/universe
  field from the effective values and preserve them in provenance.

Contract / launcher interface

- Exact immutable source, config, input, D1 dependency, service-unit, and
  environment hashes.
- Preparation/synthetic command cannot be reused as a production command.
- Explicit execution-enable token/state defaults false and fails closed.
- Launcher validates hashes and frozen thresholds before opening production
  arrays or creating the production output namespace.
- Managed job is independent of chat/terminal lifetime; dummy disconnect test
  for any new manager; persist job ID, command, stdout/stderr, start/end state,
  and exit status.
- State whether interruption restarts the whole bounded diagnostic or resumes
  from an independently verified checkpoint.

Resource / stop policy

- CPU, RAM, threads, wall time, aggregate input bytes, output bytes, and local
  scratch bounds are explicit and enforced where practical.
- No voltage, GPU, fit, replay, sort, RF/holdout, or large transfer.
- Stop before outcomes on hash/schema/clock/row-time/ancestry drift, empty
  common support, resource breach, or any attempt at transitive identity.
- Preserve failure namespace, logs, partial receipts, and exit status; never
  overwrite prior packets.

Required pre-outcome gates

- Full DARTsort final/H5 row counts and `times_samples` equality, not a bounded
  probe; matching labels remain distinct from final labels.
- REF original and curated internal `full_st/full_clu/kept` ancestry validated
  empirically; results explicitly do not reconstruct historical executable
  semantics.
- A exact run-bound exporter/settings and consumed-array ancestry validated.
- Half-open `[0,314204094)` AP-frame-zero clock and inclusive +/-15-sample
  matching preserved.
- Four anchors and three pair names exact; no hidden candidate cap.

Output schema / closure

- DARTSORT_STAGE_EVENTS includes parent_row_id, sample_time, matching_label,
  final_label.
- KILOSORT_PARENT_RELATION includes unique/ambiguous/unmatched status, every
  duplicate-key parent candidate, same-key children, and deleted-key rows.
- PAIR_EVENT_RELATION includes pair/side, pair-local and source indices,
  exclusive partner indices, all-neighbor candidates/degrees, and unmatched.
- PAIR_SUMMARY recomputes exactly from event relation and accepted matcher.
- DARTSORT_CANDIDATE_RANKING preserves complete universe, dense ties, effective
  thresholds, passing set, and no cap.
- SUMMARY and PROVENANCE bind all inputs, actual config, resources, attempts,
  failures, and interpretation limits.

Synthetic managed-fixture review

- Known-answer fixtures must exercise exact threshold boundaries (99/100
  events, 24/25 matches, retention just below/equal 0.01), zero/one/multiple
  candidates, exact ties, input-order invariance, row-time mismatch, ambiguous
  REF keys, schema drift, hash drift, disabled-production attempt, and resource
  stop behavior.
- Final managed-fixture delta can be reviewed incrementally only if D1 adapter,
  contract, and accepted matcher hashes remain unchanged.

Direct-review findings

- Published packet direct rerun: 2 failed, 1 passed. Both intended success and
  hash-failure cases instead hit FileNotFoundError because request payload paths
  point to an ephemeral `/tmp/..._stage/fixtures/synthetic_payload.json` rather
  than an immutable packet-relative/bound staged input.
- Runner accepts policy from any passed contract and does not independently
  compare that contract to an externally frozen reviewed hash before using its
  request whitelist, pairings, support, resources, and dependency paths/hashes.
- Non-fresh output raises inside `run`, after which `main` writes FAILURE.json
  into the already populated namespace; this can overwrite prior failure
  evidence and violates the fresh-namespace/no-overwrite rule.
- Required before managed final evidence: repair all three, rerun direct tests,
  then preserve managed success/failure unit properties, environment, argv,
  journals, outputs, terminal status, and final snapshot-to-seal diff.
