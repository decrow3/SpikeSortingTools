# H1 final incremental review — D1 adapter repair v2

## Verdict

**GO_TO_FREEZE_BOUNDED_EXECUTION_CONTRACT_WITH_GATES.**

The four D1 repair requirements are implemented and independently reviewed.
This verdict permits the coordinator to freeze a separate bounded saved-output
execution contract. It does not authorize production-array traversal, voltage,
replay, sorting, RF/holdout work, or a production run by itself.

The repaired adapter now separates A and REF provenance without inventing
unrecoverable REF executable bytes; preserves unique, ambiguous, and unmatched
curated parent relationships; emits explicit DARTsort parent/matching/final
labels and event-level pair relations; and freezes an uncapped deterministic
DARTsort candidate rule with dense ties and minimum support. Synthetic tests
and independent fixtures pass.

## Provisional-to-final binding

Reviewed provisional packet:
`dartsort_divergence_D1_adapter_repair_provisional_h5_20261007_v2`, MANIFEST
`b4b9897c222e4d0871d6f99540b724ec0e269025f3c0e14ba43c3231c8b380c2`.

Final packet:
`dartsort_divergence_D1_adapter_repair_h5_20261007_v2`, MANIFEST
`c32b16fc8398424283b74ff85a4ee51496213df812c6417e5e9b971fbf504091`,
COMPLETE
`4dd4dbd9dba3774cf14fb042fb73deb18cea26476b58d29cb77cc9515ed75c9c`.

The following review-bearing members are byte-identical from provisional to
final: `BINDINGS.json`, `CONTRACT.json`, `REVIEW_REQUEST.json`, both source
files, both test files. `SOURCE_QUALIFICATION.md` and
`FIXTURE_TEST_RECEIPT.json` change only the recorded pytest result from 16 to
17 passed. The provisional-only changed-interface inventory and snapshot are
not final members; final adds `README.md`. No executable dependency changed,
so the incremental source review remains valid.

All ten final manifest members and the final COMPLETE binding independently
verify. H1 reran the final suite: 17 passed. H1 reran the independent fixture:
6 parent-map cases, 500 random event-relation cases, and one ranking case pass.
The unchanged accepted matcher retained its earlier independent 176,400-case
audit and was not redundantly re-exhausted.

## Four repaired interfaces

1. **Per-arm provenance:** A binds the run's patched Kilosort 4.0.27 `io.py`,
   saved parameters/effective settings, wrapper evidence, producer identity,
   and curator source. REF explicitly marks historical sorter/package/wrapper
   bytes unavailable and limits original intermediate claims to relationships
   empirically validated at execution. REF curator project source is exactly
   bound. A's wrapper hash plus pre-run mtime is supporting historical evidence,
   not as strong as a run-receipt byte binding; saved parameter and empirical
   ancestry gates remain required.
2. **Curated parent mapping:** the frozen `(sample_time,
   detection_template)` relation emits `unique_parent` only for one-to-one key
   cardinality, retains every original candidate for duplicated keys, emits
   unmatched rows, identifies only keys with no curated child as deleted, and
   rejects out-of-order inputs. It does not guess occurrence rank.
3. **Event-level relations:** DARTsort rows separately expose
   `parent_row_id`, `matching_label`, and `final_label`. Pair-local relations
   retain source indices, accepted matched pairs, all-neighbor candidate sets
   and degrees, and all unmatched rows. The emitted match count is checked
   against the accepted counter; pair names remain separate and no transitive
   identifier is emitted.
4. **Candidate selection:** the default rule uses all final labels >=0 with at
   least 100 common-support events; ranks by matches, smaller-train retention,
   and F1; assigns dense ties; reports every passing candidate and the complete
   eligible table; and applies no top-N cap or depth/waveform/state/identity
   gate.

## Gates for the separate execution contract

- Hard-bind the production candidate thresholds to `unit_events=100`,
  `exclusive_matches=25`, and `smaller_train_retention=0.01`, or change the
  adapter's `candidate_universe` string to derive from the effective
  `minimum_unit_events`. The current function accepts overrides while its prose
  always says `>=100`; H1 reproduced `actual_min=1` with metadata still saying
  `>=100`. This does not affect the frozen production defaults, but runtime
  provenance must prevent or disclose an override.
- Require empirical `bind_kilosort_export` validation separately for original
  and curated REF arrays; do not promote REF relationships to exact historical
  executable semantics. For A, retain the exact patched exporter and saved
  parameter bindings and validate the consumed arrays as well.
- Require exact DARTsort final/H5 full-row time equality before feature or
  matching-stage use. Preserve a failure packet if the full traversal disproves
  the bounded prior probe.
- Freeze resource bounds, chunking, output schemas, exact four anchors, and the
  no-transitive/no-depth-gate rule. The synthetic managed-runner preparation
  remains non-production and cannot enable production execution.

## Implementation checks

- Done: final integrity -> all ten member hashes/sizes, final MANIFEST, and
  COMPLETE independently verified.
- Done: incremental diff -> executable source, tests, bindings, contract, and
  review request unchanged; only final test-count text and packaging members
  changed.
- Done: per-arm provenance -> A and REF records reviewed independently; REF
  historical byte unavailability is explicit and correctly scope-limiting.
- Done: parent relation -> code and fixtures cover unique, duplicated,
  one-to-many, many-to-one, unmatched, deleted-key, wrong-template, and
  out-of-order cases with closure.
- Done: event relations -> source-index preservation, half-open support,
  inclusive matching, all-neighbor sets, unmatched rows, accepted-count
  reproduction, pair separation, and absent transitive identity checked.
- Done: candidate rule -> zero/one/multiple candidate behavior, exact ties,
  input-order invariance at duplicate times, no-pass preservation, thresholds,
  and uncapped reporting inspected; mutable-threshold metadata caveat recorded.
- Done: final tests -> 17 pytest cases plus 507 independent cases pass.
- Not done: production arrays or scientific outcomes -> prohibited; a separate
  frozen execution contract and later evidence are required.
- Can establish: repaired adapter synthetic correctness and readiness to freeze
  a bounded execution contract subject to the gates above.
- Cannot establish: any real divergence, unique ancestry inside duplicate keys,
  exact historical REF executable semantics, biological identity, purity,
  sorter-only causality, or production benefit.

