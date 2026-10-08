# Candidate 3 equality result evaluator v3 correction

V3 preserves v2 and repairs its narrow deterministic-output NO-GO.  The v2
review identified two correctly sealed malformed schemas that raised instead of
returning a frozen category.  V3 catches malformed FAILURE JSON as
`REJECT_SEAL_OR_SCHEMA_STOP`, type-checks receipt paths before `Path`, catches
receipt parse/file errors as `REJECT_PROVENANCE_STOP`, checks receipt top-level
object types, and gives the CLI a final fail-closed sealed-category guard.

All 12 controls pass: the v1 stock and adversarial cases, fully bound synthetic
real-schema reachability, and the two exact v2 reviewer fixtures.  No scientific
threshold, equality rule, resource bound, or coordinator decision changed.

Implementation checks
- Done: malformed sealed FAILURE and integer real receipt-path fixtures now
  return their exact frozen categories without exception.
- Done: accepted v3 synthetic success remains synthetic-only.
- Not done: real H5 packet inspection.
- Can establish: deterministic sealed coordinator output for all reviewed
  compact-schema cases.
- Cannot establish: real equality or any downstream scientific result.

Requested verdict: GO only as the future exact v3 equality packet consumer.
