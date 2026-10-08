# Candidate 3 equality result evaluator v2 correction

V1 remains preserved as NO-GO (review manifest
`68d3018a129976a588039b53d5586ca0bcaa5acaa9e8b5262fd0114649a05d9d`,
complete `550e3a4ceb439911ba4324c93397802e8ebcae6d805529308b5753426e19a79b`).

V2 closes all three independent fail-open fixtures:

- every lazy/eager output digest must be present lowercase 64-hex as well as
  identical;
- real provenance requires exact canonical argv reconstruction, strict receipt
  digest syntax, rehashing all three receipt files, and full review, dispatch,
  resource, path, limit and cross-binding validation against v3;
- handled failure requires MANIFEST, COMPLETE and FAILURE statuses all equal
  `FAILED_PRESERVED_NO_RETRY` plus the frozen failure fields.

Ten controls pass: the original six, the three reviewer adversarial fixtures,
and a fully bound synthetic construction of the real-pass schema.  The latter
tests evaluator logic only and is not a real H5 outcome.

Implementation checks
- Done: exact v3 request frames/shapes and zero tolerances remain unchanged.
- Done: all v1 adversarial failures now resolve to the predeclared stop category.
- Done: a fully cross-bound synthetic real-schema fixture reaches the real-pass
  category, proving it is reachable only through the strict join.
- Not done: real H5 result inspection or real receipt availability.
- Can establish: v2 consumes the accepted v3 schema fail-closed under tested
  integrity, provenance, equality, positive-control, resource and failure cases.
- Cannot establish: real equality, sorting benefit, identity, purity, or later
  experiment authorization.

Requested verdict: GO only as the frozen consumer for a future exact v3 packet.
