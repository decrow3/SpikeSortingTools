# Independent N0/N1 validator v3 rereview

Verdict: **GO_METADATA_VALIDATOR_ONLY**.

V3 repairs the two remaining v2 boundary defects, and no new structural evasion
was found in the bounded rereview. Both historical failures remain immutable:
v1 at commit `1a59c341c3b61813cbaffdc35915b89ed2e687e2` / COMPLETE
`2be9d4fa1380b8cca8c47d46865b5e65895bae39fb1236f326f0d22e500ae183`,
and v2 at commit `e1e461dc2b8ac09272a8738a51d78055ff25035a` / COMPLETE
`4e10d71c6e8268009adc95b97eda26b907f9794adc5fa855458a6e1032251732`.

## Findings

- The code delta is confined to schema versioning, NUL rejection before
  `PurePath`, a conservative full-match ASCII service-unit grammar, and the
  255-byte service-name limit. The compatibility commit only clarifies the NUL
  error message.
- All four formerly failing v2 controls pass. All ten prior new controls remain
  green, including normalized checkpoint maps, exact K10 pins, clock links,
  canonical/ancestral output checks, manager/receipt checks, and exact stops.
- The semantic v1 replay remains 44/44. Author tests pass 43/43 and the three
  packet-local suites pass 101/101.
- Seventeen independent boundary controls pass. A 255-byte simple `.service`
  name is accepted and the otherwise identical 256-byte name is rejected.
  Whitespace, empty-stem, NUL, non-ASCII, wrong-suffix, and slash-containing
  service names fail closed. NUL is rejected in recording, motion, N0-output,
  and N1-output paths.
- Local and coordination shared v3 manifest members and hash chains verify.
  Both local/shared v1 and v2 `NO_GO` review chains also verify.

This verdict is deliberately limited to metadata structure. It does not prove
that any referenced bytes exist or match, that the service or resources are
available, that implementation behavior beyond this validator is correct, or
that N0/N1 is approved to execute.

## Implementation checks

- Done: full v3 contract/source/tests and v2-to-v3 delta read -> commits
  `160a2866b5120e82e95fa8c857a70cf038c4ac59` and
  `66e1788a8afa4728546455aaddf84fdcf2bea51a` implement only the frozen guards.
- Done: historical regression replay -> v2 controls 14/14, split into exact four
  former failures and ten former passes; semantic v1 replay 44/44.
- Done: author and packet regression -> 43 author and 101 combined tests pass.
- Done: independent path/service boundaries -> 17/17 pass against implementation
  SHA-256 `0fc63eba0cc560c313e180e086a4b7f728be5f4a122ff276d04afc62c34df734`.
- Done: provenance/hash verification -> local/shared v3 chain verifies at
  MANIFEST SHA-256
  `2b0a20f4cd632be669c539fab6e970df48a58739ee41a89e9ba77ecd07261c21`
  and COMPLETE SHA-256
  `b0ba209aa383fee392679f72284446f3f7928e638480e373966eb839c6431f56`;
  prior `NO_GO` chains remain preserved.
- Not done: referenced-byte checks, real data, live resource preflight,
  production execution, GPU/service/sort, RF/holdout, or C4 -> explicitly
  excluded.
- Can establish: all known metadata-validator structural and boundary defects
  are repaired within the frozen v3 schema and tested input domain.
- Cannot establish: referenced input correctness, live resource readiness,
  scientific validity, execution authorization, or N0/N1 outcome.

