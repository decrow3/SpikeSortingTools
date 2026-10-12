# Independent N0/N1 validator v2 rereview

Verdict: **NO_GO_PENDING_PATH_AND_SERVICE_NAME_GUARDS**.

V2 correctly repairs every frozen v1 finding. The v1 `NO_GO` remains immutable
at commit `1a59c341c3b61813cbaffdc35915b89ed2e687e2` and COMPLETE SHA-256
`2be9d4fa1380b8cca8c47d46865b5e65895bae39fb1236f326f0d22e500ae183`.
However, two new fail-closed boundary defects remain in fields introduced or
tightened by v2, so the validator is not yet ready as a load-bearing execution
gate.

## Requested rereview results

- The v1 semantic replay passes 44/44. Selected independently, the exact 15
  former failures pass and all 29 formerly passing cases remain green.
- The author suite passes 37/37 and the packet-local author plus replay suite
  passes 81/81.
- The delta matches the frozen v2 contract: session-clock links, `half_open`
  bounds, common-event identity, normalized per-window checkpoints, exact K10
  hashes, placeholder/container guards, structured managed-service receipt,
  exact stops, and canonical/disjoint output namespaces are the substantive
  changes. The compatibility commit changes only an error message.
- Ten independent controls pass for case-normalized checkpoint equality and
  exact key sets, exact K10 pins, clock links, dot/dot-dot/trailing aliases,
  ancestral and input-overlapping outputs, manager/receipt booleans, and exact
  order-independent stop IDs.

## Residual defects

1. **NUL paths pass.** `PurePath` accepts embedded NUL characters, and
   `_canonical_path` only tests absolute form, dot-dot components, and string
   normalization. Recording, motion-field, and output strings containing NUL
   therefore receive `STRUCTURALLY_VALID_METADATA_ONLY`, although no filesystem
   operation can use them as paths.
2. **Invalid systemd unit names pass.** The service check requires only a
   `.service` suffix, no slash, and unchanged outer whitespace. It accepts
   `bad unit.service`, `.service`, and a NUL-containing unit. These are not valid
   simple resolved service-unit names, so the managed-execution receipt remains
   evadable.

## Exact repair

- In `_canonical_path`, reject `"\0"` before constructing `PurePath`; preserve
  every existing canonical/absolute/ancestry check.
- Validate `service_unit` with a full-match conservative grammar for the frozen
  “simple unit” domain, such as `[A-Za-z0-9:_.@-]+\\.service`, require a nonempty
  stem, and enforce the systemd unit-name byte-length limit. This also rejects
  whitespace, NUL, separators, and an empty `.service` stem.
- Add the four failing parameterized controls as regressions. Rerun all 10 new
  passing controls, the semantic v1 replay split (15 and 29), the 37 author
  tests, and the 81 packet-local combined tests unchanged in a new immutable
  version. Preserve this v2 failure packet.

## Implementation checks

- Done: full v2 contract/source/test and v1-to-v2 delta read -> requested repair
  is exact at commits `0fac090b2dc1f1f22c91aee37a4ab17d5d537863` and
  `464ed8910a576181cd584e080a4120f3af01ce08`.
- Done: semantic v1 replay and author suites -> 15 former failures repaired, 29
  former passes preserved, 37 author and 81 combined tests pass.
- Done: independent checkpoint/K10/clock/path/service/stop probes -> 10 pass and
  4 fail against implementation SHA-256
  `ab6c0435188e534acd67033c1285e4aa919cee93c62cb1c8289f9ad5b40ee3d4`.
- Done: provenance/hash verification -> local and coordination-copy v2 manifest
  members verify at MANIFEST SHA-256
  `ec9fb52cd73d018fc9f5cc4b9eae0a474d932ed62488b110151d13fa1a00e629`
  and COMPLETE SHA-256
  `2667d2074620422e5a3775e0cd695fc2eed33f41fdf8714eec3ef69f2176fbf5`;
  v1 `NO_GO` chains remain preserved.
- Not done: referenced-byte validation, real data, live resource preflight,
  production execution, GPU/service/sort, RF/holdout, or C4 -> explicitly
  excluded.
- Can establish: v2 repairs all known v1 evasions but still accepts unusable
  path and service-unit strings at newly reviewed boundaries.
- Cannot establish: a fully fail-closed execution gate, referenced-byte
  correctness, live resource readiness, scientific validity, or N0/N1 outcome.

