# Focused independent review — Candidate2 v8 stable systemd identity

## Verdict

**GO** for the contract-defined next boundary: create the exact typed `AUTHORIZATION.json`, perform fresh reconciliation, and make at most one managed v8 attempt. This review does not authorize reuse or retry and is not evidence that production execution has begun.

## Load-bearing findings

### Stable command identity

The final preflight obtains each unit object through `org.freedesktop.systemd1.Manager.GetUnit`, then reads `Service.ExecStart` and `ExecStartPre` with exact signature `a(sasbttttuii)` (`candidate2_cache_managed_preflight.py:51-117`). It rejects query failure, malformed JSON/envelopes, the wrong signature, malformed row cardinality/types, and unsupported properties. For every command it retains only the stable fields required by the contract: executable path, complete argv array, and boolean `ignore_errors`. List and row iteration preserve command order/cardinality and argv argument boundaries; an empty command array remains an explicit empty list. It does not parse or strip human-formatted `systemctl` command strings.

Authorization capture and preflight checking use the same typed implementation. The complete typed dictionaries compare exactly (`lines 265-283`), while non-command scalar properties and dependency membership remain separately strict. Downstream isolation/monitor checks consume the typed argv arrays (`lines 297-308`). The frozen deployed capture's target `ExecStart`, target `ExecStartPre`, monitor `ExecStart`, and empty monitor `ExecStartPre` independently match the packet service directives exactly, including path, order, cardinality, flags, and argument boundaries.

The rescue-production typed fixture passed independently. It preserved a two-word single argument, accepted an empty command array, and rejected bad signature, short row, non-string argv, non-boolean `ignore_errors`, and query failure.

### Complete preflight and Path repair

The systemctl loop now stores stdout in `systemctl_output`; the attempt output remains a `Path` from its construction through freshness checks, free-space checks, parent creation, and final receipt serialization (`lines 234-251,269-280,334-348`).

I independently reran the actual preflight `main()` with the rescue-production interpreter against temporary realistic user services captured inactive and then started active. It traversed real source hashing, synthetic input metadata, host memory/PSI, live GPU/self classification, process scan, systemctl scalar/dependency checks, typed D-Bus identity, monitor-ready validation, installed Kilosort identity, environment binding, and receipt serialization. The positive receipt was valid; independently generated receipt SHA-256 was `53a637b69a8b36d87ec723e04c5549e594b3fb1770c0f745a7ec396e9b4c9633`. Changed executable path, argv cardinality/boundary, and `ignore_errors` each rejected. The synthetic fixture read zero production-recording bytes, ran no sorter, created no production output, and removed its temporary units.

The producer's preserved full-preflight receipt SHA-256 `1d718ddd7426fbb6c6349ac26a9d8d6531d9067d72c8e96139ed815b78e345fc` contains the complete ordered receipt fields, including resources, loaded units, typed identity, monitor readiness, final timestamps, and explicit `recording_opened=false`/zero bytes.

## Unchanged and deployed checks

- Exact reviewed bindings: contract `4769968e7bef087d07097bb61b745de7882afaa9e38c2d20f7c75eb4071d6c68`; 31-member source manifest `db2660fbb9d63e706780cef5a9b60855b8b7f1ca190d882705eec40de39a9408`; preflight `2c3d0e07f4e6a723aa7af16bd865ecc353007d611927f5fe31932753e0eb7da7`; unchanged runner `f62f8c5f42165414233f3532c4ba1a334563bae0bf947b9ac4fe1ae3c5f2da2b`.
- Packet and installed source inventories contain exactly the same 31 declared members with no mismatch or extra member. Installed bootstraps and source manifest also match the packet.
- Packet and installed unit hashes match: target `aecc25da47aaaf9213b8635dd2b5f56360e6c6bc5958624d67c0cebd8a195a11`; monitor `30b82b2396705f11fc1e7c2177ace6e73a6b74ee27446307dd126d0a071391e7`.
- Frozen deployed typed identity SHA-256 `9e1126c0faec08a6bdfc2ce0d49511752cefb8ab8e9c65e0a6a1fb87f6bcf744` exactly matches the unit directives. Deployment evidence SHA-256 `65303e1a3019e114133cadeedbdfa526ed7af9765148c67292aaefbf44aad9c5` agrees with independently checked installed bytes and current inactive/dead PID-zero state.
- The v7 terminal failure is preserved: MANIFEST SHA-256 `e24bfdfce47d4d59fad84047f811e0a04abc04189b548af33fd0f9a8ea7e889e`, six of six members verified, status terminal pre-read failure/no retry.
- Independently repeated unchanged tests passed: exact-self GPU five cases, late external-GPU/sorter arrival gates, runner scientific-parent join, monitor forwarding, 11 monitor readiness/classification cases, and systemd dependency-set canonicalization.
- Candidate2 science, full parameter equality, effective `nblocks=1`, three cache operations, output derivation, no-retry lifecycle, and absence of hard memory/runtime/CPU caps are unchanged.
- Production authorization, output, preflight receipt, and job receipt are absent. Production units are inactive/dead with PID 0; the target is disabled. No production launch occurred.

## Scope

No production recording or voltage was opened, read, or hashed. No production service or sorter was started. The only service execution was the bounded synthetic dummy lifecycle explicitly required to test the complete preflight path.

## Implementation checks

- Done: inspected exact D-Bus query/decoding and authorization comparison -> stable command identity is exact and fail-closed without regex stripping (`candidate2_cache_managed_preflight.py:51-117,265-308`).
- Done: compared deployed typed identity to both packet service directives -> every command path, argv, flag, order/cardinality, and empty array agrees (artifact SHA `9e1126c0...`).
- Done: reran actual full preflight main and negative authorization cases -> valid receipt serialized; path/argv/ignore-errors changes rejected; Path-shadow failure is closed.
- Done: rehashed packet/installed closure, units, v7 failure, and repeated unchanged fixed-small gates -> exact bindings and preserved behavior pass.
- Not done: production authorization creation, fresh production preflight, recording validation, production start, sort, curation/QC, or outcome review -> these remain prospective.
- Can establish: the exact v8 bytes and inactive deployment are technically fit to proceed to one exact-authorized, freshly reconciled managed attempt.
- Cannot establish: future resource state, execution success, output validity, memory outcome, or scientific benefit.
