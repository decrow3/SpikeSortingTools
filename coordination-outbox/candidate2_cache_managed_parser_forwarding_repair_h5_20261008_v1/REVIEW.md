# Focused independent review — Candidate2 parser-forwarding repair v1

## Verdict

**GO** for the focused parser-forwarding and deployed-binding progression. This review does not install, authorize, or start the services.

## Defect and repair

The preserved v5 failure packet rehashes cleanly (MANIFEST `9b758a590494c2a115ca61671f1b16c2502844cd41f88ee958ee6eb7ca530be0`, five members). Its service evidence shows that the monitor exited with status 2 before preflight, job receipt, recording access, or GPU work. With argparse abbreviation enabled, monitor-bootstrap parsing consumed `--target-unit` as an abbreviation of `--target-unit-file`; the later exact option replaced the value, leaving the child without required `--target-unit`.

The repaired `monitor_bootstrap.py` differs from the sealed reviewed v5 parent by exactly one executable line: `ArgumentParser(add_help=False, allow_abbrev=False)`. This makes `--target-unit` unknown to the bootstrap parser and preserves it in `parse_known_args()` forwarding, while the exact `--target-unit-file` remains bootstrap-owned.

## Regression evidence

I independently ran `tests/test_monitor_bootstrap_forwarding.py`. The test:

- extracts the actual `ExecStart` option names and their exact order from the frozen v6 monitor service;
- asserts that order includes `--target-unit` immediately before `--target-unit-file`;
- substitutes only synthetic contract/authorization/unit files and an inert monitor child;
- launches the real repaired bootstrap as an isolated subprocess; and
- proves the child receives the exact target unit, target-unit file, monitor-unit file, output, and service-log values.

The finalized regression (`21717ec9c9558aaa4aab69ba8bd1e58c05074106c1fd892d7d9ebd5ea730e10e`) accepts explicit bootstrap/unit paths. I ran it twice: once against packet files and once against the actual installed monitor bootstrap and unit. Both subprocesses returned zero and printed: `actual service argv crossed monitor bootstrap boundary with --target-unit intact`.

`TEST_RECEIPT.json` SHA-256 `4b25822c39ebabf7eca4bdfc31f0bf4faeadc4857663db6ddff30408838b2c8b` binds the final contract, repaired parser, final regression test, both packet/installed unit hashes, both successful boundary runs, and the eleven unchanged monitor known-answer cases. Its no-recording/no-GPU declarations agree with this review's actions.

This exercises the real parser/forwarding boundary; it is not merely a generic argparse or string test. The unchanged monitor known-answer suite also passed all eleven readiness/classification cases.

## Delta checks

- All 30 runtime source members and `SOURCE_MANIFEST.json` are byte-identical to the sealed reviewed v5 packet.
- `bootstrap.py` and `preflight_bootstrap.py` are byte-identical. Monitor implementation/classification is byte-identical.
- Candidate2 sorter/science parameters, effective `nblocks=1` gate, input bindings, and all three cache-managed operations are unchanged.
- Contract changes are limited to parent/failure history, the single parser repair declaration, fresh v6 attempt/path bindings, repaired monitor-bootstrap hash, and release wording.
- Target and monitor service diffs are mechanical fresh-v6 names/paths, packet/contract hashes, and the repaired monitor-bootstrap hash. Raw input target, command structure/order, isolation, dependencies, `Restart=no`, `OOMPolicy=stop`, infinite timeout, and absence of hard resource/CPU/runtime caps are unchanged.
- Fresh v6 output, results, partial, launch, and monitor paths are absent. `AUTHORIZATION.json` is absent.
- The failed v5 namespace/evidence is preserved; this is one separately bound fresh attempt, with no automatic retry.

## Installed execution-disabled binding

- All 30 installed source members match `SOURCE_MANIFEST.json`; the installed manifest and all three bootstraps are byte-identical to the reviewed packet.
- Installed target unit SHA-256 is `c5717f5a77930907079d0aee11d9eb88841a09f282a94dee22da2acff6231d71`; installed monitor unit SHA-256 is `464ad9d9d1d6ced7ecd25898404e7625202ae1990a92cc36fb840fb89460fc35`. Both equal the reviewed packet files.
- Manager-loaded fragment paths resolve to those exact installed files. Both units are loaded but `inactive/dead`, have empty invocation IDs and `ExecMainPID=0`, and have not been started.
- The loaded target `ExecStartPre`/`ExecStart` and monitor `ExecStart` contain the exact reviewed v6 paths, contract `f0319985...`, manifest/bootstrap/source hashes, authorization path, raw target, and adjacent `--target-unit` / `--target-unit-file` order.
- Effective lifecycle/resource properties are `Restart=no`, `TimeoutStartUSec=infinity`, `RuntimeMaxUSec=infinity`, `OOMPolicy=stop`, `MemoryHigh=infinity`, `MemoryMax=infinity`, and `CPUQuotaPerSecUSec=infinity`. Target `Requires`, `BindsTo`, and `After` reference the v6 monitor exactly.

No blocking defect was found in the requested delta.

## Scope

No recording binary or voltage was opened, read, or hashed. No GPU/CUDA operation, installation, systemd start, sort, or output namespace creation was performed.

## Implementation checks

- Done: inspected and rehashed the preserved v5 failure packet -> exact pre-read parser-abbreviation cause confirmed.
- Done: byte-diffed executable/source artifacts against sealed reviewed v5 -> only monitor-bootstrap `allow_abbrev=False` changed.
- Done: ran the finalized permanent subprocess regression through both packet and installed service option names/order -> inert child received `--target-unit` and adjacent exact options intact.
- Done: reran unchanged monitor readiness/classification fixture -> eleven cases passed.
- Done: compared installed source/bootstrap/unit bytes and manager-loaded properties -> exact reviewed bindings, inactive/dead state, no invocation, no caps/retry.
- Done: diffed contract/services and checked fresh output paths -> mechanical v6 binding changes only; no cap/retry/science/cache drift.
- Not done: authorization, managed preflight, service start, recording validation, GPU work, sort, or outcomes -> explicitly excluded.
- Can establish: the exact v5 parser-forwarding failure is repaired at the real bootstrap-to-child boundary without changing scientific or monitoring semantics.
- Cannot establish: post-authorization fresh preflight, launch readiness, successful execution, memory outcome, or scientific benefit.
