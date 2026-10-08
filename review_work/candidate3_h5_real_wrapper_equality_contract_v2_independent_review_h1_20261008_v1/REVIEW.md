# Independent delta review: candidate-3 H5 real wrapper equality contract v2

## Verdict

`NO_GO_REPAIR_EXACT_INVOCATION_AND_OUTPUT_BINDING`

V2 closes most of the v1 defects and its scientific comparison remains coherent,
but it does not close the v1 requirement that the release authority pre-bind and
the runner enforce the exact invocation and fresh output topology. Real execution
must remain disabled. Preserve v2 unchanged and repair this narrow release join in
a distinct namespace.

Reviewed immutable inputs:

- author MANIFEST `d3cda8d18062cd0ec1c60d45533a631f15f58996046df0c56f07054c15d9e724`
- author COMPLETE `da4550bf427bfa38186dfd5bcc6e6a373709483c0fb0b441bce15682e4a47496`
- contract `432b6aebab7469fc1f71350625feec9f50c56dcd105bf5034d89faf7a1507110`
- runner `3db928375cbcd75e9f79f1ab593978ef0af736f5ea3adbc0b9a0e46853c217fc`
- prior independent v1 NO-GO MANIFEST `3bf7158d17e89e39accfb2379b94df0e2ae70df870d6ede299677bf852539dba`
- prior independent v1 NO-GO COMPLETE `eeca0ac3b8f8ca1c02ca8879eba4842684b43f0033c6d0512b9ec561858be8db`

## Repairs that pass

1. **Runner and contract source identity.** The committed runner and contract
   exactly match the packet. The runner checks its own SHA-256 against the
   contract before output creation. Real mode also checks the interpreter,
   repository root, contract path and appended Kilosort site before any binary
   access. Imported scientific source files remain independently hashed.

2. **Cache/environment repair.** The runner creates one private temporary root,
   places `TMPDIR` and `NUMBA_CACHE_DIR` beneath it before scientific imports,
   resets Python's cached temporary root, accounts the tree every 0.1 seconds,
   and removes it at termination. The independent self-test succeeded without a
   caller-supplied cache variable, repairing the exact v1 environment failure.

3. **Focused-review gate.** Real mode now requires the exact focused-review
   schema and `GO_EXECUTION_READY` status and checks its contract and runner
   bindings. The dispatch receipt binds the hashes of the supplied review and
   managed-resource receipts as well as the contract and runner.

4. **Managed resource and outer failure requirements.** The resource receipt
   must claim the exact contract/runner and all frozen resource values, identify
   a manager job and name a persistent outer-receipt path. The contract correctly
   leaves current H5 reconciliation and actual external enforcement to a later
   prestart and dispatch. No current H5-readiness claim is made.

5. **Active internal guards.** The runner lowers `RLIMIT_AS` and `RLIMIT_FSIZE`,
   installs an active real-time alarm, watches its private temporary tree, fixes
   Torch intra-op/inter-op thread counts, checks logical binary-read bytes, and
   projects durable terminal bytes including terminal member, MANIFEST and
   COMPLETE before writing any of them. External process-group enforcement is
   still mandatory, appropriately, because an in-process guard cannot preserve
   evidence after every kill or pre-output failure.

6. **Terminal behavior and fixtures.** Source order writes terminal member,
   MANIFEST and COMPLETE in that order, and refuses to add a second terminal
   member. The author success fixture contains only `RESULT.json`; the injected
   source-hash failure contains only `FAILURE.json`. Both seals bind their
   members. Shared-copy birth times for the nested fixtures do not preserve the
   original write order, so they are not used as chronology evidence. The code
   order and independent fresh self-test establish handled-path COMPLETE-last
   behavior; an outer manager receipt remains necessary for abrupt/pre-output
   failures.

7. **Independent full-shape reproduction.** The exact committed runner and
   contract completed both 419,998-by-384 synthetic windows without a caller
   cache setting. All nine lazy/eager requests were bitwise equal, both positive
   controls differed, cached synthetic input was 645,116,928 bytes, recorded
   binary reads were zero, terminal temporary usage was 5,001 bytes, and the
   packet sealed exactly one `RESULT.json`. This validates execution mechanics,
   not recorded-voltage equality.

## Remaining blocking defect

The v1 review required the coordinator release to bind the exact command,
environment and fresh output path, with the runner verifying every binding
before binary access. V2 does not do so:

- the contract contains no expected runner execution path;
- the runner verifies its content hash and derived repository root but accepts
  the same bytes from any repository-local path with the same two-level parent;
- the dispatch receipt schema has no command, argv, runner path, output path or
  environment fields;
- the resource receipt schema has no output path or managed temporary/output
  topology fields; and
- `--output` is checked only for nonexistence. Any fresh absolute path is
  accepted, even if the external monitor/quota described by the resource receipt
  applies to a different path.

The later outer receipt records argv post hoc, but that is failure evidence, not
a pre-execution authorization or an enforcement join. Consequently a caller can
satisfy every current runner check while selecting an unbound output namespace.
This is especially material because the contract claims external enforcement of
the persistent-output limit; the runner cannot verify that the selected output
lies inside the topology covered by the resource receipt.

## Minimal repair

In a distinct version:

1. Add the exact expected H5 runner path and verify `Path(__file__).resolve()`.
2. Make the coordinator dispatch bind a canonical invocation or its hash,
   including runner, contract, all receipt paths and hashes, extra site, real-mode
   flag, exact fresh output path, and the environment policy.
3. Make the managed-resource receipt bind the same exact output path and the
   externally enforced temporary/output topology. Have the runner compare these
   fields to resolved CLI values before binary access.
4. Preserve the later outer receipt requirement for exact argv, manager state,
   stdout/stderr, exit status and sealed-terminal assessment.

No change to window arithmetic, eager/lazy semantics, scientific acceptance,
resource magnitudes or the author fixtures is requested.

## Implementation checks

- Done: packet/nested hashes, committed source equality, contract/runner delta,
  imports, release joins, internal guards and terminal source order -> all
  inspected directly; most v1 defects are repaired.
- Done: exact full-shape local synthetic command without caller cache state ->
  9/9 primary comparisons passed, both positive controls differed, no binary
  voltage bytes read, and one sealed result was produced.
- Done: exact invocation/output authorization trace -> dispatch/resource schemas
  do not bind the runner path, argv or output/resource topology; runner accepts
  any fresh output path.
- Not done: H5 state, real voltage equality, external service/cgroup enforcement,
  GPU/sort/RF/holdout or full-session execution -> prohibited and still gated.
- Can establish: v2 repairs the cache failure, source/review joins, active
  internal guards, bounded synthetic behavior and handled terminal sealing.
- Cannot establish: that the exact later invocation and its output location are
  the ones authorized and covered by external enforcement; v2 is therefore not
  execution-ready.
