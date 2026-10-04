# Candidate-G production-bootstrap v5 final review

## Verdict

`NO_GO_EXTERNAL_COMPACT_ACCOUNTING_INCOMPLETE`

Two of the three frozen blockers are closed. Installed production/finalizer units are byte-bound to reviewed templates with meaningful mutation negatives, and the authoritative systemd receipt binds the actual v5 production entry and gate to a 17-member pre-execution closure that matches this sealed packet.

The storage blocker is only partially closed. All files beneath the run root are classified disjointly, including all four runtime-cache subtrees. The real compact output, however, is not beneath the run root: bootstrap creates it as a separate child of the compact parent and Bubblewrap binds it independently. `categories(attempt)` looks for `<attempt>/compact`, so it cannot observe or enforce the compact category cap on the actual writable compact root.

The producer's 311-byte known-answer fixture hides this mismatch by creating an artificial `<attempt>/compact`. A topology-matched independent fixture wrote 47 bytes to a sibling compact root; `categories(attempt)` returned `compact_publication=0` and total zero.

The narrow repair is to pass the exact compact path through sampling and account for that tree once, or enforce and receipt an equivalent hard bound inside the already bounded compact publisher. The regression fixture must use sibling run/compact roots matching `bootstrap_v5` and exercise the compact cap. No new systemd smoke, science, voltage, agglomeration or GPU work is needed unless that repair changes the executed bootstrap/lifecycle boundary.

## Accepted evidence

- Packet MANIFEST, COMPLETE and 64-product inventory verify.
- Twenty-two focused tests pass independently.
- Installed-unit identity is hard-pinned to the reviewed production and finalizer templates; all four requested mutation controls reject.
- All 17 smoke closure entries match their immutable packet counterparts, including the actual v5 entry, gate, lifecycle, policy, units and publisher.
- The H5 receipt reports systemd success, status 0, zero restarts and sealed result/terminal/compact evidence.
- Canonical release, attempt, compact output and production units remain absent at paths visible from huklaban1.

Implementation checks
- Done: what ran -> inspected the v5 gate, templates, unit constructor, bootstrap/entry/finalizer, lifecycle/accounting, smoke source/release, tests and authoritative receipts.
- Done: installed-unit identity -> exact byte/hash/parsed-directive equality is enforced; independent mutation controls reject.
- Done: executed-source provenance -> the 17 receipt entries independently match the sealed packet and identify the actual v5 entry and gate.
- Done: attempt-local storage -> result, temp, complete cache root, logs and launch metadata are disjoint and fail on unclassified files.
- Done: topology falsification -> production bootstrap creates compact outside the attempt, while the sampler only scans the attempt; a sibling-root fixture reproduced the omission.
- Done: provenance -> verified MANIFEST `6e2f75cf07a783bdc4a8fad829f291b1ab1cf9b2ec0f22659e39b901a22c5d64`, COMPLETE `78725f621cf9f4204cc36ad319bfa0a97c408f7d321a3ac41e0bad51340ef58e`, request `8338f2b3472eb8e867b8d2303b078f5400da5399e48075caa385a0231c0ec58f`; supplied coordination hash `8fcd7c64a4c2a94b63365e8b029d607a1f40c7cc7ccb2e390e99c0cb1d139ae1`.
- Not done: independent opening of the two H5-local state files -> unavailable on huklaban1; 28 other production-closure entries verify.
- Can establish: unit identity and smoke executed-source provenance blockers are repaired, and all run-root writable files are classified once.
- Cannot establish: complete accounting or cap enforcement for every writable production root because the actual external compact tree is invisible to the sampler.

No release, installation, continuation, recording access, agglomeration or GPU work is authorized.
