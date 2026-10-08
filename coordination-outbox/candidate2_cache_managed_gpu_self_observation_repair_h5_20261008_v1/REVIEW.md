# Focused independent review — Candidate2 v7 final authorization-reconciliation repair

## Final verdict

**GO** for the contract-defined next boundary: create an exact hash-bound `AUTHORIZATION.json`, perform the fresh reconciliation, and make at most one managed v7 attempt. This is not evidence that execution has started or that the sort will succeed.

This version supersedes both my initial `NO_GO` review of the provisional v7 bytes (`REVIEW.md` SHA-256 `f22e60ac...`; `REVIEW_STATUS.json` SHA-256 `c8bdd152...`) and my first repair `GO` (`REVIEW.md` SHA-256 `82adb2de...`; `REVIEW_STATUS.json` SHA-256 `3c57283a...`). The exact final bytes close both findings.

## Blocking repair closed

The final runner imports the same strict GPU metric and gate used by preflight (`candidate2_cache_managed_full.py:20,50-54,113-115`) and independently scans for competing sorter processes (`lines 57-71,116-119`). The single `require_resource_gates()` implementation is called both before the 241,309,358,592-byte accepted-content validation (`line 179`) and after whole-file `POSIX_FADV_DONTNEED` plus the five-second settle (`lines 184-189`), before output creation and `run_sorter_config` (`lines 191,215`). Thus a late external GPU PID or competing sorter is no longer accepted merely because free GPU memory remains above 20 GiB.

The actual rescue-production late-arrival subprocess fixture passed independently: a clean first gate passed, an external GPU PID introduced only at the second gate failed, and a newly introduced competing sorter failed. The isolated real-runner import/join also passed and stopped only at the deliberately absent preflight receipt; this proves the restored dependency is available before any recording access.

## Authorization-reconciliation repair

The final preflight canonicalizes only the ordering of whitespace-delimited tokens in systemd's set-valued `Requires`, `BindsTo`, and `After` properties (`candidate2_cache_managed_preflight.py:22,39-46,204-206`). Role membership, property keys, token membership/multiplicity, and all scalar strings remain part of the exact dictionary comparison. Independent execution with the rescue-production interpreter confirmed that reordered set tokens compare equal, while a changed dependency member and a changed `ExecStart` scalar both compare unequal. This removes nondeterminism without weakening the authorization binding.

## Final-byte and deployed checks

- Contract SHA-256: `4323bf1f4c83a0f03351200f8e2769555ea8f819868b90bc9cee40988c066f48`.
- Source manifest SHA-256: `2d1ff9cae67dc0d8993a6a00495adea5092860dec407fff15173725acf3b4ec2`. All 31 declared files exist, no undeclared file remains in the installed source tree, and every packet and installed member matches its manifest hash.
- Runner SHA-256: `f62f8c5f42165414233f3532c4ba1a334563bae0bf947b9ac4fe1ae3c5f2da2b`. Relative to the sealed v6 source closure, executable deltas are the already-reviewed preflight self-observation repair, this runner gate repair, and restoration of the previously omitted dependency.
- Final preflight SHA-256: `e1fbc08a0960021abad0ad3b3768b8ef0c41c69e9835f48feceeb961d23d78da`; its only delta from the previously accepted repair is the three-property canonicalization above.
- Restored `pipeline/kilosort_cache_compat.py` SHA-256: `3b28a67b84ea2e259d5d1923d549c1715c97c17490e6dfd3990ea615e8f4da68`; it exactly matches commit `3c9a1e2` and is required by the bounded-reader module's import closure.
- Packet and installed target service SHA-256: `c2cb7f3f4b668b8e1b195f825da134ffe9710de37b6707b5cc2832ad1e03a712`. Monitor service: `7fd09574eb2e52554ca36079d71e83fd7f4d352ae35be0c987c213ab642f3713`.
- Independent manager query confirmed both units are loaded but inactive/dead with `MainPID=0` and `ControlPID=0`; `Restart=no`, `TimeoutStartUSec=infinity`, `RuntimeMaxUSec=infinity`, `OOMPolicy=stop`, and memory/CPU limits are infinity. Loaded commands bind the exact contract/source hashes and isolated bootstraps.
- `TEST_RECEIPT.json` SHA-256 `e3d52a4f9c8e125713d8a51c84ff5be4e46c6e535a2252e94d15cd5c82aba885` agrees with independently repeated canonicalization, late-arrival, runner-join, GPU, monitor-forwarding, and monitor tests.
- `DEPLOYMENT_REVIEW_EVIDENCE.json` SHA-256 `6d1ff5be07f068d2e6e96bae8e752a549d50fa6b3d3ada1cd67c61ecefb3df39` agrees with independently checked installed bytes and manager state.
- The fresh v7 output and `AUTHORIZATION.json` are absent. No hard cap, finite timeout, restart, automatic retry, science change, parser/monitor/cache-mode change, or new attempt path was introduced.

## Scope and residual limits

The accepted exact-self rule excludes only the current runner PID. Query errors, malformed/unknown rows, non-self compute applications, low free memory, and competing sorter processes fail closed at both runner gates. As with any point-in-time prelaunch gate, this does not guarantee that the host remains idle after the final check; the contract's managed isolation, monitoring, stop conditions, and single-attempt rule govern execution.

No recording binary was opened, read, or hashed in this review. No service was started, no authorization was created, and no sorter or scientific outcome was inspected.

## Implementation checks

- Done: traced the exact runner source from preflight receipt through both resource gates to `run_sorter_config` -> the prior late-arrival hole is closed (`candidate2_cache_managed_full.py:104-120,179-215`).
- Done: independently executed the actual rescue-production late-arrival and isolated-runner join fixtures -> external GPU and sorter arrivals fail before output/sort; restored closure imports successfully.
- Done: independently executed the actual rescue-production systemd-set fixture -> ordering alone is ignored; changed membership and changed scalar bytes fail comparison.
- Done: verified all 31 packet and installed source members, bootstraps, service bytes, loaded commands and lifecycle/cap properties -> exact bindings agree and units remain inactive.
- Done: repeated the five-case GPU self-observation, monitor forwarding, and 11 monitor known-answer tests -> accepted unchanged behavior remains intact.
- Not done: authorization creation, live managed preflight, recording validation, service start, full sort, curation/QC, or scientific comparison -> outside this focused execution-disabled review.
- Can establish: the final v7 implementation and deployed inactive bindings are technically fit to proceed to one separately authorized, freshly reconciled managed attempt under the exact contract.
- Cannot establish: future resource availability, execution success, output validity, or scientific benefit; those require the prospective managed run and terminal review.
