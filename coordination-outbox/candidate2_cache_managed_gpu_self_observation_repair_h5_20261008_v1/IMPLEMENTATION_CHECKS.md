# Implementation checks

- Done: 28/30 v6 manifest members and all three bootstraps are byte-identical; preflight and runner change, and the exact historical cache-compat dependency omitted by v6 is restored as member 31.
- Done: accepted monitor-bootstrap `allow_abbrev=False` repair and its exact service-boundary regression remain unchanged.
- Done: changed metric records raw post-init compute rows plus current PID and excludes only an exact PID match; external processes, query errors, malformed/unknown rows and low memory fail closed.
- Done: actual rescue-production interpreter subprocess passes self-only, self-plus-external, low-memory, query-error and malformed-data case groups.
- Done: actual isolated runner import passes the scientific-parent contract join and reaches the deliberately absent preflight boundary; this caught and repaired the omitted dependency before launch.
- Done: a clean first runner gate followed by either a late external GPU PID or a late competing sorter fails at the final gate.
- Done: repeated real `systemctl show` reads exposed nondeterministic set-token order before launch; the comparison now canonicalizes only `Requires`/`BindsTo`/`After`, and a fixture rejects changed membership and changed scalar bytes.
- Done: Candidate2 input, full parameter equality rule, effective nblocks=1 rule and three-operation cache-managed mode are unchanged.
- Done: fresh attempt v7 derives output, results, partial, launch, monitor and unit identities consistently.
- Done: service changes are limited to attempt names/paths, new source-manifest/contract hashes and packet path; no restart or hard cap was added.
- Done: v5 and v6 failure packets and verdicts remain preserved and explicitly bound.
- Done: final source and units are installed inactive; loaded argv hashes match; both units are dead with PID 0 and retain infinite MemoryHigh/MemoryMax/RuntimeMax.
- Not done: focused rereview of the final runner-gate repair, authorization, fresh preflight and execution remain pending.
- Can establish: exact-self CUDA observation is distinguished from external GPU competition without weakening the free-memory threshold.
- Cannot establish: launch readiness until rereview; run completion, memory outcome or scientific benefit until actual execution evidence is reconciled.
