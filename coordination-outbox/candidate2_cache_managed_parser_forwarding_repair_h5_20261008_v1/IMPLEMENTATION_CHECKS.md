# Implementation checks

- Done: v5 source tree, source manifest, target bootstrap and preflight bootstrap are byte-identical.
- Done: monitor-bootstrap executable delta is one parser construction argument, `allow_abbrev=False`.
- Done: permanent subprocess regression executes the actual bootstrap with option names/order extracted from the frozen service and proves the inert child receives `--target-unit` intact.
- Done: Candidate2 input, full parameter equality rule, effective nblocks=1 rule and three-operation cache-managed mode are unchanged.
- Done: fresh attempt v6 derives output, results, partial, launch, monitor and unit identities consistently.
- Done: service changes are limited to attempt names/paths, corrected monitor-bootstrap hash, new packet path and new contract hash; no restart or hard cap was added.
- Done: v5 failure packet and verdict remain preserved and explicitly bound.
- Not done: focused local independent review, installation, loaded-unit capture, authorization, fresh preflight and execution remain pending.
- Can establish: the known argument-forwarding defect is corrected at the actual bootstrap-to-child boundary without changing science or monitor behavior.
- Cannot establish: current launch readiness, run completion, memory outcome or scientific benefit until actual deployment/execution evidence is reconciled.
