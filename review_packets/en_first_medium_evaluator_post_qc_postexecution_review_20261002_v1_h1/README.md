# H1 independent post-QC evaluator review

Verdict: `EXECUTION_ACCEPTED_ZERO_REPRODUCED_ENDPOINT_NOT_DECISION_VALID_REPAIR_AND_MEASUREMENT_REDESIGN_REQUIRED`.

The H5 service and immutable result packet are valid: one start, successful exit,
zero retries, exact accepted source/config/QC/clock bindings, and no prohibited
access. H1 independently reconstructed the reported `0/61 = 0.0` population
fraction from the emitted tables.

That zero is not a valid B-specific efficacy failure. A saved-table REF
self-support control shows that only 4 of the 61 interior REF units can satisfy
the frozen two-window and 50%-of-600-seconds support rule. The endpoint therefore
cannot reach its required 31/61 even with a perfect candidate. This matches the
historical warning that common-time amplitude completeness covered only a few
percent of eligible units and could not rank the population.

There is also one concrete load-bearing evaluator defect. `spike_positions.npy`
selects the spatial denominator and interior matches, but its bytes and
coordinate frame are not bound in the arm identity or execution-time input
receipt. This can matter: 31 of all 140 primary pairs meet the amplitude-support
rule, while none of the 14 selected interior pairs do. Earlier aligned hashes
and byte-identical derived metrics reduce evidence for an actual mutation, but
they do not cure the missing execution binding or establish coordinate meaning.

The targeted correction is narrow: bind and validate each position file's exact
hash and explicit frame, include it in identity/receipts/manifests, and reject
mutation, row permutation, or wrong-frame declarations before scoring. Preserve
this result; use a fresh namespace for one later saved-output replay.

Do not launch a new sort from this result. First redesign and freeze a feasible
amplitude/window endpoint, require a real REF self-control to pass, define the
currently undefined primary benefit statistic and null, and freeze complete
content-bound early/middle/late physical-channel waveform evidence. The 11,430
ambiguous rows are nonprimary correspondence-graph edges, not biological events
or verified split/merge instances.

Identity remains `UNRESOLVED`; waveforms remain `UNMEASURED`.

Implementation checks
- Done: verified the result manifest/COMPLETE, service lifecycle, executed source,
  effective config, saved-array/QC hashes, clocks, output inventory, and scope.
- Done: independently reconstructed all 61 denominator reasons, all 14 support
  failures, common-time intersections, matching/count closures, and `0/61`.
- Done: saved-table REF self-support control found `4/61`; consistent 450-label
  bijective permutation preserved all 14 pair support rows and the zero exactly.
- Done: traced spatial eligibility to an unbound position file and reviewed the
  ambiguity, primary-benefit, and waveform semantics.
- Not done: no new evaluator run, H5-local array rehash, recording/voltage,
  sorting/training/detection, repaired/repeat arm, waveform capture, RF, or
  holdout work.
- Can establish: exact execution and arithmetic, endpoint infeasibility, and a
  load-bearing spatial provenance/frame defect.
- Cannot establish: B-specific inferiority, biological identity, waveform
  stability, prospective repair efficacy, repeatability, or advancement.
