# Review of longitudinal development contributions (`dde66c8`)

**Fix follow-up:** all five findings below have been addressed in the working
tree. The focused suite now passes 58 tests, including a real temporary-Git
receipt, tiny synthetic strip materialization/validation/reuse, mocked sorter
dispatch with real slice validation, downstream-settings/completion refusals,
and lost-unit/no-correspondence coverage cases. Comparison schema v2 retains
the fixed baseline denominator and exports `baseline_eligibility.csv`. No sort
was launched, and the user-requested execution hold remains in place. These
fixes do not implement checkpoints or validate job survival.

Reviewed the development contract, strip preparation, arm execution, shared
comparison, CLI and focused tests. No sorts were run; the full-session hold
remains in place. This is a targeted review, not an exhaustive audit.

The separation of contracts, preparation, execution and comparison is useful.
However, the workflow is not ready for expensive execution or scientific
advancement decisions.

## Findings

1. **P1 — repository receipt always raises.**
   `testing/development_strip.py:37–40` requests binary subprocess output and
   calls `.encode()` on the resulting bytes. A direct call in the real repository
   raises `AttributeError: 'bytes' object has no attribute 'encode'`. This blocks
   `run-arms` and fails strip preparation after materialization has already done
   its expensive work. Hash the bytes directly. Exercise the real helper in a
   temporary Git repository; current runner tests mock it away.

2. **P1 — prepared strip receives the wrong recording schema.**
   `testing/development_strip.py:201–204` sets the accepted-recording schema and
   then expands `request`, which overwrites it with
   `longitudinal-development-strip-v1`. Production validation requires
   `rescue-recording-manifest-v2`, so after fixing finding 1 the newly prepared
   recording still cannot pass `run-arms` or cache reuse. Set the recording
   schema after expanding the request and retain the strip schema separately.
   Test a tiny materialization through the real accepted-recording validator.

3. **P1 — coverage excludes units lost by the candidate.**
   `testing/sort_comparison.py:440–448` divides measurable pairs by surviving
   interior primary matches. A small fixture with three baseline units, only
   one retained candidate unit, and valid fits returns measurable fraction 1.0
   and `ready_for_pareto_review`. The unmatched baseline units disappear from
   this gate. Report matched-pair conditional coverage separately and use a
   fixed baseline-eligible cohort for the decision denominator, with unmatched
   and non-interior partners explicit. This report does not itself auto-promote,
   but its coverage result can wrongly qualify a candidate for the next review.

4. **P1 — arm execution validates ancestry, not the contracted slice.**
   `testing/development_runner.py:56–58` accepts any valid recording with the
   source request digest, or any strip derived from that source. It does not
   check the requested time range, processing/scoring bands, geometry, or exact
   prepared-strip request. Passing the full recording or a different strip
   through `--recording-dir` therefore launches all arms and labels their output
   with a contract they did not execute. Bind the prepared request/manifest to
   the contract and reject mismatched time or channel selections before sorting.

5. **P2 — reused downstream output is not checked against the curation profile.**
   `testing/development_runner.py:22–36` checks only the source sort identity in
   the curation/QC receipts. The same sort can have downstream results produced
   using different curation thresholds or QC settings. Such outputs are accepted
   and reported as the frozen profile; incomplete receipts or missing outputs
   are not rejected there either. Validate the saved stage requests against the
   specified shared settings, completion and required artifacts before reuse.

## Validation and next action

48 tests passed across `test_development_comparison`, `test_development_runner`,
`test_development_strip`, `test_sort_comparison`, `test_ladder_sorter`, and
`test_luke_full_session_compare`. The real provenance helper and dropped-unit
coverage reproductions above were additional checks, not covered by that suite.

Fix these issues and add a tiny prepare → validate → mocked-sort → compare
integration test before running a recording. Negative-control labels in an
evaluation JSON are not substitutes for executing the negative controls.

The new Luke configuration describes six arms, including threshold variants;
that is broader than the single rigid comparison last authorized in this chat.
Treat it as a separate proposal, not permission to launch a sweep or bypass the
current cancellation hold. Independent job ownership and checkpoint validation
remain separate execution requirements.
