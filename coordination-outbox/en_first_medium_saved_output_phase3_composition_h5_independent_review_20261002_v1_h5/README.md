# H5 independent review: corrected Phase3 v2

Verdict: `BLOCK_PAIR_OUTPUT_PROVENANCE_UNBOUND`.

The corrected v2 packet, COMPLETE, coordination status, and every manifest
member verify. Its `waveform.pair_coverage` correction is present, the compact
schema validates, and all seven managed-entry tests pass independently. The
strict 0.05 boundary, REF-first ordering, amplitude ancestry, missing-evidence
semantics, and no-false-completion mutation path behave as reported.

One load-bearing attribution defect remains. `_verify_phase3_pair_receipts`
accepts a pair `COMPLETE.json` without `execution_states` and does not relate
the request's repaired/repeat `curated_output` paths to completed arm paths or
hashes. The independent known-answer fixture runs the actual managed entry
with repaired/repeat data outside the pair namespace and a structurally thin
pair receipt; it returns zero and writes authoritative Phase3 COMPLETE.

Repair only this binding boundary in a fresh immutable namespace. Require and
verify both arm execution states and bind request output paths to those exact
completed namespaces. Preserve the estimator, matcher, cohort, thresholds,
schema, waveform/QC semantics, and v1/v2 evidence.

No real project data, outcomes, voltage, evaluator run, service, RF, holdout,
sort, or training was accessed or executed.
