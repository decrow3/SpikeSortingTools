# Candidate 3 H5 equality contract v2 correction

This packet preserves and supersedes the execution-readiness claim of v1.  The
independent v1 review was a NO-GO (`MANIFEST`
`3bf7158d17e89e39accfb2379b94df0e2ae70df870d6ede299677bf852539dba`,
`COMPLETE`
`eeca0ac3b8f8ca1c02ca8879eba4842684b43f0033c6d0512b9ec561858be8db`).
Neither packet is overwritten.

V2 repairs the reported blockers:

- the contract binds the exact runner hash, interpreter, repository, contract
  path, appended Kilosort site, invocation fields, and preserved argv;
- the runner creates and owns its Numba/TMPDIR cache, so the caller no longer
  supplies an undocumented environment variable;
- review is a non-substitutable exact schema/status binding to both contract and
  runner; coordinator dispatch binds contract, runner, review, and managed
  resource receipt hashes;
- a hash-bound managed-resource receipt is required for external enforcement
  and identifies the persistent outer receipt that captures failures occurring
  before the runner can create its output;
- the runner additionally lowers address-space and file-size limits, installs
  an active wall timer, and watches its owned temporary tree every 0.1 seconds;
- terminal packet size is projected before any terminal member is written, and
  the sealed packet contains exactly one of `RESULT.json` or `FAILURE.json`.

The success fixture ran at both frozen full-window shapes with no caller-set
cache environment, passed all nine exact comparisons, used 5,001 temporary
bytes at the terminal check, and sealed exactly one result.  The independent
negative fixture changed the expected wrapper source hash; it failed before
array construction and sealed exactly one failure.  Neither fixture contacted
H5 or read recorded voltage.

## Implementation checks

- Done: v2 source and contract hashes are exact and self-checked before real
  binary access.
- Done: ordinary/transition arithmetic and equality semantics are unchanged
  from the independently reviewed coherent v1 design.
- Done: success and negative-control packets validate COMPLETE-last behavior
  with exactly one terminal member.
- Done: internal resource guards are active; external enforcement is a required
  hash-bound prestart input and persistent outer receipt, not an inferred
  property of the runner.
- Not done: H5 environment/receipt validation or real voltage equality; those
  remain gated on this focused review and a later exact coordinator dispatch.
- Can establish: the repaired contract and portable runner inputs close the v1
  execution-specification defects on local metadata/source/synthetic evidence.
- Cannot establish: that H5 currently satisfies the frozen paths/resources,
  real H5 equality, sorting benefit, biological identity, purity, or
  full-session performance.

Requested verdict: `GO_EXECUTION_READY` only if the exact v2 contract and runner
are sufficient for a later managed H5 job that supplies all three bound release
receipts.  Otherwise preserve a new precise NO-GO.  No H5 execution is requested
by this review.
