# Candidate 3 H5 equality contract v3 correction

V3 preserves v1 and v2 and repairs the sole remaining v2 NO-GO.  The v2
independent-review hashes are `0851c6142bdff132a21a1538f97556e166910978f8446c1c32ab572f5b8d9cd0`
(manifest) and `9ddbbecb9a6011fb9e2b7b0a31421bf1717913898b4791496e7a2f7c1ffd3426`
(complete).

The contract now freezes absolute H5 runner, interpreter, repository, contract,
Kilosort-site, output, and outer-receipt paths.  Real mode compares all of them.
The coordinator dispatch must bind every semantic command argument, including
the exact review and resource-receipt paths and `execute-real` mode.  The
managed-resource receipt must independently bind the same resolved execution
and output topology plus all resource limits.  The output path is no longer a
caller choice and remains fresh by `mkdir(exist_ok=False)`.

No scientific logic, windows, producer, tolerance, or resource budget changed
from v2.  The exact v3 full-shape synthetic self-test passed 9/9 requests with
zero mismatch, both positive controls differed, and recorded-voltage bytes were
zero.

Implementation checks
- Done: exact path/argument/output fields are traced from contract to real-mode
  comparisons in `RUNNER.py`.
- Done: dispatch and resource receipts independently converge on the identical
  resolved output and execution topology.
- Done: v3 source hashes and a full-shape success seal are included.
- Not done: H5 prestart receipts or real equality; these require a later exact
  managed dispatch.
- Can establish: the v2 release-join defect has a concrete enforced repair.
- Cannot establish: H5 readiness, real voltage equality, sorting benefit,
  biological identity, purity, or full-session performance.

Requested review status is `GO_EXECUTION_READY` only if this exact v3 contract
and runner close the v2 blocker.  This review does not authorize H5 execution.
