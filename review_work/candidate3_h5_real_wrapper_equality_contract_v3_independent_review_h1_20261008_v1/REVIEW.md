# Independent delta review: candidate-3 H5 real wrapper equality contract v3

## Verdict

`GO_EXECUTION_READY`

The exact v3 contract and runner close the sole v2 NO-GO: the real-mode
invocation and output topology are now frozen, independently joined through the
dispatch and managed-resource receipts, and enforced before output creation or
binary access. This is a scoped contract-readiness verdict. It is not a claim
that H5 is currently ready and does not itself authorize or launch execution.

Reviewed immutable identities:

- author MANIFEST `0dadacfd850286456345d3b9d447fb1bc38a94da171c889347a2a3e914c032f3`
- author COMPLETE `eb42b4446d4818b25d8abbfadfc7096d83503bcb0fd54707e7fd94c61fc3c438`
- contract `6cb5c29b806b63021bf9edef87535718f24efc9ff27842d9e4e34c3e06e56037`
- runner `0bed993b8718f29591b92884e88b65f2bf8a6780cce49490b869ba82823f1f7e`
- preserved v2 NO-GO MANIFEST `0851c6142bdff132a21a1538f97556e166910978f8446c1c32ab572f5b8d9cd0`
- preserved v2 NO-GO COMPLETE `9ddbbecb9a6011fb9e2b7b0a31421bf1717913898b4791496e7a2f7c1ffd3426`

## Delta checks

1. **Exact execution identity.** The contract freezes absolute H5 paths for the
   interpreter, repository, runner, contract and appended Kilosort site. Real
   mode resolves and compares every one. Runner content is also self-hashed
   against the contract before output creation.

2. **Exact output identity.** The contract freezes
   `/mnt/NPX/Luke/DARTsort_motion_experiments/candidate3_h5_real_wrapper_equality_h5_20261008_v1`.
   Real mode resolves `--output` and rejects any other path before creating it.
   `mkdir(exist_ok=False)` then enforces freshness.

3. **Coordinator invocation join.** The hash-bound dispatch must match the exact
   task/status, contract and runner hashes, review and resource receipt hashes,
   interpreter, runner, contract, output, appended site, resolved review and
   resource receipt paths, and `execute-real` mode. The runner compares these
   semantic fields before binary access.

4. **Independent managed-topology join.** The separately hash-bound resource
   receipt must repeat the identical interpreter, runner, contract, output and
   Kilosort-site paths; exact contract/runner hashes; all memory, wall, CPU,
   temporary, persistent-output and GPU limits; a manager job identifier; and
   the frozen outer receipt path. This closes the v2 possibility that a valid
   limit claim covered a different output topology.

5. **Outer receipt.** The frozen outer receipt path is required to preserve
   exact argv, manager identity/settings, stdout/stderr hashes, exit state,
   release hashes and sealed-terminal assessment even for pre-output failure.
   Its later existence and content remain mandatory H5 prestart/terminal facts;
   this review does not assume them.

6. **Unchanged scientific and guard semantics.** V3 changes only the release
   join. The windows, arithmetic, eager/lazy construction, positive control,
   sequential two-window read bound, internal resource guards, compact output,
   and exactly-one handled terminal behavior remain the reviewed v2 code.

7. **Fixture.** The exact v3 full-shape synthetic packet is internally sealed,
   contains one result, reports zero recorded-voltage bytes, passes all nine
   lazy/eager requests with zero mismatches, preserves both differing positive
   controls and uses the exact v3 contract and runner hashes. Re-running the
   unchanged scientific path was unnecessary because the independent v2
   full-shape reproduction already passed and the v3 diff is confined to
   pre-execution release binding.

## Scope and remaining gates

GO means only that this immutable specification is ready for a later exact
managed H5 prestart and dispatch. Before any execution, H5 must independently
produce the exact managed-resource receipt, coordinator dispatch must bind this
review COMPLETE and that receipt, the exact output and outer-receipt paths must
be absent/fresh as required, and all current resource/dedup checks must pass.
Any path, hash, receipt, resource, manager or freshness mismatch invalidates the
GO and fails closed. No retry is authorized by this review.

## Implementation checks

- Done: packet seals, committed source identity and v2-to-v3 source/contract
  delta -> exact path and receipt fields are present and consumed before output.
- Done: dispatch and resource schemas traced field-by-field -> both converge on
  the identical resolved execution and output topology.
- Done: exact v3 synthetic result and seal -> 9/9 comparisons pass, both positive
  controls differ, one result exists and recorded-voltage bytes are zero.
- Not done: H5 paths/state, live manager/cgroup limits, later receipts, recorded
  voltage equality, GPU/service/sort/RF/holdout/full launch -> outside this
  review and mandatory where applicable before execution.
- Can establish: exact v3 contract/runner execution readiness conditional on all
  frozen later H5 prestart, receipt and dispatch gates.
- Cannot establish: present H5 readiness, successful real equality, sorting
  benefit, biological identity, purity or full-session performance.
