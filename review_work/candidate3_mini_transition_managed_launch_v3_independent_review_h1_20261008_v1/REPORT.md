# Candidate-3 mini transition managed-launch v3 independent review

Verdict: `BLOCKED_PENDING_LAUNCH_RECEIPT_AND_CONSUMPTION_REVALIDATION`.

V3 correctly repairs all three tree/resource false passes found in the v2 review. The four-file input and 22-file retained ordinary inventories are complete and exact; independent positive, same-size mutation, extra-file, and symlink-directory fixtures behave correctly. Full preflight passes with 57,081,256 input bytes + 128,211,669 ordinary bytes = the exact 185,292,925-byte baseline, and the editable DARTsort checkout is clean at `edcfe1b5...`.

Two launch-closure gaps remain:

1. The exact argv still invokes `systemd-run` directly. If unit creation fails, `ExecStopPost` never runs and no frozen executable atomically records the outer argv, return code, acceptance time, or failure. The finalizer still records only `SERVICE_RESULT`, `EXIT_CODE`, `EXIT_STATUS`, and presence of `FINAL_STATUS`; it has no timestamp or frozen terminal-property snapshot. Add the frozen launch receipt/terminal collector required by the v2 review, and require those fresh receipts for completion.

2. `verify_tree` runs only once before the transition child. The reporter later consumes mutable `/tmp` inputs and ordinary artifacts after the transition completes. A post-preflight change can therefore affect final accounting without detection. Re-run both exact tree verifications immediately before and after reporting (recording their results), or copy the verified inputs/ordinary artifacts into an owned immutable attempt area and report only from that snapshot.

Verified: subject MANIFEST `19d0fad9...`, COMPLETE `2d1ee71d...`, and RUN_CONTRACT `c5d4b4c2...` match; six members and COMPLETE-last ordering pass. Six focused tests pass in 2.09 seconds. The v5 service is `not-found`/inactive/dead with no timestamps; no launch occurred.

Implementation checks
- Done: complete input/ordinary membership, size, hash, symlink, baseline, and DARTsort clean-state gates -> independently pass and close the v2 false passes.
- Done: cap/finalizer/split-root mechanics -> unchanged reviewed v2 code; author tests pass.
- Done: timing of validation relative to consumption -> manifests are not rechecked after the transition or report.
- Done: outer failure closure -> direct `systemd-run` has no durable failure receipt; finalizer lacks timestamp/full property snapshot.
- Not done: launch, transition outcome, real voltage, RF/holdout, or D3 -> excluded.
- Can establish: v3 has exact launch-time tree provenance and baseline accounting.
- Cannot establish: that reporter-consumed trees stayed unchanged through use, or durable closure if unit creation itself fails.

Required decision: `NO_START`. Preserve v3 and repair in a fresh namespace.
