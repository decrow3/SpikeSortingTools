# Implementation checks

- Done: v7 failure is preserved and explicitly bound; no v7 retry occurred.
- Done: executed preflight source now uses `Manager.GetUnit` and typed `ExecStart`/`ExecStartPre` `a(sasbttttuii)` values.
- Done: stable identity retains exact executable path, argv array, `ignore_errors`, command order/cardinality and argument boundaries; unsupported schema/query failure rejects.
- Done: non-command scalars and dependency membership remain separately strict; no regex/text stripping or new authorization layer was introduced.
- Done: the preflight's output `Path` survives the systemctl loop and reaches receipt-parent creation.
- Done: actual rescue-production full preflight main passed a realistic inactive-to-active dummy-service lifecycle and serialized a valid receipt; path/argv/ignore-errors negatives rejected.
- Done: exact-self/external/low-memory/query/malformed GPU fixtures, late-arrival runner gates, monitor forwarding, monitor known answers and source closure remain covered.
- Done: Candidate2 input, full parameter equality, effective nblocks=1, cache-managed three-operation mode, no-retry lifecycle and absence of hard memory/runtime caps are unchanged.
- Done: focused independent review returned GO after independently rerunning the actual full-preflight dummy lifecycle and all required negative cases (REVIEW_STATUS.json SHA-256 91be0ce096640ef0c24ad347c17e4dafa43e74b5d10f4dcac68cd8c29204d775).
- Done: a fresh final deployed no-start capture reproduced the reviewed typed command identity exactly; its sole scalar text delta was dependency-set ordering, which the reviewed comparator canonicalizes (evidence/FINAL_PRESTART_SYSTEMD_IDENTITY.json SHA-256 23a67b885d43a954d61a06b2a25e566d8086cdbeac92aca3be5bcf1c3a261323).
- Done: AUTHORIZATION.json binds the exact contract, 31-member source manifest, installed service bytes, review GO, v7 preserved failure, loaded scalar properties and typed execution identity for exactly one fresh v8 attempt with no retry (SHA-256 2baa2148dadc8aaa76a56c2c574d7e18c9950eec2d457bc6e5f3eea2ccdba83e).
- Not done: production preflight, recording validation, sort, curation/QC, memory outcome and scientific comparison remain pending the single managed start.
- Can establish: the complete preflight path, stable systemd identity, deployed bindings and one-attempt release are implementation-reviewed and reconciled without recording access.
- Cannot establish: run success, memory outcome, complete outputs or scientific benefit before execution and terminal reconciliation.
