# Candidate-3 mini transition managed-launch v2 independent review

Verdict: `BLOCKED_PENDING_PREFLIGHT_BINDING_AND_LAUNCH_RECEIPT_REPAIR`.

The v2 repair closes the shared-source, child/process-group, in-root cache, cap-monitor, and split-root reporting mechanics. It is not yet safe to launch because the executable preflight does not bind every artifact that the frozen reporter consumes, does not recompute the retained/input baseline, checks an editable DARTsort checkout by HEAD only, and still cannot durably report an outer `systemd-run` failure or the promised timestamped terminal manager state.

## Blocking findings and exact repair

1. **Consumed artifacts can change without rejection.** `reporter.py:182-292` reads `truth_events.npy`, enumerates every retained ordinary H5/NPZ, loads every `*labels.npy`, and uses threshold/matching contents. `RUN_CONTRACT.json` binds the input receipt and transition boundary, but not `truth_events.npy`; it binds ordinary config, timing, and final sorting, but not threshold, matching, label, model, receipt, motion, or other enumerated members. The independent fixture changed an unbound input and an unbound ordinary member after a valid preflight; both altered states still passed. Repair by freezing a canonical recursive inventory of every input and retained-ordinary regular file (path, size, SHA256), rejecting symlinks, extras, missing members, and hash/size changes. At minimum, explicitly verify the receipt's nested truth hash, but the reporter's recursive enumeration makes a complete tree inventory the safer exact boundary.

2. **The storage baseline is trusted rather than measured.** The monitor adds the scalar `baseline_bytes` to the fresh-root size but preflight never recomputes the input/ordinary inventory. Current regular-file totals independently measured as 57,081,256 input bytes plus 128,211,669 ordinary bytes = 185,292,925, whereas the frozen scalar is 185,309,309 (16,384 bytes higher). A conservative overcount is not a cap escape today, but mutation or growth can become one. Repair by deriving the baseline during preflight from the same canonical inventory, requiring exact agreement with the frozen expected total, recording the measured value, and passing that measured value to both child monitors.

3. **The executed editable checkout is not content-clean-bound.** The DARTsort checkout is currently clean at the expected `edcfe1b5...` commit, but `verify_preflight` only calls `rev-parse HEAD`. The venv imports DARTsort from `/home/huklab/Documents/DARTsort/src`, so an uncommitted source edit can execute while HEAD still matches. Repair by rejecting nonempty `git status --porcelain --untracked-files=no` (and preferably freezing hashes for the load-bearing imported source or a reviewed tree digest).

4. **Manager failure closure remains incomplete.** `ExecStopPost` correctly saves `SERVICE_RESULT`, `EXIT_CODE`, and `EXIT_STATUS`, including when the main wrapper cannot finish, but its receipt has no timestamp or terminal property snapshot. If `systemd-run` itself fails before unit creation, `ExecStopPost` never runs and no frozen executable captures the outer return code. Repair with a small frozen launch wrapper that validates MANIFEST/COMPLETE/GO and service/path freshness, atomically records the exact outer argv and launch return/acceptance, and a terminal collector/finalizer that durably records UTC time plus the frozen `systemctl show` property set. These receipts must be fresh/nonsymlink and required by completion.

## Verified closures

- Author packet MANIFEST `149a2d3...`, COMPLETE `558b10ec...`, and RUN_CONTRACT `f8c2f23f...` match exactly; all six manifest members match and birth ordering is payload < MANIFEST < COMPLETE.
- The exact systemd argv executes shared immutable packet source. Runner, producer, and reporter match `92879fe4...`, `5d55bb1e...`, and `e627b1aa...`.
- The five author tests pass in 2.07 seconds. A named stale hash is rejected; a symlink attempt is rejected; the child is terminated on cap breach; the finalizer records the systemd terminal tuple.
- The frozen reporter independently traversed a temporary split-root symlink composition (ordinary linked as both arms only as a plumbing fixture) and emitted ACCOUNTING.json `1bdadcb8...` (159,003 bytes) and ACCOUNTING.npz `216a0dd1...` (1,361,143 bytes). Both arms reloaded with identical known-answer counts.
- The service remains `not-found`/inactive/dead with no start or exit timestamp; attempt root and both external receipts are absent. No launch occurred.
- Failure paths preserve the fresh attempt and are not automatically retried. The transition remains a whole-stage restart in a new namespace after investigation, not an intra-sort checkpoint.

## Implementation checks

- Done: actual runner/finalizer/argv -> shared source, process-group handling, `Restart=no`, in-root cache, cap monitor, and split-root report path work as scoped.
- Done: positive, stale-hash, symlink/freshness, cap, finalizer, and split-root fixtures -> expected behavior reproduced; two unbound-artifact false passes found.
- Done: source/environment binding -> exact source hashes and current clean DARTsort HEAD confirmed; dirty-state enforcement absent in code.
- Done: resource arithmetic -> current baseline independently measured and shown not to equal the frozen scalar; launch-time recomputation absent.
- Done: failure closure -> inner wrapper and ExecStopPost tuple work; outer-launch failure and timestamped terminal snapshot remain unimplemented.
- Not done: service launch, transition sort, real voltage, RF/holdout, or D3 -> expressly excluded.
- Can establish: v2 substantially repairs the managed mechanics and the reporter can consume a split-root synthetic composition.
- Cannot establish: immutable provenance of the reported ordinary/truth comparison, exact cap accounting at launch, or complete durable manager closure until the four repairs above are independently reviewed.

Required decision: `NO_START`. Preserve v2 unchanged and publish a fresh v3 repair namespace.
