# Candidate-3 mini transition managed-launch closure v3

Verdict: `FROZEN_AWAITING_INDEPENDENT_REVIEW_DO_NOT_START`.

V3 preserves blocked v1 and v2 and changes only launch provenance/resource
closure. The synthetic input, producer, reporter, DARTsort settings, 512 MiB
cap, 2,700-second runtime, 16 GiB memory limit, and scientific scope are
unchanged.

The runner now binds every regular file in the four-file input tree and every
regular file in the 22-file retained ordinary tree by relative path, exact size,
and SHA-256. It refuses missing, extra, changed, or symlinked files/directories.
This covers `truth_events.npy`, both H5 stage outputs, every intermediate label,
model/template artifacts, resolved config, timing, receipt, and final sorting
read or enumerated by the frozen reporter. Preflight recomputes input and
ordinary regular-file totals (`57,081,256` and `128,211,669` bytes) and requires
their sum to equal the frozen `185,292,925`-byte cap baseline. It also requires
the DARTsort worktree to be clean as well as at commit `edcfe1b5...`.

The prior v2 closures remain: shared immutable execution, exact contract/source
hashes, fresh nonsymlinked contained attempt, signal-safe child group,
`ExecStopPost` manager receipt, cache-in-root two-second cap monitoring, and
split-root compact accounting. Six focused tests pass, including adversarial
extra-file and same-size content mutation refusal. The v5 service/output/failure
receipts are absent; no launch occurred.

Implementation checks
- Done: v2 false-pass findings mapped to complete tree manifests, recomputed baseline, and clean-worktree gate.
- Done: six focused tests passed in 2.10 seconds; local full preflight is required to pass before release.
- Not done: focused independent v3 review -> mandatory before launch.
- Can establish: launch provenance/resource closure for the one missing synthetic transition arm if review passes.
- Cannot establish: transition outcome, real-input readiness, benefit, identity, purity, causal attribution, or full-session behavior.
