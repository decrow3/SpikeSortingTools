# Candidate-3 mini v3 executed-source recovery

Verdict: `RECOVERED_EXACT_EXECUTED_SOURCE_REUSE_ELIGIBLE_WITH_CONFIG_CAVEATS`.

The exact wrapper source used to launch the ordinary synthetic arm has been
recovered from the retained Codex root-session patch history. Its SHA-256 is
`5d55bb1e0257afbb87b60d9d0d9dd5fcb4290a49ce2f67f8656969929e18801a`,
exactly matching the prelaunch/running receipt. The recovered bytes are retained
at `source/candidate3_mini_e2e_stage_capability.py`.

## Reconstruction

The retained session JSONL was filtered to source-changing tool calls before the
ordinary launch at `2026-10-08T07:10:20.340Z`. Exactly three patches touched the
wrapper:

1. `2026-10-08T07:08:04.513Z`: initial full-file addition.
2. `2026-10-08T07:08:29.846Z`: replace rounded motion-cell timing with exact
   `N_FRAMES / FS` timing and record duration.
3. `2026-10-08T07:09:40.449Z`: set `detection_type="threshold"` and disable the
   defective `save_everything_on_error` path.

Replaying those edits yields the retained 9,468-byte source and the exact
prelaunch hash. The next source edit did not occur until
`2026-10-08T07:25:36.021Z`, after native output completion; it expanded only
`summarize()`. The subsequent `07:26:06.354Z` edit changed only that reporter's
loader call. A direct diff confirms `prepare()` and `run_arm()` are byte-identical
between the recovered launch source and commit `697d2bc`; differences begin at
the post-run `summarize()` function.

## Reuse decision

The ordinary native outputs are eligible for reuse as the ordinary arm of the
same synthetic mini-stage comparison because exact producer source, input hashes,
resolved configuration, DARTsort commit, native outputs, and accounting are now
bound. This does not waive the independent review's configuration caveats:
initial detection used the resolved defaults `subsampling_spikes_per_channel=5000`,
`subsampling_presence=0.025`, and `chunk_sampling="kmeanspp"`; the 47,624-event
mini run remained below its 120,000-event implied cap, processed all ten chunks
once in shuffled order, and does not define a full-recording detection contract.
The run's interactive launcher is completed evidence, not an approved launch
method for a future real or long sort.

Implementation checks
- Done: replayed all source-changing session patches before launch -> recovered source hash exactly equals the recorded prelaunch hash.
- Done: diffed recovered source against commit `697d2bc` -> `prepare()` and `run_arm()` are byte-identical; differences are post-run reporting/accounting only.
- Done: compiled the recovered snapshot -> syntax is valid.
- Not done: transition execution -> requires a separately frozen amended resource contract and reviewed launch receipt.
- Can establish: exact ordinary-arm producer-source recovery and reuse eligibility for the same synthetic mini comparison.
- Cannot establish: real-input readiness, transition outcome, benefit, biological identity, purity, causal attribution, or full-session behavior.
