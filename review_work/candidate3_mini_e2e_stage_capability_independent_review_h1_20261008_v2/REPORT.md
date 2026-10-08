# Candidate-3 miniature stage-capability independent review

Verdict: `PARTIAL_GO_NATIVE_OUTPUT_CAPABILITY_WITH_CONFIG_AND_LAUNCH_CAVEATS`.

The retained native ordinary-arm outputs support the packet's narrow capability claim: the synthetic prewhitening boundary entered DARTsort with `preprocessing="none"`, threshold detection ran, intermediate clustering/refinement labels were saved, one final matching pass ran, final agglomeration labels were saved, and the final NPZ reloads. This does not establish real-input behavior, performance, identity, purity, transition-arm behavior, or exact-A earliest-divergence attribution.

Publication note: independent-review v1 reached shared storage only as an unsealed staging packet before the exact-source recovery arrived. It has no `COMPLETE.json`, was not overwritten, and must not be consumed. This v2 namespace incorporates the recovery and is the only sealed independent-review result.

## Material findings

1. **Exact executed wrapper source was recovered after the initial correction.** The pre-launch status receipt binds the executed full script to SHA-256 `5d55bb1e0257afbb87b60d9d0d9dd5fcb4290a49ce2f67f8656969929e18801a`; commit `697d2bcf38087e6b2ba9ec8116c969b63a63debb` contains SHA-256 `e627b1aaba5355a6a156aab908bd6bc4e19f82cfffbb359ffcaf1ee1ef67d32e`. Deterministic replay of the retained root-session Add File and the only two pre-launch patches reconstructs a source file whose SHA-256 is exactly `5d55...`. Independent diff shows `prepare` and `run_arm` are byte-identical to the committed source; the only differences begin in `summarize` and are post-run reporting/accounting additions. This resolves producer-configuration inspection for the completed ordinary run, while the later committed summarizer is the source that exactly reproduces the packet accounting. Recovery from a session transcript is weaker operational practice than freezing source before launch, so a pre-launch source snapshot remains mandatory for the real snippet.

2. **Initial detection retained consequential defaults.** Resolved configuration has `subsampling_spikes_per_channel=5000`, `subsampling_presence=0.025`, and `peeler_sampling_cfg.chunk_sampling="kmeanspp"`. The implied cap was 120,000 detections. The miniature run produced 47,624 detections, so the cap did not bind, and all ten 30,000-sample chunks appear exactly once. They were processed in shuffled order `[240000,0,60000,30000,180000,270000,120000,210000,150000,90000]`; `last_chunk_start=90000` is the last processed chunk, not the recording tail. This supports whole-mini-input capability but cannot be generalized to complete detection on a longer real input unless subsampling/order are explicitly frozen.

3. **Row ordering and stage boundaries must remain explicit.** Detection, matching, and final times are not globally sorted (5, 1, and 2 adjacent inversions respectively). Final channels equal matching channels, but 79/2,297 final sample times differ from matching times by +1 sample, and final labels differ after agglomeration. The reported 1,034/1,263/46,590 exact multiset accounting is specifically detection-to-matching using pre-final matching `(sample, channel)` keys; it is not final-row identity or causal attribution. Any downstream sort-by-time must apply one stable permutation to every row-aligned array.

4. **Resource stop is supported by retained state, not by a managed launch receipt.** The ordinary output directory is 128,223,957 bytes, above both 120,000,000 bytes and 120 MiB. The transition input exists and is hash-bound, but no v3 transition run directory or transition output exists; stopping before that run is consistent with the frozen rule. The recorded launch identifier is an interactive tool session rather than an independent job manager, and no immutable launcher/final-status receipt proves disconnect survival or external stop enforcement. The completed outputs remain reviewable, but this run does not validate a production launch method; the real snippet must use a managed service or scheduler.

5. **Exact-A claims are appropriately asymmetric when read as packet-bound metadata.** `INPUT_BINDINGS.json` SHA-256 `7c045a88be1819d5411534867ee717ceadf76661c3bb3ba176dd525ed5c4e3b1` binds the accepted recording identity and final `spike_times`, `spike_clusters`, `spike_templates`, `templates`, and `ops` exports, while its README explicitly says the model-derived template summaries are not the unavailable pre-extraction detection bank. No bound matching-candidate or refinement-transition artifacts are listed. Prospective capture of immutable candidates, matching decisions/scores, pre-refinement labels, and row/time/template-linked membership transitions is a supported requirement for future exact-A-style stage attribution; its sufficiency and minimality still require implementation review against the future Kilosort path. It cannot recover historical A.

## Independent recomputation

- Packet v3: 6/6 members match; MANIFEST `31114d69be1871d5f0ca44078c34c5bef69e527a9d558188a748dde8cf58fb45`; COMPLETE `03a99e7caacd36dd3344561af97b85bfcf0b5874b4a60ec69f995d0dffb0d3d4`; birth times place COMPLETE after every payload and MANIFEST. Failed v1 and v2 are also internally sealed with `complete=false` and remain unchanged.
- Provenance correction v1r1: 2/2 members and all rebound native hashes match; MANIFEST `4d7aed65488a56517307c41b976eb5136950035e3e7a3e018da41c65d672ae45`; COMPLETE `abf39aa132e6a6bf5a4336e03d7daf0cfbc7b40fbd0c453deaa3232c32693d65`; COMPLETE birth time follows payload and MANIFEST. Its then-correct unresolved verdict is superseded only by the subsequently recovered exact source packet bound below.
- Executed-source recovery: 3/3 members match; recovered source SHA-256 exactly equals the pre-launch `5d55...` binding; MANIFEST `747618557284507285754a868e44f124811d15554a9c94a108ed5cf84dd0d2ff`; COMPLETE `11602c0bcf06e9cdff5064e562d799f4b0e3578b167d7aedbf883c40caea9364`; strict payload-before-MANIFEST-before-COMPLETE birth ordering passes.
- Current summarizer rerun against retained scratch reproduced packet `ACCOUNTING.json` and `ACCOUNTING.npz` byte-for-byte.
- Detection: 47,624 unique `(sample, channel)` rows; exact `times_seconds=times_samples/29999.835983263598`; all packetized times/channels equal scratch.
- Matching: 2,297 rows, 18 nonnegative labels, 95 used template indices, zero negative matching labels; every packetized matching field equals scratch.
- Final: 2,297 rows, 11 nonnegative units, 4 negative rows; saved final times and labels equal scratch; row order is unsorted.
- Every one of the 11 intermediate/final label snapshots equals retained scratch, with reported row, negative-label, and nonnegative-unit counts reproduced.
- Independent `Counter` multiset arithmetic gives matched `1,034`, added `1,263`, removed `46,590`, satisfying both row-count identities.
- Exact clock repair is correct: `duration=297000/29999.835983263598=9.900054125818931 s`; the last sample is `9.900020792303355 s`. V1's fixed `[0,9.9)` support excludes it; v2/v3 duration-derived cells cover `[0,duration)`.

## Implementation checks

- Done: adapter/producer bindings -> retained input hashes and adapter/Kilosort library-source hashes match the input receipt; the recovered executed `prepare` source binds exact half-open cells, zero-fill remap, channel centering, median CAR, 300 Hz high-pass, and no whitening.
- Done: actual DARTsort state -> editable checkout is clean at `edcfe1b51d672b4136eb13cc78c0875da804b851`; resolved config and native H5/NPZ files show threshold detection through final agglomeration.
- Done: clock/frame support -> independently recomputed v1 exclusion and v2/v3 exact-duration coverage; H5 seconds equal samples divided by the saved sampling frequency.
- Done: counts/lineage/accounting -> independently recomputed all key row, label, template, noise, ordering, and exact multiset counts directly from retained H5/NPY/NPZ artifacts.
- Done: resource and transition closure -> ordinary retained bytes exceed the frozen cap; no v3 transition run directory/output exists.
- Done: exact-A boundary -> inspected the bound INPUT_BINDINGS and README without opening remote H5 voltage or sorter arrays.
- Done: exact executed wrapper-source comparison -> transcript reconstruction hashes to the pre-launch `5d55...` receipt; `prepare`/`run_arm` are byte-identical to committed source and only `summarize` differs.
- Not done: managed-job survival/stop proof -> launch receipt identifies only an interactive tool session.
- Not done: transition run, real voltage, RF/holdout, D3, or scientific comparison -> excluded by this review.
- Can establish: saved native artifacts demonstrate ordinary synthetic stage capability and exact compact accounting within the resolved mini-run configuration.
- Cannot establish: the entire commit `697d2bc` was executed unchanged (its reporting block is later), full-detection behavior on longer data, transition or real-input behavior, safe production launch, earliest exact-A divergence, or scientific benefit.
