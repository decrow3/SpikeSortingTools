# Candidate 3 miniature stage-capability result

Verdict: `PARTIAL_GO_ORDINARY_END_TO_END_TRANSITION_STOPPED_RESOURCE_BOUND`.

The ordinary synthetic arm executed the intended boundary and native DARTsort entrypoint through threshold detection, initial clustering/refinement, template estimation, matching, assignment and final agglomeration. Final output reload succeeded with 2,297 events, 11 nonnegative units and four noise-labeled events. Training here was only to exercise native stages; these counts make no benefit, transfer, purity, or biological-identity claim.

Detection produced 47,624 event rows. Every intermediate label snapshot is summarized and stored in the compact `ACCOUNTING.npz`; negative rows and nonnegative label counts are separated at every stage. Matching produced 2,297 rows. Under exact `(sample, channel)` multiset accounting, 1,034 matching rows correspond to detection rows, 1,263 are added and 46,590 detection rows are removed. These are construction/accounting categories only, not causal attribution or biological identity.

The ordinary run took 842.35 s, dominated by initial clustering/refinement (777.10 s). Its output occupied 128,223,957 bytes (about 122.3 MiB), exceeding the frozen 120 MB limit. The transition input boundary was generated and hashed, but its DARTsort run was not started. Its downstream stages are therefore unavailable in v3; no resource expansion is inferred.

Exact A remains asymmetric. Its input and final spike/template/cluster exports are saved, but the bound packet explicitly says the pre-extraction detection bank is unavailable. No matching-candidate or refinement membership transitions were found. Therefore cross-sort earliest-divergence attribution against A is unavailable. Candidate-3 internal lineage and final-to-final comparison may still be useful, but they cannot establish where A first diverged.

Minimal prospective Kilosort instrumentation is needed only if a future exact-A-style comparator must support stage attribution: save immutable detection candidates, accepted matching events/scores, pre-refinement assignment labels, and each refinement membership transform with row/time/template lineage. It cannot retroactively recover historical A. No additional Kilosort instrumentation was needed to exercise the already bound prewhitening input boundary here.

## Exact missing links and ownership

- Transition downstream execution: unavailable because v3 hit its byte cap; owner H1/Hub scope choice if a new bound is desired.
- Real Arm-A Candidate-3 snippet: not run; owner H1 after host/input/storage route and real contract are frozen.
- Exact A early stages: historical inputs absent; owner is a future prospective comparator instrumentation task, not replay of A.
- Large Candidate-3 detection/matching HDF5 durability: retained only in `/tmp`; compact complete row/accounting snapshots are packetized. Owner H1 at the next real snippet to bind durable large-output storage.
- H1 versus H5: remains pending real input/storage/I/O choice; H1 GPU runtime is GO and H5 is fallback.

Implementation checks
- Done: actual input adapter -> source-bound `ExactLatticeRemapRecording` and installed Kilosort `BinaryFiltered.filter` generated finite ordinary/transition float32 boundaries with full hashes and voltage statistics.
- Done: actual DARTsort entrypoint -> ordinary arm ran at commit `edcfe1b5...` with saved resolved config, timing, detection/matching HDF5 schemas, intermediate labels and final NPZ.
- Done: clocks/frames -> exact 29,999.835983263598 Hz, 297,000 frames and exact duration-derived motion cells; v1 uncovered-tail failure is preserved.
- Done: event/label/template accounting -> all ordinary rows and lineage fields are in `ACCOUNTING.npz`; ambiguity/noise and exact added/removed/matched counts are separate.
- Done: reload -> final NPZ reloads to 2,297 events; its row order is not time-sorted, so no sorted-order assumption is made.
- Done: A capability inspection -> input/final exports are bound; pre-extraction bank is explicitly unavailable; no early-stage agreement inferred.
- Not done: transition DARTsort stages -> stopped on the frozen resource bound after ordinary output reached 128,223,957 bytes.
- Not done: real Arm-A equality/padding or scientific performance -> require the separately frozen real snippet.
- Can establish: the ordinary synthetic intended boundary can exercise native Candidate-3 stages and save compact assignment/refinement lineage; exact-A earliest-divergence attribution is unavailable from current artifacts.
- Cannot establish: performance benefit, transfer, identity/purity, real-voltage behavior, transition-stage behavior, or full-session readiness.
