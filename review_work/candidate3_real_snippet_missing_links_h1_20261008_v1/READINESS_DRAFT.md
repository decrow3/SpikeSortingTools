# Candidate 3 real-snippet missing-link draft

Status: metadata-only preparation, not a frozen execution contract and not authorization to launch.

## Cheapest adequate real check

Use two 10 s accepted Arm-A windows with 2 s read padding on each side. This is cheaper than a medium/full run and can establish real input-boundary execution, saved Candidate-3 stage lineage, and ordinary-versus-rounded-state-transition implementation behavior. It cannot establish longitudinal performance or benefit.

Provisional windows at the exact sampling rate 29,999.835983263598 Hz:

| arm | central seconds | central frames `[start,end)` | padded frames `[start,end)` | rounded-field context |
|---|---:|---:|---:|---|
| ordinary | `[6500,6510)` | `[194998934,195298932)` | `[194938934,195358932)` | inside the long 0 um rounded-state run `[6478.75,7254.75)` |
| transition | `[7249.75,7259.75)` | `[217491311,217791309)` | `[217431311,217851309)` | spans the 0 to +40 um rounded-state transition |

Frame conversion uses `floor(seconds * fs + 0.5)` with origin zero. Before freezing execution, the transition boundary must be resolved from the actual materializer's step-cell convention: field centers change at 7254.75 s, so the cell boundary is expected near 7254.625 s for 0.25 s cells. The selected window covers both possibilities by about five seconds.

## Proposed execution boundary

1. Read only the padded accepted Arm-A int16 frames and geometry on the selected host.
2. Apply the reviewed installed Kilosort `BinaryFiltered.filter` boundary with the accepted settings and `whiten_mat=None`; stream bounded batches rather than materializing full-session float32.
3. Present float32 to DARTsort with `preprocessing="none"` and crop accounting back to the central frame interval.
4. Run native Candidate-3 stages with `save_intermediate_labels=true`, the reviewed fresh whole-stage restart wrapper, and a run-specific `.partial` directory.
5. Save compact event/time/channel/template/assignment/refinement lineage and stage schemas. Preserve noise, ambiguous, removed, added and unmatched rows separately. Do not interpret exact row matches as identity.

## Exact unresolved links before freeze

- Host/input: the accepted materialization is H5-local and absent on H1. Select H5, or bind a reviewed H1 streaming reconstruction whose bounded output is checked against accepted H5 materialization. Owner: Hub/H1 host selection.
- Storage: bind a writable run-specific target with at least 3 GiB free for two padded real snippets plus failed-attempt retention; never use the full-session 482.6 GB materialization. Owner: execution host.
- I/O: if H1 is selected, run a separately frozen bounded read-throughput check on the CIFS REF route; no throughput evidence currently exists. Owner: H1.
- Runtime context: repeat the 4-byte CUDA allocation through the same managed-service context that will launch the snippet, per DARTsort host rules. Interactive success alone is insufficient. Owner: execution host.
- Adapter equality: bind the exact accepted materializer source/config and verify the two padded windows, including transition edge/padding, against the accepted H5 Arm-A samples before sorting. Owner: H1 implementation, independent reviewer.
- Effective DARTsort config: v3 synthetic exercised stages but used a capability-only threshold/refinement config and default subsampling/order. A real contract must freeze the actual Candidate-3 config and explicitly state whether detection is full coverage or subsampled. Owner: H1/Hub scientific choice.
- Checkpoint: bind the reviewed whole-stage restart wrapper, constrain attempt IDs to slash-free generated IDs, reject symlinked attempt roots, and state maximum recomputation. Owner: H1.
- Comparator scope: historical exact A has no saved pre-extraction detection bank or matching/refinement transitions. Real Candidate-3 internal lineage and final-to-final A comparison are possible; earliest cross-sort divergence is unavailable without a new prospectively instrumented Kilosort comparator. Owner: future comparator task if required.
- Resource contract: freeze byte, wall-time, GPU-memory and event caps using the ordinary synthetic observation (842.35 s, 128,223,957 bytes) only as a planning lower bound, not a real-data estimate. Owner: H1/Hub.

## Required output states

For input, detection, matching, assignment and refinement, record one of `saved`, `qualified_replay_only`, or `unavailable`. A stage is `saved` only when its row/time/template membership artifact is in durable run storage and manifest-bound. Large HDF5 may remain external, but its exact hash/schema/path and a compact sufficient lineage snapshot must be published.

## Stop rules

- Stop before voltage access if host, input identity, adapter equality, writable target, managed runtime, or effective config is unresolved.
- Stop an arm on any input/frame/padding mismatch, stage exception, byte/wall/GPU cap, or lineage-length mismatch; retain the attempt.
- Do not automatically start the transition arm if ordinary exceeds its frozen bound.
- Do not start a medium/full run from snippet success; require a separate representative-window decision.

Implementation checks
- Done: actual field metadata -> rounded states and stable runs were recomputed from the bound field SHA `4c769125...`; provisional windows avoid edge ambiguity except the deliberately included transition.
- Done: clocks/frames -> frame indices use the bound exact sampling rate and explicit rounding rule.
- Done: current stage capability -> synthetic v3 and whole-stage restart packets define observed capabilities and limitations.
- Not done: real samples, transition edge equality, host I/O, managed-service CUDA, actual config, or storage target -> prerequisites above.
- Can establish: the exact decisions and bindings still required for the cheapest real snippet.
- Cannot establish: real readiness, scientific performance, host choice, or permission success for the eventual voltage read.
