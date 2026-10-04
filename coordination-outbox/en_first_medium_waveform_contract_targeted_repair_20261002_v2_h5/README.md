# First-medium waveform-contract targeted repair v3

Status: `READY_FOR_H1_REVIEW_EXECUTION_DISABLED`.

This fresh packet repairs only H1 findings WF1–WF3. It preserves the reviewed
parent packet and H1 rereview by exact MANIFEST/COMPLETE hashes. It contains no
spatial-loader change, real arm binding, voltage result, or launcher.

WF1 is closed by a content-bound preprocessing closure: the accepted Kilosort
`io.py` and previously missing `preprocessing.py`, frozen CPU float32 high-pass
impulse, source/config constants, runtime lock, package metadata, and a narrow
executable adapter are all bundled and hashed. The operation is correctly named
common-median reference: per-channel time means are removed, then
`torch.median` across channels is subtracted per sample. A four-channel known
answer distinguishes it from average reference, and the adapter exactly equals
the bound Kilosort operation on a deterministic 384×1,145 record.

WF2 is closed by an exclusive/exhaustive endpoint partition over all crop event
rows. Stable pairs common to every maximum-cardinality/minimum-lag solution are
retained; globally nonunique partner or matched-state endpoints are ambiguous;
every other unmatched REF endpoint is lost and candidate endpoint is gained,
even if it had an admissible edge consumed competitively. Unsupported anchors
and provenance conflicts are pre-match ambiguous. A >80-um edge is merely
inadmissible and does not contaminate an otherwise valid endpoint. Synthetic
asymmetric-competition, tie, far-anchor, and unsupported-anchor fixtures assert
partition closure.

WF3 is closed by atomic retained-pair sampling. At most 64 retained pair IDs per
comparison/epoch are selected, and both endpoint rows are read; the other three
categories retain 128 endpoint slots each. Thus the frozen maximum remains
4,608 reads and 4,052,090,880 requested bytes, with no extra partner reads.
Correlation uses common physical channels and common absolute signal frames,
requires both selected endpoints, at least two channels and 26 frames, and has
explicit invalid/low-N behavior. API-requested and returned physical payload
bytes are receipted separately from kernel/device diagnostics.

All prospective arm and anchor hashes remain unbound. Both execution and voltage
gates remain false. Ten synthetic contract tests pass directly and from a
byte-identical `/tmp` stage. No voltage, waveform, evaluator, sorter, RF,
holdout, or job was accessed or run.

Implementation checks
- Done: verified H1 review MANIFEST `2c2cbe71...` and COMPLETE `35f776bf...`,
  plus parent MANIFEST `80021b93...` and COMPLETE `63170aff...`.
- Done: traced `BinaryFiltered.filter` and `fft_highpass`; bundled and hashed
  their exercised source/constants/runtime closure; checked exact output against
  the accepted Kilosort code.
- Done: tested category closure, asymmetric competition, co-optimal ties,
  anchor handling, atomic pair selection, correlation alignment, and cap-boundary
  arithmetic.
- Not done: no prospective hashes, real event ledger, voltage, evaluator,
  waveform result, sorter, RF, or holdout was accessed or generated.
- Can establish: a fully specified, dependency-bound, internally closed but
  execution-disabled waveform contract that addresses WF1–WF3.
- Cannot establish: voltage launch readiness before independent review and fresh
  bindings, or any physical support, efficacy, identity, purity, or advancement.
