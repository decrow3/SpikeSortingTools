# CY fixed-bank rematching independent preflight review

## Verdict

The original worker required targeted source repairs before launch. The
V0/CD1_FULL coordinate-descent contrast itself is correctly specified, and the
frozen 30,000-policy bank is the intended bank. The repaired worker now states
that this is fresh fixed-bank redetection/rematching and uses a new row
namespace. Its first real REMATCH0 attempt exposed a separate missing-whitener
precondition; the repaired whitening mediation is qualified below.

## Decisive source finding

In the exact DARTsort source identified by the producer, `match()` uses the
provided `template_data` directly and skips template estimation.  It then
deletes `sorting` before constructing the matching peeler.  Consequently the
`provisional` labels, GMM candidates, likelihoods, responsibilities, and unit
proportions constructed by `cx_fixed_bank_worker.py` do not initialize or
constrain matching.  They are lineage assertions only.  Matching redetects
events from the recording against the fixed bank, so `matching2.h5` has a new
event-row namespace and cannot be advertised as a transformation of the
641,588 captured rows.

This was checked against local source hashes matching the producer audit:

- `dartsort/main.py`: `e4c03417c1256384c4f8fbf0250560003bd795b7274932ef1e49dcde217bd249`
- `dartsort/peel/matching.py`: `eddc983bbca8e609d9ffeb8d7b34a0b3ea250472a97ce1519734cb811a89fe41`
- frozen bank: `cb50e88196a24cc811073a5b8e474b9385bd62eaf13d5d038b9957b298c59243`

## Checks that pass

- V0 (`cd_iter=0`, `coarse_cd=true`) performs one full-resolution round.
- CD1_FULL (`cd_iter=1`, `coarse_cd=false`) performs two full-resolution
  rounds, so the intended scheduling contrast is valid.
- Both arms point to the same 747-unit, 121-sample, 206-registered-channel
  saved bank; unit IDs are exactly 0--746, trough offset is 42 samples, and
  sampling frequency is 29,999.7591667 Hz.
- TPCA initialization is deterministic here: the accepted configuration has
  `tpca_from_templates=true`, and the matching peeler initializes it from the
  supplied bank. Sampling seeds in the accepted fit/refinement configuration
  are zero.
- The CE bank alone is **not** executable with the accepted matching
  configuration: it says `whiten_strategy=none` and contains no whitener or
  covariance, while `prewhiten_postapply` asserts that both exist. The first
  REMATCH0 attempt therefore failed before peeling at `drifty.py:251`.

## Whitening repair review

Attaching the accepted W2 matcher's recording-level whitener, covariance, and
temporal kernel is mathematically consistent with this bank. In
DARTsort's `peelreduce` template constructor, `prewhiten_postapply` does not
prewhiten the template waveforms during reduction; it leaves the templates in
their plain representation and stores the nuisance arrays for the matcher.
The drifty matcher then applies those arrays to template convolutions and raw
traces. Thus this repair restores omitted recording-level state without
changing the CE templates, unit IDs, counts, geometry, or TSVD.

The producer stopped its initial in-memory repair before peeling for this
independent representation audit, then materialized one immutable compatible
bank used by both arms. All CE arrays other than the strategy metadata are
byte-identical; only the three nuisance arrays and
`whiten_strategy=prewhiten_postapply` were added. The compatible-bank hash is
`298258e2cf740320be5591b08755b3dc38dd3ded33bd64eaa1e62cbd96525819`.

The accepted nuisance source has the same 206-row registered geometry, 42-sample
trough offset, and 29,999.7591667-Hz sampling rate. The service-context mock
successfully constructed the peeler and completed matching-template
precomputation. Recorded hashes are whitener `4af8476d...`, covariance
`4b814e0e...`, and temporal kernel `dbaf7e8...`.

Qualification: the repair is valid for this exact W2 recording/configuration,
not a generic permission to mix whitening assets across recordings. The worker
should assert all frozen manifest hashes at execution time; the preparation
step currently asserts the bank and field hashes while merely recording the
remaining hashes.

## Repairs required before execution

1. Keep the captured-state checks only as bank/provenance checks; never use
   exact row-count preservation as an outcome assertion.
2. Add explicit output provenance that maps new detections to prior rows only
   in downstream QC, with unmatched and multiply matched events retained.
3. Freeze and assert source, bank, recording, field, whitening source, full effective config,
   model-initialization, RNG, and computation hashes/values in the worker
   receipt before expensive execution.
4. Preserve the first failure evidence and treat the repaired run as an
   infrastructure retry, not a scientific-parameter change.

This review launched no arm. The external producer launched the repaired
REMATCH0 only after its constructor/precompute mock passed. The source tree is
dirty, so immutable file hashes—not a branch name alone—must identify the
executable code.
