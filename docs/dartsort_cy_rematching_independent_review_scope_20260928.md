# CY fixed-bank rematching independent preflight review

## Verdict

**Do not launch the two arms from the current worker without targeted source
repairs.**  The V0/CD1_FULL coordinate-descent contrast itself is correctly
specified, and the frozen 30,000-policy bank is the intended bank.  However,
the worker and README incorrectly describe the run as beginning from the
captured pre-force provisional identities.

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
- Whitening metadata is internally compatible with the accepted configuration:
  the bank says `whiten_strategy=none`, while matching performs its declared
  prewhiten/postapply handling through the matching configuration.

## Repairs required before execution

1. Remove the claim that either arm starts from captured provisional
   identities.  State that both are fresh fixed-bank redetection/rematching
   passes with a new row namespace.
2. Keep the captured-state checks only as bank/provenance checks, or remove the
   unused `ephemeral_replace`; never use exact row-count preservation as an
   outcome assertion.
3. Add explicit output provenance that maps new detections to prior rows only
   in downstream QC, with unmatched and multiply matched events retained.
4. Freeze and assert source, bank, recording, field, full effective config,
   model-initialization, RNG, and computation hashes/values in the worker
   receipt before expensive execution.
5. Run the already-required service-context dummy through both matching and
   refinement on the destination filesystem before launching V0.

No arm was launched by this review.  The source tree is dirty, so immutable
file hashes—not a branch name alone—must identify the executable code.
