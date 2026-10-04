# EN first-medium Phase 3 provenance repair candidate v3

Status: `EXECUTION_DISABLED_PENDING_H5_INDEPENDENT_REVIEW`.

This packet adds one coordinated producer/consumer provenance edge to the
previously accepted Phase 3 composition. It does not change an estimator.

The production contract and Phase 3 phase flag are both false. Every real-run
token remains prospective. The packet contains no recording access, voltage,
sorter, training, RF, holdout, service-launch, or automatic advancement path.

V1 and v2 remain preserved. H5's independent interface trace established that
the existing producer COMPLETE chain binds arm completion receipts but not the
artifact inventories. V3 therefore adds each deterministic inventory path and
SHA256 to the producer execution state; authoritative pair COMPLETE already
binds that exact state map. The consumer requires that transitive edge before
it opens any arm.

The consumer then requires exactly `REF384_repeat` and `repaired_B384` states,
validates completion path/hash/status/association, resolves both producer
configs, derives REF `sorter_output` separately from repaired
`native_invocation.results_dir`, and verifies every consumed saved-file hash
and repaired size against the authenticated arm inventory. Historical REF and
existing-B keep their accepted independent bindings.

## Frozen Phase 3 decision surface

- Arm order is exactly `REF384`, `existing_corrected_B384`, `repaired_B384`,
  `REF384_repeat`.
- REF is loaded and compared with the successful Phase 1 receipt before any
  candidate arm is opened.
- Primary margin rule: one-sided 95% lower bound from 20,000 paired fixed-REF-
  unit composition-bootstrap draws, seed `20261002`, must be strictly greater
  than `0.05`.
- Repeat diagnostic: same-window `R(REF384_repeat) >= 0.98`. The criterion is
  bound to source receipt SHA256
  `b2ef938c0087fdf5f75102c1178ddb6ac2841c086766dfe78d67e596c66d7ede`.
- Missing truncation QC or accepted pair-bound waveform evidence is
  `UNMEASURED` and cannot pass. Advancement is always false in this composer.
- `amplitudes.npy` is not an input. The accepted loader derives event
  amplitudes from `full_st[kept_spikes][:,2]`.

## Immutable prerequisite

The completed launcher-v3 review is reused without reopening:

- packet: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_no_mask_ref_repeat_launcher_r2_finalization_h1_review_20261002_v1_h1`
- MANIFEST SHA256: `7fde2fc6d93a85d3453471176ad66f6e3e8cb14349a449f38ea7e7e44d0fa6c5`
- COMPLETE SHA256: `8765639ef124917879254aa5c5d9f2c189426522e6bbc46aaaf5d50b7255058b`
- shared status SHA256: `5169ce5003914c215d4c07d085b471910953f6561e25b001548aaf04eb2cc526`

## Verification

Run from the repository root:

```bash
PYTHONPATH=coordination-outbox/en_first_medium_saved_output_phase3_provenance_repair_candidate_20261002_v3_h1/source:coordination-outbox/en_first_medium_saved_output_phase3_provenance_repair_candidate_20261002_v3_h1/producer \
PYTHONDONTWRITEBYTECODE=1 \
pytest -q coordination-outbox/en_first_medium_saved_output_phase3_provenance_repair_candidate_20261002_v3_h1/tests/test_phase3_composition.py
```

The tests use the accepted producer finalizer with lightweight stub arm
artifacts, then invoke the actual managed consumer entry. They do not use real
project outcomes.
