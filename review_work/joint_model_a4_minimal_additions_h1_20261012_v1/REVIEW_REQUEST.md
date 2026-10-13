# A4 injection and thin-entry focused boundary review

Review the source/config snapshots and combined fixture in this directory.

Load-bearing checks:

1. Injection consumes already-placed event waveforms after channel mapping/sign inversion and before temporal centering/CAR.
2. Physical-recording and padded-batch clipping, overlap addition, and source immutability are correct.
3. The thin entry performs exactly one call and forwards only chunk starts, the two frozen feature switches, and a create-only output directory.
4. H5 must bind its exact reviewed callable from `run_restored_c4.py` SHA `4bd8d30…`; no CLI/callable name is inferred here.
5. The combined fixture covers the frozen session-clock boundary shape, moving left/right placement, fixed overlap, and exact matching1 times/labels round trip.

Implementation checks
- Done: producer source/config audit and five passing synthetic fixtures.
- Not done: H5 callable boundary inspection, real donor ledger, production execution, independent acceptance.
- Can establish: bounded component and synthetic integration behavior.
- Cannot establish: P1/P2/I2 or scientific outcome validity.
