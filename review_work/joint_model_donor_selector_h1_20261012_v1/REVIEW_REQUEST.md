# I2 donor selector focused review

Review commit `2ea5ee8` and these frozen snapshots.

Load-bearing checks:

1. Exactly four donors are selected from each stable amplitude third.
2. At least four selected donors are within the inclusive 200 µm risk radius, including one from each tier.
3. Isolation is strict nearest-template cosine `< 0.9`.
4. Tie-breaking is deterministic and does not inspect arm outcomes.
5. H5 must bind the exact local metadata construction and execute beside the retained bank; this packet contains no waveforms.

Implementation checks
- Done: frozen rule and independent-value known-answer fixture -> reviewable, not accepted.
- Not done: actual H5 ledger and source-bound metadata review.
- Can establish: selector behavior on its declared metadata contract.
- Cannot establish: biological identity, actual donor quality, or N0/N1 benefit.
