# REF-repeat criterion provenance correction

Verdict: `R_repeat >= 0.98` is an accepted, applicable pre-outcome engineering gate.

The exact origin artifact hashes to `b2ef938c0087fdf5f75102c1178ddb6ac2841c086766dfe78d67e596c66d7ede`. It froze a maximum exclusive-retention shortfall of `0.02` before the new pair outcomes. Phase3 v1 bound that rule; v2 and v3 explicitly preserved every threshold and interpretation; H5's reviews required the same preservation before the final provenance GO.

This packet supersedes only the `NOT_DEFINED` repeat-threshold documentation in the post-run freeze v2. It preserves all paths, sources, cohort, matcher, bootstrap, primary threshold, missingness, and scope restrictions. Primary `DeltaR` and the repeat gate are reported separately; neither makes the other pass, and `advance` remains false.

One exact contract conflict remains. The origin artifact says repeat failure or unmeasured makes the comparison inconclusive, while the accepted consumer source maps a numeric repeat failure to overall `reject`. Per instruction, this packet publishes the exact sources and makes no code change. Resolve that mapping before evaluator invocation.

No outcomes, voltage, RF, or holdout data was accessed.
