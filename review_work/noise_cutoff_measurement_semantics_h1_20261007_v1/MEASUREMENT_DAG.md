# Noise-cutoff measurement DAG

This DAG is traced from the corroborating snapshot SHA-256 `a31f1349854eff0e4c732063993be37e8cf53de5d206628f66d68a65ff7f341e`. The declared accepted evaluator SHA-256 `415f29d7f26102606143d0bd4fbb8ed9a136199a6642e664d50c2a6716b7c423` is not available on H1, in the shared packet tree, or from the configured Git remote; therefore this is corroborating rather than exact executed-source proof.

```text
curated output/amplitudes.npy (one sorter-amplitude scalar per event)
  + spike_clusters.npy
  + spike_times.npy
          |
          v
per-unit stable cluster grouping, then event-time ordering
          |
          +---------------- full: all unit amplitudes -------------------+
          |                                                             |
          +-- state assignment from saved event sample + displacement --+-- q0 subset
                                                                        +-- displaced subset
          |
          v
finite filter -> invalid-input gates -> unit/subset-specific max
          |
          v
100 equally spaced EDGES from 0 to max => 99 histogram bins
          |
          v
peak index -> occupied bins after peak (excluding last bin)
          |
          v
high_start = ceil(0.5 * occupied_top + peak_index)
          |
          v
positive counts in high tail; second occupied-bin count = first_low
          |
          v
noise_cutoff = (first_low - mean(high counts)) / population_std(high counts)
          |
          +-- undefined if required support/variance absent
          |
          v
finite and noise_cutoff < 5 (strict) -> saved pass flag
          |
          v
compact per-unit scalar/pass tables -> census status -> audit arm-level rates
```

## Semantics and dependencies

- Input units are sorter amplitude units as stored in Kilosort `amplitudes.npy`; the corroborating source does not convert them to microvolts. The derived cutoff is dimensionless because both numerator and denominator are histogram counts.
- `first_low` is a bin count, not the location or amplitude of a low-amplitude bin.
- `quantile_length=.25` affects an index offset based on the number of occupied bins; it is not a 25th amplitude quantile.
- The last histogram bin is excluded from `occupied_top` but can enter the high-tail mean and standard deviation.
- The rightmost maximum is included by NumPy in the final bin.
- Because every unit/subset uses its own maximum, an extreme high amplitude rescales all edges. Full, q0, and displaced values do not share fixed amplitude support and are not paired effect estimates.
- State eligibility (`q0_spikes >= 100` and `displaced_spikes >= 100`) belongs to the downstream through-displacement rule. The scalar helper itself has no 100-spike minimum.
- The bootstrap path in the corroborating source caches 99-bin block/state histograms on a full-unit maximum, then recombines counts. Those caches are in-memory construction state and are not present in the published compact census packet.

## Source trace

| Claim | Corroborating source lines |
|---|---:|
| IBL commit declaration | 38 |
| scalar construction and strict pass | 59–87 |
| in-memory histogram helper | 206–225 |
| curated amplitude load/group/order | 246–271 |
| dynamic 100-edge block histogram cache | 280–288 |
| full-unit call and saved fields | 289–305 |
| state-subset calls | 309–318 |
| bootstrap reconstruction from cached counts | 323–335 |

Exact executed-source verification remains unresolved.
