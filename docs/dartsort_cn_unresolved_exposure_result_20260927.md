# CN unresolved-default sensitivity and exposure forecast

## Verdict

**Keep the frozen pass-only primary.** Accepting every unresolved edge is a
useful post-primary sensitivity, but it is not supported as a new default. In
W3 it accepts 112/122 direct edges and produces 175 force relations, only 45
relations short of all-force. The graph contains the primary by construction
and reconnects none of the ten resolved failures, but `unresolved` still means
that the prespecified gate could not decide—not that the pair is one neuron.

The stationary full-session forecast says that longer observation could make
the support minima evaluable for many edges. At the observed-rate scenario,
185/222 W2 edges and 99/122 W3 edges meet all independent-rate support minima;
the shoulder-derived calculation gives 165/222 and 84/122. These are exposure
forecasts only. They do not forecast gate passage, identity, recovery, or
stable-unit yield, and they assume rates and availability continue unchanged.

## Saved-window result

| Window | Frozen primary | Accept-unresolved sensitivity |
| --- | ---: | ---: |
| W2 direct / force / union | 6 / 6 / 21 | 213 / 413 / 424 |
| W2 indirect / rejected reconnected | 0 / 0 | 200 / 2 |
| W3 direct / force / union | 7 / 8 / 13 | 112 / 175 / 178 |
| W3 indirect / rejected reconnected | 1 / 0 | 63 / 0 |

The W2 sensitivity reconnects two resolved-failed direct pairs through paths;
the W3 sensitivity reconnects none. This graph consequence is reported, not
treated as evidence for either identity decision.

## Full-session support forecast

The exact retained canonical-rest support is 239.715 s in W2 and 194.769 s in
W3, versus 8,210.05 s over the full session. Every direct edge was recomputed,
including early failures whose saved diagnostic row contained zero sentinels.
Zeros remain zeros; they are not treated as missing values.

| Rate scenario | W2 independent / shoulder | W3 independent / shoulder |
| --- | ---: | ---: |
| 0.5x | 119 / 142 | 74 / 78 |
| 1.0x | 185 / 165 | 99 / 84 |
| 2.0x | 196 / 173 | 100 / 88 |

At 1x this is 83.3% / 74.3% of W2 direct edges and 81.1% / 68.9% of W3.
Three W2 edges and one W3 edge have zero independent-rate expectation; 49 W2
and 34 W3 edges have zero shoulder expectation. Their corresponding required
duration is unbounded rather than silently imputed.

## W3 consumer sensitivity

The post-primary W3 consumer ran once from the sealed boundary. It did not rerun
the prefix, templates, CCG statistic, raw voltage, sort, or motion fit, and it
reused the exact sealed all-force control by hash. Independent reconstruction
from `SENSITIVITY_GRAPH_ARRAYS.npz` confirms 112 accepted direct edges, 175
strict-upper-triangle force relations, 178 union relations and 500 connected
components. All 571,935 source rows close in every branch.

| Policy | Assigned / noise | Final units | Rest ISI 9–29 | Episode ISI 9–29 |
| --- | ---: | ---: | ---: | ---: |
| All-force | 565,063 / 6,872 | 488 | 0.010781 | 0.009358 |
| Primary pass-only | 566,611 / 5,324 | 585 | 0.006020 | 0.004603 |
| Accept unresolved | 565,356 / 6,579 | 497 | 0.008894 | 0.007829 |
| No force | 566,629 / 5,306 | 589 | 0.006002 | 0.004593 |

These are actual-final-clock ISI fractions. Splitting mechanically lowers
within-unit short-ISI counts, so they cannot select a policy. For all-force the
previous fixed-clock value was 0.009329 (1,625/174,183) whereas actual final
times give 0.009358 (1,630/174,183). The five-interval difference is caused by
within-state one-sample timing changes; zero rows crossed the episode mask.

The 45 accept-unresolved split-template comparisons have exact common physical
positive support (20–32 positions; median 24), with median cosine 0.820 and
19/45 at least 0.90. That descriptive compatibility is not independent identity
evidence. The next decisive test should predeclare waveform-first identities on
a longer observation and test direct-voltage waveform continuity and competing
matches independently of the force graph.

## Integrity and scope

All 45 output products and 28 executed-source/adaptor products match their
manifests. The source-at-execution supplement closes CM's prior source-access
caveat: primary replay and sensitivity services have no capture dependency;
the code restores captured post-TMM times/channels, separates finite and
nonfinite replay comparisons, and invokes no downstream consumer RNG. CN did
not repeat the all-force consumer.

Forecast packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cn_force_gate_exposure_forecast_20260927/`.
W3 sensitivity packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-cn-default-sensitivity-v1/`.
