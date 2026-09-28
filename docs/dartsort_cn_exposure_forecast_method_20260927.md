# CN unresolved-default and exposure forecast: pre-outcome method

The primary gate is unchanged: only `secondary_pass` direct edges receive the
force route; resolved failures and unresolved edges are withheld. A post-primary
sensitivity accepts `secondary_pass OR unresolved`, withholding only resolved
failures. Both masks are applied before connectivity and then unioned with
unchanged QDA and explicit SI state. No CCG statistic, threshold or status is
recomputed to select a favorable default.

For every direct W2 and W3 edge, sufficient statistics are recomputed from the
sealed fixed-clock source rows and exact complete canonical-rest blocks. This
includes edges whose saved early-exit rows contain zero sentinels. Report:

- total eligible-rest events for each unit, including zero-event blocks;
- event-positive common blocks and scoring-domain events after the frozen
  within-segment two-block rule;
- observed central/shoulder counts and shoulder-derived expected central count;
- unconditional unit rates over all exact retained support;
- independent-rate expected central count using exact finite discrete exposure
  for lags ±9..29 after the 89-sample symmetric trim.

Forecasts assume stationary continuation and are not gate-power or identity
forecasts. At 0.5x, 1x and 2x both unit rates, calculate duration multipliers for
100 expected events/unit, 20 expected common-positive 5-s blocks, and expected
central count 20. The common-block calculation uses independent Poisson
occupancy on actual block durations. Count an edge as potentially meeting
support only if every projected support minimum fits within the independently
verified full-session canonical-rest duration. Shoulder-zero cases remain
unbounded for the shoulder-derived forecast.

The full session is 10,473.55 s; the merged canonical mask covers 2,263.5 s, so
verified canonical rest is 8,210.05 s. No W2/W3 unit IDs are joined, and no
stationarity of identity, rates, missingness or field quality is assumed beyond
the labeled scenarios. Passing bootstrap/null or biological identity is outside
this calculation.
