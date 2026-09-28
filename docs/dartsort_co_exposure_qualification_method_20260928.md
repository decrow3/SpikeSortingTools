# CO exposure qualification: frozen method

This is a saved-data qualification of CN, not a new force gate or a gate-power
calculation. Preserve CN's output unchanged.

Read the full imec1 AP frame count and sampling rate from the SpikeGLX metadata,
checking that `fileSizeBytes / (2 * nSavedChans)` is integral and that frames / rate
equals `fileTimeSecs`. Verify the canonical censor-mask hash, merge touching or
overlapping intervals after clipping them to the exact recording interval, and
form the exact complement. Tile each complement segment independently into
half-open 5 s frame blocks using the frozen ceiling conversion. Retain the
89-sample symmetric trim used by the gate's lag statistic.

For each W2/W3 direct edge, estimate unit rates from all complete rest blocks in
that saved window as CN did. Under stationary independent Poisson rates at
0.5x, 1x and 2x, project onto every exact full-session block. A block is common
positive when both units have at least one event. Apply the frozen within-segment
rule in expectation: a block contributes only when it is common positive and at
least one other block in the same rest segment is common positive. Calculate:

- expected retained common-positive blocks;
- expected scoring-domain events for each unit;
- expected independent central count using exact discrete lag exposure;
- shoulder-derived expected central count by scaling the observed saved-window
  shoulder count by the ratio of full-session eligible lag exposure to the exact
  lag exposure of its observed scoring blocks.

An edge clears the qualified expected-support screen only when expected common
blocks are at least 20, both expected scoring-event totals are at least 100, and
the relevant expected central count is at least 20. These are expectations under
stationarity and independence, not probabilities of realizing the minima, gate
pass probabilities, identities, or recovery/yield forecasts. Zero observed
shoulder support remains unbounded in the shoulder-derived screen.

Also summarize the saved state-complementary table without changing its
thresholds. Count pairs with fewer than 20 spikes in either named child and join
the already saved per-state rates. Those rows are exposure-sensitive descriptive
cross-products within all-force parents, not recovered/lost biological units.
