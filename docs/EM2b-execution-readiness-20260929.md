# EM.2b W2 execution readiness — 2026-09-29

**Status: scientifically specified and resource-bounded; no sort launched.**
EM.2a selected a nonredundant comparison set. The exact scorecard is frozen in
`configs/em2b_w2_scorecard.v1.json` (SHA-256
`5ab747781f930fdd35faabf3dd6049481cccce8e2848b6ffac116686f6ffeee7`). A harmless systemd user-service dummy
continued after its launching shell exited and completed with status 0, so the
required persistent launch mechanism is available. Its receipt is
`testing/outputs/em2b_launcher_dummy_20260929/receipt.json` (SHA-256
`9a86ff4a75174761c1880c01656ac65763b993e7a7690097c38224c146b5027b`).

The cheaper first stage is two new sorts: unrounded kriging, the primary
standard comparator, and rounded kriging, the field-rounding bridge. They are
compared with the existing S0 and exact-DD W2 controls. Unrounded IDW and
nearest remain frozen stage-2 arms and run only if stage 1 cannot distinguish a
field-rounding effect from an operator effect. This can answer the main question
with two new sorts instead of four; resolving differences among all three
unrounded kernels would still require stage 2.

The existing D2L W2 sort is the resource measurement. Its preserved
`sort-complete.json` reports exit status 0 and 1,514.141 seconds; its complete
run directory occupies 1.3 GB. Each new arm is capped at 2,700 wall seconds and
5 GiB scratch, with one GPU and four CPU jobs. Stage 1 is capped at 5,400 wall
seconds. Timeout or cap breach is a preserved failure and does not authorize a
silent retry or larger budget. The network mount currently has about 231 GB
free, while local scratch has about 1.2 TB free; new working directories should
therefore stay on local scratch and publish only the compact receipts and saved
sort products required by the scorer.

The scorecard uses exact piecewise-linear W2 domains and arm-local units. A unit
is eligible with at least 100 flat-domain events in the full W2 point dataset;
that eligible set remains fixed through 2,000 common 5-second-block bootstrap
draws. Every draw uses the same block multiplicities for every arm. The primary
quantity is `rho_lattice - rho_unrounded_kriging`, where rho is an arm-local
rate-rank continuity proxy. No unit is matched across arms, and rho is not
interpreted as identity or purity.

A lattice advantage requires delta rho at least 0.05, a 95% CI lower bound
above zero, no more than 5% relative yield loss, and no domain's segment-safe
9--29-sample ISI fraction increasing by more than 0.01. Practical equivalence
requires the entire CI inside `[-0.05, 0.05]`, yield within 5% in either
direction, and every ISI fraction within 0.01. Fewer than three eligible units,
a constant rank vector, a non-finite point estimate, or fewer than 1,900 valid
bootstrap draws is unresolved rather than zero-filled.

Before a real launch, the run packet still needs the exact production recording
construction, source/config hashes, output paths, and systemd unit definitions
for the two stage-1 arms. Each new sort requires explicit post-report run
authorization under the EM plan. No RF work, outer-holdout access, production
change, or automatic W3 run is authorized.
