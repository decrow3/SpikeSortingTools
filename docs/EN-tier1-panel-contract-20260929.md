# EN fixed Tier-1 panel contract

This contract freezes the implementation of the EN Tier-1 panel before either
new full-session sort is available. The machine-readable contract is
`configs/en_tier1_panel.v1.json`.

## Cohort and state comparisons

Tier 1 uses all raw Kilosort clusters for REF, XR, A, and B. KS-good counts are
reported as secondary descriptive yield. No curation outcome is allowed to
change the Tier-1 cohort.

Each candidate is evaluated against REF on the time states defined by the
field that produced that candidate. A and REF use the rounded AM.3 field. XR,
B, and REF use the rounded XR field. This keeps B versus XR on identical field
time and prevents either field from defining the other's apparent continuity.
The arm-local state/rate Spearman rho is a continuity proxy; it is not an
identity or purity claim.

## Frozen metric details

- Yield includes deterministic full-session raw/KS-good unit counts and spike
  counts, plus common 300 s block-bootstrap rate and active-unit summaries.
- Short-interval fractions use positive adjacent within-unit lags below 1 ms
  and never cross a contiguous rounded-state boundary. Exact duplicates, lag 8,
  and lag 30 are retained separately rather than folded into that result.
  Because the aggregate exact-duplicate fraction uses all segment-safe adjacent
  pairs as its denominator, it is reported separately for every field/arm
  pairing; the top-level arm summary contains only field-invariant quantities.
- Chance-aware coincidence uses the existing 0.5 ms, 75 µm marked-spike rule
  and a deterministic circular shift for each cluster.
- Presence uses full-session 300 s half-open bins, including the clipped final
  bin. Edge units have median saved y inside the 200–3640 µm processing domain
  but outside the 300–3540 µm scoring domain. Coincidence uses events inside
  that processing domain from units whose median saved y is also inside it.
- Time-only REF correspondence uses exclusive one-to-one event matches within
  ±0.5 ms, retains the full edge graph, and calls a primary only when the best
  edge is reciprocal, unique, and retains at least half of both trains.
- All inferential summaries use 2,000 300 s common-block bootstrap draws with
  seed 20260929. Undefined quantities remain unavailable with their denominator
  and cause; they are never replaced with zero.

Tier 1 produces no composite score. Tier 2 runs only for arms whose full Tier-1
profile is favourable overall. RF remains out of scope and the outer holdout
remains sealed.
