# Depth-aware lighthouse tracking

## Current preferred discovery method

Use [waveform-only, whole-probe candidate discovery](lighthouse_candidate_discovery.md)
for new lighthouse searches. Absolute depth must not determine candidate ranking,
identity competition, or a preference for the seed location. Preserve relative
multichannel geometry, search all available complete-support probe positions,
and reveal absolute depth after matching. Start with cached evidence and retain
strict, lower-score, ambiguous, and unmatched results separately. Inspect the
waveform observations over depth/time peak scatters before building consensus.

This supersedes the original nearby-only competitor and bounded seed-depth
search below as the default for finding candidates. The historical experiments
remain controls; geometry limits and any deliberately restricted diagnostic
search must be explicit. Promotion concerns candidate discovery, not certified
biological identity or calibrated displacement.

## Historical bounded implementation and continuing measurement requirements

User direction: trace lighthouse cells through depth for all new validation. Historical fixed-template tracks remain preserved as diagnostic controls; their sparse or stationary-looking accepted events do not establish stationary neurons or failed DREDGE motion.

The initial bounded scope was unit445 over930–1030s, with a predeclared depth range independent of DREDGE and nearby identity and depth hypotheses. For the current whole-probe workflow, retain an unmatched state and expose depth ambiguity and geometry/search boundaries. Do not force trajectories through gaps, fit search bounds or continuity to DREDGE, or promote additional matches to biological identity proof.

A first coarse bank can use exact same-column geometry translations at40µm increments to test whether fixed support caused dropout. Coarse winning shift is a template hypothesis, not calibrated physical displacement. Waveform centroids must use a correspondingly translated spatial window; otherwise the readout can retain the original support bias even when matching is depth-aware. Fine spatial interpolation requires separate checks because it changes amplitudes and can bias localization.

Required comparison figure: original fixed-template observations, depth-aware accepted observations and uncertain alternatives, waveform support/counts, and DREDGE added only after matching. Evaluate DREDGE at accepted event times for summary comparisons. Display search limits and unsupported regions. Report identity acceptance separately from depth-resolution confidence. Do not use an aggregate agreement score alone to choose a tracker.

Keep detection sensitivity, waveform preprocessing, acceptance gates, and competitor treatment explicit. Expanding a search changes the chance of incidental matches even with unchanged cosine thresholds; the first result remains exploratory until sensitivity and competitor controls support it. Use known geometry translations to verify shift sign and implementation before inference, and preserve full score alternatives for subsequent review.

Run substantial extraction independently under the existing managed-job policy, with completed stage outputs saved for reuse.

## Updated population priority

The user now requests approximately20–30distinctive waveform candidates across the whole probe, initially in930–1030s, rather than concentrating on2000–2400µm or requiring one cell to cover every transition. Prioritize this population expansion over further single-unit445 tuning. Individual units may drop in and out; preserve useful local observations without demanding perfect full-interval identity or coverage.

First inspect direct observations and establish shared movement across independently supported identities. When those checks justify aggregation, use adaptive bins per unit with explicit event counts and temporal spans, then consider depth-resolved consensus across independently represented identities. Keep waveform training/reference times compatible across units: independent centering on each cell's observed times can create spurious motion as cohort membership changes. Give each independent identity/family one vote rather than weighting purely by spike count. Require several independent units for a consensus claim, expose lower-support regions separately, and do not assume motion is rigid across depth.

Show actual temporal support, mixed depth hypotheses, and missing coverage. A0.25s evaluation grid is not0.25s measurement resolution when contributing waveform bins span seconds. Compare DREDGE on the same event/time support before interpreting apparent temporal smoothing or amplitude attenuation; preserve its full-resolution trace separately. No DREDGE data may enter candidate selection, identity tracking, or consensus construction.
