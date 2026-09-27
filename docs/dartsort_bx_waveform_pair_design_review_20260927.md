# BX: independent review of the proposed 18-pair waveform test

## Verdict

**Needs revision before execution, but the bounded question is useful.** The
balanced 18-pair panel, split-half lag lock, common-bin requirement and fixed
waveform support are a credible diagnostic of whether selected post-TMM groups
have reproducibly similar or different waveforms. They cannot certify biological
identity, truth-label a merge, or distinguish a duplicated detection from two
overlapping spikes.

The proposed calls should be renamed accordingly:

- reliability ≥ 0.90 and held-out cross cosine within 0.03 of within-unit
  reliability supports **waveform compatibility with merging on the qualified
  state/support**;
- held-out cross deficit > 0.10, reproducible signed A−B difference cosine
  > 0.8 and a dependence-aware confidence interval supports **waveform evidence
  against that merge on the qualified state/support**.

Neither is biological identity truth. Similar cells can have similar waveforms;
one cell's waveform can change with motion, support truncation or time.

This review applies the repository's waveform-only and depth-aware principles:
retain relative multichannel geometry, expose incomplete support and ambiguity,
avoid absolute-depth identity claims, and treat candidates as verification
evidence rather than certified cells (`docs/lighthouse_candidate_discovery.md`;
`docs/luke_depth_aware_lighthouse_policy_20260908.md`).

## Material design corrections

### 1. Freeze the estimand and unit of replication

Keep the preselected 18 pairs: six force-exclusive (including the stated
parents 71, 91 and 29), six QDA-only, and six different-parent comparisons.
Freeze the exact pair IDs and selection fields before waveform extraction.
Different-parent pairs are specificity controls, not known-different-cell truth;
they may still be fragments or lookalikes.

Pairs sharing a parent are not independent. Report every pair and parent, and
give each source parent at most one vote in an aggregate. Do not use an
18-independent-pair binomial interval. The pair panel is selected from routing
outcomes, so its result is conditional on this panel and cannot estimate the
prevalence or accuracy of all force/QDA merges.

Define halves explicitly. If chronological halves are retained, the held-out
half tests temporal transfer but is confounded with slow waveform/noise change;
that is a feature of the harder question, not an IID replicate. Freeze the
half boundary before reading waveforms and require the same eligible state
definitions in both halves.

### 2. Preserve motion state and complete waveform coverage

The proposed same 18 physical channels centred on the matching-channel midpoint
is fair only when both units have complete, comparable signal support there.
Luke's approximately 200-µm episodes can move the useful footprint beyond that
window. For every pair, half and state, record:

- exact physical channel IDs and two-column geometry, midpoint rule and probe-edge
  status;
- each unit's peak/main channel and energy fraction inside the 18-channel window;
- expected observed support across the frozen motion range and whether any
  required channels are absent/interpolated/bad;
- event counts before/after support, partner-proximity and state exclusions.

Choose the midpoint and channel list from frozen pre-waveform metadata, never
from the resulting cosine. Reject or mark unresolved any pair/state where either
unit lacks a predeclared minimum coverage; do not repair it by shifting the
window after inspection. A wider fixed physical window is preferable when it
fits the bounded footprint, but it must remain identical for A and B and retain
relative lateral geometry.

### 3. Make rest primary and episodes a separate transfer supplement

Use independently frozen motion masks. The primary analysis should be
rest-stratified because it offers stable physical support and directly tests
whether the selected labels differ when large displacement is absent. Do not
pool rest and episode events.

Add an episode-limited supplement only for pair/half combinations meeting the
same count, common-bin and coverage gates during episodes. This asks whether a
rest-qualified relationship transfers through large motion. Sparse or truncated
episode support is unresolved, not a failed or successful merge. The episode
supplement does not replace the original full-session routing question; it
exposes the state where registration or fragmentation could alter the answer.

Also tabulate mask/state composition by pair and half. A cross-half difference
is uninterpretable if one half is mostly rest and the other mostly 200-µm
episodes.

### 4. Tighten waveform, lag and sampling definitions

The 121-sample window (42 before, 79 after) and common 18-channel geometry are
reasonable. Freeze preprocessing, gain, referencing/filtering, dtype, bad-channel
handling and baseline subtraction. Verify the sign and sample origin with a
known shifted waveform. Declare whether channels or samples containing missing
support enter cosine as zeros, NaNs or exclusions; zeros can artificially
increase shared-background similarity.

At most 100 evenly time-spaced events per unit/half is preferable to a dense
burst sample, but it is deterministic coverage rather than a random sample.
Preserve selected row IDs and selection ranks. Require ≥10 common eligible
5-s bins *after all exclusions*, and report events and bins separately. Cap each
bin's contribution so a high-rate bin cannot dominate the template.

Fit exactly one directed lag in `[-2,2]` on half 1 and lock it before half 2.
Record the convention as `A_time - B_time` (or its explicit alternative), the
padding/cropping rule and how the inverse B→A comparison changes sign. Do not
re-optimize lag per half, state, direction or bootstrap draw. The lag search
itself inflates half-1 cosine, so decisions should use held-out half 2.

### 5. Control amplitude and noise without making cosine the whole result

Cosine suppresses amplitude differences and can be high because of shared
background, referencing or noise. Alongside cosine report:

- peak-to-peak amplitude and fitted gain ratio for A and B by half/state;
- robust per-channel noise from waveform-free local times, estimated separately
  by state/half but applied symmetrically to both units;
- noise-whitened and ordinary cosine on the same support;
- energy concentration and residual RMS after the locked lag and one declared
  amplitude convention.

Do not select pairs or channels by these held-out values. A reproducible A−B
difference requires the same signed definition, support, preprocessing and scale
in both halves. Difference-vector cosine > 0.8 can still reproduce a channel
artifact or support bias, so require the deficit and coverage/noise controls too.

### 6. Use dependence-aware uncertainty

Individual snippets are temporally dependent, and evenly spaced selection does
not make them IID. Bootstrap common 5-s bins (or larger prespecified blocks if
autocorrelation supports it), carrying all events in a selected block together.
Nest this within pair and parent; do not resample 7,200 snippets independently.
Report the bootstrap unit, number of blocks, effective parents and interval
definition.

For a negative waveform call, require the lower confidence bound for the
held-out cross deficit to exceed 0.10 and for the signed difference reproducibility
to remain above its frozen threshold. For compatibility, require the lower bound
on both within reliabilities to meet 0.90 and an interval for the held-out
cross-minus-within gap to lie within the declared ±0.03 tolerance. If the small
panel cannot resolve those bounds, report inconclusive rather than changing the
threshold.

### 7. State exactly what the partner exclusion can show

Excluding events within 30 samples of the partner avoids extracting the same or
near-synchronous event into both unit templates. Report excluded counts and
directional asymmetry by pair/state/half. This exclusion **cannot directly
distinguish** duplicate assignment/double matching from two real overlapping
spikes: both generate close timestamps and both are removed. It conditions the
waveform comparison on noncoincident events. The close-time population needs a
separate collision/duplicate analysis and must not be silently interpreted from
this test.

## Corrected minimal output

For each pair and state, save eligibility, parent/category, exact channel IDs,
row IDs, half/state/bin counts, exclusion counts, support coverage, amplitudes,
noise, locked lag, within cosines, held-out cross cosine, cross deficit,
signed-difference cosine, residual RMS and block-bootstrap intervals. Include a
small contact sheet with identical axes/support and separate rest/episode line
styles. Summaries should show pair-level values and parent-weighted category
medians; they should not collapse the panel into a truth accuracy.

The original maximum of 7,200 snippets follows from 18 pairs × 2 units × 2
halves × 100 events. At 121 samples × 18 channels, that is about 62.7 MB of
float32 selected waveform payload. Reading full 384-channel int16 rows is about
669 MB decimal (about 638 MiB) before filesystem chunk amplification, filtering,
buffers and temporary arrays, so the proposed 635 MB is a plausible nominal
payload but not a measured I/O total. The proposed 900 s CPU, 2 GB peak RAM and
25 MB output are unmeasured estimates. Preflight actual source dtype/chunking,
deduplicated read intervals, filter padding, batch size and temporary-array
factor; report projected and actual values. Full-row reads may exceed nominal
payload substantially if storage chunks are repeatedly touched.

## Readiness and next dependency

Execution should wait for a compact frozen manifest containing the 18 pairs and
parent grouping; half and rest/episode definitions; construction/event-row
namespace; preprocessing and source hashes; channel/support rules; row-sampling
and partner-exclusion rules; signed lag/difference conventions; metrics,
bootstrap hierarchy and confidence-bound decisions; and footprint/runtime
preflight. With those corrections, this remains a bounded qualified-waveform
diagnostic. It is not biological truth and does not authorize merge changes.

No voltage, raw data, GPU, held BH payload, scientific computation, framework,
test campaign or launch was used in this review.

