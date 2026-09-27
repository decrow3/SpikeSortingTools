# CH: CF field-sensitivity packet and episode-residual design review

## Verdict

CF now answers its bounded operator question correctly. The actual-field and
zero-field native routes use the same 641,588-row source population, but their
motion-dependent registered geometries and finite pair domains differ. The
fixed common operator maps both banks to the same 182 physical coordinates,
uses the same event-derived zero-motion weights masked to support observed in
both banks, and gives exactly the same 8,968 finite upper-triangle pairs.

Under that fixed operator, the actual and zero arms have almost identical net
direct-edge totals (214 versus 213), yet 65 individual direct edges flip and 194
component relations change. Net counts therefore conceal substantial graph
rewiring. This is an operator-sensitivity result, not a causal percentage of a
historical deficit and not evidence that either field is correct.

No GPU work, raw voltage read, new template bank or graph calculation was run.
All checks use the sealed CF arrays plus existing BZ templates and event metadata.

## Source and population lineage

Packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-cf-field-sensitivity-v1/`.
Every manifest product matches its declared hash and size.

The receipt's full-population hashes independently match their actual sources:

- row IDs and post-TMM labels/times match BQ `capture/arrays.npz`;
- physical channels match the 641,588-row `matching1.h5` channel vector;
- all arrays are row-aligned and the four raw-byte SHA-256 values exactly match
  CF's receipt.

Both native arms therefore used the corrected full pre-grouping population for
radial weights, not construction rows in one arm. Both template banks have the
same dense 748 unit IDs, per-unit construction counts, rank-5 TSVD basis/mean,
configuration and 299,635-row membership. The actual bank is the exact corrected
BZ 3,000-sample bank; the zero bank is fresh. BZ did not save its historical RNG
state, so exact RNG-state equality cannot be established retrospectively. The
zero branch matches the captured seed-zero code contract. This limitation is
reported rather than silently converted into an equality claim.

## Native and fixed operators

| Operator/arm | Finite pairs | NN median | Direct edges | Linkage relations | Components |
|---|---:|---:|---:|---:|---:|
| actual native | 12,083 | 0.3378 | 229 | 483 | 559 |
| zero native | 8,968 | 0.3588 | 213 | 435 | 577 |
| actual fixed | 8,968 | 0.3500 | 214 | 465 | 572 |
| zero fixed | 8,968 | 0.3588 | 213 | 435 | 577 |

Native finite-domain overlap is 8,409 pairs, with 3,674 actual-only and 559
zero-only pairs. Native actual-versus-zero comparison has 90 direct flips
(53 actual-only, 37 zero-only) and 236 linkage flips (142/94). These changes mix
template content, registered geometry, empirical weights, spatial IoU and pair
availability.

The fixed operator's finite masks are array-identical. Its 65 direct flips are
33 actual-only and 32 zero-only; its 194 linkage flips are 112 actual-only and
82 zero-only. The saved common mapping is exact in both banks, `observed_both`
reconstructs from per-channel counts, fixed weights equal the single zero-motion
base weights on observed support and are zero elsewhere, and every unit has at
least 178 common observed physical sites (median 182).

The fixed operator is a controlled projection of two different template banks,
not a factorial decomposition of field, geometry, support and waveform effects.
Subtracting native and fixed gaps does not identify a causal fraction. Likewise,
a small distance does not prove identity and a large distance does not prove
field error. The prior failed BW/BQ criterion remains failed.

## Physical-coordinate template check

The compact figure uses raw template amplitudes—no unit normalization—on the
same physical contacts in both banks. Examples are frozen by stable sorting of
all units' common-support cosine distance at q10/q50/q95:

- unit 399: 0.00584;
- unit 591: 0.01093;
- unit 264: 0.21439.

The top row shows the strongest common contact's raw waveform; the bottom row
shows raw peak-to-peak amplitude over physical depth. Colours are Okabe–Ito blue
and vermillion with solid/dashed lines. The examples demonstrate both near-
identical and materially changed templates without choosing cases by graph
outcome. Figure:
`testing/outputs/ch_h1_cf_review_v1/CF_PHYSICAL_TEMPLATE_EXAMPLES.png`.
The final 1,872×1,044 PNG was visually inspected: all six panels, raw-scale
axes, physical-depth labels, legend and line-style distinctions are present and
unclipped.

## Conditional minimal episode-residual diagnostic — frozen proposal

This diagnostic may start only after CG reports qualified **rest-reproducible**
CE-native anchors. Rest reliability is necessary but does not prove that an
anchor is observable or stable during episodes.

### Episode coverage and cache

1. Use only qualified CG units and their frozen corrected-3,000 peak/support.
   Apply canonical episode, bounds and collision masks before sampling.
2. Tile accepted episode time on the deployed 0.25-s grid. A bin is measurable
   only with at least 20 total events from at least three anchors, each
   contributing at least five events. Require at least ten measurable bins from
   at least three catalogue episodes; otherwise stop as unresolved.
3. Cap each anchor at 200 additional episode events, balanced deterministically
   across episode and bin. Read all 182 contacts, preserving row IDs and
   competitors. Maximum additional cache:
   `8 × 200 × 121 × 182 × 4 = 140,940,800` bytes. With CG's rest cache, the
   maximum logical waveform payload is 281,881,600 bytes (268.8 MiB).

Expected additional resources are one reader, two CPU threads, under 20 GB RAM,
about 1 GB physical reads, under 220 MB added products and 6–10 CPU minutes.
Together with CG the final packet remains below 500 MB and retains the 30 GB
free-space guard. Support can be sliced before the per-channel median: for finite
arrays this is exactly equivalent to median-then-slice and reduces bootstrap
working memory; verify equality on the point statistic and first draw.

### Rigid-bin update and sign

Build each anchor's reference template only from its frozen rest rows. On each
measurable episode bin, evaluate a rigid residual grid from -40 to +40 µm in
2-µm steps through DARTsort's existing physical-coordinate interpolator. The
support, zero temporal lag, unit gain and half-1 noise remain frozen. The loss is
the equal-anchor median of per-anchor median noise-scaled squared waveform error;
event count cannot give one anchor extra weight. Choose the smallest loss, with
ties going to zero and then smaller absolute shift. No temporal smoothing or
post-hoc support/gain/lag optimization is allowed.

Sign convention: `corrected = observed - (Q displacement + residual)`. Positive
residual is passed as a positive addition to the deployed Q displacement through
the same DARTsort transform; no bespoke array-index shift is accepted. Save the
candidate grid, objective curve, support and transform receipt for every bin.

### Held-out claims and controls

- **Unit-heldout:** alternate qualified anchors by physical depth into fit and
  evaluation sets. Estimate each bin from fit anchors and score the untouched
  anchors at the same times. This is contemporaneous unit holdout only.
- **True time prediction:** separately fit, on chronologically early development
  episodes only, a fixed ridge rule predicting residual from Q displacement and
  its first difference. Freeze coefficients and regularization before applying
  to later episodes. Later anchors may score but never fit that rule. Without
  this arm, make no time-heldout claim.
- Compare against zero residual and sign reversal. Add 1,000 seeded nonzero
  circular episode-bin phase shifts of the proposed update, preserving its
  values, missing-bin pattern, run lengths and first-difference scale while
  breaking temporal alignment. This is the matched random null.
- Primary success requires held-out-unit waveform-loss improvement over zero,
  a unit/5-s-block bootstrap interval excluding zero, and improvement beyond the
  95th percentile matched-null result. Report sign reversal descriptively.

### Label-free T8 check

For every covered accepted episode, rerun the label-free depth×x shift reference
with peak/bootstrap resampling. Report the 10-µm native grid's ±5-µm quantization
floor and a bootstrap interval for measured shift and episode error. Compare the
total `Q + residual` displacement against Q alone on identical accepted episodes.
The residual arm may not worsen frozen episode error by more than 3 µm. If the
waveform and T8 criteria disagree, the result is inconclusive and no field is
adopted.

This remains a preregistered diagnostic. It does not authorize an actual field
fit, deployment or sort.

## CD sidecar closure

The sibling CD supplement at
`luke0804-imec1-cd-fullchannel-v1-supplement-v1/` matches its manifest and all
four original source hashes. For pair 1 and pair 5, support indices/AP IDs,
finite noise arrays, fitted lag/gain, point metrics and every interval exactly
match the original tables and new sidecar. Each A1/A2/B1/B2 bank has 100 unique,
ordered rows; common block counts are 20/23 and 29/34 respectively. Independent
helper-v3 qualification with 1,000/1,000 finite draws returns limited waveform
compatibility for both. This closes the prior packet-reproducibility caveat, but
still does not establish biological identity or merge truth.
