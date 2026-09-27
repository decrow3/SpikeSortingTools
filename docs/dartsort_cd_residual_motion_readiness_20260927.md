# CD residual-motion one-update readiness inventory

> **WITHDRAWN / FAILED PROPOSAL (CE correction, 2026-09-27).** The 90 donor IDs
> and BC reliability measurements come from the separate static-W2 final sort,
> not BQ's prospective D2L run. No cross-run unit correspondence was proved.
> Therefore the inverse-`depth_reorder` mapping, 48-anchor count, anchor lists,
> row count and all downstream split claims below are invalid. They are retained
> only to preserve the failed proposal. See
> `docs/dartsort_ce_fullchannel_and_anchor_correction_20260927.md` for the
> authoritative audit and replacement design. No residual update was fitted.

## Verdict

**Superseded claim:** A small causal one-update test is source-ready without a voltage read, but it
must use a conservative 48-anchor rest-only cohort and explicit label-namespace
mapping.** This is a prospective residual-motion diagnostic, not an identity
test, motion-field fit, or sorting evaluation. No loop or update was run.

## Exact reusable inputs

- Corrected short-construction bank:
  `luke0804-imec1-bz-construction-v2/3000/template_data.npz` (SHA-256
  `60a807...`) and `decision_arrays.npz` (`abbd2b...`), 748 stage units,
  `[748,121,206]` registered templates.
- BQ fixed clock and namespaces:
  `capture/arrays.npz` (`234b6e...`) and `contrast_arrays.npz` (`636cad...`).
  The former supplies row IDs, pre-stage labels/times, maps and long-bank force
  graph; the latter supplies `fixed_current_labels` in the **pre-depth-reorder
  component namespace**.
- Matching features: the 1.13 GB `matching1.h5` (`6734ba...`) has exactly
  641,588 rows and times identical to BQ's saved pre-stage times. Its own labels
  are not equal to BQ pre-stage labels and must not be substituted. By exact row
  position it supplies channel, score, scaling, physical `[182,2]` geometry and
  point-source localization `[x,y,z]`; DARTsort uses column 2 as depth and
  applies motion separately (`cluster_util.py:372-374`).
- Existing anchor evidence: 90 final-label candidates have frozen count/support
  measurements; BC v2 supplies disjoint-half spatial reliability over seven
  states. These IDs are in the final depth-reordered namespace.
- Independent T8 input: already validated pilot-frontend peak caches
  `full_block_007` through `full_block_010` cover session 840–1320 s and can be
  restricted to W2, 900–1240 s. No sort labels enter T8.

## Frozen anchor rule and namespace-safe exclusions

Map each final candidate ID `f` to pre-reorder component `p` using the unique
inverse of `depth_reorder`, where `depth_reorder[p] == f`. Then require the
already defined evidence only:

1. existing `first_three_pass` and `scalar_and_waveform_support_pass`;
2. BC minimum split-half centred cosine >=0.90 and minimum normalized agreement
   >=0.90;
3. exclude each pre-reorder current component that spans more than one saved
   no-force component (85 disputed force-dependent groups);
4. additionally exclude a current component containing any stage unit incident
   to an off-diagonal force-linkage relation in either BQ's accepted long bank
   or the corrected short bank;
5. on W2's two contiguous 170-s halves, require >=100 rows and >=10 nonempty
   5-s bins in each half.

The evidence rules retain 76/90 before force exclusions, 67 after disputed-
component exclusion, 62 after the union-of-force-graphs exclusion, and 48 after
the contiguous-time count gate. Sparse missing reliability never becomes a
stable anchor.

For unit holdout, sort the 48 by `(depth_um, final_unit_id)` and alternate:

- fit units: 6, 13, 22, 74, 88, 93, 139, 184, 196, 275, 307, 332, 336, 352,
  415, 437, 451, 461, 480, 493, 496, 498, 509, 512;
- held-out units: 12, 18, 48, 86, 92, 124, 143, 191, 222, 291, 329, 334, 343,
  407, 435, 443, 454, 477, 484, 494, 499, 505, 511, 526.

The time-development half is local `[0,170)` s / session `[900,1070)` s; the
untouched time-held-out half is local `[170,340)` / session `[1070,1240)`.

## Minimal cache and one-update design

No waveform extraction is needed. Cache only the 56,658 selected row-aligned
records (about 2.4 MB uncompressed): immutable row ID, pre-component/final ID,
time, point-source x/z, channel, score, scaling and unit/time-fold flags. Add the
48-row anchor table, exact hashes above, final-to-pre mapping, physical geometry,
field hash/sign and the frozen support/exclusion masks. Do not cache or infer new
labels. The corrected short templates and BC support arrays remain immutable
side inputs.

For each anchor, centre motion-corrected depth on its median in the development
half. In each 5-s bin, form per-unit residual depth, cap every unit at one vote,
and estimate one rigid residual update from fit units only. Freeze every robust
aggregation/smoothing choice using the development half, then apply exactly
once in held-out time. Under the deployment sign convention
`corrected = observed - displacement`, add the measured residual to the field.
Never iterate or reselect anchors from the outcome.

Controls are paired on identical rows/bins:

- zero update;
- sign-reversed update `-delta(t)`, which exactly preserves update magnitude
  and first-difference smoothness while reversing the causal direction;
- the proposed `+delta(t)` update.

Primary evaluation uses held-out units during held-out time. Success requires
the proposed update to reduce the parent-equal median absolute per-unit residual
versus **both** zero and sign-reversed controls, with a 95% interval formed by
resampling 5-s blocks and units (not events) excluding zero improvement. It must
also preserve support/count eligibility and not worsen the independent T8
label-free shift error by more than Q's frozen 3-um tolerance. Report T8 and
unit evidence separately; disagreement is failure/inconclusive, not a reason to
tune.

## Limits and one next hypothesis

The point-source depth is an estimator output from the accepted matching file,
not biological ground truth. Unit splits reduce circularity but cannot remove
fragmentation or localization bias. The corrected 3,000-sample bank changes
force structure and is used only to define/exclude disputed support here.

The one useful next hypothesis is: **a single small residual update learned from
force-independent, split-half-reliable anchors reduces held-out anchor residuals
in the correct direction and agrees with label-free T8, whereas a sign-reversed
update does not.** This is the cheapest causal check before any loop, new field,
sort, or broader estimator work.
