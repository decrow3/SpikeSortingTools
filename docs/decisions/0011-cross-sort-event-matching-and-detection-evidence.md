# 0011 — Cross-sort identity must be exclusive; detection claims need a null

**Status:** Adopted 2026-09-03  
**Retracts empirical results in:** [0009](0009-cross-sort-comparisons-must-be-unit-matched.md),
[0010](0010-rescue-yield-is-relabelling-not-detection.md), and the Phase A/A2
and full-session stitching reports.  
**Does not retract:** the composition-confound rule in 0009, the raw 301 versus
228 KS-good counts, or the synthetic logistic-model characterization of the
truncation fitter.

## Problem found

The earlier cross-sort matcher assigned each source spike its nearest target
spike independently. One target event could therefore be counted more than
once, and swapping source and target changed the matches. The reported 101
matched / +200 unmatched / -127 unmatched decomposition is consequently not a
valid identity result.

The subsequent claim that unmatched units were nevertheless detected in the
other pipeline used any spike anywhere on the probe within 0.5 ms. At the
observed whole-probe event rates, that condition covered about 87% of session
time for rescue and 89% for legacy. A reported median of 100% “found anywhere”
therefore cannot distinguish a shared detection from chance coincidence.

Phase A2 also computed the proposed merge's refractory statistic on the clean
anchor train, not on the union of fragment trains that would actually be
merged. The 92–95% clean-merge result did not test its stated claim.

## Rule

1. Cross-sort identity uses maximum-cardinality, one-to-one event pairs before
   cluster-pair fractions are calculated.
2. A shared-detection claim must be spatially plausible and exceed a fixed
   circular time-shift null. Whole-probe temporal coincidence alone is
   descriptive only.
3. A proposed merge is evaluated on the complete union of member spike trains,
   including events not matched to the anchor.
4. Corrected analyses write versioned outputs. Old outputs remain historical
   artifacts and cannot be cited. Conclusions remain unresolved until v2 is
   regenerated.

## Consequence

We currently do **not** know whether rescue detects anything new, loses legacy
detections, or is better than legacy. The +200/-127 composition, 80 MUA
promotions, 27 demotions, 92–95% clean merges, and the full-session stitching
result (2 recovered / 4 destroyed) are all withdrawn pending rerun.

The raw sort counts and directly computed similar-template gate count remain
observations, but they do not establish pipeline superiority or a mechanism.
