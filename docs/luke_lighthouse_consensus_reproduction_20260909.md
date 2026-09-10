# Lighthouse consensus reproduction

## Technical summary

The previously saved 17-label waveform-traced lighthouse consensus is exactly
reproducible from the cached strict-event table. An independent implementation
recovered every per-label 5-second track to within `1.14e-13` um and the
population median to within `3.55e-15` um. These are floating-point differences,
not analytical discrepancies.

Collapsing likely-dependent labels 673 and 675 into one identity-family vote does
not change the qualitative answer. The resulting 16-family consensus correlates
`r=0.979` with the saved consensus. Thus the earlier shared-motion result was not
created by counting 673 and 675 separately.

![Consensus reproduction](../testing/outputs/luke_lighthouse_consensus_reproduction_v1/01_consensus_reproduction.png)

## What was independently rebuilt

The audit reads the 897 cached `strict_accepted` events from 930--1030 s. For
each identity it computes the median `relative_um` in non-overlapping 5-second
bins, subtracts that identity's median over 930--940 s, and takes an unweighted
median across identities in every bin with at least three votes. It does not
import the original implementation's binning or centering helpers.

The resulting unit tracks and consensus were compared with the frozen rows in
`luke_kilosort_family_plausibility_v3/binned_displacement_tracks.csv`.

For the dependence sensitivity, events assigned to labels 673 and 675 were
pooled before binning and centering. That pooled family then contributed one
vote, alongside the other 15 labels.

## The shared result survives deduplication

Among the 16 family tracks, 67 of 120 pairs have at least four common 5-second
bins. Their median Pearson correlation is 0.255 and 68.7% are positive. Thirteen
families have enough support for a leave-one-out comparison; their median
correlation with the median of the other families is 0.433 and 12/13 are
positive. Independent circular shifts of each sparse track put both summary
statistics beyond the 2,000-draw null (`p<=0.001`).

This is materially stronger agreement than the static premerge-cluster-centroid
control. The difference is expected: the earlier waveform matcher can assign
one identity to translated support at different absolute depths, whereas a
single Kilosort cluster is spatially localized.

## Reproducible does not yet mean calibrated

The consensus itself remains an exploratory readout. Collapsing 673/675 changes
individual bins by as much as 29.2 um, despite the high overall correlation. The
deduplicated trace spans 190.4 um peak-to-peak, and support ranges from only 3 to
12 families per bin (median 9; three bins have fewer than six).

The circular-shift null tests temporal alignment conditional on the already
accepted sparse tracks. It does not remove shared template-lattice acceptance
bias, identity lookalikes, implausible per-unit depth excursions, or changes in
which identities contribute to each bin. Therefore the result verifies that the
old calculation and qualitative waveform-traced agreement recur; it does not
yet certify the trace as biological motion ground truth.

## Assessment

**Reproduction: pass.** The frozen result is exactly recoverable.

**Dependence sensitivity: pass.** Giving 673/675 one vote preserves the answer.

**Motion-ground-truth validity: share with caveats.** The agreement is real in
the accepted-event table, but the lattice and identity controls still determine
whether that agreement is biological.

## Next cheapest validation

Repeat this identical collation after restricting accepted events to a narrow
window around lattice nodes. Preserve the same identities, 5-second bins,
centering, family voting, and minimum-support rule. If the waveform-traced
consensus remains similar on support where acceptance is not phase-biased, that
would address the largest known confound without rerunning extraction.

