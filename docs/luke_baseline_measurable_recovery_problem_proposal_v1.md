# Baseline measurable recovery problem — scoped proposal

Date: 2026-09-06. Status: **closed with no intervention nominated.**

## Decision

Existing baseline QC identifies one compact, measurable recovery problem, but it
does not discriminate a defensible intervention. Do not launch another audit,
sorter run, identity replay, or external-correction run from this record.

The correct outcome is **no suitable failure–intervention pair currently exists**.

## Nominated baseline problem

Use the already frozen rescue-baseline audit selection:

- unit: `rescue_luke0804_v2v1_g0_imec0__c37__failure1`, cluster 37;
- source: cached rescue curated arrays and cached amplitude QC only;
- reference: `[6590.316, 6657.450)` s, two finite-interior historical
  1,000-spike windows, median missingness **0.667%**;
- deterioration: `[6657.452, 6810.180)` s, two finite-interior windows, median
  missingness **43.474%**;
- shortest shared interval preserving both phases: **[6590.316, 6810.180) s**,
  **219.864 s**;
- transition: **42.807 percentage points** in the existing amplitude-truncation
  diagnostic.

The four windows are contiguous in the cached baseline sequence, use the
production amplitude source `full_st[kept_spikes][:, 2]`, and reproduce the
historical fits. The audit’s exact-1,000 sensitivity remained eligible and
changed the window estimates only slightly; this proposal does not replace the
original Option A endpoint.

### Waveform and refractory evidence

The existing evidence panel read 400 assigned events on 16 frozen channels
around the peak channel, without recentering waveforms. It found:

- no samples at or above the 500 uV saturation threshold;
- median peak-to-peak amplitude **345.7 uV** in reference windows versus
  **253.7 uV** during deterioration;
- no automated voltage-integrity verdict beyond the non-saturation observation.

The existing sorter output reports `ContamPct=17.2%` for cluster 37. This is not
a clean refractory result and is retained as a safeguard, not treated as proof
that cluster 37 is one neuron. The cached evidence therefore supports a
repeatable unit-level amplitude/depth change, while leaving identity and causal
stage unresolved.

## Cohort scope and exclusions

This is a development cohort, not a revised Option A denominator:

- **1 of 301** baseline rescue KS-good clusters, **0.33%** of the baseline-good
  population;
- **1 of 710** curated rescue clusters, **0.14%** of all clusters;
- it excludes the other 300 KS-good clusters, all MUA/no-label clusters, units
  without four consecutive valid historical windows, units with boundary-pinned
  or nonfinite fits, and units outside the permitted development windows.

The matched diagnostic control is rescue cluster 666, selected from four stable
finite-interior windows, but it is not part of the failure unit’s denominator
and does not establish a candidate effect. The cohort is intentionally narrow:
it demonstrates that one baseline unit supplies a stable period and a sustained
degradation with inspectable QC, not that the population has been redefined.

## Intervention decision

The supported observation is classified only as **motion/amplitude change**:
missingness rises with a measurable depth/amplitude change and without saturation.
That makes motion-aware handling a plausible question, but it does not choose
between external voltage registration and unwarped identity handling.

No intervention is nominated because:

- identity redistribution was not tested under a frozen spatial/shift-null
  protocol;
- the prior Option B identity candidate is closed on its prespecified cluster-37
  test;
- the prior Option A comparison used a different 120 s domain and remained
  inconclusive because its primary endpoint coverage was 3.77%;
- the unit’s existing refractory/contamination evidence is not clean enough to
  support a casual family-link or merge claim.

Choosing either intervention now would be choosing an architecture from an
ambiguous observation, not following existing evidence.

## What would reject a future bounded intervention

If a future authorization supplies a feasible, predeclared endpoint protocol,
the smallest comparison would use this 219.864 s interval, the same unit-level
failure, and a fixed healthy/control safeguard. An intervention would be
rejected if it did not lower missingness in the two deterioration windows with
supported waveform evidence, or if it increased contamination/refractory
violations, created ambiguous identity relationships, or damaged the stable
reference period. A positive local result would remain exploratory and require
the existing independent-window, held-out, and second-session gates.

No such comparison is authorized by this proposal. The closed Option A result,
its 1,000-spike endpoint, and the first-release decision remain unchanged.