# Cluster 553 exact replay — intervention decision

Date: 2026-09-06. Status: **measured baseline evidence; no intervention nominated.**

## Result

The prioritized rescue-baseline unit 553 reproduces the census case under exact
indexing. The four nominal 1,000-spike fits consume 999 historical and 1,000
exact amplitudes respectively; historical replay matches the cached QC values
exactly. Exact-minus-historical changes are below 0.15 percentage points.

| phase | cached/historical missingness | exact missingness |
|---|---:|---:|
| reference 1 | 4.019% | 4.013% |
| reference 2 | 1.507% | 1.500% |
| failing 1 | 24.263% | 24.261% |
| failing 2 | 44.411% | 44.559% |

The shortest shared interval is **[5802.134, 6360.328) s**, 558.195 s. All
four fits are finite-interior and the case is not an indexing artifact.

## Bounded waveform evidence

Existing rescue raw voltage was read only around 100 evenly spaced assigned
events per window, on the same 16 channels frozen from the reference depth
(channels 294–309; reference depth 3013.58 um; peak channel 302). No candidate
sort or new calibration was run.

The waveform evidence is mixed rather than a coherent translation:

- event-level median peak-to-peak amplitudes were 283.6 and 308.2 uV in the
  reference windows, then 250.8 and 339.8 uV in the two failing windows;
- mean waveform cosine to the pooled reference was 0.971 in failing window 1
  and 0.943 in failing window 2;
- the modal peak channel remained at the same relative channel (13/15) in all
  four windows; median peak-channel offsets were 8, 9, 7.5, and 9;
- no waveform read sample reached the existing 500 uV saturation threshold;
- the cached train evidence reports 0.0% short-ISI fraction in both phases;
  the sorter export reports `ContamPct=0.7%` for cluster 553.

The later failing window shows a weaker and less similar average shape, but the
first failing window does not show a monotonic amplitude loss. Stable peak
location and the absence of a demonstrated assignment relationship do not
support external registration or identity replay. The evidence is compatible
with a local waveform/amplitude-integrity or detection effect, but does not
identify an existing processing operation whose replay would be informative.

## Decision

**No intervention is currently justified for cluster 553.**

Do not revive the closed threshold branch, launch external registration, or
apply identity replay based on this evidence. Cluster 452 remains a fallback
case, not an automatic next run.

A future intervention would need a separately frozen operation and prediction:
it would have to lower missingness in both failing windows, preserve the stable
reference waveform, and retain contamination/refractory safeguards. A failure
to do so, or a new identity ambiguity, would reject the intervention. This
record does not authorize that comparison.

Evidence receipt: `docs/outputs/luke_baseline_recovery_census_v1/cluster553_exact_replay_waveform_evidence.json`.
The closed Option A result and its population denominator remain unchanged.