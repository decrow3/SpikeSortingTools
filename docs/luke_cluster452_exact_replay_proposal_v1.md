# Cluster 452 exact replay — fallback intervention proposal

Date: 2026-09-06. Status: **bounded proposal; not executed.**

## Exact replay

Rescue cluster 452 was the fallback from the baseline recovery census. Its four
cached historical fits reproduce exactly, and exact indexing changes estimates
by at most 0.112 percentage points:

| phase | historical/cached missingness | exact missingness |
|---|---:|---:|
| reference 1 | 1.750% | 1.757% |
| reference 2 | 2.557% | 2.576% |
| failing 1 | 19.488% | 19.600% |
| failing 2 | 22.890% | 22.951% |

The shared interval is **[3869.576, 4198.765) s**, 329.189 s.

## Bounded waveform evidence

The same existing rescue raw voltage was sampled at 100 evenly spaced assigned
events per window on 16 frozen reference-depth channels (channels 222–237;
reference depth 2305.33 um; peak channel 230). No sorter or candidate output
was read.

- median event peak-to-peak amplitudes: 335.2 and 223.8 uV in reference,
  236.7 and 228.5 uV in failing windows;
- modal and median peak-channel position stayed fixed across all four windows;
- mean waveform cosine to the pooled reference: 0.790 in failing window 1 and
  0.983 in failing window 2;
- cached short-ISI fractions were 0.050% reference and 0.150% failing;
- census `ContamPct` is 1.3%.

This is not a simple monotonic amplitude loss: the first reference is unusually
strong, the second reference is already weaker, and the first failing waveform
has a marked shape change despite an unchanged peak-channel location. The
second failing waveform is more similar in shape but remains lower amplitude.

## Intervention recommendation

452 is the first case with evidence that makes a **bounded identity/mixture
check** more plausible than external voltage registration: a large local
waveform-shape change occurs without a corresponding depth/peak-channel shift.
This remains a hypothesis, not an identity result; no spatially restricted
exclusive event matching or shift-null test was run here.

The smallest future comparison would be a retained-output identity evidence
replay on this interval and a matched stable control, with a frozen spatial
candidate set, exclusive event pairing, shift-null threshold, waveform
similarity, and union refractory/contamination checks. It would test whether
the deterioration coincides with events being reassigned to another existing
cluster or a mixed waveform population.

Predicted supporting outcome: a reciprocal, spatially plausible, null-excess
event relationship appears during the failing windows, with waveform evidence
and no unacceptable union refractory increase; the identity-aware intervention
would then be worth a separately contracted development comparison.

Reject the intervention if no null-supported relationship appears, if the
relationship is ambiguous, or if the union violates the contamination/refractory
guardrail. In that case the evidence supports only unresolved waveform/detection
change, not external registration.

This proposal does not execute the replay, revive the closed cluster-37 branch,
change production QC, or alter the closed Option A result. Evidence receipt:
`docs/outputs/luke_baseline_recovery_census_v1/cluster452_exact_replay_waveform_evidence.json`.