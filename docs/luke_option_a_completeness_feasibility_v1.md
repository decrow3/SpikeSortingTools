# Option A completeness feasibility — closed

Date: 2026-09-06. Status: **engineering completion; scientific result inconclusive.**

## Decision

Do not rerun either arm and do not write a v2 comparison contract. The baseline-only
screen does not establish a duration that can measure the primary endpoint under the
frozen rules. The current run remains **inconclusive**, not positive or negative.

The limiting measurement is endpoint coverage, not sorting runtime alone: using the
fixed baseline-defined cohort of 124 good units present in the nominated interval and
depth band, no tested duration reached the required 50% of units with at least two
finite-interior 1,000-spike fits. The longest interval available inside the nominated
development window is 2,880 s and measured only 38/124 (30.6%).

## Feasibility table

The table uses the existing baseline curation and production amplitude source
`full_st[kept_spikes][:, 2]`, historical window indexing, `max_isi_s=10`, and the
frozen requirement of two finite-interior fits. The cohort was defined before this
screen from baseline good units with a retained spike in [7200, 7320] s and
[1810, 3710] um; no candidate output was used.

| centered interval | duration | nominated measured | H1 measured | H2 measured | H3 measured | limiting statuses at nominated |
|---|---:|---:|---:|---:|---:|---|
| [7200, 7320] | 120 s | 4/124 | 2/124 | 2/124 | 1/124 | 112 too few, 8 insufficient |
| [7110, 7410] | 300 s | 12/124 | 8/124 | 10/124 | 7/124 | 97 too few, 12 insufficient, 3 nonfinite |
| [6960, 7560] | 600 s | 22/124 | 18/124 | 20/124 | 19/124 | 87 too few, 7 insufficient, 8 nonfinite |
| [6660, 7860] | 1,200 s | 27/124 | 22/124 | 31/124 | 22/124 | 78 too few, 9 insufficient, 10 nonfinite |
| [6060, 8460] | 2,400 s | 34/124 | 31/124 | 36/124 | 25/124 | 67 too few, 4 insufficient, 19 nonfinite |
| [5820, 8700] | 2,880 s | **38/124** | 33/124 | 39/124 | 28/124 | 56 too few, 5 insufficient, 25 no windows |

`H1`, `H2`, and `H3` are the pre-existing healthy intervals centered at 3180,
5700, and 9660 s. They are reported as baseline feasibility controls, not as
candidate-selection evidence. Boundary-pinned fits were zero in these screens;
`no_windows`, `too_few_spikes_for_one_window`, `insufficient_finite_interior_windows`,
and nonfinite fits remain distinct failure states.

## Support and execution checks

- The field covers every tested interval on the recording clock.
- The 120 s frozen domain passes measured neighborhood support: minimum 0.975
  against the 0.95 gate for 190 channels.
- The maximum nominated interval [5820, 8700] s does **not** pass the same policy:
  minimum channel support is 0.867 and only 79 of 96 sampled depth positions reach
  0.95. The 120 s receipt cannot be reused for this extension.
- A 2,880 s run would be materially larger than the completed 120 s run and would
  still fail the measured support policy. Runtime/storage estimates therefore do
  not rescue feasibility; no longer comparison is authorized by this screen.

## Current run record

Both arms completed with the locked production environment (Python 3.12.4,
SpikeInterface 0.102.1, Kilosort 4.0.27, CUDA RTX A5000) and reached curation, QC,
and MATLAB export. The final corrected endpoint record reports:

- eligible units: **53**;
- exclusive reciprocal matches: **30**;
- measurable paired units: **2**;
- completeness coverage: **3.77%**, below the 50% floor;
- paired completeness change: **−0.259 percentage points** using
  `baseline_missing_pct - candidate_missing_pct`;
- contamination maximum change: **−0.0219**;
- waveform cosine P10: **0.9958**;
- peak-retention P10: **0.9278**;
- verdict: **inconclusive**.

The endpoint amendment was present before sort execution. Its aggregation wording
was corrected after execution to match the subtraction already used by the evaluator;
the corrected endpoint is therefore an engineering recomputation, not a claim that
the final endpoint implementation was prospectively frozen before candidate results.

No v2 contract is written. No thresholds, field, domain, gain, production gate, or
closed Option B branch was changed.