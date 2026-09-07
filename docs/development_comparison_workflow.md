# Longitudinal development comparison workflow

Comparison schema `long-sort-comparison-v2` uses all baseline units in the
scoring interior as the fixed coverage denominator (all baseline units when
no spatial region is supplied), regardless of candidate matching or labels.
`baseline_eligibility.csv` records unmatched units, partners outside the
interior, insufficient fit support, and measurable pairs. Coverage conditional
on surviving pairs is reported separately and cannot satisfy the coverage gate.
V1 comparison outputs require a new output namespace for reevaluation.

Arm execution validates the exact prepared strip against the contracted time
and physical channel selection. Reused curation and QC require matching stage
settings, completed receipts and required outputs, not just a shared source sort.

This is the executable contract layer for the
[hindsight-first prescription](spikesorting_hindsight_development_prescription.md).
It coordinates existing sorter runners and evaluators; it does not introduce a
new sorter or replace their content-bound manifests.

The capability-to-code inventory that guided the implementation is in the
[implementation map](hindsight_first_implementation_map.md).

## 1. Freeze the comparison before running it

Copy `configs/example.development_comparison.v1.json` to a versioned experiment
file. Replace the recording identity, accepted-recording digest, duration and
physical depth bounds. Define one reference and a bounded set of candidates.
All arms must name the same curation profile.

The first resolved Luke configuration is
`testing/configs/luke0804_hindsight_ladder_v1.json`. It selects 100 physical
contacts from 1400–2380 µm for processing and a 60-contact 1600–2180 µm interior
for efficacy scoring, with 200 µm real-voltage halos. Its status is
`implementation_ready_not_executed`; committing the contract does not authorize
or claim results from the expensive sorts.

The physical support contract is expressed in micrometers. The processing band
must surround the scoring band by at least `required_support_um` on both sides;
units near the unsupported processing boundaries are not efficacy units.

Validate and render the content-bound plan:

```bash
python -m testing.run_development_ladder plan \
  --config configs/my_comparison.v1.json \
  --output /path/to/experiment/comparison_plan.json
```

Planning refuses short non-full-session recordings, discontinuous intervals,
naked depth strips, moving curation, unit count as a primary endpoint, and a
custom method without a named demonstrated failure.

Prepare the full-duration physical strip, then run every named standard arm
through the shared curation and QC stages:

```bash
python -m testing.run_development_ladder prepare-strip \
  --config configs/my_comparison.v1.json \
  --output-root testing/outputs/my_comparison/recording

python -m testing.run_development_ladder run-arms \
  --config configs/my_comparison.v1.json \
  --recording-dir testing/outputs/my_comparison/recording \
  --output-root testing/outputs/my_comparison
```

For two independent workers, select whole arms explicitly. Each worker writes
one group receipt and each arm is protected by an OS lock; subset workers do
not overwrite the shared summary:

```bash
python -m testing.run_development_ladder run-arms \
  --config configs/my_comparison.v1.json \
  --recording-dir /local/my_comparison/recording \
  --output-root /local/my_comparison \
  --group-id motion-axis \
  --arm rescue_12_9_motion_off \
  --arm rescue_12_9_native_rigid \
  --arm rescue_12_9_native_nonrigid
```

After all intended workers complete, create the shared summary. Omitting
`--arm` requires every contracted arm; repeated `--arm` options finalize only
that declared subset:

```bash
python -m testing.run_development_ladder finalize-arms \
  --config configs/my_comparison.v1.json \
  --recording-dir /local/my_comparison/recording \
  --output-root /local/my_comparison
```

Long jobs must be launched by an independent manager. Wrap its command with
`python -m testing.managed_job --receipt ... --cwd ... -- ...` so the exact
command, working directory, GPU assignment, PID, timestamps and final return
code survive outside the initiating terminal. Prove manager survival first
with a cheap dummy command; the wrapper alone is not an independent manager.

Compare any candidate with the reference using the same generic comparator used
by the historical full-session rigid analysis:

```bash
python -m testing.run_development_ladder compare-arms \
  --config configs/my_comparison.v1.json \
  --recording-dir testing/outputs/my_comparison/recording \
  --output-root testing/outputs/my_comparison \
  --baseline rescue_12_9_motion_off \
  --candidate rescue_12_9_native_rigid
```

`testing/luke_full_session_compare.py` now imports its correspondence and
common-time calculations from `testing/sort_comparison.py`, so an existing or
completed full-session rigid output can be evaluated through the same code path.
For exact reuse rather than rerunning, an arm may declare `existing_sort_dir`
and `existing_downstream_root`; reuse is accepted only when the sort, curation,
and QC identity receipts match the contracted recording and effective settings.

## 2. Run existing arms and the evaluator

Run each arm against the exact accepted input in the plan. Every arm retains its
own recording, sorter, curation and QC receipts. Results must use the same
physical time support and report the eligible population as well as the
measurable population.

Before results can rank real arms, the evaluator receipt must establish:

- exclusive event matching and chance-aware coincidence;
- common physical-time support and explicit measurement coverage;
- content-bound inputs, verified clocks and preserved physical geometry;
- worse results for the known pathological claim mask and external warp; and
- rejection of a fake improvement made by dropping difficult spikes.

These are evaluator qualification controls, not efficacy results.

## 3. Evaluate without a composite score

The evaluator writes a JSON document with this shape (metric names must exactly
match the prospective contract):

```json
{
  "schema_version": "longitudinal-development-results-v1",
  "contract_digest": "digest from comparison_plan.json",
  "evaluator": {
    "invariants": {
      "exclusive_event_matching": true,
      "chance_aware_coincidence": true,
      "common_physical_time": true,
      "measurement_coverage_reported": true,
      "content_bound_inputs": true,
      "acquisition_and_selected_clocks_verified": true,
      "physical_probe_geometry_preserved": true
    },
    "negative_controls": {
      "pathological_claim_mask": {"outcome": "worse"},
      "harmful_external_warp": {"outcome": "worse"},
      "fake_improvement_by_dropping_difficult_spikes": {"outcome": "rejected"}
    }
  },
  "candidates": [
    {
      "name": "candidate-name",
      "eligible_units": 100,
      "measurable_units": 80,
      "common_time_fraction": 0.75,
      "primary_improvements": {"contracted_metric": 6.0},
      "guardrail_regressions": {"contracted_guardrail": 0.0}
    }
  ]
}
```

Include one row for every arm, including the reference. Improvements are signed
so larger is better; guardrail regressions are signed so larger is worse. The
reference normally has zeroes.

```bash
python -m testing.run_development_ladder evaluate \
  --config configs/my_comparison.v1.json \
  --results /path/to/experiment/evaluator_results.json \
  --output /path/to/experiment/advancement_decision.json
```

An arm stops on evaluator failure, inadequate coverage, a primary effect below
the prespecified meaningful margin, or any guardrail breach. Among passing arms,
only the Pareto frontier advances. If multiple arms remain incomparable, the
tool reports that an explicit selection is required; it never manufactures a
weighted score.

Advance through implementation `smoke`, `long_strip`, `full_probe`, then
`second_session`. Freeze settings before full-probe confirmation. Short snippets
are subsequent fault-isolation tools, not substitutes for these stages.
