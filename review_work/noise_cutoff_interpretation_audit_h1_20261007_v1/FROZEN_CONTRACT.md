# Frozen noise-cutoff interpretation audit contract

Frozen 2026-10-07T22:14:53-07:00 on huklaban1 before any category-outcome aggregation. Three example rows at the head of `CENSUS_UNITS.csv` and the already-published overall missingness statement had been inspected during schema qualification; no arm/probe category rates, decompositions, or state-persistence outcomes had been computed or read.

## Milestone and decision

- Milestone: M1 saved-output diagnosis.
- Decision: determine whether the saved full-session R1 noise-cutoff arm gap is mainly associated with low spike counts, q-state composition, failures persisting in both q0 and displaced subsets, displaced-only units, or ordinary deep-interior units; select the cheapest measurement repair or explicitly retain uncertainty.
- Cheapest adequate test: one read of the compact census table and its frozen source/provenance, followed by predeclared descriptive stratification. No raw voltage, waveform extraction, sorting, RF/holdout, bulk transfer, or new H5 output.
- Completion: immutable compact packet with exact sources, data-quality checks, denominators/counts, frozen stratifications, implementation checks, and scoped recommendation.

## Controlling inputs

1. Census packet `/mnt/NPX/Luke/DARTsort_motion_experiments/post_sort_structure_census_h5_20261007_v1`:
   - MANIFEST `022ad0e1c558a283a97cabe7f27aaa318701bc228197ea7819ae75bcae2f1028`
   - COMPLETE `db6c9ea4f4f0c9f3841cc3bb351947b1aaf04979f7c0ef8257f01bd634ad3d43`
   - `CENSUS_UNITS.csv` `7d74b41e37ac4d5606decbc7caf5efd0ac8cf5ad3d9cda86ac6bfff2c38cccc7`
   - `SOURCE.py` `dc33e5d4d536998cfcd74a8d317b979577ab982c20c339e158b32925f74388ac`
   - `CONTRACT.json` `e90ad62778b5fa9422102c2f0d9ffeefa0c710a38ceba68d6d79ad9553bf5ea3`
2. Accepted independent census review `review_packets/post_sort_structure_final_review_20261007_v1_h1`:
   - MANIFEST `6da6d3da2a6c6fac9b6d570e2c8772cb7c6bfbc2baa3306497c993398ca26759`
   - COMPLETE `8ba7624321de95962576e4316419156531e14ed240ec9e7a5ab2d4d91ee878e8`
   - `FINAL_REVIEW.md` `7bfcc65b1f67a4ee382eaa051d39c4797522c4f7134a598f79c87cfacabd792e`

The census contract binds the underlying executed evaluator to commit `ba21bc97ad177ba5f2a1b79fb69d37f4f3eac3ad` and SHA256 `415f29d7f26102606143d0bd4fbb8ed9a136199a6642e664d50c2a6716b7c423`. Those exact bytes are absent on H1/shared storage. A nearby published source snapshot has SHA256 `a31f1349...` and is not substituted as exact executed source. Therefore saved-field arithmetic can be audited, but source-level state assignment and estimator-version equivalence retain the existing provenance qualification.

## Grain, key, and availability rules

- Intended grain: one curated unit per `(probe, arm, unit_id)`; probes and arms are separate populations and no cross-arm identity is inferred.
- Primary population: `in_scoring_domain == True`, reported separately for imec0 and imec1.
- Primary outcome has three states: `finite_pass` for finite `noise_cutoff < 5`; `finite_fail` for finite `noise_cutoff >= 5`; `undefined` for nonfinite. Undefined is never zero, pass, or failure.
- Headline denominator: all primary-population unit rows. Report pass, failure, and undefined counts/rates together.
- Finite-only failure rate is sensitivity only: finite failures / finite values.
- Spike weighting is descriptive sensitivity only and cannot replace unit prevalence.
- Full-session and state-specific noise-cutoff fields are computed on different event sets. They use the same named estimator and threshold according to the saved source lineage, but each histogram has subset-dependent support and stability. Their values are not paired effect estimates.

## Frozen categories

Use only pre-existing fields and thresholds:

- Spike-count strata: `0_to_99`, `100_to_999`, `1000_to_9999`, `ge_10000`; “low count” means `<1000` spikes only.
- Q activity: `displaced_only_q0_fraction_lt_0.10`, `q0_dominant_displaced_fraction_lt_0.10`, `mixed` exactly as saved. Applied q0 is not called physical rest.
- Depth: `inside_ge_200_um` is the “ordinary interior” category; `inside_lt_100_um` and `inside_100_to_lt_200_um` are boundary-near comparators. Excluded-domain rows are not mixed into the primary population.
- State measurement eligibility: at least 100 q0 spikes and at least 100 displaced spikes, and both state-specific noise-cutoff values finite.
- State persistence classes among eligible units: `both_fail`, `q0_only_fail`, `displaced_only_fail`, `neither_fail`. Units failing either count or finite-value requirement are `state_unavailable`, never `neither_fail`.

## Frozen comparisons and interpretation rule

For each probe independently:

1. Headline arm gap is `A all-unit finite-failure prevalence - REF all-unit finite-failure prevalence`, with the two undefined prevalences beside it.
2. Report the same arm gap within every frozen spike-count, q-activity, and depth category, retaining zero cells and unavailable values.
3. For each candidate compositional explanation, standardize arm-specific failure rates to pooled stratum weights over strata represented in both arms. Report raw gap, standardized gap, common-support coverage, and composition reduction `1 - abs(standardized_gap)/abs(raw_gap)` when the raw gap is nonzero.
4. Call a factor a **primary compositional explanation** only when common-support coverage is at least 90%, the standardized gap has the same sign or crosses zero, and the absolute gap falls by at least 50%. Otherwise describe concentration or persistence without “primarily explained.”
5. Call failure **persistent across q states** only from the eligible state subset and report `both_fail` as a fraction of (a) all primary units, (b) state-eligible units, and (c) full-session finite failures. No state-effect or causal-motion claim follows.
6. Call the gap **not confined to boundary/displaced-only units** if, in both probes, the ordinary-interior or mixed-q arm gap has the headline sign and at least half its absolute magnitude. A one-probe result is probe-specific, not general.
7. Sensitivity is limited to the predeclared finite-only denominator and exclusion of low-count units; no threshold search, post-outcome regrouping, or inferential p-value.

## Implementation-first prerequisites

- Verify packet seals and consumed member hashes.
- Recompute row count, composite-key uniqueness, required-column presence/type, boolean domains, missingness, `num_spikes=q0_spikes+displaced_spikes`, fractions, saved strata, and saved full noise status/pass consistency.
- Verify state-specific pass flags against finite value `<5` and treat saved booleans on undefined values as false, not evidence of failure.
- Trace the saved source fields and note the missing exact executed evaluator snapshot.
- Preserve any contradiction or failed check; do not repair values in place.

## Stop condition

Stop after the compact saved-table audit and immutable packet. No new measurement generation, H5 request, production gate, or experimental launch.
