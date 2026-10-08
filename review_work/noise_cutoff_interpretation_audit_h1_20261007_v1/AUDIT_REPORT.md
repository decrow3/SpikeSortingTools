# Noise-cutoff interpretation audit

Verdict: **the Arm-A finite-failure gap is real in the saved tables but is not primarily explained by low counts, q-state mix, displaced-only units, or scoring-edge composition.** It persists in mixed-q and ordinary-interior units on both probes. Failures in both q0 and displaced subsets are common in both arms, but do not consistently explain the arm gap.

## Dataset and grain

The controlling census has 2,766 unique `(probe, arm, unit_id)` rows. The primary in-scoring-domain population has 2,243 units: imec0 A/REF 653/586 and imec1 A/REF 553/451. Numeric `unit_id` overlaps across arms are identifiers within separate sort outputs, not biological matches; the full composite key is unique.

The saved full-session noise status is three-valued: finite pass `<5`, finite fail `>=5`, and nonfinite/undefined. Undefined is reported separately and never converted to zero or failure. Full, q0, and displaced cutoffs are scalar summaries of different amplitude subsets; a shared threshold does not make their numeric differences a motion effect.

## Headline gap

| Probe | A finite fail | REF finite fail | A−REF gap | A undefined | REF undefined |
|---|---:|---:|---:|---:|---:|
| imec0 | 434/653 (66.46%) | 308/586 (52.56%) | +13.90 pp | 24/653 (3.68%) | 5/586 (0.85%) |
| imec1 | 401/553 (72.51%) | 287/451 (63.64%) | +8.88 pp | 17/553 (3.07%) | 5/451 (1.11%) |

Finite-only gaps are larger, +15.99 pp and +10.46 pp, so differential undefinedness does not create the headline direction. Spike-weighted failure shares are high in both arms and higher in A: 93.12% vs 82.84% on imec0 and 98.24% vs 94.86% on imec1. These are descriptive population differences, not matched-cell or sorter-only effects.

## Frozen explanations

### Low spike counts

Not supported as the primary explanation. Standardizing to pooled spike-count strata slightly increases the gap on both probes (imec0 13.90→14.65 pp; imec1 8.88→14.11 pp). Excluding units below 1,000 spikes leaves +12.34 pp on imec0 and +5.77 pp on imec1, short of the frozen 50% reduction rule.

Low counts mainly affect availability: undefined units typically have few spikes, while finite failures are concentrated in much larger units. Median spike counts for A finite failures are 12,262/17,702 on imec0/imec1, versus 219/261 for A undefined units. Failure prevalence increases across the saved count strata rather than being confined to sparse units.

### Q-state composition and displaced-only units

Not supported as primary compositional explanations. Standardizing q-activity strata leaves +13.10 pp on imec0 and +11.20 pp on imec1, with 100% common-stratum support. Displaced-only units show same-sign gaps (+13.75 pp, +7.49 pp), but mixed-q units show still larger gaps (+18.91 pp, +17.59 pp). Thus the gap is not created by an excess of displaced-only units.

### Ordinary interior units

The gap persists in units at least 200 µm inside the scoring domain: +15.54 pp on imec0 and +8.93 pp on imec1. Depth-stratum standardization leaves +13.66 pp and +9.00 pp. The gap is therefore not confined to scoring-domain boundaries.

### Persistence within q0 and displaced subsets

Among units with at least 100 events in each state and finite cutoff values in both states, both-state failure is common:

| Probe | A both fail / eligible | REF both fail / eligible |
|---|---:|---:|
| imec0 | 194/430 (45.12%) | 161/418 (38.52%) |
| imec1 | 227/378 (60.05%) | 199/332 (59.94%) |

This supports a broad, state-persistent measurement pattern. It does not explain the arm gap consistently: the eligible both-state difference is only +6.60 pp on imec0 and +0.11 pp on imec1. Moreover, 375 q0 and 189 displaced scalar cutoffs are nonfinite across the full census, far more than the 62 full-session nonfinite values, so state-subset availability is count/support sensitive.

## Data quality and implementation findings

- All 16 census packet members match the immutable manifest and COMPLETE binding.
- Composite keys are unique; required columns and booleans are valid; `num_spikes=q0_spikes+displaced_spikes` holds for every row.
- Saved full/q0/displaced pass flags exactly match finite cutoff `<5`; all frozen count, q-activity, and depth strata recompute with zero discrepancies.
- Probe-specific clocks and state exposures are constant within probe. Exact field membership remains provenance-qualified because the declared executed evaluator (`415f29d7...`) is not published on H1/shared storage.
- The availability ledger confirms that per-unit amplitude distributions are absent. Only derived cutoff scalars are available, so the current tables cannot distinguish tail-histogram instability, an estimator-support problem, and genuine low-amplitude truncation.

## Recommendation

Do not change the `<5` threshold or commission raw-voltage work from this result. The cheapest repair is to publish the exact executed evaluator snapshot and, in the next compact QC export, save per-unit full/q0/displaced histogram sufficient statistics: finite event count, bin range/counts, peak index/count, occupied-tail count, high-tail count/mean/standard deviation, first-low count, cutoff margin, and explicit undefined reason. A nearby nonbinding source snapshot constructs amplitude histograms, suggesting this may be a compact export repair, but the exact executed source must confirm that route.

Then recompute the frozen cutoff from those saved statistics and test whether Arm-A’s gap reflects estimator construction/support or true distribution truncation. Raw voltage or waveform extraction is not yet the cheapest adequate diagnostic.

## Implementation checks

- Done: source identity and packet integrity -> all 16 census members, manifest, COMPLETE binding, and consumed hashes verified (`SOURCE_VERIFICATION.json`).
- Done: grain, missingness, duplicates, joins -> 2,766 unique composite-key rows; 2,658 rows reuse a numeric unit ID when probe/arm is ignored, confirming that numeric IDs cannot be joined across sorts; no join was performed.
- Done: status and denominator semantics -> full/q0/displaced pass flags, nonfinite values, all-unit and finite-only denominators, state eligibility, and frozen strata independently recomputed (`DATA_QUALITY.json`, `run_audit.py`).
- Done: clocks/state counts -> one clock/exposure tuple per probe; all event counts partition exactly into q0 plus displaced.
- Done: predeclared sensitivity -> only count exclusion and count/q/depth standardization used; no threshold or category search.
- Done: independent known-answer fixtures -> threshold boundary plus pure-composition and within-stratum standardization fixtures, 3/3 passed (`TEST_RECEIPT.json`).
- Not done: exact executed evaluator byte review -> declared SHA `415f29d7...` is absent; nearby SHA `a31f1349...` was inspected only as corroborating, not substituted.
- Not done: amplitude-histogram reconstruction -> amplitude distributions/sufficient statistics are not saved in the compact tables.
- Can establish: the saved finite-failure gap is present on both probes and is not primarily a low-count, q-composition, displaced-only, or depth-edge composition effect under the frozen rules; state-persistent failures are common.
- Cannot establish: why the amplitude histogram has this shape, a causal pipeline stage, physical amplitude completeness, matched-cell harm, biological identity/purity, or a motion effect.
