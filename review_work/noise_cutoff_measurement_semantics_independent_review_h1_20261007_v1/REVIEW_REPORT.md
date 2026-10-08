# Independent review: noise-cutoff measurement semantics

Verdict: **Share with caveats — ACCEPT `NON_DECISIVE` within the reviewed scientific scope.** The recommendation follows from the inspected evidence. The saved scalar-table gap is reproducible, while the exact accepted evaluator bytes and per-unit/subset construction sufficient statistics are unavailable. Those gaps prevent attribution to a sorting-signal defect or within-neuron deterioration.

Reviewed immutable snapshot:
`/mnt/NPX/Luke/DARTsort_motion_experiments/noise_cutoff_measurement_semantics_h1_20261007_v1`

No production launch, H5 output, raw-voltage or waveform read, RF/holdout access, sort, or parameter sweep was performed.

## Scope and principal checks

- Verified all 11 manifest-listed member sizes and hashes; no unexpected or missing packet files were found.
- Verified that `COMPLETE.json` binds the observed `MANIFEST.json` SHA-256 `e1fc1f66fbffc95d39d4d7501deb0ef117b4d8cbccf705f50ec611541861edae`.
- Independently verified every member and seal in the two frozen input packets.
- Independently recomputed the four in-domain arm/probe counts and all-unit finite-failure prevalences from 2,766 census rows. The A-minus-REF gaps are +13.9028 percentage points on imec0 and +8.8772 points on imec1; there are no duplicate `(probe, arm, unit_id)` keys and no scalar/pass mismatches.
- Inspected the corroborating evaluator snapshot (SHA-256 `a31f1349...`) at lines 59–87 and 246–335. The packet's DAG correctly describes 100 edges/99 bins, dynamic subset-specific maxima for direct calls, count-based `first_low`, population standard deviation, undefined branches, and strict `<5` passing. The exact declared commit is absent from the local Git object database, so this remains corroborating rather than executed-source proof.
- Independently checked the fixture receipt's two manual formulas, explicit histograms, three undefined branches, strict threshold boundary, and outlier-induced histogram change. These fixtures test the transcribed semantics without calling the production helper.
- Verified that the census availability ledger marks amplitude distributions unavailable for all four probe/arm rows. The scalar/pass fields are enough to reproduce downstream status, but not the histogram construction or branch diagnostics.

## Material finding

1. **Packet seal chronology does not demonstrate `COMPLETE.json` was written last (P1 provenance protocol).** Nanosecond mtimes show `COMPLETE.json` at `2026-10-07 22:41:23.542658900 -0700` and `MANIFEST.json` at `22:41:23.543089800`, about 0.431 ms later. Current bytes are self-consistent and all hashes pass, so this does not overturn the scientific recommendation; however, it violates or at least fails to prove the required publication ordering. **Fix proposed:** do not overwrite this packet. Publish a new version whose staging receipt records manifest creation/fsync before `COMPLETE.json`, then verify ordering from the publication receipt rather than relying only on mtimes.

## Recommendation review

`NON_DECISIVE` is the only defensible option among `RETAIN`, `REPAIR`, and `NON_DECISIVE` under the frozen rule. `RETAIN` would require exact executed-source and construction evidence that are absent. `REPAIR` would overstate the evidence because no specific construction defect has been observed; only sensitivity and insufficient retained intermediates are established.

The proposed cheapest next test is appropriate: export compact histogram sufficient statistics and exact evaluator provenance from the existing source-host arrays before any waveform or voltage work, then reconstruct the saved scalar and separately test maximum leverage and count sensitivity. This can distinguish provenance/implementation failure and estimator sensitivity without claiming signal harm. Its future contract still needs to freeze deterministic handling of tied maxima, count-matching eligibility/support, sampling seed, comparison statistic, and pass/fail interpretation before outcomes are inspected.

Even if that test reproduces the gap, unmatched arm populations remain a separate limitation; it cannot establish within-neuron deterioration without a frozen identity/correspondence design.

## Review scorecard

Totals are scoped inventory counts, not an accuracy percentage.

### Artifact quality

| Category | Observed defects | Assessment |
|---|---:|---|
| Usefulness and completeness | 0 / 5 | Contract, DAG, source ledger, sufficiency table, fixtures, and recommendation answer the scoped decision. |
| Analytical clarity | 0 / 4 | Units, thresholds, eligibility, unmatched populations, and construction limits are explicit. |
| Artifact integrity and publication | 1 / 3 | Hash closure passes, but filesystem chronology does not demonstrate COMPLETE-last publication. |

### Analytical correctness and robustness

| Category | Observed defects | Assessment |
|---|---:|---|
| Source authority and confidence | 0 / 3 | The exact-source gap is correctly disclosed; corroborating source is not substituted. |
| Value accuracy | 0 / 4 | Four group counts/rates and two arm gaps independently reproduce. |
| Data-quality controls | 0 / 4 | Grain, duplicates, missing construction data, undefined values, and availability are handled correctly. |
| Conclusion support | 0 / 3 | `NON_DECISIVE`, causal limits, and the cheapest next test follow from the checked evidence. |

## Implementation checks

- Done: current packet integrity -> all 11 listed members match declared byte counts and SHA-256 hashes; `COMPLETE.json` binds MANIFEST SHA-256 `e1fc1f66...` (`REVIEW_RECEIPT.json`).
- Done: frozen input integrity -> all 16 members in each input packet hash correctly and both COMPLETE files bind their manifests (`REVIEW_RECEIPT.json`).
- Done: saved scalar pattern -> independently recomputed 2,766 rows, four group counts/rates, +13.9028 pp and +8.8772 pp gaps, zero duplicate keys, and zero pass mismatches (`review_packet.py`; `REVIEW_RECEIPT.json`).
- Done: measurement construction -> inspected corroborating source lines 59–87, 246–318, and 323–335; packet DAG matches those bytes (source SHA-256 `a31f1349...`).
- Done: fixture independence and arithmetic -> independently recomputed manual expected values and checked boundary/undefined/outlier receipts; production helper was not used (`REVIEW_RECEIPT.json`).
- Done: publication-order check -> nanosecond mtimes place COMPLETE 0.431 ms before MANIFEST, so COMPLETE-last is not established (`REVIEW_RECEIPT.json`).
- Not done: exact executed-source review -> declared commit `ba21bc97...` is absent from the local object database and declared SHA-256 `415f29d7...` bytes were not located.
- Not done: scalar reconstruction from construction sufficient statistics -> per-unit/subset histograms, ranges, retained/invalid counts, and branch intermediates are unavailable.
- Not done: causal or within-neuron attribution -> arm populations are unmatched and the review intentionally did not expand to waveform/voltage work.
- Can establish: current packet contents are hash-consistent; the saved finite-failure gap is reproducible; the corroborating estimator is construction-sensitive; `NON_DECISIVE` and the compact saved-data/source test are supported.
- Cannot establish: exact executed semantics, the cause of the gap, a specific repair, within-neuron deterioration, biological identity/purity, causal pipeline stage, or arm superiority.
