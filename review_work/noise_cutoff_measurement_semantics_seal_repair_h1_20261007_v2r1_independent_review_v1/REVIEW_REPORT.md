# Independent seal/provenance review: v2r1

Verdict: **GO** for the changed seal/provenance boundary.

Scope was limited to the immutable packet at `/mnt/NPX/Luke/DARTsort_motion_experiments/noise_cutoff_measurement_semantics_seal_repair_h1_20261007_v2r1`. No scientific outcome was re-inspected. The carried-forward scientific verdict remains exactly `NON_DECISIVE`.

## Checks

- The packet contains exactly the four members declared by `MANIFEST.json`, plus `MANIFEST.json` and `COMPLETE.json`; there are no missing or unexpected files.
- Every declared member byte count and SHA-256 matches the filesystem object.
- `COMPLETE.json` binds the observed `MANIFEST.json` SHA-256 `e33a1a8bec2f950486cb518a0df0fd5cb3ebad7fd7019af2b328ba212e22ad00`.
- Nanosecond filesystem mtimes satisfy strict ordering: `MANIFEST.json` is 16,127,268,400 ns later than the latest member, and `COMPLETE.json` is 17,151,935,500 ns later than `MANIFEST.json`.
- Original v1, its accepted review, preserved failed v2, and the source-host preparation contract exist at the declared paths and match every declared SHA-256.
- The v2r1 scientific-verdict file, provenance, boundary receipt, and completion record all carry forward `NON_DECISIVE` without a replacement scientific claim.

## Focused decision

The v2r1 packet repairs the prior seal-chronology caveat. The strict order `COMPLETE > MANIFEST > every member` is independently demonstrated. GO applies only to this provenance boundary; it does not resolve the exact-evaluator or histogram/intermediate sufficient-statistics prerequisites and does not establish within-neuron deterioration or a causal sorting defect.

Implementation checks
- Done: complete membership, byte counts, SHA-256 values, COMPLETE-to-MANIFEST binding, and nanosecond mtime ordering -> all pass (see `FILESYSTEM_RECEIPT.json`).
- Done: predecessor and source-host-contract reference hashes -> all pass (see `FILESYSTEM_RECEIPT.json`).
- Done: verdict carry-forward -> exactly `NON_DECISIVE`; no new scientific claim found in the reviewed boundary files.
- Not done: scientific outcome/evaluator/statistic reconstruction -> deliberately outside this changed-boundary review.
- Can establish: v2r1 is correctly sealed with the required strict filesystem chronology and declared provenance bindings.
- Cannot establish: the scientific cause of the noise-cutoff gap or any within-neuron deterioration.
