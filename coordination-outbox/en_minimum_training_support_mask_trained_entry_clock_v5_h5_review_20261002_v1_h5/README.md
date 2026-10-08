# H5 independent review: trained-entry clock v5

Verdict: **GO for the clock repair only**.

The candidate correctly treats native `full_st[:,0]` as crop-local and exported
`spike_times.npy` as global acquisition samples, with the exact relation
`exported == selected_local + crop.imin`.  Contract validation is fail-closed
before bound source or recording access.  Both clocks use exact half-open crop
bounds and signed-int64 guards.  Previously accepted kept-row ordering,
cluster ancestry, amplitude ancestry, and nondecreasing checks remain.

This verdict does not authorize a real trained launch, the no-mask REF-repeat
pair launcher, RF/holdout access, or any scientific-effect claim.  The reviewed
production config remains `execution_enabled: false`.

One provenance correction is recorded: the candidate member
`bindings/parent_v4_delta/source__kilosort_support_mask_trained.py.diff` is a
stale v3-to-ordering diff despite its directory label.  H5 independently
reconstructed the actual accepted-v4-to-candidate-v5 delta from the two
manifest-bound source files and verified the declared old/new hashes.  The
stale convenience diff must not be cited as the clock delta; this review and
`EXACT_DELTA.json` provide the corrected binding without modifying the
immutable candidate.

## Implementation checks

- Done: candidate integrity -> every manifest member passed `sha256sum -c`; manifest `58c8a3be3f449c65b3bdd0da40f83f6856acc0906659a2cabce9ef62d3bdb6fa`, COMPLETE `425a4f9d78c303412896b50a27dab72a7395d14d9502a799fb0613d01682ec5e`.
- Done: actual parent/source delta -> accepted v4 source `aafc740beda0ef67fb3e62c37a44d56ba22351b7fa9b1962ed146c0a8b0935ee`; candidate v5 source `46bdf606c7d83263fd4ef70e0482b55d2df61dc2b074c07d0fe6c7f657956cd3`.
- Done: native frame semantics -> reviewed Kilosort `io.py` hash `767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd`, where export adds `imin` before duplicate selection and separately saves unshifted `st` as `full_st` (lines 381-402, 472-478).
- Done: fail-closed clock contract -> required clock fields and `_validated_output_clock` are checked during structural prevalidation before bound-path validation (candidate source lines 137-199).
- Done: terminal semantics -> local/global bounds, exact offset, overflow, ordering, and row ancestry are checked at candidate source lines 455-498.
- Done: isolated candidate suite -> 14 passed in 40.05 s from an independently staged candidate root.
- Done: independent H5 known-answer fixture -> exact nonzero origin passed; 11 lower/upper-bound, offset, overflow, ordering, cluster, and amplitude negatives rejected.
- Done: execution gate -> production config line 4 remains false.
- Not done: real trained execution, recording voltage, covariance, detection, training, sorting, pair launch, RF, and holdout -> outside this changed-path review and not needed to decide the clock implementation.
- Can establish: the reviewed v5 clock implementation/config is suitable for downstream integration while execution remains disabled.
- Cannot establish: runtime success on real data, scientific benefit, biological identity/purity, GPU readiness, or pair-launcher readiness.
