# Trained-entry global/crop clock repair candidate v5 (H1)

Status: `READY_FOR_H5_CHANGED_PATH_REVIEW`; execution remains disabled.

This packet narrowly repairs the accepted v4 trained terminal validator. Native
Kilosort saves `full_st[:,0]` in crop-local samples but exports
`spike_times.npy = full_st[kept_spikes,0] + crop.imin` in global acquisition
samples. REF384 (1,856,371 retained rows) and B384 (1,854,340 retained rows)
both show the exact offset 208,498,882 for every retained row
(`bindings/CLOCK_ORIGIN_KNOWN_ANSWER.json`, sha256
`f9e69cec194295002e3cdbedd5b9fb26f64dba9f8f1ebeb7888296bdbf8876f5`).

The repair adds a config-bound output-clock contract, exact crop/global
half-open bounds, signed-int64 checks, and the correct ancestry relation while
retaining accepted v4 uniqueness, strict native row ordering, cluster and
amplitude ancestry, and nondecreasing-time checks. The real production config
is bound to crop `[208498882, 226498783)` and remains
`execution_enabled: false`.

Validation completed without recording access, voltage processing, detection,
training, sorting, RF, or holdout access:

- isolated focused suite: 14 passed;
- vertical orchestration fixture: success completion plus all prior failure and
  ancestry/order controls passed;
- saved-array clock fixture: zero origin and exact 208,498,882 origin passed;
  wrong offset, off-by-one, local upper bound, and signed-int64 overflow were
  rejected.

No other defect in the trained-entry terminal validator was found in this
bounded repair. That does not make the pair experiment executable. The missing
reviewed heterogeneous/no-mask `REF384_repeat` pair launcher remains a separate
open blocker and is intentionally not implemented here. GPU availability, the
unmounted coordinator strategy, and the downstream evaluator scientific
decision also remain as recorded in `bindings/PREPARATION_READINESS.json`.

Accepted parent bindings:

- v4 candidate manifest: `91bcf5ed0560bf5617815e4929a3d97f82b386f0040ef31dc6a61991a45fd36c`
- v4 candidate COMPLETE: `9ab3e67a41af42b65cbf56509fc64fc407a39c79496a2391852de39aeb1055df`
- H1 v4 review manifest: `9ff42c9e56251e2659153de194ade4dce986aadcf24a0e419cbb01dd0d16804a`
- H1 v4 review COMPLETE: `54af269b287b3cfa06bcf6dc7f28fc5bd9ac3188cbf845baa8a331a37acaf34f`

## Implementation checks

- Done: traced actual Kilosort clock semantics -> `io.py:381` adds `imin` to
  exported spike times while `io.py:472-478` saves crop-local `full_st`; bound
  Kilosort `io.py` sha256 is
  `767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd`.
- Done: checked saved REF384/B384 arrays -> every retained row has exact global
  minus crop-local offset 208,498,882 (bound known-answer artifact above).
- Done: inspected and tested changed source -> output-clock contract is checked
  before bound data access (`source/kilosort_support_mask_trained.py:137-199`),
  and terminal ancestry/bounds/overflow are enforced at lines 422-498.
- Done: preserved accepted row semantics -> unique, strictly increasing native
  `kept_spikes`, exact cluster/amplitude ancestry, and both local/global
  nondecreasing clocks are covered by the focused and vertical fixtures.
- Not done: independent H5 review of this changed path -> requested by
  `REVIEW_REQUEST.json`; no launch GO is asserted.
- Not done: real recording or real trained output execution -> deliberately
  outside this repair packet and prohibited by its frozen execution-disabled
  contract.
- Can establish: the saved-array terminal validator now accepts the documented
  Kilosort global-vs-crop clock relation and rejects the bounded clock/ancestry
  counterexamples tested here.
- Cannot establish: GPU readiness, pair-launch orchestration, real-run success,
  scientific benefit, biological identity/purity, or evaluator advancement.

