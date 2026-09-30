# EN execution evidence ledger — 2026-09-29

This ledger maps the frozen EN design to authoritative evidence while the two
new full-session arms are in flight.  `pending` means the requirement is not
yet satisfied; it is not inferred from a running service or a planned action.

| Requirement | Status | Authoritative evidence |
|---|---|---|
| Pinned imec0 parent and q=0 identity | pass | `testing/outputs/en_q0_full_hash_v3/RECEIPT.json`: 241,309,358,592 bytes read; actual and expected SHA-256 `152f8d43a9360a95be1e9fc6bd66bce6e0fe0c091836787b37d50941f8682482` |
| Production environment | pass | Published preflight records `/environments/rescue-production/.venv/bin/python`, Python 3.12.4, SpikeInterface 0.102.1, and Kilosort 4.0.27 |
| AP191 is imec0's sole accepted bad channel | pass | Published `PREFLIGHT.json`, `checks.bad_channel_is_imec0_ap191=true` |
| Sign and temporal convention | pass | Published preflight binds the +40 µm one-channel fixture and SI half-open temporal cells with explicit extrapolated edges |
| Rigid rounded remap on all 384 channels | pass for B; pending for A | B: 1,920 installed-SI mapping pairs with zero failures and byte-equal fast-adapter checks on every rounded state. A awaits the exact field. |
| Arm B field identity | pass | SHA-256 `547ff39d1c91d819b758246c701603d868bfd4d50bd6b791e3d5b7140c01c71d`; median reference 14.522145247697315 µm; rounded states -120, -80, -40, 0, 40 µm |
| Arm B zero-fill accounting | pass | 10,105 channel-seconds, 0.0025125286841728703 of full-probe channel-time; `PREFLIGHT.json` is bound by published packet manifest SHA-256 `909c9acc4f634f91314da4c03da9ba03ec83d8b216f4cdd7f15dd3e240f98c80` and matching `COMPLETE.json` |
| Arm B smoke | pass | 205 units, 59 KS-good, 281,457 spikes; saved settings validate no internal correction and CAR enabled |
| Arm B full recording | pass | Atomic accepted manifest: request digest `c877d6697c675124df3f36cb2b3aeebb0542e568e44d27d08b73dbe3962dd7b8`; binary SHA-256 `a63d61d57d214d7a90a9698143c0fd1ea5c872a73bcac498b683a6d99efed8a2`; recording-content digest `867218a76666c0a35a4cfaaab8a3bdedcb5938217baf4cf61df407393182b289` |
| Arm B full KS 12/9 sort | pass | Persistent service `en-rounded-ks129-arm-b-20260929-v2.service` exited 0. Atomic completion and independent checks agree on 812 raw units, 332 KS-good units and 31,052,960 spikes. Native `ops.npy` confirms thresholds 12/9, CAR on, effective `nblocks=0`, and `dshift=None`. Compact packet `/mnt/NPX/Luke/DARTsort_motion_experiments/en_arm_b_full_sort_result_20260930_v1`, manifest SHA-256 `9faeba59801dd5c7bf44b7ed85c65ed6d99f9a8ecb80b73de6c15ea501afee54`; see `EN-arm-b-full-sort-result-20260930.md`. |
| Exact arm A field | pending | Requested path `/mnt/NPX/Luke/DARTsort_motion_experiments/en_am3_imec0_field_request_20260929_v1/payload/luke0804_imec0_two_layer_motion.npz` is absent; required SHA-256 `4c769125020aad3263bf6eaa1889cdaa7eccbf19dcf0aa3a068b185221fadb9f` |
| Arm A smoke and full sort | pending, persistent waiter live | `en-rounded-ks129-arm-a-20260929-v1.service` waits for the exact field before hashing, mapping checks, smoke, or full execution |
| Raw XR Tier-1 arrays | pending | Destination `/mnt/NPX/Luke/DARTsort_motion_experiments/en_xr_tier1_raw_handoff_20260929_v1` is absent. The curated handoff is provably insufficient by 1,517,602 raw events. |
| Existing XR paired comparison | complete, unfavorable | Compact result packet `/mnt/NPX/Luke/DARTsort_motion_experiments/en_xr_paired_comparison_result_20260929_v1`, manifest SHA-256 `d9a52ff18516eaeaecd7b7d20c6985e39d8c9620800d188d95321672da3e5304` |
| Frozen four-arm Tier 1 | pending, persistent waiter live | `en-tier1-panel-20260929-v3.service`; REF and B ready, XR and A pending. The evaluator reports segment-dependent exact-duplicate fractions and CIs separately for every field/arm pairing and uses common REF-unit resampling for correspondence intervals across candidates. Focused synthetic and queue tests pass 9/9; the full-scale REF preflight remains complete. |
| Tier 2 | not started by design | Runs only after profile review finds an arm favorable overall on complete Tier 1; no composite score or automatic rank is allowed |
| Optional imec1 secondary arm | unavailable | The raw imec1 12/9 reference exists, but matching curation and legacy/standard QC receipts do not; see `EN-imec1-secondary-readiness-20260929.md` |
| RF and outer holdout | sealed | Published packets record no RF run and no outer-holdout access |

All three active EN jobs use persistent systemd user services with recorded
commands, memory caps, logs, and terminal receipt hooks.  Kilosort has no
within-sort checkpoint; an interruption during a full sort requires preserving
the failed evidence and restarting that sort.
