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
| Exact arm A field | pass | Exact NPZ present at `/mnt/NPX/Luke/DARTsort_motion_experiments/en_am3_imec0_field_request_20260929_v1/payload/luke0804_imec0_two_layer_motion.npz`; verified SHA-256 `4c769125020aad3263bf6eaa1889cdaa7eccbf19dcf0aa3a068b185221fadb9f` |
| Arm A smoke and full sort | pass | `en-rounded-ks129-arm-a-20260929-v1.service` exited 0. Atomic completion and independent checks agree on 841 raw units, 352 KS-good units and 31,716,724 spikes. Native saved settings confirm thresholds 12/9, CAR on, effective `nblocks=0`, and internal correction off. |
| Raw XR Tier-1 arrays | pass | Complete packet `/mnt/NPX/Luke/DARTsort_motion_experiments/en_xr_tier1_raw_handoff_20260929_v1`; manifest SHA-256 `e241d3b02204d78a4e898ccaefd2793277778c29e1bc5e0be3dbc66722dabc91`; expected sort identity `06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555` |
| Existing XR paired comparison | complete, unfavorable | Compact result packet `/mnt/NPX/Luke/DARTsort_motion_experiments/en_xr_paired_comparison_result_20260929_v1`, manifest SHA-256 `d9a52ff18516eaeaecd7b7d20c6985e39d8c9620800d188d95321672da3e5304` |
| Frozen four-arm Tier 1 | pass, production decision unfavorable | `en-tier1-panel-20260929-v3.service` exited 0. All 28 products and all four raw arm inputs independently verify. A and B improve yield and some continuity proxies but have worse state-specific short-interval burden than REF during nonzero motion. Keep motion-off. Compact packet `/mnt/NPX/Luke/DARTsort_motion_experiments/en_tier1_rounded_field_result_20260930_v1`, manifest SHA-256 `cb885a0243891ba52f408218ec7551c4e4487554bfb7a3d39618895e1b4ffa9c`; see `EN-tier1-rounded-field-result-20260930.md`. |
| Tier 2 | not started by design | Neither rounded arm is favorable overall on Tier 1 because the frozen contamination guardrail is adverse. |
| Optional imec1 secondary arm | unavailable | The raw imec1 12/9 reference exists, but matching curation and legacy/standard QC receipts do not; see `EN-imec1-secondary-readiness-20260929.md` |
| RF and outer holdout | sealed | Published packets record no RF run and no outer-holdout access |

All three active EN jobs use persistent systemd user services with recorded
commands, memory caps, logs, and terminal receipt hooks.  Kilosort has no
within-sort checkpoint; an interruption during a full sort requires preserving
the failed evidence and restarting that sort.
