# EM.2b stage-0 measurement and stage-1 readiness — 2026-09-29

**Verdict:** the rounded exact-DD W2 sorting and sorting-only QC products are
complete and usable as the frozen lattice arm. The run produced 477 assigned
units and 603,125 accepted events. The production launcher did not publish its
top-level success receipt because DARTsort changed the mtime of the linked
1.10-GB detection HDF5 during sort reuse. Every scientific stage completed
with exit status 0, and the saved sorting used by QC is byte-identified by
SHA-256 `85f1537a4ffbac430d36e84b1f71e7ed5c0760287338115724ee9d80d052e61a`.
The failure is retained rather than rewritten or retried.

The bounded audit is
`testing/outputs/em2b_w2_rounded_exact_dd_measurement_20260929/AUDIT.json`;
its packet manifest SHA-256 is
`ff267fc3adf7ecd83e33fa3ca9305d5c7871a60ec5ee69a0aa96e0a7f4db15d9`.
It verifies all stage receipts, every recorded artifact size, every artifact
hash that the runner recorded, the sorting hash in the independent QC receipt,
and the absence of RF evaluation. Its one artifact difference is the mtime of
`detection/subtraction.h5`: bytes remain 1,103,987,288 and the current SHA-256
is `19558088485e6788b5ecdbaff821cbe958fc8ef888722a5f7ae989342d1dc4b6`.
The original detection receipt did not hash this large file, so the audit does
not claim byte identity between its pre-sort and post-sort states. That limit
does not alter the identity of the saved sorting used by the scorecard.

The failure mechanism is concrete. The runner asked DARTsort to reuse detection
with `link_from`; DARTsort symlinked `subtraction.h5` into the sort directory,
and its persistent-feature path opens a parent HDF5 in `r+` mode. The runner's
final inventory correctly noticed the mtime change and withheld `receipt.json`.
Later-arm configs set `network_sort_final_markers=true` and
`avoid_output_symlinks=true`. This makes the runner copy detection inputs into
the sort directory before DARTsort opens them, preserving the detection-stage
artifact and avoiding this publication failure.

The systemd service ran for 1,066.229 seconds of wall time and consumed
2,556.378 CPU seconds. Preprocess took 18.204 seconds, detection 502.094,
sort 527.437, and sorting-only QC 8.764. Observed scratch peaked at 10.377 GiB
and finished at 9.281 GiB. Kernel I/O byte accounting was unavailable; the
persisted recording cache is 7,425,540,304 bytes. The service remained below
the original 10,800-second and 15-GiB stage-0 limits.

The measured result freezes later-arm limits at 3,600 seconds wall time and
13 GiB scratch per arm, with one GPU and four CPU jobs. Each prepared config
allocates 1,200 seconds to cache materialization, 720 to detection, 60 to the
static-motion marker, 1,080 to sort, and 240 to QC, for 3,300 seconds of stage
budgets. A timeout or scratch-cap breach remains a preserved failure; it does
not authorize a larger retry. The revised frozen scorecard SHA-256 is
`714fd29c41f9c4aa6142d534232f030438af0819133613177ca404f4331642c6`.

The two stage-1 recording descriptors implement the production order: accepted
383-channel preprocessing with AP191 absent, AP191 interpolation, SI kriging
motion remap, then crop to the 182 targets. Both reload under the preparation
environment (SpikeInterface 0.104.7) and DARTsort's runtime (0.104.8) as
10,199,918-frame, 182-channel float32 recordings. Their hashes are:

| Arm | Descriptor SHA-256 | Config SHA-256 |
|---|---|---|
| unrounded kriging | `df46afa998ee15f9901c6e84a8bf3451d88a9923fc05a939a0f9547401fab7ac` | `212983e94271af9ed70ef4e4f8748db17ef225c2c1e262018e387b8ceac21582` |
| rounded kriging bridge | `66c9c3a6479c23b0e0a44b8b5cdd714cf7ddc162ddfd522291e2f62fee979c9b` | `0464e246e015c3ec36ff9dd0bfd1f89774e47ef8c2798aea332f1ce33d4b514a` |

The SpikeInterface file implementing the spatial kernels has identical SHA-256
`a3e7e03c7580915d0b64317a623fe33f1d3fe3ec68968ccfbfa9af7491121fee`
in 0.104.7 and 0.104.8. Direct kernel evaluation is byte-identical across the
versions for all eight rounded W2 states and the 44 frozen unrounded screen
states, including the W2 extrema. This audit reads no voltage and is saved as
`testing/inputs/em2b_w2/kernel_version_audit.json`.

Both configs pass the production runner's validation and a persistent systemd
runtime preflight after launcher disconnection. The preflight uses the exact
future service environment, loads the custom descriptor, checks CUDA, reads no
voltage, and launches no sort. The production runner normally limits
`PYTHONPATH` to DARTsort, so `testing/em2b_launch.py` explicitly adds this
repository and records the extractor source hash; its SHA-256 is
`3d97b099021f74a88fa98bd94956c8e1561a55548df3b03fbd57b4f9c613d739`.
The preflight packet manifest hashes are
`d1549335f8b990ec7fb41f8e8e2f196f3a6349fff3511ef11ac945f223e79312`
for unrounded kriging and
`b73190f0da357c075f2c68ccb4f30f061db8e3e3300779e7b741d91f35e6d5f6`
for the rounded bridge. Both imported the extractor with SHA-256
`b7b3ed0c3d092fb351c224e9b8d0064c34779bdd21f63995a598039216e6c0a2`.
The production runner also resolved each complete recording dependency set
without voltage access: 18 files per arm, with fingerprint object hashes
`37822221266e696e40c28b739be65ab8bf0c8507ba3ce6f641337478721d90f9`
and `d8daa8fd9bf4ac1bd3661667633d61574558857f30fb961edc18af36642edb1c`.
The compact receipt is
`testing/inputs/em2b_w2/runtime_input_fingerprints.json` (SHA-256
`3f96ce54dde9d62df0b769b24594894555c09fcd81d0f3a41fe9b01baabcef51`).

No RF input or outer holdout was accessed. The two stage-1 sorts have not been
launched. They require a separate authorization after this report, as frozen
in the EM plan.
