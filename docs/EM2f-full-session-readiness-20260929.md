# EM.2f full-session rounded-kriging readiness

**Verdict:** the selected rounded-kriging pipeline is prepared for a persistent
full-session imec1 sort. The descriptor preserves the accepted preprocessing
graph, covers all 314,204,094 frames (10,473.554 s), and uses the frozen
full-session 40 µm lattice. No voltage was read and no sort was launched while
preparing this packet.

## Frozen implementation

The recording descriptor is
`testing/inputs/em2f_full/rounded_kriging_recording.json` (SHA-256
`c4ce9cc969324d61035de9d3bc8d01d9377a7ffb674c3b0c43051d30bf72b0cf`).
Its preparation receipt is `testing/inputs/em2f_full/PREPARATION.json`
(SHA-256 `4ff1ab086f4977270cd490750874f2de73088f89d28ee81f343d7101131fe59a`).
The preparation removes only the W2 `FrameSliceRecording` (frames 26,999,783
through 37,199,701) from the serialized accepted source graph. Every surrounding
operation remains in the same order: high-pass, phase shift, AP191 exclusion,
reference, scale, reference, and float32 conversion. The selected adapter then
interpolates AP191, applies rounded kriging, and crops to the 182 target sites.

The copied full-session lattice is byte-identical to its source (SHA-256
`92d7a28ccc808f76924a6ba392eb71c1a9b0b867796615a15307d1c59bc7df4f`).
It contains 41,895 quarter-second knots and states from -280 through +80 µm.
The +80 µm state occurs only in the full-session table. Its effective kernel is
byte-identical under SpikeInterface 0.104.7 and 0.104.8; the audit is
`testing/outputs/em2f_full_kernel_version_20260929/RESULT.json` (SHA-256
`24d52daaa4294d8b0f4144d90fca6b8bbb1cf41cecef8df0c7f45170f100f28c`).

## Resources and recovery

The W3 run measured 9.2 GiB final scratch and about 19.7 minutes wall time for
340 s. Linear scaling projects about 284 GiB and 11 hours for the full session.
The frozen contract requires at least 734 GiB free before launch and preserves
450 GiB free during the run. Stage budgets total 16 hours plus launcher margin.

The preprocessing cache is resumable at a durable completed prefix of aligned
one-second chunks. Completed later stages may be reused only after receipt
validation. Detection and sorting have no within-stage checkpoint; interruption
during either stage requires preserving failure evidence and restarting that
stage. The job must run under the independent user systemd manager.

## Lighthouse evidence boundary

The six frozen imec1 lighthouse source files are absent from their canonical
paths, all searched local data roots, the retained 67-member archive, and Git
objects. Their expected hashes remain recorded, but hashes cannot reconstruct
the data. The availability result is
`testing/outputs/luke_imec1_lighthouse_input_availability_20260929/AUDIT.json`
(SHA-256 `43d5eb6c5b814c4c5f94e336ca277b783304ade4eecce102c006938ef7b2f60a`).

This gap does not invalidate the W2/W3 scorecards or event-overlap results, but
those results do not establish biological identity. The initial full-session
analysis therefore keeps lighthouse identity explicitly pending and does not
replace it with cached proxy metrics. RF remains outside this run.

## Validation

The production runner accepted
`configs/em2f_full_rounded_kriging.v1.json`. Nine focused tests passed for the
descriptor, cross-version kernel, resource cap, lighthouse audit, and selected
pipeline contract.
