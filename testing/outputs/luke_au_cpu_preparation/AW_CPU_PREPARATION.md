# AW CPU preparation for AU

Status: **CPU fixtures pass; waiting for the staged AV bundle. Not worker-ready.**

This preparation performs no sorting, raw-voltage read, GPU work, new motion
fit, or continuous-trajectory injection. It is safe to run beside AM.3 with
`CUDA_VISIBLE_DEVICES=''`, at most two numerical threads, one input reader and
low process priority. No active AM.3 source or output is an input.

## Frozen preparation

| Item | Frozen value |
|---|---|
| Evaluation interval | imec1 W2, 900--1240 s (340 s) |
| Donor target | 30 accepted static DARTsort S units |
| Train | regular 5 Hz, 1 s guard at both ends, 1,690 events |
| Train array SHA-256 | `17e3cd15fb4965d973dc7d328deda5d0c3525387e17b207f4aefb770c9a13314` |
| Motion states | exact same-column multiples of 40 µm only |
| Matching tolerance | inclusive +/-0.4 ms = 12 samples at 30 kHz |
| Qualification | every donor/state: energy retention >0.99, round-trip PTP ratio 1.00 +/-0.02, centred cosine >=0.99 |
| Resources | CUDA disabled; <=2 CPU threads; one reader; nice 10; working-set target <=8 GB |
| Persistent-output ceiling | <=5 GB; current preparation is below 0.1 MB |
| Disk guard | stop below 30 GB free on the output filesystem |

The donor rule is frozen before any injected outcome is visible. Eligibility
requires the accepted static-bank label `good`, refractory-violation fraction
<=0.005, rest-spike fraction >=0.80 and isolation score >=0.80. Eligible units
are divided into six equal depth strata. Within each stratum they rank by larger
static PTP, then smaller refractory fraction, then unit id; five are selected.
Unfilled quota slots are filled globally by that same ranking. Fewer than 30
eligible units is a hard gate and does not relax the thresholds.

## Motion-state semantics

DARTsort does not consume a distinct displacement for every spike during
matching. In the inspected checkout (`edcfe1b51d672b4136eb13cc78c0875da804b851`),
`peel/matching.py:260-269` computes
`chunk_start_samples + chunk_length_samples // 2`, converts that sample to
seconds, and asks the template bank for its state at that time.
`templates/template_util.py:62-78` then evaluates the external field and turns
the result into pitch shifts used to select static-channel template support.

AW's *injected trajectory* samples the exact D2L two-layer v1 field at those
actual chunk centres, then rounds half away from zero to verified 40 µm
same-column states. It will report quantization error separately for rest and
canonical episode time. The injection operator will not interpolate donor
voltage between states and will not use AI-v2 or the imec0 AM.3 field.

That does **not** describe all matching implementations. In the inspected
DARTsort checkout, the default `drifty` matcher constructs a
`FromFullProbeInterpolator` whenever motion is active
(`peel/matching_util/drifty.py:105-123`). At each chunk centre it evaluates the
unrounded external displacement and spatially kernel-interpolates the registered
template basis onto `geom - displacement`
(`util/interpolation_util.py:1264-1292`). The external rigid field itself is
linearly interpolated in time by `dredge.motion_util.RigidMotionEstimate`.
`MotionInfo.pitch_shifts` separately rounds displacement/pitch to an integer
(`util/motion.py:329-380`), and `templates_at_time` uses those integers for
static-channel support selection (`templates/template_util.py:62-80`), but that
is not evidence that a `drifty` run omitted fractional spatial interpolation.
The exact S/D2L matching config from AV decides which path was used.

The remapper preserves x-column identity and requires an exact site at
`(x, y + state_um)`. Missing sites are explicit; ambiguous or many-to-one maps
fail. The qualification applies the forward map and its exact inverse. No donor
or state may be silently clipped, attenuated, time-shifted, rescaled or dropped.

## Corrected scorer and fixtures

The scorer follows decision 0014: exclusive interval-order matching is run
between one truth train and one output cluster, not between truth and the pooled
spike river. Candidate-cluster competition is downstream of that match. The AU
fixture additionally freezes the requested +/-0.4 ms boundary: offsets of 12
samples match and offsets of 13 do not. One output event cannot satisfy two
truth events. Nine focused AW CPU regressions and 35 combined
AW/injected-truth/scorer regressions pass.

The August injected-truth adapter is used only for its validated float32,
no-clipping/no-truncation, immutable-template contract. C2-v4 supplies the
corrected per-cluster scorer and exact-lattice operator controls. The old C2
5/11/22 µm fractional arms are not reused because their forward model attenuated
rather than faithfully translated compact donors.

## AV handoff dependencies

The following are required before a worker manifest or HDF5 can be called
ready. There is no local substitute:

1. The accepted static DARTsort **S/W2 900--1240 s** template bank and unit
   metadata, including the preprocessing identity, channel ids/geometry,
   sampling frequency, template time origin/support, quality fields used by the
   frozen donor rule, and authoritative hashes.
2. The exact **two-layer v1 field used by D2L**, with authoritative SHA-256,
   time grid, displacement array, sign convention and canonical episode mask.
3. The frozen DARTsort matching settings/config and code identity used by S and
   D2L, especially the matching chunk length and recording time mapping.

AV's 13:36 PDT update identifies the only saved S bank as a shallow crop:
654 x 121 x 182, AP202--AP383, SHA-256
`a99b12c3075f038fbad8c05c36c96f63221fd0eac5ba71caee8f18ff97acb75c`;
the final sorting hash is
`be6106ed0cb0f99759fd23629dd3bcb5f50657dbffa238ad3721c15d1b21fba7`.
The h5-local source and AV manifest are not mounted on this host. A compact
shared-path bundle and authoritative D2L-v1 provenance have been requested.

### Crop feasibility decision

The crop can be screened for an *interior, crop-specific* cohort. A necessary
check requires >99% of each donor's **observed crop energy** to lie at least
`max_abs_state + interpolation_radius` from both crop edges, followed by all
per-state exact-remap qualifications. The preparation now implements that check
and labels its denominator explicitly as observed support.

It cannot establish the frozen full-probe >99% energy requirement: voltage on
AP0--AP201 is absent, so its energy is unknown rather than zero. Therefore the
current strict full-probe donor design requires new full-probe donor extraction
or an equivalent saved full-probe waveform/template source. A restricted
interior cohort is feasible only if the scientific scope is explicitly narrowed
to reproducing this shallow crop's deployed matcher; AW does not make that scope
change on its own.

A local file named `luke0804_imec1_two_layer_motion.npz` has SHA-256
`85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9`, but it is
recorded only as a candidate locator. It is not accepted as the D2L authority
until AV confirms the same hash.

## Stage dependency order

1. Validate every staged AV file/hash and preprocessing/channel identity.
2. Run the frozen donor selection and save all eligible, selected and excluded
   rows with reasons.
3. Sample and quantize D2L at the real matching chunk centres; report rest and
   episode state/error distributions.
4. Qualify every selected donor in every occupied state. All rows must pass.
5. Build and hash the immutable 5 Hz truth contracts and only then render the
   small worker assets. A `worker_ready` assertion remains false until steps
   1--4 pass.

The historical C2-v4 output archive is about 82 GB and is intentionally not
copied. Its small prespec/results and source hashes are sufficient for this
preparation; AU must use the AV bank, not the 14 compact C2 donors.
