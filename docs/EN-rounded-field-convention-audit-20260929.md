# EN rounded-field convention audit — 2026-09-29

**Verdict:** the executable Arm B path implements the frozen EN remap
conventions.  Its field identity, reference, rounding, time assignment, sign,
rigidity, full-probe support, and sorter settings are bound by the published
preflight packet and the accepted full-recording manifest.  Arm A remains
unverified because its exact field is not yet present; no values are inferred
for it.

## Executed environment and evidence

The published preflight packet at
`/mnt/NPX/Luke/DARTsort_motion_experiments/en_preflight_smoke_20260929_v1`
records Python 3.12.4 from `environments/rescue-production/.venv`,
SpikeInterface 0.102.1, and Kilosort 4.0.27.  Its manifest SHA-256 is
`909c9acc4f634f91314da4c03da9ba03ec83d8b216f4cdd7f15dd3e240f98c80`;
the matching `COMPLETE.json` was written last.

Arm B uses field SHA-256
`547ff39d1c91d819b758246c701603d868bfd4d50bd6b791e3d5b7140c01c71d`.
The full-session median reference is `r = 14.522145247697315 um`; applying
40-um half-away-from-zero rounding yields states `[-120, -80, -40, 0, 40] um`.
The formula is implemented at `testing/en_rounded_field.py:8-21`, and the
field is loaded and frozen before correction at
`testing/en_rounded_ks129_queue.py:127-141`.  The config states the same
reference and rounding contract at `configs/en_rounded_field_ks129.v1.json:28-38`.

## Coordinate and time conventions

- **Sign:** the scientific convention is `corrected = observed - displacement`.
  SpikeInterface obtains that correction by sampling voltage at
  `location + displacement`.  The configuration freezes both statements at
  `configs/en_rounded_field_ks129.v1.json:32`; the exact adapter states its
  lookup direction at `npx_preprocessing/motion/lattice_remap_si.py:9-11`.
  The constant `+40 um` synthetic sign fixture at
  `testing/test_en_rounded_field.py:25-28` shifts each supported target to the
  source 40 um above and zero-fills the unsupported outer sites.  It passed in
  the production preflight.
- **Time cells:** each sample belongs to the half-open cell around a supplied
  center; an exact shared boundary belongs to the later cell.  The direct
  implementation and `searchsorted(..., side="right")` are at
  `npx_preprocessing/motion/lattice_remap_si.py:319-360`.  The SI adapter uses
  explicit extrapolated edges so SI 0.102.1 reconstructs the exact supplied
  uniform-grid centers (`testing/en_rounded_field.py:24-67`).  The q=0 and
  stepwise boundary test is at `testing/test_en_rounded_field.py:31-39`.
- **Rigid Motion:** the SI construction supplies one displacement column and
  one spatial bin (`testing/en_rounded_field.py:54-60`).  Arm B already arrives
  as a rigid vector.  Arm A will be projected by the frozen median-across-depth
  rule only after its exact time-by-depth input passes validation.
- **Spatial operator and borders:** the frozen operator is nearest-neighbour
  with `border_mode="force_zeros"`
  (`testing/en_rounded_field.py:61-67`).  The production exact-coordinate
  implementation initializes zeros and copies only valid lattice mappings
  (`npx_preprocessing/motion/lattice_remap_si.py:400-425`).  For every Arm B
  rounded state it matched installed SI nearest exactly: 1,920/1,920
  target-state pairs, with byte equality on sampled voltage at all five states.

## Pipeline order and full-probe scope

The accepted parent was produced by phase correction, 500-uV bilateral
blanking, then bad-channel interpolation in that order
(`pipeline/preprocess.py:241-281`).  AP191 is the sole accepted imec0 bad
channel.  The EN queue loads that accepted parent and then applies the remap
(`testing/en_rounded_ks129_queue.py:127-142`).  It materializes the entire
recording without a channel slice (`testing/en_rounded_ks129_queue.py:188-211`):
384 channels, 314,204,894 samples, and 241,309,358,592 int16 bytes.  There is no
182-site crop in EN.

Kilosort receives thresholds 12/9, internal motion correction disabled, and
CAR enabled (`pipeline/sorting.py:23-45`).  The sorter wrapper independently
rehashes and validates the accepted recording before loading it and launching
the sorter (`pipeline/sorting.py:190-268`).  The effective Kilosort setting
`nblocks=0` is part of the frozen config
(`configs/en_rounded_field_ks129.v1.json:40-48`) and is checked against the
saved sorter parameters.

## Quantitative checks

- The q=0 adapter's complete 241,309,358,592-byte parent hash matched the
  accepted reference SHA-256
  `152f8d43a9360a95be1e9fc6bd66bce6e0fe0c091836787b37d50941f8682482`.
- Arm B contributes 10,105 zero-filled channel-seconds, or
  0.0025125286841728703 of full-probe channel-time.
- The 120-s Arm B smoke sort completed with 205 units, 59 Kilosort-good units,
  and 281,457 assigned spikes.
- The full Arm B recording was atomically accepted with binary SHA-256
  `a63d61d57d214d7a90a9698143c0fd1ea5c872a73bcac498b683a6d99efed8a2`
  and recording-content digest
  `867218a76666c0a35a4cfaaab8a3bdedcb5938217baf4cf61df407393182b289`.

Arm A's required SHA-256 is
`4c769125020aad3263bf6eaa1889cdaa7eccbf19dcf0aa3a068b185221fadb9f`.
Until that exact file arrives, its clock mapping, median reference, rounded
states, mapping equivalence, zero-fill accounting, and voltage checks remain
pending.
