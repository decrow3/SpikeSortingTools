# Independent H1 post-execution review: A four-unit capture

Review date: 2026-10-01.

## Verdict

**Technical execution: valid within the frozen bounded-capture contract.**
**Physical-signal and biological interpretation: not yet established.**

The execution packet manifest
`9714697d213368f932c0b4f8cafbc152307fffb3dd195ff6aa1bea1960c514ca`
and COMPLETE
`af4f860729c8bb901bede19d93af0a610bd4d02ab4842302c67fcad40a1b5320`
match the delegated values, and all nine manifest members verify. The executed
receipt is byte-identical to the final H1 v3 receipt at
`5268a41d4cae91cc030e8be782566d0d491d7a5e8443495820dba5b371b8d2f4`.
Prestart evidence binds the installed config, service, capture source and
validator to the reviewed hashes and records fresh production paths.

The lifecycle evidence is consistent with exactly one accepted start and no
retry: the inventory records one request/one acceptance/zero retries; the
journal has one start sequence; the terminal unit is loaded/static,
inactive/dead, PID 0, zero restarts, `Result=success` and exit status 0. The
completion marker is present and no failure marker was observed.

## Count, window and archive closure

Independent recomputation from the frozen 20-row selection receipt and the
40-row diagnostic table gives:

- 20 unique slots in each of `original` and `corrected_A`;
- eight short-pair, eight ordinary-event and four q0-control slots per
  representation, covering units 271, 278, 445 and 588;
- every table input range equals the source's half-open
  `[frame_1-512, frame_2+513)` pair rule or
  `[frame_1-512, frame_1+513)` single rule;
- 41 reads and 32,492,544 logical bytes, including the one repeated 1,054-sample
  original-pair read;
- 3,912,192 retained numeric bytes total, exactly 1,956,096 bytes per archive.

Each archive is reported as 20 `int16` arrays. From the executed source and
table, the expected member shapes are `(384, 121 + lag_samples)` for each pair
and `(384, 121)` for each single, totaling 2,547 time samples and 1,956,096
numeric bytes per representation. The archive files are each 1,961,518 bytes,
with frozen SHA-256 values `05520535...f3838` and `91b8fc7a...fb513`.

The compact packet is sufficient to review execution without transferring
voltage: identities, aggregate member count/dtype/bytes, expected shapes and
archive hashes are preserved. H1 did not independently reopen or rehash the
H5-local archives, and the packet does not enumerate each NPZ member's observed
shape. Any downstream use must first verify the local archive hashes, exact 20
slot keys and member shapes against this contract.

## Axes, clock and preprocessing semantics

Frames are global recording frames with origin zero and common sampling rate
29,999.835983263598 Hz. Every saved selection timestamp equals `frame/fs`.
Input binaries are interpreted as time-major int16 `(samples, 384)` and
transposed to channel-by-time `(384, samples)`; no depth column or registered
coordinate is used during capture. Pair and retained windows are half-open.

The 40 CSV rows are aggregate diagnostics over all 384 channels and the full
padded input windows, not unit-local waveform measurements and not summaries of
only the retained interval. Kilosort's frozen `BinaryFiltered.filter` always
subtracts each channel's temporal mean. `car` additionally subtracts the
per-timepoint across-channel median; `highpass` adds the saved Fourier-domain
kernel. `chan_map=None`, whitening and drift matrices are absent. The frozen
ops require `artifact_threshold=Infinity`, so no artifact-zeroing branch is
active; sign inversion and the high-pass kernel remain bound to the saved ops
hash. There is no top-N truncation, threshold search or rounding in the capture
source.

The repeated-read positive control compares one original, first-pair padded
window byte-for-byte and passed implicitly because the bound source reached its
completion marker. It establishes repeatable access for that exact window, not
all reads or cross-representation identity. The four q0 rows have exactly equal
aggregate original/corrected metrics at all three stages, supporting the
zero-shift control at this summary level only.

## Next decision supported

The captured evidence can support a separately frozen, H5-local waveform-QC
decision: **do the four selected units show visible, channel-local event
waveforms, and does corrected A preserve q0 controls while changing displaced
states in the direction expected from the remap?** The cheapest adequate next
step is direct plots plus simple precommitted channel-local amplitudes/shapes
from the already retained arrays, after archive key/hash/shape validation. It
needs no new recording read, sort, RF loop or holdout access. Only if that check
shows replicated usable waveforms should a representative medium-window test
be considered.

## Implementation checks

- Done: executed source/config/receipt identities -> exact frozen hashes and
  byte-identical v3 receipt.
- Done: lifecycle -> one accepted start, zero retries/restarts, successful
  terminal unit and durable completion with no failure marker.
- Done: selection, clock, half-open windows, table rows and byte arithmetic ->
  independently recomputed exact closure.
- Done: preprocessing and silent behavior -> full-channel/global-window metric
  scope, mean/CAR/Fourier filtering, no whitening/drift/artifact zeroing, and
  saved-ops dependencies traced to executed source and frozen Kilosort code.
- Done: controls -> one exact repeated original read passed by completion; four
  q0 aggregate metric rows are exactly equal across representations.
- Not done: independent H1 archive reopen/hash/member-shape inspection or any
  channel-local waveform analysis -> voltage stayed H5-local by contract.
- Can establish: one technically valid bounded capture produced hash-bound,
  byte-closed local artifacts suitable for a frozen waveform-QC analysis.
- Cannot establish: biological identity, purity, channel-local signal quality,
  motion-correction benefit, covariance validity or sorter performance.
