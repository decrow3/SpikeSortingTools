# Independent H1 post-execution review: support-mask covariance/W smoke

Review date: 2026-10-01.

## Verdict

**Implementation-valid for the frozen covariance/whitening smoke.** The result
is sufficient to advance to a frozen first trained-comparison **design**, but it
is not a launch authorization and does not establish sorting efficacy.

The execution packet manifest
`07de38b2fa8d0b3be8a625d8864d856b2ed722e032f7244a5e8316c8bb86d34e`
and COMPLETE
`d44c704d5621158ed8eb0042b1ff9ffc405d639a2af032d64c2ac916140105f9`
match the delegated values; all 17 manifest members verify. The prestart packet
binds the final GO, candidate, conditional review, installed preflight, prior
NO-GO and metadata supplement. Installed config, service, candidate/helper,
launcher/tests, resolved Python and five Kilosort modules matched their frozen
hashes immediately before the one start.

## What actually ran

One retained systemd oneshot was requested, with no retry and `Restart=no`.
Kilosort 4.0.27 under Python 3.12.4 on one NVIDIA RTX A5000 opened the frozen
384-channel int16 recording crop `[208498882,226498783)`, corresponding to the
configured 600 seconds at 29,999.835983263598 Hz. It used identity full-probe
channel mapping, CAR, a 300 Hz high-pass, no Kilosort drift matrix, and the
enabled pre-whitening pointwise support mask.

The candidate temporarily replaced Kilosort's `BinaryFiltered` and
`get_whitening_matrix`. Each selected padded batch was natively filtered with W
disabled, unsupported channel/time values were set to zero, and covariance was
formed after removing 61 padding samples from each side. For each batch the
code accumulated `(batch @ batch.T) / 60000`, averaged the 12 results, validated
C, called native `whitening_local(C, ..., nrange=32)`, saved/validated W, wrote
the stop receipt, and deliberately raised `SmokeStopAfterWhitening`.

Kilosort logged that exception as an error, but the candidate caught that exact
exception, wrote `run_outcome.json`, returned the required smoke status, and the
launcher verified passing C and finite W before exiting 0. Accordingly systemd
reports `Result=success`, `ExecMainCode=CLD_EXITED`, `ExecMainStatus=0`, PID 0,
zero restarts, and terminal `active/exited` because `RemainAfterExit=yes`.
This is the reviewed success path, not an unhandled run failure.

## Independent numerical and accounting checks

H1 independently loaded the published non-voltage C and W arrays:

- C: float32 `(384,384)`, all finite, exactly symmetric, rank 384, condition
  number `215.0203399658203`, below the frozen `1e8` threshold; raw-array digest
  `155944a56cfdad66c4ddadfc6607b7eaa5b79de7f35510585d014088e4b70554`;
- W: float32 `(384,384)`, all finite; raw-array digest
  `752be427fce43138bb6a8fa043f79087d2f27a715c8544e07ab3d830b619c077`.

The W receipt binds that digest, the same contract digest
`25102362ea5a890c4662a80beceda5e00170d982d24924db4bc411b89240d9c4`
and the exact 12 batch indices. This verifies construction/binding and
finiteness, not that W improves sorting.

The schedule is exactly `0,25,...,275`. Twelve interiors contain 720,000
columns (24.0001312141065 seconds); twelve padded reads contain 721,464 columns.
At 384 channels and two bytes, the journal-derived logical read total is exactly
554,084,352 bytes. This is a logical native-read accounting identity, not an OS
physical-I/O measurement.

All 384 rows have interior covariance support. Counts are:

- 372 rows at 720,000 valid samples;
- four rows each at 678,815, 652,581 and 536,331 samples.

The interior support table therefore contains 1,169,092 unsupported values.
The executed padded-batch journal contains 1,172,020 unsupported values; the
2,928-value difference belongs to padding and must not be conflated with the
interior covariance support count. Both the planned interiors and executed
padded journal contain 14 state-transition boundaries, and all 12 journal rows
mark transition samples retained.

C and W contain 1,179,648 numeric bytes total. The complete retained run root
is 1,216,754 apparent bytes, below the 33,554,432-byte ceiling. The packet
contains the reservation, read plan, complete 12-batch journal, C/W arrays and
validations, stop/outcome receipts, native log, unit/journal state and resource
accounting. Run, covariance and whitening failure markers are absent.

## Invalid systemd condition

The service used `ConditionPathNotExists=...`, which systemd explicitly ignored
as an unknown `[Unit]` key. That defense did not operate and must not be credited.
Freshness nevertheless held for this consumed run because the immediate
external prestart check found every namespace path absent and the launcher then
claimed the exact run root with atomic `mkdir(exist_ok=False)` before native
recording access. The reservation receipt proves that claim succeeded; a race
that created the same root first would have failed closed at `mkdir`.

This combination suffices for the historical run. It does not make the invalid
unit reusable. Before any future launch, replace the key with valid negative
systemd condition syntax, use a fresh service/config hash and namespace, verify
the new manager behavior with a cheap dummy, and obtain independent review.

## Scope and next decision

This smoke establishes that the exact reviewed support-mask implementation can
reach Kilosort's native covariance boundary on the frozen 12-batch schedule,
retain every physical row and all observed transitions, produce a finite
full-rank C below the preregistered condition ceiling, construct a finite
hash-bound native W, stay within read/artifact ceilings, and stop before
training/detection.

It does **not** establish behavior on the unsampled training/detection batches,
template learning, detection, clustering, sorting completion, unit quality,
biological identity, motion-correction benefit, long-window generalization, or
superiority to an unmasked baseline. The permissive `1e8` condition threshold
is a fail-closed gross-feasibility gate, not evidence that condition 215 is
scientifically optimal.

The result is sufficient to design a first trained comparison. The cheapest
adequate design is one frozen representative-window baseline/candidate pair
with identical recording, crop, channels, environment, settings, seed and
downstream stages; only the reviewed support-mask policy differs. Because the
mask intentionally changes C and W, that comparison estimates the end-to-end
pipeline effect, not a mask-only mechanism. Freeze completion/QC endpoints,
failure rules, resource bounds, evidence capture and interpretation before
outcomes. Defer fixed-W attribution ablations unless the first pair shows a
replicable difference. Do not use RF or the sealed holdout. Repair/review the
systemd condition before launching either trained arm.

## Implementation checks

- Done: executed source/config/environment/decision chain -> exact prestart
  hashes and immutable packet lineage verified.
- Done: lifecycle and stop semantics -> one start, no retry/restart, deliberate
  caught `SmokeStopAfterWhitening`, launcher exit 0 and terminal retained unit.
- Done: schedule, masks, row support, transition exposure and accounting ->
  independently recomputed from plan/journal, with interior versus padding
  semantics separated.
- Done: C/W shape, dtype, finiteness, C symmetry/rank/condition and raw-array
  hashes -> independently recomputed from the saved arrays.
- Done: artifact completeness and prohibited stages -> complete boundary
  receipts/logs present; failure markers, training, detection, RF and holdout
  outputs absent.
- Done: invalid systemd condition -> ignored in fact; external absence check and
  atomic mkdir support this consumed run only, with mandatory future repair.
- Not done: trained sorter, baseline comparison, unsampled-batch behavior,
  biological validation or sorting efficacy -> outside the frozen smoke.
- Can establish: numerical and lifecycle feasibility of this exact masked
  covariance/W boundary on the frozen 12-batch schedule.
- Cannot establish: downstream efficacy or authorization to launch a trained
  comparison.
