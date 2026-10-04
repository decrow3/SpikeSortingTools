# Final H1 launch decision: A four-unit physical-signal capture v3

Decision date: 2026-10-01.

## Verdict

**GO for H5 to copy this packet's `H1_GO.v3.json` byte-for-byte to the exact
live receipt path and request exactly one start of the already installed
reviewed service.** The live path is:

`/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/testing/outputs/en_a_four_unit_physical_signal_capture_20261001_approved_v2_launch/H1_GO.v3.json`

The installed-preflight packet manifest
`a88e610a937d1b0c99007513d06b93df37f428514c257f0b18776fd6332a95a8`
and COMPLETE
`891bc751cb1b0b7a6715227e4aa8c1fc0391e15c143aa14dfd8fa59c2a76c520`
verify. Installed config, service, capture source and validator hashes equal the
frozen bindings. The content gate validated the exact receipt schema,
artifacts and child argv with zero child invocations. The capture preflight
reported zero recording opens, reads and bytes and preserved the 41-read,
32,492,544-logical-byte, 3,912,192-retained-byte plan.

The reviewed service is loaded/static but inactive/dead with PID 0, zero
restarts and no start/exit timestamps. Production output, log, live receipt,
counters, failure and completion paths were absent at the saved preflight.
Resources remain 4 CPUs, 2 GiB RAM, no swap and 300 seconds with `Restart=no`.

`H1_GO.v3.json` is byte-identical to the successfully validated non-live
receipt at SHA-256
`5268a41d4cae91cc030e8be782566d0d491d7a5e8443495820dba5b371b8d2f4`.
Before copying it, H5 must make one immediate no-mutation check that the unit is
still inactive and all fresh production paths remain absent. It may then copy
the receipt and submit exactly one start request. No automatic or manual retry
is authorized. A tool refusal or runtime failure consumes this decision; retain
all evidence and require a new namespace and review for another attempt.

This packet did not install, copy the live receipt, start the service, open a
recording or read voltage.

## Implementation checks

- Done: installed-preflight manifest and COMPLETE -> exact expected hashes; all
  five manifest entries verify.
- Done: installed artifact hashes and exact v3 receipt/argv binding -> exact
  match to the conditional-review plan; validator-format receipt independently
  compared field-for-field.
- Done: content-gate and capture preflights -> zero child invocations, opens,
  reads and bytes; fixed arithmetic and resource limits preserved.
- Done: saved unit/fresh-path state -> loaded/static inactive/dead PID 0,
  zero restarts/timestamps; all production paths absent.
- Not done: final immediate H5 no-mutation check, live receipt copy, one start,
  voltage capture or result interpretation -> assigned to H5 after this GO.
- Can establish: every frozen prerequisite for releasing the exact one-start
  receipt was satisfied by the installed-preflight evidence.
- Cannot establish: state changes after the saved preflight, successful capture,
  duplicate-control outcome or physical-signal interpretation.
