# Covariance/W v2 installed-chain preflight

Status: **installed and full zero-voltage preflight passed; pending final
independent launch decision; not started**.

The candidate and H1 conditional-review packets verified. Every reviewed file
was installed byte-for-byte, including the one-line fresh-root source policy,
and the static user unit was reloaded without starting it.

The installed-path launcher preflight validated the exact config and full
contract, saw CUDA and one NVIDIA RTX A5000, found readable recording metadata
with `recording_content_opened=false`, and confirmed the v2 run root absent. The
installed candidate/helper suite passed 34/34.

The unit is loaded/static but inactive/dead, PID 0, zero restarts and no
timestamps. Run root, native-results, launch-evidence and reservation remain
absent. Bounds remain 554,084,352 padded input bytes and 33,554,432 retained
bytes, with `Restart=no`, non-resumable behavior and mandatory post-W stop.

No service was started and no voltage was opened. A single final independent
launch decision is still required; this packet is not GO.

Implementation checks
- Done: every installed hash, full validator/CUDA preflight and 34-test suite ->
  exact and passing.
- Done: unit/fresh namespace -> loaded/static inactive/dead PID 0; all absent.
- Not done: service start, voltage, real C/W, training or detection.
- Can establish: installed v2 chain is ready for final independent decision.
- Cannot establish: execution, numerical covariance/W validity or benefit.
