# H1 waveform-method and REF-repeat launcher independent review

Overall verdict: `TARGETED_LAUNCHER_REPAIR_REQUIRED`. No scientific execution
GO is issued.

The actual-H5 waveform method receives `GO_METHOD_FREEZE_ONLY`. Its 89-object
runtime closure and numerical probe match, the separate 241,309,358,592-byte
full-voltage read exactly reproduced SHA256 `152f8d43...`, the compact future
preflight reads zero voltage bytes, and prospective arm/anchor bindings remain
disabled. Clock-v5 retains `GO_CLOCK_REPAIR_ONLY`.

The launcher core is coherent: REF uses the historical no-mask entry, repaired
B uses the support-aware clock-v5 trained entry, both share crop and output
clock, REF is always first, only completion state is read between arms, terminal
row/clock ancestry is checked, namespaces are fresh, and every current gate is
disabled.

Execution GO is blocked by three implementation defects:

1. No durable one-start claim exists, and preflight runs before any namespace or
   failure receipt. A no-data fixture reached preflight twice after the same
   injected failure.
2. The declared 16 GiB aggregate output cap is not enforced by source or unit.
3. The accepted full-voltage receipt/lstat continuity check is declared but not
   performed by the pair launcher before REF voltage access.

The exact repair is bounded to those three launcher lifecycle paths. Do not
enable the contracts, install the proposed service, or start scientific work
before a fresh independent changed-path review.

## Implementation checks

- Done: verified 60 manifest members across waveform, voltage, clock, and
  launcher packets and every delegated identity.
- Done: inspected executed entry paths, intended arm differences, clocks,
  ancestry, serial ordering, between-arm field access, namespaces, resources,
  service behavior, and fail-closed gates.
- Done: reran 6 waveform and 9 launcher tests, validate-only, and the independent
  pre-root retry fixture.
- Not done: no enablement, service installation, scientific execution,
  recording/voltage read, evaluator, RF, or holdout work.
- Can establish: waveform method freeze is acceptable for actual H5 and the
  launcher core scientific routing is coherent.
- Cannot establish: reliable one-shot bounded launch or any scientific result
  until blockers R1-R3 are repaired and independently reviewed.
