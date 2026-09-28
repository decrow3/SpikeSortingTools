# CX masked-donor qualification scope

Frozen before inspecting modified-donor held-out outcomes. CW remains the active
priority; CX begins only after the corrected CW packet is closed.

## Question

Can one source-coordinate support/taper, learned only from saved training-side
reliability and physical geometry, remove unstable donor energy while retaining
enough stable waveform signal to turn the existing 5/90 donor qualification
failure into a finite, explicitly narrower modified donor set?

This is a donor-qualification check, not a sorter comparison or a claim of
pristine validation. One bounded, versioned qualification is allowed.

## Frozen construction

1. Reuse the existing 90 donor arrays, split halves, counts, physical geometry,
   and BC lineage. Preserve the original 5/90 result unchanged.
2. Fit one smooth support/taper in donor coordinates from training-side channel
   reliability and geometry only. It travels with the source waveform; it is
   never tied to an injection location, sorter outcome, or held-out score.
3. Use a held-out spike half when it exists. If the existing arrays reuse halves
   or a shared basis, label the result exploratory and report swapped-half
   stability; do not call it independent validation.
4. Evaluate lattice shifts first on the exact physical 182-channel geometry,
   including all occupied positions, time phase, sign, boundaries, and native-
   clone analytic controls. Fractional forward modelling remains a separate,
   unqualified question because of the prior dimming behavior.
5. Apply the 99% retained-energy tests to the modified source, not retroactively
   to donors that failed under the original source.
6. Predeclare stress cases from known injected identities: similar distinct
   sources including real overlaps, and an isolated strong source with controlled
   morphology/amplitude mismatch. Include zero and unmodified-background controls.
7. Audit corrected truth assignment for pooled-background stealing; unmatched
   background events are not automatically false positives.

## Required outputs

For all 90 donors report stable signal removed, unstable energy, PTP, shape,
support changes, original-source qualification, and modified-source
qualification. Return a finite qualified modified set or a narrower engineering
stress set plus a concrete benchmark plan. Do not launch a sorter here.

If inputs are missing, stop with exact paths, hashes, access dependency, and a
finite generator/truth-assignment code audit rather than another broad inventory.

## Limits

1,200 CPU seconds of all work, two threads, one reader, 20 GB RAM, zero GPU,
at most 256 MiB saved-donor source reads, 2 GB scratch, 500 MB final output, and
at least 30 GB free. H1 cumulative before CW/CX is 16,343.22 seconds; charges are
appended without transferring H5 work. The prior 19,300-second ceiling is
extended by 1,200 seconds to **20,500 seconds** before any work could exceed the
prior allowance.

`DARTsort-method-variants-20260928.md` was not present on the H1 filesystem at
scope freeze time; the coordinator-provided adopted priority summary governs
this bounded step unless the exact document becomes available.
