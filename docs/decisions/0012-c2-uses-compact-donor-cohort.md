# 0012 — C2 uses the compact donor cohort; pilot plateau donors are forbidden

**Status:** Adopted 2026-09-03  
**Supersedes:** the donor specification in C2 v2  
**Evidence:** [`luke_20250804_d2b2_donor_cohort.md`](../luke_20250804_d2b2_donor_cohort.md)

## Problem

C2 v2 corrected the geometry operator but still named T01/T04/T06 from the
original pilot. D2b-2 showed those waveforms are not compact neuron footprints:
T04/T06 are broad common-mode/LFP plateaus and T01 is near the noise floor.
Their weak spatial gradients make motion and interpolation artificially easy.
Correct geometry applied to the wrong stimulus would still give an
uninterpretable drift penalty.

## Decision

C2 v3 injects all 14 sealed D2b-2 real compact donors unchanged. It records
amplitude band and polarity and tests both rescue and `legacy_style` by default.
A donor enters the primary moving-minus-static comparison only when its static
accuracy is at least 0.8 under both configurations. Failures remain reported
but cannot be interpreted as drift penalties.

The donor archive and manifest hashes are frozen in the prespec. Pilot `T*`
identifiers are rejected by code. C2 v2 and all results derived from its pilot
donors remain historical and retracted.

Because the donors originate on imec0 and are injected into an imec1 strip,
placement must also preserve the donor crop's relative four-column x/y geometry.
The source geometry is hash-frozen; the code chooses the nearest central target
site with an exactly translated relative geometry and fails if none exists.

Downstream non-rigid and stitching evaluations consume only C2 v3 outputs.
D2b-1 remains blocked until C2 v3 and its compact-donor oracle evaluation are
complete, at which point a D-donor focus can be frozen without reusing the old
post-hoc T-donor list.
