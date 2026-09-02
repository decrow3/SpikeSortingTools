# 0004 — Physical channel 191 is interpolated and sorted

**Status:** Adopted for this probe/session family

## Decision

Interpolate physical channel 191 and **include** it in Kilosort rather than
dropping it.

## Why

Channel 191 is a genuine bad channel, and how it is treated propagates into
whitening and spatial localization for its neighbours — dropping it is not
neutral. Interpolation keeps the geometry contiguous.

## Guardrails measured

On the imec1 full-probe rescue:

- 0.35% of assigned spikes lie within 40 µm of the repaired channel-191 depth.
- Only four KS-good templates peak in the repaired zone.

That is a small enough footprint that the repair is not manufacturing units.
Re-measure both numbers on any new probe where this policy is applied.

## Scope limit — read before reusing

Do **not** copy imec1's explicit AP191 repair mask onto imec0, or onto any other
probe, as a default. The policy that generalizes is *"identify and interpolate
genuinely bad channels, then measure the guardrails above."* The specific channel
index does not generalize; it is a property of one probe.

This is recorded as an explicit note in the frozen acceptance criteria.

## Evidence pointers

- `docs/luke_20250804_rescue_status_and_test_plan.md` § Channel 191
- `configs/rescue/imec0_legacy_acceptance_criteria.json` (notes)
- `testing/luke_bad_channel_interpolation_audit.py`,
  `testing/luke_interpolation_implementation_audit.py`
