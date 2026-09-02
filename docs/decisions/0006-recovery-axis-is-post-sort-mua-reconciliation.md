# 0006 — The recovery axis is post-sort MUA family reconciliation

**Status:** Adopted as the direction of work. No promotion is currently authorized.

## The finding

Lost neurons are mostly **not** lost at detection. Of 128 reused, manually
reviewed neural events, the repaired full-probe no-motion KS4 sort contains
**119 (93.0%)** in its final output. Exactly one reviewed event appears in the
learned event table but is absent from the final table.

The large transition is the **unit-quality boundary**:

- 61/128 (47.7%) have a nearby event belonging to a KS4 `good` unit
- 58/128 (45.3%) are present only under an `mua` label

The independent sealed automatic holdout points the same way (307/432 = 71.1% in
the final table; 145/432 = 33.6% near a good unit; 162/432 = 37.5% final-MUA-only),
though it is raw-extrema derived and is not neural ground truth.

This is event-level, and several events can map to one unit — so it does **not**
prove 58 neurons were lost. It localizes where intervention is worth attempting.

## Decision

> Review and reconcile supported MUA unit families after sorting, without
> changing voltage preprocessing, detection, or the accepted KS4 event table.

This axis is reversible and operates only on spikes KS4 already accepted.

Any family link must be reversible and must require: coherent raw-voltage
templates across fragments; complementary rather than duplicate temporal support;
acceptable refractory burden after union; improved event-centred residual energy;
and preservation of already-good neighbouring units.

## Explicitly not supported by the evidence

- Globally lowering detection thresholds
- Relaxing duplicate removal
- Promoting MUA units wholesale
- Automatically merging KIAsort-nominated fragments (see [0005](0005-dartsort-kiasort-deferred.md))

Each would create new events or identities where the evidence only supports
targeted reconciliation.

## The rescue configuration is already doing this automatically

Recorded 2026-09-02. On Luke0804 imec0, 80 of the 200 KS-good units that rescue
has and legacy does not are legacy `mua` clusters relabelled good — roughly two
thirds of each unit's spikes coming from a single legacy MUA cluster. That is
this record's recovery axis, performed automatically by the curation stage
rather than by the reversible, evidence-backed family linkage specified below.

The refractory evidence is reassuring: median violation fraction 0.16% with 6.3%
exceeding 1%, against 0.09% and 1.0% for units both sorts agree on. But that is
not the screen this record specified.

Either the screen below is relaxed deliberately and on the record, or those 80
promotions are not yet authorized. See
[0010](0010-rescue-yield-is-relabelling-not-detection.md).

## Current state

Under the conservative screen (≥2 MUA-only reviewed neural events, contamination
≤10%, refractory fraction ≤1%, ≥100 spikes, presence in ≥half of 300 s bins),
**no unit is currently safe to promote.** Unit 389 carries the strongest reviewed
support (22 events) but 20.8% estimated contamination. Cleaner MUA units lack
unique reviewed support.

The bounded next step is to ask whether unit 389 can be **split** into a clean,
waveform-coherent component — not to relabel it wholesale.

## Evidence pointers

- `docs/luke_ks4_neuron_loss_audit.md`
- `python testing/luke_ks4_neuron_loss_audit.py` → `testing/outputs/luke_ks4_neuron_loss_audit/`
