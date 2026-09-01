# Where the repaired unwarped KS4 pipeline loses supported neural events

## Answer

The dominant measured exclusion is **not** KS4 duplicate removal and is not a
large failure to place reviewed neural events in the final event table. Of 128
reused, manually reviewed neural events, the repaired full-probe no-motion KS4
sort contains 119 (93.0%) in its final output. Only one reviewed event is found
in the learned event table but absent from the final table.

The large transition is the unit-quality boundary: only 61/128 reviewed neural
events (47.7%) have a nearby event belonging to a KS4 `good` unit, while 58/128
(45.3%) are present in the final output only under an `mua` label. This does not
prove that 58 neurons were lost—the endpoint is event-level, and several events
can map to the same unit—but it localizes the safest useful intervention:

> Review and reconcile supported MUA unit families after sorting, without
> changing voltage preprocessing, detection, or the accepted KS4 event table.

The independent sealed automatic holdout gives the same direction but is not
neural ground truth: 307/432 events (71.1%) are in the final table, 145/432
(33.6%) are near a `good` unit, and 162/432 (37.5%) are final-MUA-only.

## Unit-level clue: fragmentation rather than final-event deletion

The geometry-valid KIASORT comparison produced two isolated units that passed
the prespecified short-window waveform-consistency screen. They are hypotheses,
not confirmed neurons. KS4 contains events near about half of each candidate's
events, but those matches are scattered across 16–19 KS4 units and almost all
land in MUA clusters. No single KS4 unit captures more than 22.2% of either
candidate's events. Naively unioning the implicated KS4 labels yields 67.4% and
75.4% refractory violations, so these are explicitly **not safe merge sets**.

This is exactly the phenotype expected from incomplete unit-family
reconciliation: a plausible waveform family is partly detected, but its events
are distributed among labels that are individually weak. It does **not** yet
justify a hard merge. A safe follow-up should first create a reversible family
link and require:

- coherent raw-voltage templates across fragments;
- complementary, rather than duplicate, temporal support;
- acceptable refractory burden after union;
- improved event-centered residual energy; and
- preservation of already-good neighboring units.

## What is and is not safely recoverable

The safest recovery axis is bounded post-sort review of MUA clusters containing
supported events, followed by unit-family linkage. This is reversible and works
with spikes already accepted by KS4.

Under a deliberately conservative screen (at least two MUA-only reviewed neural
events, contamination at most 10%, refractory fraction at most 1%, at least 100
spikes, and presence in at least half of the 300-second bins), **no unit is safe
to promote yet**. Unit 389 contains the strongest reviewed support (22 events)
but has 20.8% estimated contamination. Cleaner MUA units have too little unique
reviewed support to clear the screen. The bounded next step is therefore to ask
whether unit 389 can be split into a clean, waveform-coherent component—not to
relabel it wholesale.

The present evidence does not support globally lowering detection thresholds,
relaxing duplicate removal, promoting all MUA units, or automatically merging
the KIASORT-nominated fragments. Those interventions would create new events or
identities where the evidence only supports targeted reconciliation.

## Reproduction

Run:

```bash
python testing/luke_ks4_neuron_loss_audit.py
```

The machine-readable outputs are written to
`testing/outputs/luke_ks4_neuron_loss_audit/`. Reviewed neural events are a
reused discovery cohort, learned-stage depth uses the final cluster median as a
spatial proxy, and the KIASORT evidence comes from one 120-second, 32-channel
window.
