# Lighthouse tracker method audit, 2026-09-09

Four controls on the waveform-only whole-probe tracker itself, none of which
existed before and none of which needed new voltage extraction. Results:
`testing/outputs/luke_lighthouse_method_audit_v1/`. Reproduce with
`python -m testing.luke_lighthouse_method_audit_v1` (about 15 s; no managed job
required, no recording read). No motion estimator, absolute-depth prior or
held-out selection enters any control. Nothing here validates or refutes a
motion field.

## Headline: the 40 um patch lattice biases acceptance toward its own nodes

Each column of this probe samples every 40 um (x=0,32 from y=20; x=16,48 from
y=0), so 40 um is the only exact translation available and the half-step at
20 um is the worst case. Placing each candidate's own template at known depth
offsets and matching it by the tracker's own rule gives:

| True offset from lattice node | 0 um | 10 um | 20 um | 30 um | 40 um |
|---|---|---|---|---|---|
| Strict acceptance, noiseless | 100% | 85% | 32% | 97% | 100% |
| Strict acceptance, at recorded noise | 44% | 13% | **0%** | 19% | 43% |
| Correct identity retained, noiseless | 100% | 94% | 45% | 94% | 100% |

Linear and cubic interpolation kernels agree closely, so this is probe geometry
rather than an artifact of the interpolator used to build the test. When a match
is accepted the centroid readout is close to unbiased (gain 1.05 at 20 um), so
the failure is in acceptance, not in the depth measurement.

Consequences that matter for how existing results are read:

- Accepted tracks are biased toward 40 um lattice nodes.
- Dropout at intermediate depths is partly geometric. Absence of accepted
  events is not evidence that a cell stopped firing or stopped moving.
- Apparent stationarity can be manufactured by the lattice, so any comparison
  that rewards a flatter motion field is confounded until this is controlled.
  This applies directly to the current full-session rigid-versus-nonrigid
  comparison, where rigid presently scores better.

Caveats. The test is synthetic. Templates are spike-averaged, so the noisy
condition adds single-sample channel noise to a mean waveform and does not model
per-spike amplitude variability, bursting, or collisions; the peak channel is
taken from the clean template so patch assignment is deterministic. Absolute
acceptance percentages are indicative. The node-versus-half-step contrast, which
is a within-test comparison, is the result.

## Track plausibility

Median held-out (940-1230 s) temporal coverage is 11%; only unit 673 exceeds
50%. Within-second centroid scatter is 1.7 um, so measurement precision is
100-1000x finer than the excursions being compared and is not the limiting
factor. Units 125, 161, 557, 698 and 705 place strict matches across more than
500 um of probe (up to 2,880 um), which tissue drift cannot produce; those
tracks are most consistent with waveform lookalikes. Unit 161 is one of the
three candidates currently driving the >100 um nonrigid discrepancy flag.

Span and coverage are post-hoc readouts of the resulting track. They are not
gates applied during matching and must not become one, or the depth-blind
guarantee is lost.

## Selection null and identity ambiguity

Holding cached seed spike times fixed and circularly shuffling each candidate's
accepted-event train within the 930-940 s selection window (2,000 draws), 16 of
16 testable candidates recover their own seed spikes above chance, with null
recovery near zero throughout. The selection rule is doing real work and is not
a coincidence of detection density. This is the first null this step has had.

Against that, for 82% of suprathreshold detections a runner-up identity also
clears cosine 0.86. Nearly all identity discrimination therefore rests on the
0.025 margin over a highly degenerate 247-template bank, which is a thin basis
for treating a match as an identity.

## Candidate independence

Single linkage on the cached timing-aligned template cosine gives 11 families at
0.80, 13 at 0.85, 15 at 0.90 and 17 at 0.95. The 17 templates are 17 independent
families only at the highest threshold; 557/673/675/632/657 are mutually similar
and where that block splits is a choice. Report the curve rather than one
number, and give each family one vote in any consensus, per the existing policy.

## What this does not do

This audit does not calibrate a whole-probe false-positive rate. That still
requires decoy templates (time-reversed and vertically flipped) added to the
bank and rescored against real detections, which is a managed re-extraction. The
82% runner-up figure bounds ambiguity among real identities only.
