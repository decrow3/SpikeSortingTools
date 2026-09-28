# CX masked-donor qualification

## Verdict

The frozen donor-coordinate taper produces a **25-donor dual-half exploratory
engineering set**, compared with the preserved 5/90 unmodified qualification.
This is a modified-full-source feasibility result, not held-out donor
qualification: the left/right halves train alternate tapers, but each trained
taper is evaluated on the modified full source. The halves are disjoint yet
share the temporal basis and preprocessing. No sorter should launch until
background provenance is added to the hybrid scorer.

| Result | Donors |
|---|---:|
| Original unmodified source, preserved | 5/90 |
| Left-trained taper applied to full source | 28/90 |
| Right-trained swapped-half taper | 28/90 |
| Passed both constructions | **25/90** |
| Qualification agreement, pass or fail | 84/90 |
| Left-only passes | 3 (303, 307, 352) |
| Right-only passes | 3 (88, 336, 491) |

All five original qualifiers remain qualified. The 25 dual-pass IDs are 13,
30, 139, 143, 184, 275, 276, 291, 296, 325, 332, 372, 407, 415, 425, 433,
438, 445, 451, 461, 465, 480, 493, 499, and 512.

## What the taper changed

The single frozen rule uses temporally centred training-half channel energy,
40-um Gaussian spatial smoothing, a raised-cosine transition from 5% to 10% of
peak amplitude-equivalent energy, and only the <=45-um connected component
containing the peak. It is defined in source-donor coordinates and travels with
the waveform; it never sees placement or sorter outcomes.

Across all 90 donors:

- median held-out-half *total* energy removed: 5.68% (maximum 15.50%); this
  mixes reproducible signal and noise rather than isolating either one;
- median split-half difference energy removed: 41.86%; the difference also is
  not a pure unstable/noise component;
- median modified/original centred cosine: 0.9934;
- median modified/original PTP ratio: 1.0000;
- median support is a soft full-span taper (384 nonzero channels), not a binary
  compact mask. This is why the result is a modified-source engineering set,
  not evidence that all far-field signal is absent.

The six swapped-half disagreements and median mask cosine of 0.55 show that
mask details remain estimator-sensitive even though the feasibility decision
agrees for 84/90 donors. The 25-donor intersection is therefore an exploratory
source set for a future bounded benchmark, not a validated donor bank.

## Geometry and stress controls

The qualification evaluated every frozen 40-um lattice placement and all seven
occupied states on the exact 182-channel target. Independent controls verify
one-to-one/inverse closure at -240, -40, +40, and +240 um and correctly localize
boundary losses by sign. Exact self-clone is a positive control; 0.5x and 2x
amplitude variants keep cosine 1 but fail the PTP clone gate; zero is negative.
A nonwrapping +/-1-sample temporal mismatch of isolated unit 30 reduces cosine
to about 0.932. The predeclared similar overlapping pair 484/487 has cosine
0.900 unmodified and 0.913 modified. These labels are not established as two
distinct biological neurons; this is only a waveform-similarity stress case,
not an executed overlap or identity experiment. Fractional-shift controls
remain unqualified and were not run.

The implementation's target slice `202:384` remains a hard-coded assumption.
Its physical-channel meaning must be asserted from metadata before any future
benchmark; preserve the present counts until that check is made.

## Truth-assignment audit

The existing corrected scorer still lacks immutable background provenance.
`score_with_frozen_association` counts every event in a donor's chosen output
label as output and defines FP as output minus injected-truth matches. Thus
pre-existing background in an assigned label is charged as injection FP, while
events in unassigned labels are absent from donor FP counts. Association can
also move between labels through its precision ranking. Unmatched background
is therefore neither uniformly nor causally classified.

Because a new sort cannot preserve input row identity by construction, do not
invent exact background lineage. Score injected TP/FN and injected identity
confusion exactly. Separately compare the paired unmodified and injected
outputs with an explicit ambiguity class, reporting supported preserved/stolen
background and novel output without charging every unmatched background event
as FP or assigning unexplained novel output to injection truth.

## Deliverables and resources

The complete packet remains local under
`testing/outputs/cx_masked_donor_qualification_v1/run/`. The shared experiment
root contains status/scope only; the waveform-derived taper packet was not
exported. The local packet includes all-90 denominators, the 25-donor
exploratory set, tapers, stress controls, source, preregistration, truth audit,
hashes and resource receipt. It read 46.7 MB of saved donor arrays, used no GPU
or raw voltage, launched no sorter, and is conservatively charged 180 H1 CPU
seconds.
