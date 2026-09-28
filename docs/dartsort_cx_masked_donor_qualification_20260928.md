# CX masked-donor qualification

## Verdict

The frozen donor-coordinate taper produces a **25-donor dual-half exploratory
engineering set**, compared with the preserved 5/90 unmodified qualification.
It is suitable for a bounded benchmark design, but it is not pristine
validation: the two spike halves are disjoint yet share the temporal basis and
preprocessing. No sorter should launch until background provenance is added to
the hybrid scorer.

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

- median held-out-half energy removed: 5.68% (maximum 15.50%);
- median split-half difference energy removed: 41.86%;
- median modified/original centred cosine: 0.9934;
- median modified/original PTP ratio: 1.0000;
- median support is a soft full-span taper (384 nonzero channels), not a binary
  compact mask. This is why the result is a modified-source engineering set,
  not evidence that all far-field signal is absent.

The six swapped-half disagreements and median mask cosine of 0.55 show that
mask details remain estimator-sensitive even though qualification agrees for
84/90 donors. The conservative set is therefore the 25-donor intersection.

## Geometry and stress controls

The qualification evaluated every frozen 40-um lattice placement and all seven
occupied states on the exact 182-channel target. Independent controls verify
one-to-one/inverse closure at -240, -40, +40, and +240 um and correctly localize
boundary losses by sign. Exact self-clone is a positive control; 0.5x and 2x
amplitude variants keep cosine 1 but fail the PTP clone gate; zero is negative.
A nonwrapping +/-1-sample temporal mismatch of isolated unit 30 reduces cosine
to about 0.932. The predeclared similar overlapping pair 484/487 has cosine
0.900 unmodified and 0.913 modified, so tapering does not manufacture an
identity claim. Fractional-shift controls remain unqualified and were not run.

## Truth-assignment audit

The existing corrected scorer still lacks immutable background provenance.
`score_with_frozen_association` counts every event in a donor's chosen output
label as output and defines FP as output minus injected-truth matches. Thus
pre-existing background in an assigned label is charged as injection FP, while
events in unassigned labels are absent from donor FP counts. Association can
also move between labels through its precision ranking. Unmatched background
is therefore neither uniformly nor causally classified.

Before a sorter benchmark, carry immutable background row IDs through
injection/sorting and report four separate quantities: injected TP, injected
FN, preserved/stolen background, and genuinely novel output. Unassigned
background must remain unassigned rather than becoming automatic FP.

## Deliverables and resources

The packet is under
`testing/outputs/cx_masked_donor_qualification_v1/run/` and is mirrored to the
shared experiment root. It includes all-90 denominators, the 25-donor set,
tapers, stress controls, source, preregistration, truth audit, hashes and
resource receipt. It read 46.7 MB of saved donor arrays, used no GPU or raw
voltage, launched no sorter, and is conservatively charged 180 H1 CPU seconds.

