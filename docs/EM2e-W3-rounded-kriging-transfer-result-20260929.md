# EM.2e W3 rounded-kriging transfer result — 2026-09-29

## Verdict

**Direct W3 transfer passes.** The W2-selected production-order rounded
kriging arm is practically equivalent to the existing W3 rounded exact-lattice
sort on the frozen arm-local continuity scorecard. It also reproduces the W2
cross-arm event association at a window near 8,000 seconds.

This strengthens the selected imec1 remap across two widely separated
windows. W3 was already exposed prospective transfer evidence, so this is not
an untouched holdout. No RF or outer holdout was accessed, and event
coincidence is not biological identity or purity proof.

## Frozen continuity result

| Arm | Assigned units | Eligible units | rho, negative versus flat |
|---|---:|---:|---:|
| rounded exact lattice | 485 | 394 | 0.741152 |
| rounded kriging | 478 | 386 | 0.752584 |

Exact lattice minus rounded kriging has delta rho -0.011432 with paired 95%
CI [-0.022919, 0.000459]. The full interval lies inside the frozen ±0.05
equivalence band. Yield differs by 1.44%, and the directional and symmetric
segment-safe short-ISI guardrails pass in every domain. The decision is
`practical_equivalence` across all 2,000 valid common block-bootstrap draws.

The scorecard packet is `testing/outputs/em2e_w3_scorecard_v2_20260929`.
Its manifest SHA-256 is
`f9c1f1e8f12876c1ddf6c43a0e95d1ccfe16ec4efa71ab66326fff9a914b8850`,
and `RESULT.json` has SHA-256
`2c6a87e43075d3fd77ea8557ca70c56fec9cb6a7569d388cc64c40f0f3b64176`.
This reuses the frozen W3 domains and remains an arm-local population
continuity proxy.

## Cross-arm event association

At the frozen ±2-sample tolerance, 344 unit pairs are reciprocal best. Their
median event F1 is 0.548, with p10 0.199 and p90 0.913. Reciprocal pairs contain
40.29% of exact-lattice reference events and 40.59% of rounded-kriging events;
190 pairs have F1 at least 0.5 and 87 have F1 at least 0.8. Ninety-two
reference best matches and 104 comparator best matches are marked ambiguous;
two reference units and one comparator unit have no five-event candidate.

The exact-sample sensitivity yields 345 reciprocal pairs, median F1 0.407, and
31.34% reference-event coverage. The packet retains all pair scores,
directional best matches, ambiguity, and unmatched cases. It is
`testing/outputs/em2e_w3_event_overlap_20260929`, with manifest SHA-256
`3b172d678da2e87b877e6c8a23523db37a0f018c8c0f05f0f6589a2c57fed731`
and result SHA-256
`dd6468b6ca2576ae3b33347050397fadb9d44e03671daef84998a577e99fd664`.

## Execution and resources

The candidate ran as persistent systemd service
`em2b-0a16a35f75669d90.service` and finished inactive/dead with result success
and exit status 0. It produced 478 assigned units, 531,975 accepted events, and
2,351 negative labels. The sorting SHA-256 is
`e82b0402057197a47f16fb5fefd98701fc18c2ebfdfe824f0d89722a62b5ec1b`.

| Measure | Value |
|---|---:|
| service wall to QC receipt | 1,180.4 s |
| systemd CPU | 3,030.2 s |
| preprocess | 214.7 s |
| detect | 490.2 s |
| sort | 457.3 s |
| QC | 9.0 s |
| periodic peak scratch | 11 GiB |
| final scratch | 9.2 GiB |

The compact measurement audit is
`testing/outputs/em2e_w3_measurement_20260929/AUDIT.json` (SHA-256
`e0e5b8fe32222595091cdec97757c411cd547af593031f6e2e57e478f0af34b3`);
its packet manifest SHA-256 is
`9a376ca703fa532626c41ff937c66c92d0bb86e987d8ba126a80923a015dd480`.

## Pipeline implication

Rounded production-order SpikeInterface kriging is now directly supported on
W2 and W3 by practical-equivalence scorecards and cross-arm event association.
The field-rounding advantage also transfers on W3 against static in the prior
exact-lattice experiment. The remaining risk before an approximately
10,474-second full-session run is operational scale and direct waveform
identity evidence. A full run is estimated near eleven hours from the measured
window runs and has no within-sort checkpoint, so its launch should follow a
durable full-session resource plan and recovery of the frozen imec1 lighthouse
event packet.
