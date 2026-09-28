# Luke0804 imec1 cross-seed lighthouse replication, 2026-09-12

## Result

A second, independently seeded sorter-free waveform candidate reproduces the
large held-out depth changes of `p06_f000`. The pair is a strong positive local
replication lead, not yet certified biological ground truth.

The new 300--320 s discovery window was fixed by uniform temporal spacing within
the dot-RF epoch. No motion estimate, candidate track, or apparent quietness
entered its selection. The v3 global phase/rival/decoy/gain gates were unchanged.

The new bank contained 97,613 seed detections and 192 waveform proposals. Three
families passed the depth-blind seed qualification:

| Candidate | Strict seed recovery | Global rival cosine | Seed P90 span |
| --- | ---: | ---: | ---: |
| `p09_f020` | 27/97 (27.8%) | 0.896 | 250.7 um |
| `p09_f038` | 26/77 (33.8%) | 0.936 | 246.3 um |
| `p09_f024` | 26/80 (32.5%) | 0.936 | 258.6 um |

The original 120 um localized-seed flag rejected all three. That flag was too
coarse for the user's prior scientific allowance that motion up to approximately
250 um can be plausible. The output is preserved unchanged. Future reports now
separate localized candidates (<=120 um) from motion-scale candidates
(>120--300 um), with the latter requiring independent replication rather than
automatic promotion.

## Frozen primary pair

`p09_f020` was the primary new candidate before held-out comparison because it
ranked first, had the lowest global rival cosine, and did not belong to the
high-similarity `p09_f038`/`p09_f024` pair. It was compared with the original
bank's sole localized candidate, `p06_f000`.

| Diagnostic | Result |
| --- | ---: |
| Cross-template cosine | 0.608 |
| Exact common relative channels | 11 |
| Minimum overlap-energy fraction | 0.644 |
| Strict events, `p06_f000` / `p09_f020` | 76 / 55 |
| Exact shared raw detections | 0 |
| Reciprocal nearest pairs within 250 ms | 20 |
| Paired relative-depth correlation | 0.923 |
| Centered median absolute paired difference | 7.8 um |
| Same-sign relative-depth observations | 80% |
| Within-window circular-shift p-value | 0.002 (2,000 draws) |

Candidate selection did not use these replication metrics. The null independently
circularly shifts the second candidate's times within each of the six frozen
held-out windows, repeats reciprocal pairing, and asks how often the correlation
meets or exceeds the observed value.

The low template cosine and zero exact shared detections argue that these are not
simply two names assigned to the same detected events. Their seed reference
depths are nearby (approximately 3582 and 3527 um), so this establishes local
rather than probe-wide replication. Shared lattice acceptance or another local
measurement artifact remains possible.

## Temporal resolution

The two candidates together provide 131 strict events over 52 s of held-out
windows, about 2.5 strict events/s before accounting for temporal clustering.
This does not support an independent >4 Hz lighthouse trace. Lower-score events
can increase display density but must not be promoted to strict identity solely
to meet a temporal-resolution target.

The correct next validation is candidate-specific lattice/phase recovery and
inspection of paired waveforms at common large excursions. A new >=5 Hz motion
fit is premature until that measurement control passes or additional strict
identities raise temporal support.

## Artifacts and verification

- Second discovery:
  `testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/`
- Managed job evidence:
  `testing/outputs/luke_imec1_dots_sorterfree_waveform_v3_s300_job/`
- Direct audit: `testing/luke_imec1_cross_seed_replication_v1.py`
- Audit outputs: `testing/outputs/luke_imec1_cross_seed_replication_v1/`
- Main figure: `01_primary_replication.png` / `.pdf`
- Pair-level observations and all six candidate-pair audits are preserved as CSV.

The 300--320 s managed service completed with exit status 0. The direct cached
audit read no motion estimate and completed successfully. Nine focused tests
pass across the phase-aware scorer, gain gate, complete-link audit, reciprocal
pairing, and within-window shift machinery. No production sort, correction,
motion field, or concurrent full-probe job was modified.
