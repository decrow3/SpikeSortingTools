# CZ fixed-bank rematching outcome review

## Verdict first

Both completed CY arms pass independent saved-output accounting and exact
AP-frame clock checks. The extra full-resolution coordinate-descent round does
not improve the frozen accepted-episode adjacency endpoint: after final
refinement REMATCH0 and CD1_FULL each have 104 accepted-state 9--29-sample
pairs. CD1_FULL instead produces 17,915 more final rows than REMATCH0 and 27
more final units, with a slightly larger all-state endpoint (4,453 versus
4,347). These are descriptive sorter outcomes, not biological identities.

The cleanest finite recommendation is therefore to retain **REMATCH0** as the
method variant if the producer's paired metrics confirm no material CD1_FULL
benefit. Do not start another calibration, RF evaluation, or sorter arm. The
one useful next step would be transfer of this already-frozen REMATCH0 method
to a genuinely held-out window/session, conditional on the paired result.

## Independent accounting

| Output | Rows | Assigned | Noise | Final units | 9--29 all | 9--29 accepted |
|---|---:|---:|---:|---:|---:|---:|
| Accepted D2L baseline | 641,588 | 635,734 | 5,854 | 566 | 7,493 | 312 |
| REMATCH0 | 648,483 | 645,448 | 3,035 | 586 | 4,347 | 104 |
| CD1_FULL | 666,398 | 663,388 | 3,010 | 613 | 4,453 | 104 |

Every row closes as assigned plus noise. All outputs use exactly
29,999.7591667 Hz, 182 physical channels, local sample coordinates inside the
340-s W2 recording, and the frozen mapping
`source_frame = 26,999,783 + local times_samples`. Ten accepted catalogue
segments overlap W2. The endpoint counts require consecutive events within a
final label and both event times in the same accepted segment.

At the matching stage before refinement, REMATCH0 has 648,483 rows, 743 used
bank units, 2,639 all-state endpoints, and 49 accepted endpoints. CD1_FULL has
666,398 rows, 742 used bank units, 3,288 all-state endpoints, and 50 accepted
endpoints. Thus the additional CD round changes the broad event partition but
does not improve this accepted-state endpoint.

## Model and interpretation audit

Both arms use the same compatible-bank hash
`298258e2cf740320be5591b08755b3dc38dd3ded33bd64eaa1e62cbd96525819`,
same whitening arrays, configuration, seed, and source. Their saved TPCA and
common waveform-basis tensors are exactly equal. Their adaptive localization
encoder tensors are not equal, however. Consequently the matching-stage
contrast is the cleaner CD comparison; final-refinement differences include
arm-specific adaptive localization fits and are not a perfectly pure CD-only
contrast.

The new matching rows have no valid index join to baseline rows. A temporal
candidate within +/-7 samples is only a correspondence candidate, not an
identity assignment; unmatched and multiply matched events must remain
explicit. ISI, yield, or unit count alone cannot select an arm.

## Scope qualifications inherited from CW/CX

- CW's rank-1-versus-other 248-pair subset is only a descriptive
  amplitude-proxy partition. It admits tiny/reassigned children and does not
  establish a general 50% bound for a changed consumer.
- Sharing a no-force final child is not the same as sharing a native
  constituent, and the current data do not apportion historical BQ effects.
- CX's 25 modified-source donors are exploratory feasibility sources, not the
  old AZ qualification set.

## Artifacts

- `testing/outputs/cz_rematching_outcome_review_20260928/INDEPENDENT_ARM_ACCOUNTING.json`
- `testing/outputs/cz_rematching_outcome_review_20260928/MODEL_STATE_COMPARISON.json`
