# DA W2 development RF result

## Verdict first

The original accepted **D2L** final has the strongest descriptive result of the
three saved W2 outputs on this frozen development-only RF screen. It retains
more eligible units (80) and has higher aggregate and mean cross-validated SNR
than either fresh rematching result. This is not a pipeline winner based on
eligibility alone: the arm populations are unmatched and selection changes the
denominator.

**CD1_FULL does not show a functional improvement over REMATCH0.** It has fewer
eligible units (68 versus 75), a slightly lower median cvSNR (0.221 versus
0.235), and a lower all-eligible score sum (45.66 versus 52.96). This provides
no functional reason to add the coordinate-descent round.

| Arm | All units | Eligible | Coverage | Median cvSNR | Mean cvSNR | Sum, all eligible | Positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| D2L | 566 | 80 | 14.1% | 0.262 | 1.046 | 83.68 | 48 |
| REMATCH0 | 586 | 75 | 12.8% | 0.235 | 0.706 | 52.96 | 40 |
| CD1_FULL | 613 | 68 | 11.1% | 0.221 | 0.671 | 45.66 | 36 |

The existing static same-support context, reused rather than rerun, has 67
eligible units, median cvSNR 0.388 and mean 1.270. It is contextual only and is
not mixed into the three-arm DA table.

## Validation and limits

All frozen config, stimulus, split, gaze and arm hashes pass. The analysis uses
the exact imec1 W2 AP-frame interval `[26999783, 37199701)`, the original
61-development/20-sealed-holdout split, the same 12 complete W2 development
trials, 7/5 inner folds, and 6,052 valid gaze/lag frames. The five complete W2
holdout trials were not evaluated. Eligibility remains 500 total and 200 per
fold.

No cross-arm label join was used. The population summaries therefore compare
saved pipeline outputs, not matched biological neurons; neither raw ISI nor
eligibility alone selects a winner. TPCA/common waveform bases are exact, but
adaptive localization differs between rematching arms, so the final-output
contrast is total-pipeline mediation. The matching-stage CZ counts remain
descriptive and do not override this functional result.

## Runtime and resources

The successful evaluator took 28.46 seconds internally and 30.35 seconds
service wall, using 30.84 user plus 2.73 system CPU seconds and 2.93 GB peak
RSS. The first attempt failed before scoring because it used an environment
without `mat73`; the unchanged run succeeded in the existing locked Rowley
environment. Directly metered evaluator CPU across both attempts is 50.38
seconds. Setup, validation and reporting are conservatively charged 69.62
seconds, for a total DA charge of 120 seconds. H1 cumulative usage is therefore
17,453.22 seconds, below the 20,500-second ceiling.

No sort, raw voltage, GPU, gaze/lag refit, threshold tuning, holdout evaluation,
or count calibration was performed.
