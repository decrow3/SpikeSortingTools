# Focused independent rereview: joint-model K7 frame-score fixture v2

Proposed verdict: **GO_SYNTHETIC_SCORE_ORACLE_ONLY**.

V1's independent review found six failing fixtures across five defects: weakened
support floors, coupled training/held-out thresholds, int64 separation overflow,
unstable residual-ratio arithmetic, and skipped validation of noncontributing
rows. V2 applies only those requested repairs and adds focused author
regressions. The original v1 failure packet remains immutable.

Verify the exact six preserved failures now pass against v2, all 27 previously
passing independent cases remain green, the expanded 34-test author suite
passes, and the source delta is confined to the requested repairs. Verify the
local/shared v1 NO-GO and v2 hash chains.

No reviewer production action. Do not implement or inspect template learning,
matched-filter detection, whitening, threshold calibration, real units/arms,
recordings/voltage, DARTsort, GPU, sorts, RF/holdout or C4. Publish a distinct
immutable review.

Implementation checks
- Done: v1 NO-GO preserved; exact requested guards and regressions implemented.
- Not done: template learning, detection, whitening, calibration or real frame classification.
- Can establish: whether the known synthetic K7 score-oracle blockers are repaired.
- Cannot establish: full K7 validity, real labels, N3/N4 or N0/N1 readiness.
