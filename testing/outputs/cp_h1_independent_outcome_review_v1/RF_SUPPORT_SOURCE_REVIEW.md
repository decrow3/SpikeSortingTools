# CP W2 RF support and source review

Status: W2 has usable development-only RF support. No W3 dots support is documented or present. The outer 20-trial holdout remains unopened.

## Actual W2 support

- Exact source interval: `[26,999,783, 37,199,701)` imec1 AP frames at 29,999.759166666667 Hz.
- Explicit imec1-to-NIDAQ RANSAC maps it to `[900.078594, 1240.108198)` NIDAQ seconds.
- All 17 overlapping valid dots trials are complete in W2: ordinals 66--82, totaling 169.733333 s.
- The original full-universe outer split assigns 12 of these to development (66, 67, 69, 70, 71, 74, 75, 78, 79, 80, 81, 82) and five to the unopened outer holdout (68, 72, 73, 76, 77).
- The original full-universe inner split gives the 12 development trials as 7 fold-0 and 5 fold-1 trials, 119.816667 s total and 7,093 lag-safe 60 Hz frames after removing the first seven frames of each trial.
- Unit-level eligibility still depends on the three actual CP W2 outputs: at least 500 development spikes total and at least 200 in each inner fold.

## Frozen inputs and clock

- Stimulus table SHA-256: `6a950b31bff1031456b394271d88af7796fbf7d38d526168ba504a7d147b62f0`.
- Frozen inner split SHA-256: `399858bbd6dbd689d43e5b333c24f6e850910f5f5ff3b18709477747a9e1a18c`.
- Accepted development-only gaze CSV SHA-256: `3a72672721ad83ad4810e764ea365d268794bafd2ceb65c955bc51f8c3e2e2a2`; its receipt proves training on the original 61 development trials and exclusion/non-evaluation of the 20 outer holdout trials.
- imec1-to-NIDAQ timing SHA-256: `3b0c46443ddc0448840a91a5f40dec6c74d4d996e2f3983a72955f6187776a43`. The actual MAT contains one timing struct with `imec1Time` and `NidaqTime`; 10,473/10,473 pairs are RANSAC inliers. Fitted slope 1.000095110566873, intercept 6.851202309 samples, residual MAD 2.116099 samples (70.5 us), maximum absolute residual 8.565135 samples (285.5 us).

## Load-bearing evaluator findings

The accepted evaluator is suitable only with a lightweight CP adapter; it must remain unchanged.

1. `evaluate_dots_rf.py` defaults `window_start_frame` to zero whenever `result.json` is absent. Bare CE/CL exported NPZ files therefore cannot be supplied directly. CP must require the explicit W2 source start frame and sampling frequency.
2. The evaluator builds its mask from gaze validity, lag safety, and development membership, but does not intersect response or lagged stimulus support with the sorting window. CP must restrict both to complete W2 trials; otherwise missing early-session W2 spikes become artificial zero responses.
3. Both outer and inner assignments must be generated on the original 81-trial universe before selecting the W2 subset. Recomputing either split on ordinals 66--82 is invalid.
4. The accepted shared gaze calibration is conditional on all 61 development trials, so the two-direction inner CV is not fully independent of gaze calibration. It remains a common, symmetric comparison across the three CP W2 outputs.
5. The generic clock helper chooses the first matching field from `imec0Time`, `imec1Time`. CP must explicitly require `imec1Time` and assert the timing source and units.

## Execution decision

Run the bounded common W2 RF comparison after the H5 arrays arrive if all three outputs carry exact source-frame provenance and yield eligible units in both frozen folds. Use the existing stimulus/gaze cache and CPU evaluator machinery, the same 12 development trials and eligibility domain for all three arms, and RF-independent correspondence. Do not evaluate the five W2 outer-holdout trials. If provenance or eligibility fails, return a finite unavailable endpoint rather than changing thresholds or opening the holdout.
