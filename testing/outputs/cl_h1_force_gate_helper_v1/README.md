# CL portable force-gate helper packet

This packet publishes the parameterized, frozen CJ-v2 computation for use on a
same-run W3 captured state. `cl_force_gate.py` accepts explicit post-state,
route, mask, interval-metadata and preregistered mask-hash paths. SI must be
declared `absent` or supplied as an array; it is never silently dropped.

The W2 equivalence check is exact: both CSVs are byte-identical to CJ v2 and all
five graph arrays are elementwise identical. W2 explicitly declares SI absent.

The frozen primary continues to use 42/90. That value is not literally the exact
finite-window pair-exposure ratio after symmetric trimming. Across W2's 48
blocks (39 of length 149,999 samples and nine of length 149,998), exact ratios
range from 0.4668162453 to 0.4668162463 versus nominal 0.4666666667, a relative
difference of about 0.0321%. The one-sample block-duration effect is about 1e-9
absolute. The closest passing bootstrap upper margin is 0.0387 and the closest
failing margin is 0.1716, so no current W2 decision is marginal. This is a
qualification/sensitivity audit, not a replacement or tuned statistic.

The derangement null preserves genuine segment, nonwrapping and shared support,
but nonstationary rates and event-positive common-block conditioning remain.
It is not biological identity proof.
