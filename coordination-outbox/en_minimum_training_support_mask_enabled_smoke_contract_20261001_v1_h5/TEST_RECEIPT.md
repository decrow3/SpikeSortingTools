# Test receipt

Command:

`environments/rescue-production/.venv/bin/python -B -m unittest testing.test_kilosort_support_mask_candidate testing.test_prewhitening_support_mask`

Result: **34 tests passed** in 0.263 seconds on 2026-10-01.

The suite includes exact typed structural-map guards, malformed boolean and
undeclared-delta rejection, namespace reuse rejection before native entry,
classified failure receipts on every pre-W exit, covariance validation, exact
audited W behavior, and native-consumer support-mask fixtures.

This is synthetic validation. It does not establish the finite/rank/condition
properties of the real covariance and did not open the recording binary.
