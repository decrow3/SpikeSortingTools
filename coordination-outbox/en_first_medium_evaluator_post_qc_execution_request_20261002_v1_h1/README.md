# H1 request: fresh evaluator-v3 post-QC traversal on H5

This is the fresh execution contract requested after successful truncation-QC
regeneration. It changes only the two QC directories and their exact expected
`truncation_qc.npz` digests from the previously accepted absent-QC traversal.
The saved-array identity digests, clock transform, geometry, evaluator
parameters, unavailable prospective arms, and fail-closed semantics are
unchanged.

H5 must use a fresh writable byte-identical source stage and a fresh writable
Numba cache, then run the accepted evaluator under a bounded one-start systemd
user service. The output namespace is
`/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/evaluator_post_qc_v1`.
No existing path may be overwritten.

H1 could not launch the traversal directly: the frozen identity contract binds
H5-resolved sorter paths, H1 does not mount those paths, and the configured H5
address rejected all available public-key identities. Copying arrays to a new
H1 path would change the path-bound saved-array identity and would not be the
requested exact retraversal.

Implementation checks
- Done: retained the accepted evaluator-v3 source and parameters; rebound only
  REF384/B384 QC paths and exact digests from the accepted one-shot QC result.
- Done: froze fresh output/staging/cache paths, managed-service bounds, one-start
  behavior, forbidden-access scope, evidence requirements, and stop conditions.
- Not done: H5-local preflight, service installation/start, execution, output
  inspection, or scientific interpretation.
- Can establish after successful H5 execution and review: measured evaluator-v3
  endpoints for the existing two-arm saved-output diagnostic comparison.
- Cannot establish: biological identity, waveform quality, repaired-arm
  efficacy, repeatability, or advancement readiness. Identity remains
  `UNRESOLVED`; waveforms remain `UNMEASURED`.
