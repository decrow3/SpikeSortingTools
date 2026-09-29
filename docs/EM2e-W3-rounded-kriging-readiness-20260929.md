# EM.2e W3 rounded-kriging transfer readiness — 2026-09-29

## Decision

Run one W3 rounded-kriging arm against the existing exact-lattice W3 sort.
This is now justified by the W2 cross-arm event result: W3 can test whether the
selected implementation preserves event populations at approximately 8,000 s,
far from the W2 development window. IDW and nearest remain out of scope.

The frozen transfer contract is
`configs/em2e_w3_transfer_contract.v1.json` (SHA-256
`a87c6828e9f43a52faae038feea772b4d800ade0bf1ce45454825c9c370beb26`).
W3 is already exposed prospective transfer evidence, not an untouched
holdout. No RF is used.

## Prepared graph

The no-voltage preparation verifies the W3 provenance, knot table, source and
target geometry, accepted-cache q0 identity, and existing exact-lattice sort
before serializing the production-order graph. It uses SpikeInterface 0.104.7
and produces 10,199,918 float32 frames by 182 target channels.

The descriptor is `testing/inputs/em2e_w3/rounded_kriging_recording.json`
(SHA-256
`577f9b42eb752f63cc2405d21eb978129a8489b2c29541ee67799672120d85e2`).
Its preparation receipt is `testing/inputs/em2e_w3/PREPARATION.json` (SHA-256
`987d99db98cdb984ca5e991d13ca6224145ad094c691080cbc877f670a821f84`).
The graph interpolates AP191, uses the accepted rounded states {-280, -240,
-200, -160, -120, -80, -40, 0} micrometers, applies kriging, and crops to the
182 target sites.

The DARTsort runner validates
`configs/em2e_w3_rounded_kriging.v1.json`. The sort uses the same stage budgets,
four CPU jobs, GPU device, and persistent systemd launcher as the completed W2
rounded-kriging arm. An interruption within a stage is not checkpointed and
can require that stage, or the sort, to restart; completed runner stages may be
reused only when their receipts and frozen inputs validate.

The first preparation attempt is preserved at
`testing/inputs/em2e_w3_failed_v1`. It verified inputs and wrote the descriptor,
then failed while formatting a relative path for its receipt. It read no
voltage and launched no sort. The path normalization was fixed before the clean
rerun.
