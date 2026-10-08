# Candidate-3 mini v3 provenance correction — seal repair

Status: `SCOPED_RESULT_PRESERVED_WRAPPER_BYTE_IDENTITY_UNRESOLVED`

This is the seal-chronology repair for
`candidate3_mini_e2e_v3_provenance_correction_h1_20261008_v1`. The predecessor
is preserved but its shared filesystem timestamps do not prove that
`COMPLETE.json` was written after every member and the manifest. This packet is
the authoritative correction.

The correction is unchanged: the running receipt binds the launched wrapper to
SHA-256 `5d55bb1e0257afbb87b60d9d0d9dd5fcb4290a49ce2f67f8656969929e18801a`,
but those exact bytes were not retained. The first committed result source is
the later reporting version with SHA-256
`e627b1aaba5355a6a156aab908bd6bc4e19f82cfffbb359ffcaf1ee1ef67d32e`.
Therefore exact launched-wrapper reproduction is unresolved.

Resolved native configuration and output hashes still support the scoped claim
that the ordinary synthetic arm completed configured native DARTsort stages and
produced the reported final output. They do not establish wrapper-byte identity,
transition or real-voltage behavior, benefit, biological identity, purity, or
causal stage attribution. Future load-bearing runs must retain and hash the exact
executed source snapshot before launch.

Implementation checks
- Done: compared the prelaunch source binding with the first committed source -> hashes differ.
- Done: rebound the resolved config, timing, final sorting, and compact accounting hashes in `PROVENANCE_SUPPLEMENT.json`.
- Done: preserved the defective-seal predecessor and used a new repair namespace.
- Not done: byte review of the exact launched wrapper -> no snapshot exists.
- Can establish: scoped ordinary synthetic native-stage completion and inspectable output/accounting provenance.
- Cannot establish: exact launched-wrapper reproduction or any transition, real-voltage, benefit, identity, purity, or causal claim.
