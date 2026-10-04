# H1 changed-path review: waveform method repair v3

Verdict: `TARGETED_REPAIR_REQUIRED`; method freeze and execution remain disabled.

The packet is byte-valid. H1 accepts the correlation repair and confirms that
the accepted preprocessing/common-median, competitive matching categories,
sampling/alignment, and 4,608-read / 4,052,090,880-byte endpoint arithmetic are
unchanged. Prospective arm and anchor hashes remain unbound and both execution
gates remain false.

Two load-bearing repairs remain:

1. **Make the exact runtime independently verifiable and state its real scope.**
   The H5 closure is an exact host-runtime lock, not a cross-build portable
   environment. On H1 the numerical backend fields and waveform-probe digest
   match, but the verifier correctly rejects 51 H1 executable objects against
   the 89 H5 objects. H1 cannot rehash the H5 objects because their bytes are not
   in the packet or an independently accessible immutable environment artifact.
   Publish a content-addressed OCI/SquashFS/conda-pack (or equivalent) containing
   the declared Python/PyTorch/CPU-FFT runtime, or provide an independently run
   verifier receipt over those exact bytes. Call the present rule
   `exact relocatable-path identity`, not general portability. Make the test
   classify a nonmatching reviewer host as the required negative control rather
   than asserting that every reviewer must match H5.
2. **Bind current voltage bytes, not only an old receipt and current size.**
   `verify_voltage_identity` validates the historic hash receipt, then accepts
   the current binary using path and size only. A bounded adversarial fixture
   replaced every byte while preserving size; both preflights returned `PASS`.
   The initial/final stat tuple detects mutation during a future run but cannot
   establish continuity from the historic full-hash pass to run start. Bind a
   trusted immutable snapshot/object/fs-verity identity created with the full
   hash, or perform and separately receipt an explicit current full-file check.
   If neither exists, do not claim current content identity from the old receipt.
   Also use `lstat`/no-follow semantics if the contract intends an exact regular
   file rather than a replaceable symlink target.

The correlation implementation has a deterministic eight-code priority and its
numeric fixtures cover +1/-1 values, channel order/intersection, frame overlap,
NaN/Inf, zero variance, malformed inputs, and overlapping invalid conditions.

No voltage, real event data, sorting, RF, or holdout work was performed.

Implementation checks
- Done: source packet MANIFEST/COMPLETE and every manifest member verified.
- Done: 31/32 native and staged fixtures passed; the only failure is the exact H5 runtime-positive fixture on the intentionally different H1 runtime.
- Done: independently captured H1's runtime closure; backend and probe output match, while exact executable identity correctly differs.
- Done: demonstrated that same-size voltage-content replacement passes the current no-rehash preflight.
- Done: inspected and accepted correlation priority/numeric fixtures and unchanged accepted-method assertions.
- Not done: exact H5 executable objects were not independently rehashed because no immutable environment artifact or reachable producer runtime was supplied.
- Not done: no current full voltage hash or trusted immutable-storage identity was available; no voltage bytes were opened.
- Can establish: R3 and the preserved accepted method are coherent; R1 fails closed outside the H5 runtime; compact preflight causes no hidden 241-GB read.
- Cannot establish: independent identity of the declared H5 executable bytes or continuity between the historic voltage hash and the current same-sized binary.

