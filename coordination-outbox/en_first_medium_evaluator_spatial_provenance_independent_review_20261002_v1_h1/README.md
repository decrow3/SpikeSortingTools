# H1 independent review: H5 spatial-provenance candidate

Verdict: **TARGETED_REPAIR**.

The spatial loader's changed path is coherent: it hashes and parses the same position bytes, binds rows to time/cluster/amplitude, applies the explicit stable-sort lineage, validates the Kilosort source/ops/geometry/frame/clock/region, preserves the accepted physical regions, and propagates receipts into arm, pair, and four-arm artifacts. A path-only adaptation to a hash-identical H1 Kilosort install passes all 16 fixtures and the packaged CLI.

The packet is not ready for execution acceptance as published. Its tests and packaged proof hard-code H5's private Kilosort path, and the unmodified independent rerun therefore fails. The redundant `provenance.spatial_identity` claim is not checked against the validated lineage digest. Real REF/B receipt values could not be recomputed because their H5-local arrays are not mounted on H1.

The waveform document remains design-only and is not voltage-launch ready. Freeze its exact input/anchor bindings, preprocessing and exclusion algorithms, category matching/tolerance rationale, denominator and uncertainty equations, and low-N behavior before review and voltage access.

See `INDEPENDENT_REVIEW.json` for exact repairs, evidence, scope, and prospective gates. `RERUN_RECEIPT.txt` records both the native failure and the path-only diagnostic rerun. No evaluator or scientific experiment ran.
