# H1 independent re-review: H5 waveform-contract targeted repair v2

## Verdict

`TARGETED_REPAIR_REQUIRED`; execution remains disabled.

The packet is internally intact and substantially repairs the earlier method
contract. H1 independently reproduced all 10 bounded fixtures on both the
published packet and a byte-identical `/tmp` staging copy. The following
components are accepted for method freeze:

- CPU float32 waveform operation order and common-median terminology;
- the frozen 30,122-sample impulse and 1,145-sample endpoint-record semantics;
- exhaustive competitive matching categories, including asymmetric competition
  and equal-cost ambiguity;
- retained-pair atomic sampling, endpoint-read arithmetic, seed/rank rules,
  category denominators, and prospective arm/anchor gates.

Three exact repairs remain before an enabled run:

1. Bind the executable runtime, not only package metadata. The present closure
   binds nine files and four distribution `METADATA` files, but the enabled
   adapter executes Python and PyTorch/FFT compiled libraries whose bytes and
   backend configuration are not in the closure. Bind and preflight an immutable
   environment/container or an equivalent executable-byte closure, including
   Python and the loaded PyTorch/CPU FFT shared objects and configuration.
2. Remove the unbounded preflight-read alternative. The contract permits a new
   full binary hash before event reads, while its exact 4,608-read / 4,052,090,880-
   byte accounting covers endpoint `pread` calls only. Require validation of a
   reviewed existing full-file hash receipt for this run, or separately freeze,
   bound, and receipt every preflight binary byte.
3. Freeze correlation failure semantics and add numeric known-answer fixtures.
   Specify an ordered, correlation-specific reason-code priority and test +1/-1
   correlation, physical-channel intersection/order, absolute-frame overlap,
   missing endpoints, fewer than two channels, fewer than 26 frames, nonfinite
   values, and zero norms. Current fixtures check only contract text and a frame
   count for this portion.

No recording voltage was read and no sorter or RF/holdout work was launched.

## Implementation checks

- Done: packet and dependency hashes -> manifest checks passed; dependency
  aggregate independently recomputed as
  `ebcad89e8c45de0315557c33414ada30e3c3bee951b250d33f3e2ff0ad156e64`.
- Done: executed implementation and effective settings -> adapter, contract,
  runtime lock, frozen constants, bundled Kilosort sources, and tests inspected;
  native and staged test runs each passed 10/10.
- Done: operation semantics -> direct fixture proves bitwise equality with
  `BinaryFiltered.filter`; source inspection confirms per-channel centering,
  across-channel median, frozen FFT filter, and fftshift.
- Done: matching/counting semantics -> exhaustive enumerator fixtures cover
  asymmetric competition, equal-cost ties, far anchors, and unsupported anchors;
  the contract makes categories mutually exclusive and exhaustive.
- Done: read arithmetic -> 3 comparisons x 3 epochs x (384 nonretained endpoints
  + 128 retained endpoints) = 4,608; 4,608 x 1,145 x 384 x 2 = 4,052,090,880 bytes;
  retained pairs are sampled atomically and extra partner reads are zero.
- Not done: executable-byte runtime closure -> no immutable environment artifact
  or hashes of Python/PyTorch shared libraries were supplied.
- Not done: full binary identity preflight -> prohibited in this bounded review;
  the enabled contract must choose and bind a no-rehash receipt or separately
  account for the preflight I/O.
- Not done: end-to-end correlation implementation -> no executable correlation
  helper or numeric invalid-case fixture is present.
- Can establish: the repaired contract's accepted components are coherent and
  reproducible on the locked H1 runtime, subject to the three repairs above.
- Cannot establish: exact future executable identity, exact total recording I/O,
  or deterministic retained-pair correlation status/reasons; no scientific
  waveform result or biological identity/purity claim is established.

