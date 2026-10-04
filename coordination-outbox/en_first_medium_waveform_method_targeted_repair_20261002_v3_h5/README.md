# Waveform method targeted repair v3

Status: `READY_FOR_H1_REREVIEW_EXECUTION_DISABLED`.

This fresh packet preserves H1 review MANIFEST
`e8d654cc3859439457acdf7e67a48af3dd81d5d919e5c3a340149bb4937775c9`
and COMPLETE
`d51472887b424cfbbecdde4a6b8ab449eac4d011fd918b40ab93dbf35b61cb13`.
It changes only H1 findings R1-R3. The accepted preprocessing adapter remains
byte-identical (`e9abe2a9...`), and contract fixtures assert exact preservation
of matching, sampling, the 4,608 endpoint-read maximum, and
4,052,090,880-byte logical payload maximum.

R1 is closed with an executable runtime closure captured after actually running
the accepted CPU float32 median/FFT/ifft/fftshift probe. It hashes the Python
executable and all 88 other mapped ELF objects: 89 members and 3,333,088,832
bytes total. Python/platform fields, PyTorch build configuration, backend state,
and probe-output hash are bound. A clean-process verifier rehashes the closure
before voltage access. Paths may differ only when the exact multiset of role,
basename, size, and SHA-256 plus all runtime scalars matches.

R2 chooses the no-rehash approach. The contract binds the existing immutable
Q0 full-hash receipt packet (MANIFEST `909c9acc...`, receipt `0bdeec71...`) for
the 241,309,358,592-byte binary and explicitly forbids a fresh full-file hash.
Per-run preflight hashes compact receipts and the compact recording manifest,
then uses `stat` only on the binary. Future voltage access is restricted to the
unchanged bounded endpoint `os.pread` calls, with one receipt row per sampled
endpoint and exact requested/returned totals.

R3 is executable in `source/retained_pair_correlation.py`. Eight invalid codes
have one frozen priority. Thirty-two tests include +1/-1 correlations, channel
reordering and intersection, absolute-frame overlap, missing/unselected/invalid
endpoints, insufficient channels and frames, NaN, positive/negative infinity,
constant and one-sided zero-variance vectors, malformed alignment, and
overlapping failure conditions.

No recording, voltage, real event data, sorter, job, RF, or sealed holdout was
opened or run. Prospective arm/anchor bindings and both execution gates remain
disabled.

Implementation checks
- Done: verified H1 review and accepted-parent bytes; adapter hash remains unchanged.
- Done: executed and rehashed the clean CPU runtime probe; 89-member closure and portable byte-identity rule passed.
- Done: bound and internally checked the prior immutable full-hash receipt; the new preflight implementation was tested only with synthetic files and did not open the real binary.
- Done: numeric correlation and overlapping-priority fixtures passed; accepted matching/sampling/read arithmetic is asserted byte-for-byte or scalar-for-scalar against v3.
- Not done: no real voltage preflight, endpoint read, waveform measurement, or scientific evaluation; H1 rereview remains pending.
- Can establish: the three requested method-contract gaps have executable, fail-closed repairs on bounded synthetic/runtime checks.
- Cannot establish: future host availability, any real waveform result, biological identity, purity, efficacy, or advancement.
