# Candidate 3 H5 real-window equality contract and runner

## Requested decision

Review whether `CONTRACT.json` and `RUNNER.py` are execution-ready for the
single bounded H5 comparison they freeze.  A GO permits only the predeclared
ordinary and transition windows after an exact coordinator dispatch receipt.
It does not permit a sort, full-session materialization, RF/holdout access, a
retry, or any scientific interpretation beyond wrapper-versus-producer
equality on those requests.

The cheapest useful test has already run locally: the full-shape synthetic
self-test used the exact DARTsort and Kilosort imports, exercised all nine
requests, and passed with zero element mismatches.  Both context-sensitive
positive controls differed.  It read no H5 path or recorded voltage.

## Implementation checks

- Done: executed source and configuration are copied verbatim and hash-bound;
  the runner rehashes the wrapper, Kilosort producer, and three DARTsort source
  files before any real binary access.
- Done: the two arms differ only by their frozen global frame ranges; request
  lengths, padding, channels, geometry rule, filter, producer, and comparison
  are shared.
- Done: frames are half-open, sampling frequency is exact, local origins are
  checked, channel order is 0..383, and the geometry file is hash-bound.
- Done: every request filters its own exact length; the positive control proves
  that padded-filter cropping is not silently conflated with filtering the core
  alone.
- Done: primary matching is elementwise exact equality plus independent byte
  hashes, with no tolerance, nearest-time matching, truncation, or top-N cap.
- Done: synthetic execution processed one 419,998x384 int16 window at a time,
  reported 645,116,928 cached input bytes, finished in 75.82 seconds, and left
  23,203 bytes of durable evidence.
- Not done: real H5 input identity and equality; those require the reviewed GO,
  an exact coordinator dispatch, fresh prestart resource/dedup reconciliation,
  and bounded H5 execution.
- Can establish: the frozen runner plumbing exactly reproduces the eager
  immediate-prewhitening producer on a full-shape noncircular synthetic fixture
  and preserves DARTsort's no-copy lazy-recording boundary.
- Cannot establish: real H5 equality, voltage quality, sorting benefit,
  biological identity, purity, transition correctness outside the selected
  request, or full-session performance.

## Reviewer focus

Check the exact window arithmetic, transition boundary placement, source/import
composition, one-read-per-window claim, release gates, positive-control
semantics, and COMPLETE-last failure behavior.  Preserve any defect as a
reviewed NO-GO rather than editing this packet in place.
