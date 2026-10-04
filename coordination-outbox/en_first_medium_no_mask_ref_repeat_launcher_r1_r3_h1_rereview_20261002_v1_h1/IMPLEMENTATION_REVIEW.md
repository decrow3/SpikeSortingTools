# Implementation review

## Blocking finding

`execute_pair` obtains `final_output_bytes` by calling `enforce_output_cap`
before writing `repaired_B384_EXECUTION_STATE.json` and the pair
`COMPLETE.json`. It then updates `CURRENT_STAGE.json`; `run_single_attempt`
subsequently writes `ATTEMPT_COMPLETE.json`. These files are all inside the
three namespaces counted by `aggregate_output_bytes`, but no cap check covers
their final post-write state.

The independent fixture first measured the unaccounted terminal-receipt size,
then repeated with a cap between the pre-terminal and post-terminal totals. The
launcher returned success at a reported 1,777 bytes under a 1,843-byte cap,
while the same aggregate function measured 2,838 bytes after return.

## Other reviewed properties

- The exact candidate manifest and every listed member verify.
- The O_EXCL claim is created before contract, path, voltage, resource, or arm
  preflight. Supplied sequential and concurrent replay fixtures admit only one
  preflight and preserve the caught pre-root failure.
- Real arm commands use new process groups. The supplied live-crossing fixture
  terminates the group, and the second-arm crossing fixture preserves REF state
  without pair completion.
- The compact voltage helper hashes only bounded receipt/manifest material and
  the recording manifest, then uses `os.lstat` on the voltage path. Source and
  fixtures establish zero voltage-byte reads, same-size replacement rejection,
  symlink rejection, and execution before REF.
- Accepted scientific sources are byte-identical to the prior candidate. REF
  config changes only the launcher binding and disabled status text. Repaired-B
  config changes only source packet paths plus disabled reason/status text.
- The pair remains disabled, approval binding is null, and the service's
  required enabled environment binding is absent.

## Implementation checks

- Done: exact packet integrity -> candidate manifest and all members pass (`MANIFEST.sha256`; manifest SHA-256 `cefdbfeb9076513f7063dc0d5ac5fb4776d9fc915b408fd659f61e17ae67f1dc`).
- Done: executed changed launcher and voltage-preflight source review -> R1/R3 structure is present; R2 terminal receipts are outside the last cap observation (`source/en_first_medium_trained_pair_launch.py:505`, `:506`, `:516`, `:517`, `:555`).
- Done: clean isolated tests -> 10 passed in 0.41 s (`TEST_RECEIPT.txt`).
- Done: independent known-answer counterexample -> successful return above configured aggregate cap (`R2_FINAL_RECEIPT_CAP_FIXTURE.py`; `TEST_RECEIPT.txt`).
- Done: validate-only -> execution disabled, fixed REF-then-repaired order, no recording/voltage opened (`TEST_RECEIPT.txt`).
- Done: prior/new scientific delta -> accepted scientific source hashes unchanged; config differences are bindings/paths and disabled status text (`EXACT_DELTA.json` in candidate plus independent diffs).
- Not done: actual service installation/start or scientific arms -> prohibited by the review assignment and candidate remains disabled.
- Not done: actual compact preflight against the 241 GB voltage path -> not needed for this source/fixture re-review and the assignment prohibited voltage access; exact packet receipts and implementation were inspected instead.
- Can establish: the promoted repair passes the tested R1 and R3 scope, preserves the frozen scientific setup, and remains fail-closed for execution.
- Cannot establish: pair-wide 16 GiB enforcement through terminal completion receipts; therefore this packet does not authorize an enabled derivative or scientific execution.
