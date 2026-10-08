# H5 targeted implementation review: launcher R1-R3

Verdict: **READY FOR INDEPENDENT H1 CHANGED-PATH REVIEW; EXECUTION DISABLED**.

Only H1 findings R1-R3 changed. The historical no-mask REF runner, support-aware
trained entry, Kilosort sources, arm order, output clock/ancestry rules, and
scientific settings are byte-identical or exact-field equal to the original
candidate.

R1 consumes an immutable O_EXCL claim and fsyncs STARTED before validating the
contract, service, namespaces, GPU, free space, voltage continuity, or arm
inputs. All later failures write a terminal fsynced receipt with the exact last
stage. A second sequential or concurrent attempt cannot reach preflight.

R2 executes real arms in child process groups watched by a pair-wide byte
accountant covering the pair root, logs, and attempt evidence. It checks during
each child, after each arm, and before completion; reaching the threshold
terminates the process group. Crossing remains a terminal failure with partial
evidence and no pair COMPLETE.

R3 reuses the accepted voltage preflight byte-for-byte. It verifies both
full-hash packets and every current manifest member, the recording manifest,
non-symlink status, and exact device/inode/mode/size/mtime/ctime before REF.
The check reports zero voltage bytes read.

## Implementation checks

- Done: exact old/new source and 22-path contract delta recorded in `EXACT_DELTA.json`; all accepted non-launcher sources are byte-identical.
- Done: sequential and concurrent one-start replay fixtures -> exactly one preflight; durable exact-stage failure; no retry.
- Done: positive, first-arm, live-process, and cross-second-arm output-cap fixtures -> accepted completion never exceeds the aggregate cap; crossing terminates/fails without pair COMPLETE.
- Done: zero-read, same-size replacement, symlink, and ordering fixtures -> continuity is fail-closed and precedes REF.
- Done: actual disabled contract validate-only and actual compact voltage continuity passed; planned attempt/root namespaces remain absent.
- Not done: independent H1 review, enabled binding/environment file, service installation, detached survival proof, or scientific execution.
- Can establish: R1-R3 now have concrete fail-closed implementations and independent known-answer coverage while accepted arm semantics remain frozen.
- Cannot establish: execution GO, real pair completion, scientific benefit, waveform preservation, identity/purity, or evaluator advancement.
