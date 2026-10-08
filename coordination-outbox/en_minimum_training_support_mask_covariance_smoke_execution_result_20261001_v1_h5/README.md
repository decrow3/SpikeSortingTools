# Minimum-training-support-mask covariance/W smoke: execution result

Exactly one start request was issued under final GO manifest `1024277e7ac865a87cf9505c2ac317e5559e9785300079bae00de7c974dabdec`. The reviewed smoke completed successfully and intentionally stopped at the post-whitening boundary before training or detection. No retry was issued.

The actual 384 x 384 covariance was finite, full rank (384), and had condition number 215.0203399658203 against the frozen maximum of 100,000,000. The resulting 384 x 384 whitening matrix was finite and its array digest is `752be427fce43138bb6a8fa043f79087d2f27a715c8544e07ab3d830b619c077`. Independent reload of both saved arrays reproduced their shape, dtype, finiteness, covariance rank/condition, and bound array hashes.

All 384 rows had covariance support (minimum 536,331 of 720,000 sampled interior columns). The 12 executed batches retained all 14 transition boundaries and masked 1,172,020 unsupported channel-sample values before whitening. The completed audited batch journal implies 554,084,352 logical input bytes, exactly the frozen read ceiling. The complete run root occupied 1,216,754 apparent bytes, below the 33,554,432-byte retained-artifact ceiling.

The unit is in its terminal `active/exited` state because `RemainAfterExit=yes`; `Result=success`, exit status 0, PID 0, and zero restarts. The Kilosort traceback records the deliberate `SmokeStopAfterWhitening` boundary and is not a run failure.

A systemd warning revealed that `ConditionPathNotExists` was ignored as an unknown unit key. This did not defeat freshness on this run because both the immediate check and the launcher's atomic `mkdir(exist_ok=False)` guard passed. The unit-level condition should be repaired and independently reviewed before any future launch; no repair or retry is part of this consumed GO.

No waveform QC ran concurrently, and no RF or holdout was accessed.

## Implementation checks

- Done: every final-GO, candidate, review, installed-preflight, prior-decision, and metadata-supplement manifest identity verified before launch.
- Done: installed config, service, candidate/helper/launcher/test sources, resolved Python, and five Kilosort module hashes matched the frozen chain (`PRESTART_CHECK.json`).
- Done: CUDA/RTX A5000 visibility, writable output parent, inactive/dead PID-0 unit, zero restarts, fresh paths, recording metadata, and absence of a competing smoke process checked immediately before launch (`PRESTART_CHECK.json`).
- Done: saved covariance and W independently reloaded and checked for shape, dtype, finiteness, array digest, rank, and condition (`RESULT_SUMMARY.json`; saved arrays in `run_evidence/`).
- Done: actual batch indices, support/state exposure, intentional stop boundary, terminal service result, resource ceilings, journal, and absence of failure markers checked.
- Not done: downstream training, detection, clustering, sorting, RF, holdout, or biological validation; the frozen boundary prohibited them.
- Can establish: this exact reviewed implementation produced a finite full-rank covariance and finite bound W on the frozen 12-batch schedule, within the frozen resource ceilings, and stopped before training/detection.
- Cannot establish: downstream sorter performance, biological identity/purity, long-window generalization beyond this covariance schedule, or correctness of any future run after changing the invalid unit condition.
