# Enabled sparse covariance/W smoke contract and H5 readiness

Status: **technically ready, unlaunched, pending H1 changed-delta review; no GO receipt**.

This immutable-candidate packet freezes a single prospective Kilosort 4.0.27
one-shot which may read only the reviewed 12-batch covariance schedule and must
stop after durable covariance validation and finite whitening `W`. It does not
authorize or report a recording read. The real service was neither installed,
enabled, nor started; the output root is absent and no reservation exists.

The earlier v5 disabled packet remains unchanged at
`en_minimum_training_support_mask_execution_readiness_20261001_v5_h5`.

## Exact launch boundary

- Config: `en_minimum_training_support_mask.covariance_smoke.enabled.review_pending.v1.json`
  (`c4c6944fc21a7f9fa21545b569ba15c4219aabcd7b59115f5acea2a16dccae8f`).
- Candidate source: `kilosort_support_mask_candidate.py`
  (`6de6d2740dceb7c1ba85c818a34806d8f8307e9e24de4d63cd2341b7c89a7f81`).
- Launcher: `en_minimum_training_support_mask_smoke_launch.py`
  (`59e4c776276e5fe751887bfedea2ac9378ff9bd3764778215b55e79a6bd0d60c`).
- Dormant service: `en_minimum_training_support_mask_covariance_smoke_enabled_v1.service`
  (`65fe6a3339066e9bfb71eb1b157f5cdd613bfbafcf69531f4c659511f51f622d`).
- Output root: `/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/en_minimum_training_support_mask_covariance_smoke_20261001_v1`.
- Logs: user journal for the exact service; durable C/W/audit receipts under the
  output root's `launch_evidence` child. Native transient files are isolated in
  `native_results`.
- Kilosort is **not resumable within this smoke**. Interruption before the
  validated post-W stop requires a clean restart in a new reviewed namespace.

The original `/media/huklaban5/Data/...` output choice was rejected during
preflight when that mount appeared read-only. Only prospective output paths
were moved; the recording identity, crop, clock, geometry, preprocessing, and
12-batch schedule are unchanged.

## Zero-voltage host preflight

The first dummy service failed before contract validation because a file-path
launcher lacked the repository on `sys.path`. Its failure was preserved in
`HOST_PREFLIGHT.json`; the launcher was repaired explicitly and received a new
hash. A fresh dummy unit then passed with the exact runtime and config:

- Python 3.12.4 in `environments/rescue-production/.venv`;
- Kilosort 4.0.27, NumPy 1.26.4, Torch 2.6.0+cu124, SpikeInterface 0.102.1;
- CUDA available, one NVIDIA RTX A5000, CUDA build 12.4;
- recording stat size 241,309,358,592 bytes, mode 0664, readable;
- no voltage content opened;
- output root absent;
- user-systemd dummy exited 0 with `RemainAfterExit=yes`, then was stopped and
  collected (`LoadState=not-found`).

The user manager reports `degraded` because of unrelated historical failed
units; the successful isolated lifecycle establishes that the manager can own,
retain, report, and collect this exact preflight command. It does not establish
future service survival across reboot or validate the real read path.

## Bounded prospective read

The frozen schedule is batch indices `0,25,...,275` (12 batches), 720,000
interior columns, 24.0001312141065 seconds, and at most 554,084,352 input bytes.
All 384 rows have metadata support (minimum 536,331; maximum 720,000); 14
transition boundaries are retained. The conservative retained-output ceiling is
33,554,432 bytes. A later executor must treat excess, missing/failing C audit,
nonfinite W, failure to stop after W, or any namespace reuse as failure.

No sort continuation, RF, holdout, waveform transfer, or scientific comparison
is included. Real covariance finite/rank/condition and W quality remain unknown.

## Required next decision

H1 should review the enabled scalar delta, conditional expected-map behavior,
new output binding, launcher import repair, and exact dormant service. If H1
denies, stop that action. If H1 approves, a separate normal tool approval is
still required before installing/starting the exact service. This packet is not
that approval and contains no GO receipt.

## Implementation checks

- Done: executed source/config hashes and full structural validator -> enabled
  contract accepted with exact reviewed base and 13 scalar deltas.
- Done: focused candidate/support suite -> 34/34 pass.
- Done: exact user-systemd preflight -> exit 0, CUDA visible, recording stat
  readable, `recording_content_opened=false`, run root absent; dummy collected.
- Done: real unit lookup and output checks -> unit not found, run root absent,
  no launch/reservation performed.
- Not done: voltage read, real C/W, covariance rank/condition, real-service
  disconnection survival, interruption behavior, sort, RF, or holdout -> outside
  this unlaunched review packet.
- Can establish: the frozen one-shot is syntactically and host-technically ready
  for independent review and a later explicitly approved bounded smoke.
- Cannot establish: H1 approval, launch authorization, real numerical validity,
  scientific benefit, or biological identity/purity.
