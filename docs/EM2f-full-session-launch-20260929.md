# EM.2f full-session rounded-kriging launch

The authorized full-session imec1 rounded-kriging run launched successfully on
2026-09-29 under user systemd service `em2b-3d54ab4c99beb6ff.service`.

- Frozen preparation commit: `28d6b4a`
- Output: `/home/huklaban5/DARTsort_experiment_scratch/em2f_full_20260929/rounded_kriging_v1`
- Interpreter: `/home/huklaban5/Documents/DARTsort/.venv/bin/python`
- SpikeInterface: 0.104.8
- GPU: NVIDIA RTX A5000, CUDA 12.4
- Frames/channels/dtype: 314,204,094 / 182 / float32
- Service runtime cap: 58,080 s
- Scratch available before launch: 1,103 GiB
- Required minimum / in-run reserve: 734 GiB / 450 GiB

The no-voltage preflight service `em2b-c58ea41179888069.service` exited with
result `success` and status 0 after loading the exact recording signature. The
sort service remained `active/running` with `MainPID=1767166` after the launcher
exited, establishing launcher-independent liveness for this run. The exact
manager commands are retained in `preflight-dispatch.json` and
`launch-dispatch.json` under the output path.

Post-disconnection inspection found the preprocessing worker live at PID
1767245 and its first durable checkpoint committed: 16 of 10,474 aligned chunks.
The cache has an expected logical size of 228,740,580,432 bytes; the sparse file
had consumed 424 MiB at that checkpoint.

The run initially entered `preprocess`. Its binary cache writer records a
durable completed prefix of aligned one-second chunks. If interrupted, that
prefix may be resumed after receipt validation. Detection and sorting lack
within-stage checkpoints; interruption during either requires preserving the
failure evidence and restarting that stage. No RF analysis is part of this
run. The frozen lighthouse identity packet remains unavailable as documented in
`docs/EM2f-full-session-readiness-20260929.md`; population scorecards and event
overlap will remain scoped separately from biological identity.

This document records a verified launch, not a completed scientific result.
Completion requires checking the actual systemd state, final receipt, sorting
artifact, QC output, resource use, and the frozen comparison analyses.
