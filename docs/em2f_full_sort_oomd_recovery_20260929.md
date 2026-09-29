# EM.2f full-sort recovery after systemd-oomd termination

The original `em2b-3d54ab4c99beb6ff.service` completed preprocessing,
detection, and the static-motion stage. At 07:50:36 PDT on 2026-09-29,
`systemd-oomd` killed its 15 processes after sustained pressure in the user
slice. The service exited by SIGKILL; it did not reach a pipeline receipt or
write a pipeline failure receipt. This was neither a DARTsort exception nor the
configured runtime limit.

The partial sort is safe to resume. Read-only inspection of
`sort/matching1.h5` found all 10,474 chunks through index 10,473, with the final
chunk start equal to the checkpoint value. Each of the 13 appendable event
datasets has 21,822,563 rows. The matching models are also present. No final
`dartsort_sorting.npz` exists. With `save_intermediates=false`, the frozen
DARTsort code starts from the subtraction result, recognizes the completed
matching peeler checkpoint, skips matching, and reruns clustering/refinement
before finalization.

`testing/em2f_resume_after_oomd.py` refuses changed config, DARTsort source,
runner source, incomplete matching, missing stage markers, and terminal output.
It retains the scientific config and partial artifacts, uses a new service
identity, and records the exact recovery command and checkpoint audit. The
recovery service adds the previously missing resource controls:

- `ManagedOOMPreference=avoid`
- `MemoryHigh=128G` and `MemoryMax=160G`
- `CPUQuota=400%`, `TasksMax=128`, and `KillMode=control-group`
- durable stdout/stderr and a 25,800-second outer limit

An interruption during the remaining clustering/refinement work can reuse the
complete matching checkpoint, but clustering itself has no finer checkpoint
and must restart.

The first guarded dispatch exited before DARTsort because the frozen pipeline's
direct-publication worker refuses to copy link inputs over files that it had
already copied during the original attempt. This protected the files as
designed. The recovery worker now validates the original copy receipt and the
source/destination sizes, reuses those inputs, and then invokes the unchanged
frozen DARTsort stage. Its failure receipt and logs are retained in the v1
recovery job directory.

The v2 dispatch likewise exited before DARTsort because importing the frozen
pipeline as a library did not initially expose its sibling `stage_validation`
module. The wrapper now adds the frozen source directory to its import path.
The v2 logs are retained separately, and the matching checkpoint was unchanged.
