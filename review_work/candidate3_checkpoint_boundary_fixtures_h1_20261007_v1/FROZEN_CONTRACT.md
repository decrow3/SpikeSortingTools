# Candidate-3 checkpoint and boundary fixtures — H1 v1

Status: `FROZEN_CPU_ONLY_SYNTHETIC_ACTUAL_SOURCE_PATH`

## Decision

- Milestone: remove the cheapest operational blockers before any real snippet or full-session candidate-3 work.
- Decision changed: determine whether the current DARTsort within-peel checkpoint can safely resume after an interrupted append, and whether the exact Kilosort immediate-pre-whitening float output can traverse DARTsort with `preprocessing="none"` without numerical change.
- Cheapest adequate test: independent temporary HDF5 fault injection through the imported `BasePeeler.gather_chunk_result`/`check_resuming` methods, plus synthetic ordinary and transition batches through installed Kilosort `BinaryFiltered.filter` and DARTsort `preprocess(..., "none")`.
- Completion: preserved positive/negative fixture receipts and a precise repair/restart requirement; no real voltage and no sorter launch.

## Exact inputs

- Candidate inventory packet MANIFEST `0370b607b3240fb4da6b31ba1defe4907e6956a47539c9f22e6efe6f74074f7e`, COMPLETE `cfda092933eef6484fa6ea6c05e734f8fab0b48b69011a115b3f1578edc9c05a`.
- DARTsort Git `edcfe1b51d672b4136eb13cc78c0875da804b851`, `peel_base.py` SHA-256 `ac357ade5b6f23b690a403e297a2260ea93c2a1b35df13ee7798d8fc8591a2d4`, `preprocess_util.py` SHA-256 `566ce1ab60af12585978bcd7e3192cb2ca4cc220ab5e41895140c00bdbc06674`.
- Installed Kilosort 4.0.27 `io.py` SHA-256 `767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd`.
- Policy/plan SHA-256: `34414087180e5ba50812ae453c7cc2fd672e7c7b21e2f665d8a42f7a3e98cb08` / `39f55c319ca13b392fb04237829540d72aec6719336c2d43f0d67ab3a99684f8`.

## Frozen predictions and acceptance

1. Fault after chunk marker advancement but before spike append must not be treated as a completed chunk. If current source reports the next chunk, checkpoint safety fails and full-session interruption must not use that state without repair.
2. A fully appended chunk must resume at the following chunk, while changed chunk starts must reject.
3. For finite synthetic ordinary and transition batches, DARTsort `preprocessing="none"` must return the exact same float32 samples produced by the installed Kilosort pre-whitening source path (array equality, zero max absolute error), with unchanged channel order and geometry.
4. DARTsort must reject `preprocessing="none"` on the original integer input, proving the guard is active.

## Bounds and stop

2 CPU, 4 GiB RSS, 10 minutes, under 50 MiB temporary data in `/tmp`, under 5 MiB durable output. No raw/real voltage, scientific HDF5/NPZ arrays, GPU, fit, training, sort, RF/holdout, network, or production namespace access. Preserve a failing fixture; do not tune the rule to pass. Stop after the frozen fixtures and publish the scoped result.
