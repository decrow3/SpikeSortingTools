# Candidate-3 H1 CUDA visibility probe

H1 has a physical RTX A5000 and the 550.163.01 NVIDIA kernel modules are loaded, but the actual task environment exposes no `/dev/nvidia*` nodes. `CUDA_VISIBLE_DEVICES` is unset, `nvidia-smi` exits 9, and the exact DARTsort environment's CUDA 12.4 PyTorch build reports zero devices and `No CUDA GPUs are available`. This distinguishes a device-exposure/runtime-visibility problem from missing hardware or an absent host kernel driver. No reinstall or infrastructure mutation was attempted.

Candidate-3 source, boundary and checkpoint work can continue CPU-only on H1. A full run cannot be bound to the present H1 task environment. The concrete alternative is H5 after candidate 2 and only after refreshing its actual DARTsort environment, GPU, storage and managed-service state. The accepted Arm-A binary (241,309,358,592 bytes) already resides on H5, so that route needs no large cross-machine input movement. An optional float32 materialization of the pre-whitening boundary would itself be about 482.6 GB and must be explicitly budgeted or avoided with a reviewed context-correct reader.

Historical full DARTsort sort timing on comparable Luke full-session archives is 19,204–43,564 seconds (about 5.3–12.1 hours), and one local archive occupies about 104 GiB excluding a 481 GB float32 voltage cache. A provisional current planning envelope is therefore roughly 6–14 hours and at least 120 GB output headroom, plus 483 GB if a new float32 boundary cache is materialized. These are operational references, not a current H5 readiness result or launch promise. Candidate 2 and candidate 3 must not run concurrently on the same GPU.

Implementation checks
- Done: PCI hardware -> RTX A5000 present at `0000:65:00.0`.
- Done: driver -> NVIDIA 550.163.01 modules loaded.
- Done: task visibility -> no NVIDIA device nodes; `nvidia-smi` cannot communicate; visibility mask is unset.
- Done: actual DARTsort environment -> PyTorch 2.6.0+cu124, CUDA build 12.4, availability false, count zero, initialization fails.
- Done: input locality/storage/runtime references -> accepted A is H5-local; H1 REF+field are visible; no movement performed; historical timing/output sizes read from existing receipts/files.
- Not done: current H5 GPU/storage/service state -> refresh after candidate 2 before any contract/launch.
- Can establish: current H1 task environment cannot launch candidate 3 on GPU and H5 avoids a large accepted-input transfer.
- Cannot establish: H5 launch readiness, checkpoint repair, real-input equality, full-run duration or scientific benefit.
